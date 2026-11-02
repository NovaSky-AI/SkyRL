"""``patch_moe_dedup_alltoall.py``: the deduplicating all-to-all dispatcher vs megatron-core's.

Under torchrun, every rank builds a ``MoEAlltoAllTokenDispatcher`` over an expert-parallel group of all
ranks and routes random top-k tokens through a fake expert that scales each row by
``(global expert id + 1) * prob``. The combined output is then ``sum_e prob[t, e] * (e + 1) * x[t]``
over the routed experts, with gradients of the same closed form. The stock dispatcher runs first, then
the patched one in the same process; both must match the closed form, and each other, in values and in
the gradients of the hidden states and the routing probabilities.
"""

import os
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist

from skyrl.train.utils.utils import get_free_port

TOKENS, HIDDEN, EXPERTS_PER_RANK, TOPK = 2048, 256, 4, 6


def _route(dispatcher, hidden, routing_map, probs, ep_rank):
    """One MoE dispatch -> fake expert -> combine pass, in the MoE layer's call order."""
    tokens, permuted_probs = dispatcher.dispatch_preprocess(hidden, routing_map, probs)
    tokens, permuted_probs = dispatcher.token_dispatch(tokens, permuted_probs)
    expert_in, tokens_per_expert, expert_probs = dispatcher.dispatch_postprocess(tokens, permuted_probs)
    expert_ids = torch.repeat_interleave(
        torch.arange(EXPERTS_PER_RANK, device=hidden.device) + ep_rank * EXPERTS_PER_RANK,
        tokens_per_expert.to(hidden.device),
    )
    scale = (expert_ids.to(torch.float32) + 1) * expert_probs.reshape(-1).to(torch.float32)
    expert_out = (expert_in.to(torch.float32) * scale[:, None]).to(hidden.dtype)
    out = dispatcher.combine_preprocess(expert_out)
    out = dispatcher.token_combine(out)
    return dispatcher.combine_postprocess(out)


def _run(dispatcher, x, routing_map, probs, grad, ep_rank):
    x = x.clone().requires_grad_()
    p = probs.clone().requires_grad_()
    out = _route(dispatcher, x, routing_map, p, ep_rank)
    (out.float() * grad).sum().backward()
    return out.float(), x.grad.float(), p.grad.float()


def _distributed_main() -> None:
    from megatron.core import parallel_state
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
    from megatron.core.transformer.transformer_config import TransformerConfig

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=world)
    num_experts = EXPERTS_PER_RANK * world
    config = TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        ffn_hidden_size=4 * HIDDEN,
        moe_ffn_hidden_size=HIDDEN,
        num_moe_experts=num_experts,
        moe_router_topk=TOPK,
        moe_token_dispatcher_type="alltoall",
        moe_permute_fusion=True,
        expert_model_parallel_size=world,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
    )
    pg = ProcessGroupCollection.use_mpu_process_groups()
    local = list(range(rank * EXPERTS_PER_RANK, (rank + 1) * EXPERTS_PER_RANK))

    torch.manual_seed(1234 + rank)
    x = torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16)
    top = torch.randn(TOKENS, num_experts, device="cuda").topk(TOPK, dim=-1).indices
    routing_map = torch.zeros(TOKENS, num_experts, dtype=torch.bool, device="cuda").scatter_(1, top, True)
    probs = torch.rand(TOKENS, num_experts, device="cuda") * routing_map
    grad = torch.randn(TOKENS, HIDDEN, device="cuda")

    # Closed form: out[t] = sum_e prob[t, e] (e + 1) x[t]; d out / d x and d out / d prob follow.
    weight = (probs * (torch.arange(num_experts, device="cuda") + 1)).sum(-1, keepdim=True)
    ref_out = x.float() * weight
    ref_dx = grad * weight
    ref_dp = ((grad * x.float()).sum(-1, keepdim=True) * (torch.arange(num_experts, device="cuda") + 1)) * routing_map

    stock = _run(MoEAlltoAllTokenDispatcher(EXPERTS_PER_RANK, local, config, pg), x, routing_map, probs, grad, rank)
    os.environ["SKYRL_MOE_DEDUP_A2A"] = "1"
    from skyrl.backends.skyrl_train.patches.megatron.patch_moe_dedup_alltoall import (
        apply_moe_dedup_alltoall_patch,
    )

    apply_moe_dedup_alltoall_patch()
    dispatcher = MoEAlltoAllTokenDispatcher(EXPERTS_PER_RANK, local, config, pg)
    dedup = _run(dispatcher, x, routing_map, probs, grad, rank)
    used_dedup = getattr(dispatcher, "_skyrl_dedup", False)

    def close(a, b):
        return bool(((a - b).norm() / b.norm().clamp_min(1e-6)) < 2e-2)

    ok = used_dedup
    for got_stock, got_dedup, ref in zip(stock, dedup, (ref_out, ref_dx, ref_dp)):
        ok = ok and close(got_stock, ref) and close(got_dedup, ref) and close(got_dedup, got_stock)
    flag = torch.tensor([1.0 if ok else 0.0], device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if rank == 0:
        print(f"RESULT: {'PASS' if flag.item() == 1.0 else 'FAIL'} (EP={world}, dedup path taken: {used_dedup})")
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


@pytest.mark.parametrize("nproc", [2, 4])
def test_moe_dedup_alltoall_matches_stock(nproc):
    """Spawn torchrun --nproc_per_node=nproc and compare the dispatchers on every rank."""
    if torch.cuda.device_count() < nproc:
        pytest.skip(f"needs >= {nproc} GPUs, have {torch.cuda.device_count()}")
    env = dict(os.environ, MASTER_ADDR="localhost", MASTER_PORT=str(get_free_port()))
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        "--master_port",
        env["MASTER_PORT"],
        __file__,
    ]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    assert "RESULT: PASS" in (res.stdout + res.stderr), f"EP={nproc} failed:\n{res.stdout}\n{res.stderr}"


if __name__ == "__main__":
    _distributed_main()

"""Four-rank check: the node-deduplicated MoE all-to-all matches megatron-core's dispatcher.

EP2 x ETP2 on four GPUs split into two pseudo-nodes of two (``SKYRL_MOE_NODE_SIZE=2``), so each
node hosts one EP rank and every token with experts on both nodes crosses the "inter-node" hop.
Random top-4 routing over 8 experts; a stand-in expert scales each row by its routing probability
and a per-expert constant. The dispatched layout (rows per expert), the MoE output and the
gradients of the hidden states and probabilities must match the stock dispatcher (FP32
activations, so only summation order can differ).

Run with:
uv run --isolated --extra dev --extra megatron pytest -s \
    tests/backends/skyrl_train/gpu/gpu_ci/patches/megatron/test_moe_node_dedup_dispatch.py
"""

import pytest
import ray
import torch
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from skyrl.train.utils import get_ray_pg_ready_with_timeout

pytestmark = pytest.mark.megatron

_WORLD = 4
NUM_EXPERTS, TOPK, HIDDEN, TOKENS = 8, 4, 64, 257


@ray.remote(num_gpus=1)
class _Worker:
    def endpoint(self):
        import socket

        from ray.util import get_node_ip_address

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            return get_node_ip_address(), sock.getsockname()[1]

    def run(self, rank, master_addr, master_port):
        import os

        os.environ["SKYRL_MOE_NODE_SIZE"] = "2"
        import torch.distributed as dist
        from megatron.core import parallel_state as mpu
        from megatron.core.process_groups_config import ProcessGroupCollection
        from megatron.core.transformer import TransformerConfig
        from megatron.core.transformer.moe.token_dispatcher import (
            MoEAlltoAllTokenDispatcher,
        )

        from skyrl.backends.skyrl_train.patches.megatron import (
            patch_moe_node_dedup_dispatch as mod,
        )

        torch.cuda.set_device(0)
        dist.init_process_group("nccl", init_method=f"tcp://{master_addr}:{master_port}", rank=rank, world_size=_WORLD)
        try:
            mpu.initialize_model_parallel(expert_model_parallel_size=2, expert_tensor_parallel_size=2)
            cfg = TransformerConfig(
                num_layers=1,
                hidden_size=HIDDEN,
                num_attention_heads=4,
                num_moe_experts=NUM_EXPERTS,
                moe_router_topk=TOPK,
                moe_token_dispatcher_type="alltoall",
                expert_model_parallel_size=2,
                expert_tensor_parallel_size=2,
                add_bias_linear=False,
            )
            pgs = ProcessGroupCollection.use_mpu_process_groups()
            ep_rank = mpu.get_expert_model_parallel_rank()
            num_local = NUM_EXPERTS // 2
            local_ids = list(range(ep_rank * num_local, (ep_rank + 1) * num_local))
            scale = torch.arange(1, NUM_EXPERTS + 1, device="cuda", dtype=torch.float32)

            gen = torch.Generator(device="cuda").manual_seed(100 + rank)  # different tokens per rank
            x0 = torch.randn(TOKENS, HIDDEN, device="cuda", generator=gen)
            logits = torch.randn(TOKENS, NUM_EXPERTS, device="cuda", generator=gen)
            top = logits.topk(TOPK, dim=-1).indices
            routing_map = torch.zeros(TOKENS, NUM_EXPERTS, dtype=torch.bool, device="cuda")
            routing_map.scatter_(1, top, True)
            p0 = torch.rand(TOKENS, NUM_EXPERTS, device="cuda", generator=gen) * routing_map

            stock = {
                k: getattr(MoEAlltoAllTokenDispatcher, k)
                for k in ("dispatch_preprocess", "token_dispatch", "token_combine", "combine_postprocess")
            }

            def run_once():
                d = MoEAlltoAllTokenDispatcher(num_local, local_ids, config=cfg, pg_collection=pgs)
                x = x0.clone().requires_grad_(True)
                p = p0.clone().requires_grad_(True)
                h, pr = d.dispatch_preprocess(x, routing_map, p)
                h, pr = d.token_dispatch(h, pr)
                h, tokens_per_expert, pr = d.dispatch_postprocess(h, pr)
                expert_scale = torch.repeat_interleave(scale[local_ids], tokens_per_expert.to(h.device))
                y = h * (pr * expert_scale).unsqueeze(-1)
                y = d.combine_preprocess(y)
                y = d.token_combine(y)
                out = d.combine_postprocess(y)
                (out * torch.linspace(-1, 1, HIDDEN, device="cuda")).sum().backward()
                return out.detach(), x.grad, p.grad, tokens_per_expert.tolist()

            ref = run_once()
            mod._APPLIED = False
            os.environ[mod._ENV] = "1"
            assert mod.patch_moe_node_dedup_dispatch()
            try:
                new = run_once()
            finally:
                for k, v in stock.items():
                    setattr(MoEAlltoAllTokenDispatcher, k, v)
                mod._APPLIED = False

            def rel(a, b):
                return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()

            return {
                "tokens_per_expert_equal": ref[3] == new[3],
                "out": rel(new[0], ref[0]),
                "dx": rel(new[1], ref[1]),
                "dprobs": rel(new[2], ref[2]),
            }
        finally:
            dist.destroy_process_group()


def test_node_dedup_matches_stock_dispatcher(ray_init_fixture):
    pg = placement_group([{"GPU": _WORLD, "CPU": _WORLD}], strategy="PACK")
    get_ray_pg_ready_with_timeout(pg, timeout=30)
    strategy = PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=0)
    workers = [_Worker.options(scheduling_strategy=strategy).remote() for _ in range(_WORLD)]
    master_addr, master_port = ray.get(workers[0].endpoint.remote())
    results = ray.get([w.run.remote(r, master_addr, master_port) for r, w in enumerate(workers)])
    for rank, res in enumerate(results):
        print(f"rank {rank}: {res}")
        assert res["tokens_per_expert_equal"], (rank, res)
        for k in ("out", "dx", "dprobs"):
            assert res[k] < 1e-5, (rank, k, res)

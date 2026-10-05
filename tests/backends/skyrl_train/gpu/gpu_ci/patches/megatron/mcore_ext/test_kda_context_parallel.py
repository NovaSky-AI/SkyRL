"""Two-rank check: head-wise context-parallel ``KimiDeltaAttention`` matches the CP=1 module.

Both ranks build the same KDA layer twice: once on the 2-rank CP group, once on a single-rank CP
group (the reference, which sees the whole packed batch). The CP module gets each sequence split
into ``2 * cp`` chunks in megatron's load-balanced order (rank r holds chunks r and 2cp-1-r), as
``preprocess_packed_seqs`` lays it out. Outputs and input gradients must match the reference rows,
and parameter gradients summed over the CP group must match the reference's.

Run with:
uv run --isolated --extra dev --extra megatron pytest -s \
    tests/backends/skyrl_train/gpu/gpu_ci/patches/megatron/mcore_ext/test_kda_context_parallel.py
"""

import pytest
import ray
import torch
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from skyrl.train.utils import get_ray_pg_ready_with_timeout

pytestmark = pytest.mark.megatron

_CP = 2
HIDDEN, HEADS, HEAD_DIM, KERNEL = 128, 4, 32, 4
SEQ_LENS = [64, 100]  # each divisible by 2 * cp


def _cp_local_rows(seq_lens, cp_size, cp_rank):
    """Global packed-row indices held by ``cp_rank`` (per sequence: chunks r and 2cp-1-r)."""
    rows, start = [], 0
    for n in seq_lens:
        chunk = n // (2 * cp_size)
        for c in (cp_rank, 2 * cp_size - 1 - cp_rank):
            rows.append(torch.arange(start + c * chunk, start + (c + 1) * chunk))
        start += n
    return torch.cat(rows)


@ray.remote(num_gpus=1)
class _Worker:
    def endpoint(self):
        import socket

        from ray.util import get_node_ip_address

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            return get_node_ip_address(), sock.getsockname()[1]

    def run(self, rank, master_addr, master_port, exchange):
        import torch.distributed as dist
        from megatron.core import parallel_state as mpu
        from megatron.core import tensor_parallel
        from megatron.core.extensions.transformer_engine_spec_provider import (
            TESpecProvider,
        )
        from megatron.core.packed_seq_params import PackedSeqParams
        from megatron.core.process_groups_config import ProcessGroupCollection
        from megatron.core.transformer import TransformerConfig
        from megatron.core.transformer.spec_utils import build_module

        import skyrl.backends.skyrl_train.workers.megatron  # noqa: F401  (FA4 import guard)
        from skyrl.backends.skyrl_train.patches.megatron.glm5_next.layer_specs import (
            get_kda_module_spec,
        )
        from skyrl.backends.skyrl_train.patches.megatron.mcore_ext import kda

        kda._KDA_CP_EXCHANGE = exchange

        torch.cuda.set_device(0)
        dist.init_process_group("nccl", init_method=f"tcp://{master_addr}:{master_port}", rank=rank, world_size=_CP)
        try:
            mpu.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=_CP)
            tensor_parallel.model_parallel_cuda_manual_seed(0)
            single = [dist.new_group([r]) for r in range(_CP)][rank]

            cfg = TransformerConfig(
                num_layers=1,
                hidden_size=HIDDEN,
                num_attention_heads=4,
                num_query_groups=4,
                linear_num_value_heads=HEADS,
                linear_num_key_heads=HEADS,
                linear_key_head_dim=HEAD_DIM,
                linear_value_head_dim=HEAD_DIM,
                linear_conv_kernel_dim=KERNEL,
                layernorm_epsilon=1e-5,
                add_bias_linear=False,
                bf16=True,
                params_dtype=torch.bfloat16,
                gradient_accumulation_fusion=False,
                sequence_parallel=False,
                context_parallel_size=_CP,
            )
            cfg.kda_gate_lower_bound = -5.0
            spec = get_kda_module_spec(TESpecProvider())
            tp = mpu.get_tensor_model_parallel_group()
            ref = build_module(spec, config=cfg, layer_number=1, pg_collection=ProcessGroupCollection(tp=tp, cp=single))
            mod = build_module(
                spec,
                config=cfg,
                layer_number=1,
                pg_collection=ProcessGroupCollection(tp=tp, cp=mpu.get_context_parallel_group()),
            )
            assert ref.cp_size == 1 and mod.cp_size == _CP

            torch.manual_seed(0)  # identical weights and inputs on both ranks
            with torch.no_grad():
                for name, p in ref.named_parameters():
                    if name == "A_log":
                        p.copy_(torch.empty_like(p).uniform_(1, 16).log())
                    elif name == "dt_bias":
                        p.uniform_(-3.0, -1.0)
                    elif "o_norm" in name:
                        p.uniform_(0.8, 1.2)
                    else:
                        p.normal_(0, 0.3 if "conv1d" in name else 0.05)
            mod.load_state_dict(ref.state_dict())

            total = sum(SEQ_LENS)
            x = torch.randn(total, 1, HIDDEN, device="cuda", dtype=torch.bfloat16)
            out_weight = torch.randn(total, 1, HIDDEN, device="cuda", dtype=torch.float32)
            cu = torch.tensor([0] + SEQ_LENS, device="cuda", dtype=torch.int32).cumsum(0).to(torch.int32)
            params = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu,
                cu_seqlens_kv=cu,
                cu_seqlens_q_padded=cu,
                cu_seqlens_kv_padded=cu,
                max_seqlen_q=max(SEQ_LENS),
                max_seqlen_kv=max(SEQ_LENS),
            )

            x_ref = x.clone().requires_grad_(True)
            y_ref, _ = ref(x_ref, packed_seq_params=params)
            (y_ref.float() * out_weight).sum().backward()

            rows = _cp_local_rows(SEQ_LENS, _CP, rank).cuda()
            x_cp = x[rows].clone().requires_grad_(True)
            y_cp, _ = mod(x_cp, packed_seq_params=params)
            (y_cp.float() * out_weight[rows]).sum().backward()

            def rel(a, b):
                return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()

            errs = {"out": rel(y_cp, y_ref[rows]), "dx": rel(x_cp.grad, x_ref.grad[rows])}
            ref_grads = dict(ref.named_parameters())
            for name, p in mod.named_parameters():
                g = p.grad.float().clone()
                dist.all_reduce(g, group=mpu.get_context_parallel_group())
                errs[f"grad.{name}"] = rel(g, ref_grads[name].grad)
            return errs
        finally:
            dist.destroy_process_group()


@pytest.mark.parametrize("exchange", ["a2a", "allgather"])
def test_kda_context_parallel_matches_cp1(ray_init_fixture, exchange):
    pg = placement_group([{"GPU": _CP, "CPU": _CP}], strategy="PACK")
    get_ray_pg_ready_with_timeout(pg, timeout=30)
    strategy = PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=0)
    workers = [_Worker.options(scheduling_strategy=strategy).remote() for _ in range(_CP)]
    master_addr, master_port = ray.get(workers[0].endpoint.remote())
    results = ray.get([w.run.remote(r, master_addr, master_port, exchange) for r, w in enumerate(workers)])
    for rank, errs in enumerate(results):
        print(f"{exchange} rank {rank}: " + ", ".join(f"{k}={v:.2e}" for k, v in errs.items()))
        # bf16 activations; CP changes only the all-to-all layout and per-shard reduction order.
        bad = {k: v for k, v in errs.items() if v > 2e-2}
        assert not bad, (rank, bad)

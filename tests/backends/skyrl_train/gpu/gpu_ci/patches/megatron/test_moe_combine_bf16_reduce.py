"""``patch_moe_combine_bf16_reduce``: skipping the FP32 upcast before a 2-rank reduce-scatter is exact.

1. The premise, on two GPUs: a bf16 reduce-scatter equals reduce-scatter-in-FP32-then-cast-to-bf16,
   bit for bit, for values spanning several orders of magnitude.
2. The wrapper: with a 2-rank expert-TP group, ``combine_preprocess`` sees ``probs`` in the
   activation dtype (so megatron-core's ``.to(self.probs.dtype)`` is a no-op) and ``probs`` is
   restored afterwards; any other group size, or an already-bf16 router, passes through untouched.

Run with:
uv run --isolated --extra dev --extra megatron pytest -s \
    tests/backends/skyrl_train/gpu/gpu_ci/patches/megatron/test_moe_combine_bf16_reduce.py
"""

import pytest
import ray
import torch
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from skyrl.train.utils import get_ray_pg_ready_with_timeout

pytestmark = pytest.mark.megatron

_WORLD = 2


@ray.remote(num_gpus=1)
class _Worker:
    def endpoint(self):
        import socket

        from ray.util import get_node_ip_address

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            return get_node_ip_address(), sock.getsockname()[1]

    def run(self, rank, master_addr, master_port):
        import torch.distributed as dist

        torch.cuda.set_device(0)
        dist.init_process_group("nccl", init_method=f"tcp://{master_addr}:{master_port}", rank=rank, world_size=_WORLD)
        try:
            gen = torch.Generator(device="cuda").manual_seed(1234 + rank)
            scale = torch.logspace(-3, 2, 64, device="cuda")
            x = (torch.randn(2 * 200_000, 64, device="cuda", generator=gen) * scale).bfloat16()
            ref = torch.empty(x.shape[0] // _WORLD, 64, device="cuda", dtype=torch.float32)
            dist.reduce_scatter_tensor(ref, x.float())
            out = torch.empty(x.shape[0] // _WORLD, 64, device="cuda", dtype=torch.bfloat16)
            dist.reduce_scatter_tensor(out, x)
            return torch.equal(out, ref.bfloat16())
        finally:
            dist.destroy_process_group()


def test_bf16_reduce_scatter_matches_fp32_for_two_ranks(ray_init_fixture):
    pg = placement_group([{"GPU": _WORLD, "CPU": _WORLD}], strategy="PACK")
    get_ray_pg_ready_with_timeout(pg, timeout=30)
    strategy = PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=0)
    workers = [_Worker.options(scheduling_strategy=strategy).remote() for _ in range(_WORLD)]
    master_addr, master_port = ray.get(workers[0].endpoint.remote())
    assert all(ray.get([w.run.remote(r, master_addr, master_port) for r, w in enumerate(workers)]))


def test_wrapper_swaps_probs_dtype_only_for_two_rank_groups(monkeypatch):
    from megatron.core.transformer.moe.token_dispatcher import (
        MoEAlltoAllTokenDispatcher,
    )

    from skyrl.backends.skyrl_train.patches.megatron import (
        patch_moe_combine_bf16_reduce as mod,
    )

    seen = []

    def fake_inner(self, hidden_states):
        seen.append(self.probs.dtype)
        return hidden_states.to(self.probs.dtype)

    monkeypatch.setattr(MoEAlltoAllTokenDispatcher, "combine_preprocess", fake_inner)
    monkeypatch.setattr(mod, "_APPLIED", False)
    assert mod.patch_moe_combine_bf16_reduce()

    dispatcher = object.__new__(MoEAlltoAllTokenDispatcher)
    hidden = torch.randn(8, 4).bfloat16()
    probs = torch.rand(8, 3)
    for tp_size, expected in [(2, torch.bfloat16), (1, torch.float32), (4, torch.float32)]:
        dispatcher.tp_size = tp_size
        dispatcher.probs = probs
        out = MoEAlltoAllTokenDispatcher.combine_preprocess(dispatcher, hidden)
        assert seen[-1] == expected and out.dtype == expected, (tp_size, seen[-1], out.dtype)
        assert dispatcher.probs is probs  # restored

    # A bf16 router has nothing to skip.
    dispatcher.tp_size, dispatcher.probs = 2, probs.bfloat16()
    MoEAlltoAllTokenDispatcher.combine_preprocess(dispatcher, hidden)
    assert seen[-1] == torch.bfloat16

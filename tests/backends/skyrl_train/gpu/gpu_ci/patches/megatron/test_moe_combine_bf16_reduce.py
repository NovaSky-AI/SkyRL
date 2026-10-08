"""``patch_moe_combine_bf16_reduce``: skipping the FP32 upcast before a 2-rank reduce-scatter is exact.

1. The premise, on two GPUs: a bf16 reduce-scatter equals reduce-scatter-in-FP32-then-cast-to-bf16,
   bit for bit, for values spanning several orders of magnitude, with NCCL's default algorithm
   (Ring for two ranks). A forced ``NCCL_ALGO=NVLS`` does not satisfy this (<= 1 bf16 ulp off),
   which is why the patch keeps the upcast then.
2. The patch: with a 2-rank expert-TP group ``combine_preprocess`` reduce-scatters in the activation
   dtype, and in the router dtype for any other group size; the installed function differs from
   megatron-core's only in that ``.to(...)``; and the patch refuses to install over a
   ``combine_preprocess`` that no longer matches the copy it was written against.

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


def _code_lines(fn):
    """A function's statements, normalized, without its docstring or local imports."""
    import ast
    import inspect
    import textwrap

    node = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
    body = [n for n in node.body if not isinstance(n, ast.ImportFrom)]
    body = [n for n in body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    return "\n".join(ast.unparse(n) for n in body).splitlines()


def test_reduce_scatter_dtype_by_group_size(monkeypatch):
    from megatron.core.transformer.moe import token_dispatcher as td

    from skyrl.backends.skyrl_train.patches.megatron import (
        patch_moe_combine_bf16_reduce as mod,
    )

    cls = td.MoEAlltoAllTokenDispatcher
    monkeypatch.delenv("NCCL_ALGO", raising=False)
    monkeypatch.setattr(cls, "combine_preprocess", cls.combine_preprocess)  # restored at teardown
    monkeypatch.setattr(mod, "_APPLIED", False)
    seen = []

    def fake_reduce_scatter(x, group, input_split_sizes):
        seen.append(x.dtype)
        return x

    monkeypatch.setattr(td, "reduce_scatter_to_sequence_parallel_region", fake_reduce_scatter)
    try:
        assert mod.patch_moe_combine_bf16_reduce()
        dispatcher = object.__new__(cls)
        dispatcher.num_local_experts, dispatcher.output_splits_tp, dispatcher.tp_group = 1, None, None
        hidden = torch.randn(8, 4).bfloat16()
        cases = [
            (2, torch.float32, torch.bfloat16),
            (4, torch.float32, torch.float32),
            (2, torch.bfloat16, torch.bfloat16),
        ]
        for tp_size, probs_dtype, expected in cases:
            dispatcher.tp_size, dispatcher.probs = tp_size, torch.rand(8, 3).to(probs_dtype)
            out = cls.combine_preprocess(dispatcher, hidden)
            assert seen[-1] == expected, (tp_size, probs_dtype, seen[-1])
            assert out.dtype == torch.bfloat16
        dispatcher.tp_size = 1
        cls.combine_preprocess(dispatcher, hidden)
        assert len(seen) == len(cases), "tp_size=1 must not reduce-scatter"
        # NVLS reduces in the switch and is not bitwise-equal to the FP32 path: keep the upcast.
        dispatcher.tp_size, dispatcher.probs = 2, torch.rand(8, 3)
        for algo, expected in [
            ("NVLS", torch.float32),
            ("reducescatter:nvls", torch.float32),
            ("^NVLS", torch.bfloat16),
            ("Ring", torch.bfloat16),
        ]:
            monkeypatch.setenv("NCCL_ALGO", algo)
            cls.combine_preprocess(dispatcher, hidden)
            assert seen[-1] == expected, (algo, seen[-1])
    finally:
        if "_tp_reduce_dtype" in cls.__dict__:  # added by the patch
            del cls._tp_reduce_dtype


def test_patched_function_differs_from_megatron_only_in_the_reduce_dtype():
    import difflib

    from megatron.core.transformer.moe.token_dispatcher import (
        MoEAlltoAllTokenDispatcher,
    )

    from skyrl.backends.skyrl_train.patches.megatron import (
        patch_moe_combine_bf16_reduce as mod,
    )

    upstream = MoEAlltoAllTokenDispatcher.combine_preprocess
    assert mod._normalized_source(upstream) == mod._PINNED_COMBINE_PREPROCESS.strip()
    diff = [
        line
        for line in difflib.unified_diff(_code_lines(upstream), _code_lines(mod._combine_preprocess), lineterm="", n=0)
        if line[:1] in "+-" and not line.startswith(("+++", "---"))
    ]
    assert len(diff) == 2, diff
    assert "self.probs.dtype" in diff[0] and "self._tp_reduce_dtype(hidden_states)" in diff[1], diff


def test_refuses_to_install_over_a_changed_combine_preprocess(monkeypatch):
    from megatron.core.transformer.moe.token_dispatcher import (
        MoEAlltoAllTokenDispatcher,
    )

    from skyrl.backends.skyrl_train.patches.megatron import (
        patch_moe_combine_bf16_reduce as mod,
    )

    def changed(self, hidden_states):
        return hidden_states

    monkeypatch.setattr(MoEAlltoAllTokenDispatcher, "combine_preprocess", changed)
    monkeypatch.setattr(mod, "_APPLIED", False)
    assert mod.patch_moe_combine_bf16_reduce() is False
    assert MoEAlltoAllTokenDispatcher.combine_preprocess is changed
    assert not hasattr(MoEAlltoAllTokenDispatcher, "_tp_reduce_dtype")

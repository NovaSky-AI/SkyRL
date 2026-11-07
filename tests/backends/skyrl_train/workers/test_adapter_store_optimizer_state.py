"""CPU numerical tests for per-adapter DistributedOptimizer state.

Only Megatron's CUDA-dependent container types and CPU pinning are replaced.
The production AdapterStore runs all allocation/copy/swap operations, and real
PyTorch Adam updates are compared with separately trained adapter baselines.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


class _DDP:
    def __init__(self, params):
        self.params = params
        self.buffers = [
            SimpleNamespace(param_data=p.detach(), grad_data=torch.zeros_like(p), params=[p]) for p in params
        ]
        self.expert_parallel_buffers = []

    def named_parameters(self):
        return [(f"adapter.weight_{i}", p) for i, p in enumerate(self.params)]


class _ChainedOptimizer:
    def __init__(self, optimizers):
        self.chained_optimizers = optimizers


@pytest.fixture
def adapter_store_module(monkeypatch):
    # Load the real source in an isolated module namespace so CUDA-free tests
    # neither require Megatron nor leave fake Megatron imports in other tests.
    modules = {
        "megatron": {},
        "megatron.core": {},
        "megatron.core.parallel_state": {"get_data_parallel_group": lambda: None},
        "megatron.core.distributed": {"DistributedDataParallel": _DDP},
        "megatron.core.optimizer": {"ChainedOptimizer": _ChainedOptimizer},
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        module.__path__ = []
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[4] / "skyrl/backends/skyrl_train/workers/megatron/adapter_store.py"
    spec = importlib.util.spec_from_file_location("_adapter_store_cpu_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_new_pinned_like", lambda t: torch.empty_like(t, device="cpu"))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(synchronize=lambda: None))
    monkeypatch.setattr(module.dist, "is_initialized", lambda: False)
    return module


class _TrainingState:
    """DDP/shard layout with real Adam state and rank-local parameter views."""

    def __init__(self, layouts, group_step):
        self.group_step = group_step
        self.chunks = []
        self.optimizers = []
        self.param_pairs = []
        for dtypes in layouts:
            models = [torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=dtype)) for dtype in dtypes]
            # Only the second element belongs to this rank's optimizer shard.
            # FP32 shards share DDP storage; BF16 parameters have FP32 masters.
            mains = [
                torch.nn.Parameter(model.detach()[1:] if model.dtype == torch.float32 else model.detach()[1:].float())
                for model in models
            ]
            adam = torch.optim.Adam([{"params": []}, {"params": mains}], lr=0.01, betas=(0.9, 0.999))
            for main in mains:
                adam.state[main] = {"exp_avg": torch.zeros_like(main), "exp_avg_sq": torch.zeros_like(main)}
                if not group_step:
                    adam.state[main]["step"] = torch.tensor(0.0)
            if group_step:
                for group in adam.param_groups:
                    group["step"] = 0
            # Preserve empty groups, as on a rank without shards of a dtype.
            opt = SimpleNamespace(
                optimizer=adam,
                shard_fp32_from_float16_groups=[
                    [],
                    [main for model, main in zip(models, mains) if model.dtype == torch.bfloat16],
                ],
                shard_fp32_groups=[[], [main for model, main in zip(models, mains) if model.dtype == torch.float32]],
            )
            self.optimizers.append(opt)
            self.chunks.append(_DDP(models))
            self.param_pairs.extend(zip(models, mains))
        self.optimizer = self.optimizers[0] if len(self.optimizers) == 1 else _ChainedOptimizer(self.optimizers)

    def step(self, gradient):
        for index, (_, main) in enumerate(self.param_pairs):
            main.grad = torch.full_like(main, gradient * (index + 1))
        for opt in self.optimizers:
            adam = opt.optimizer
            if self.group_step:
                # TE FusedAdam stores the counter on the group, whereas
                # PyTorch Adam consumes a per-parameter counter. Translate
                # that storage convention just around the real Adam update.
                for group in adam.param_groups:
                    for main in group["params"]:
                        adam.state[main]["step"] = torch.tensor(float(group["step"]))
            adam.step()
            adam.zero_grad(set_to_none=True)
            if self.group_step:
                for group in adam.param_groups:
                    group["step"] += 1
                    for main in group["params"]:
                        del adam.state[main]["step"]
        with torch.no_grad():
            for model, main in self.param_pairs:
                model[1:].copy_(main)

    def assert_matches(self, expected):
        for (model, main), (expected_model, expected_main) in zip(self.param_pairs, expected.param_pairs):
            torch.testing.assert_close(model, expected_model, rtol=0, atol=0)
            torch.testing.assert_close(main, expected_main, rtol=0, atol=0)
        for opt, expected_opt in zip(self.optimizers, expected.optimizers):
            actual_state = opt.optimizer.state_dict()
            expected_state = expected_opt.optimizer.state_dict()
            assert actual_state["param_groups"] == expected_state["param_groups"]
            torch.testing.assert_close(actual_state["state"], expected_state["state"], rtol=0, atol=0)


@pytest.mark.parametrize("group_step", [False, True], ids=["pytorch-counter", "te-group-counter"])
@pytest.mark.parametrize(
    "layouts",
    [
        [(torch.float32,)],
        [(torch.bfloat16,)],
        [(torch.bfloat16, torch.float32)],
        [(torch.bfloat16,), (torch.float32,)],
    ],
    ids=["fp32", "bf16", "mixed", "chained"],
)
def test_interleaved_adapters_match_independent_adam(adapter_store_module, layouts, group_step):
    live = _TrainingState(layouts, group_step)
    baselines = {name: _TrainingState(layouts, group_step) for name in ("a", "b")}
    store = adapter_store_module.AdapterStore()
    signature = object()
    store.register_pristine(live.chunks, live.optimizer, signature)
    store.create("a", live.chunks, live.optimizer, signature)

    # Train A before B is created so the fresh slot must come from pristine,
    # including zero moments/counters, rather than from A's live optimizer.
    for _ in range(20):
        live.step(1.0)
        baselines["a"].step(1.0)
    live.assert_matches(baselines["a"])
    store.create("b", live.chunks, live.optimizer, signature)

    for name, gradient in [("b", -1.0), ("a", 0.3), ("b", -0.7), ("a", -0.2)]:
        store.swap_to(name, live.chunks, live.optimizer)
        live.step(gradient)
        baselines[name].step(gradient)
        # A leaked positive momentum makes B's first negative-gradient update
        # move in the wrong direction, even when its group counter is restored.
        live.assert_matches(baselines[name])

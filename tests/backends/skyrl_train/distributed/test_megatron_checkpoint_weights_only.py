"""Weights-only loading must preserve optimizer history and refresh masters."""

import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from skyrl.backends.skyrl_train.distributed.megatron.optimizer_state import (
    reload_optimizer_model_params,
)
from skyrl.backends.skyrl_train.workers.worker import Worker


@pytest.fixture
def strategy_module(monkeypatch):
    """Import the real strategy with GPU-only Megatron dependencies stubbed.

    Loading, Worker dispatch, LoRA state-dict application and master refresh
    execute normally; only distributed checkpoint I/O and CUDA dependencies
    are replaced so these tests run in the CPU suite.
    """

    def stub(name, **attributes):
        module = ModuleType(name)
        module.__path__ = []
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            parent, leaf = name.rsplit(".", 1)
            if parent in sys.modules and parent.startswith("megatron"):
                monkeypatch.setattr(sys.modules[parent], leaf, module, raising=False)
        return module

    class DistributedOptimizer:
        load_parameter_state_from_dp_reshardable = Mock()

    stub("megatron")
    stub("megatron.core")
    stub("megatron.core.parallel_state", get_data_parallel_group=lambda **kwargs: None)
    stub("megatron.core.dist_checkpointing", load=Mock())
    stub(
        "megatron.core.dist_checkpointing.serialization",
        get_default_load_sharded_strategy=Mock(),
        get_default_save_sharded_strategy=Mock(),
    )
    stub("megatron.core.dist_checkpointing.strategies")
    stub("megatron.core.dist_checkpointing.strategies.async_utils", AsyncCallsQueue=Mock())
    stub(
        "megatron.core.dist_checkpointing.strategies.fully_parallel",
        FullyParallelLoadStrategyWrapper=Mock(),
        FullyParallelSaveStrategyWrapper=Mock(),
    )
    stub("megatron.core.optimizer", DistributedOptimizer=DistributedOptimizer)
    stub("megatron.core.optimizer_param_scheduler", OptimizerParamScheduler=object)
    stub(
        "skyrl.backends.skyrl_train.distributed.megatron.megatron_utils",
        **{
            name: Mock()
            for name in (
                "load_megatron_grads_to_gpu",
                "load_megatron_model_to_gpu",
                "load_megatron_optimizer",
                "offload_megatron_grads_to_cpu",
                "offload_megatron_model_to_cpu",
                "offload_megatron_optimizer",
            )
        },
    )
    stub("skyrl.backends.skyrl_train.workers.megatron.megatron_model_wrapper", MegatronModelWrapper=object)
    path = Path(__file__).parents[4] / "skyrl/backends/skyrl_train/distributed/megatron/megatron_strategy.py"
    spec = importlib.util.spec_from_file_location("_checkpoint_test_strategy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter_weight = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))

    def sharded_state_dict(self):
        return self.state_dict()


class MasterOptimizer:
    """CPU AdamW with separate compute and master parameters, as in Megatron."""

    def __init__(self, model):
        self.model = model
        self.master = torch.nn.Parameter(model.adapter_weight.float().clone())
        self.optimizer = torch.optim.AdamW([self.master], lr=0.1, weight_decay=0.0)
        self.config = SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=False)
        self.reload_count = 0

    def reload_model_params(self):
        self.reload_count += 1
        self.master.data.copy_(self.model.adapter_weight)

    def step(self):
        self.master.grad = torch.ones_like(self.master)
        self.optimizer.step()
        self.model.adapter_weight.data.copy_(self.master)

    def sharded_state_dict(self, *args, **kwargs):
        return {}

    def load_state_dict(self, state):
        self.master.data.copy_(state["master"])
        self.optimizer.load_state_dict(state["optimizer"])


@pytest.mark.parametrize("is_lora", [False, True])
@pytest.mark.parametrize("load_optimizer", [False, True])
def test_checkpoint_load_preserves_the_correct_master_and_adam_state(
    strategy_module, tmp_path, is_lora, load_optimizer
):
    model = Model()
    optimizer = MasterOptimizer(model)
    optimizer.step()  # Nonzero Adam history must survive a weights-only load.
    original_state = copy.deepcopy(optimizer.optimizer.state_dict())
    strategy = strategy_module.MegatronStrategy(
        SimpleNamespace(dist_ckpt_optim_fully_reshardable=False), is_lora=is_lora
    )
    strategy.finalize_pending_saves = Mock()
    strategy.print = Mock()
    # A BF16 checkpoint's rounded compute weight differs from its FP32 master.
    checkpoint_weight = torch.tensor([5.0], dtype=torch.bfloat16)
    checkpoint_master = torch.tensor([5.001])
    loaded = {"model": {"adapter_weight": checkpoint_weight}}
    checkpoint_optimizer = copy.deepcopy(original_state)
    for state in checkpoint_optimizer["state"].values():
        state["step"].fill_(7)
        state["exp_avg"].fill_(0.4)
        state["exp_avg_sq"].fill_(0.2)
    if load_optimizer:
        loaded["optimizer"] = {"master": checkpoint_master, "optimizer": checkpoint_optimizer}
    strategy_module.dist_checkpointing.load.return_value = loaded
    if is_lora:
        adapter_path = tmp_path / "adapter.pt"
        torch.save({"model_state_dict": loaded["model"]}, adapter_path)
        strategy._get_rank_path = lambda directory: str(adapter_path)

    worker = object.__new__(Worker)
    worker.model = SimpleNamespace(actor_module=[model])
    worker.optimizer = optimizer
    worker.scheduler = Mock()
    worker.strategy = strategy
    Worker.load_checkpoint(worker, str(tmp_path), load_optimizer_states=load_optimizer, load_lr_scheduler_states=False)

    expected_master = checkpoint_master if load_optimizer else checkpoint_weight.float()
    torch.testing.assert_close(optimizer.master, expected_master)
    assert optimizer.reload_count == (0 if load_optimizer else 1)
    expected_state = checkpoint_optimizer if load_optimizer else original_state
    for name, value in expected_state["state"][0].items():
        torch.testing.assert_close(optimizer.optimizer.state[optimizer.master][name], value)
    assert optimizer.optimizer.param_groups[0]["lr"] == expected_state["param_groups"][0]["lr"]
    worker.scheduler.load_state_dict.assert_not_called()

    oracle_param = torch.nn.Parameter(expected_master.clone())
    oracle = torch.optim.AdamW([oracle_param], lr=0.1, weight_decay=0.0)
    oracle.load_state_dict(copy.deepcopy(expected_state))
    oracle_param.grad = torch.ones_like(oracle_param)
    oracle.step()
    optimizer.step()
    torch.testing.assert_close(optimizer.master, oracle_param)
    torch.testing.assert_close(model.adapter_weight, oracle_param.to(torch.bfloat16))


@pytest.mark.parametrize("remainders", [False, True])
@pytest.mark.parametrize("param_dtype", [torch.bfloat16, torch.float32])
def test_precision_aware_refresh_keeps_moments_and_lazy_state(remainders, param_dtype):
    param = torch.nn.Parameter(torch.tensor([5.0], dtype=param_dtype))
    uninitialized = torch.nn.Parameter(torch.tensor([7.0], dtype=torch.bfloat16))
    moments = {"exp_avg": torch.tensor([0.3]), "exp_avg_sq": torch.tensor([0.4])}
    state = {param: {"master_param": torch.tensor([1.0]), **moments}}
    inner = SimpleNamespace(
        param_groups=[{"params": [param, uninitialized], "step": 7}], state=state, store_param_remainders=remainders
    )

    def set_scaled_state(p, name, value):
        state[p][name] = value.clone()
        # Lower-precision TE master storage rescales its input in-place.
        value.mul_(2)

    inner.set_scaled_state = Mock(side_effect=set_scaled_state)
    optimizer = SimpleNamespace(
        reload_model_params=Mock(),
        config=SimpleNamespace(
            use_precision_aware_optimizer_no_fp8_or_ds_fp8=True,
            optimizer_cpu_offload=False,
            # TE disables requested remainders when master precision is not FP32.
            store_param_remainders=True,
        ),
        optimizer=inner,
        _is_distopt_quantized_param=lambda param: False,
    )
    reload_optimizer_model_params(optimizer)
    expected = (
        torch.zeros_like(param, dtype=torch.int16) if remainders and param_dtype == torch.bfloat16 else param.float()
    )
    torch.testing.assert_close(state[param]["master_param"], expected)
    inner.set_scaled_state.assert_called_once()
    assert state[param]["exp_avg"] is moments["exp_avg"]
    assert state[param]["exp_avg_sq"] is moments["exp_avg_sq"]
    assert inner.param_groups[0]["step"] == 7
    assert uninitialized not in state
    torch.testing.assert_close(param, torch.tensor([5.0], dtype=param_dtype))


def test_chained_optimizers_reload_once_and_skip_empty_shards():
    ordinary = SimpleNamespace(
        config=SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=False), optimizer=object()
    )
    empty = SimpleNamespace(config=SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=True), optimizer=None)
    optimizer = SimpleNamespace(reload_model_params=Mock(), chained_optimizers=[ordinary, empty])
    reload_optimizer_model_params(optimizer)
    optimizer.reload_model_params.assert_called_once_with()


@pytest.mark.parametrize("precision_aware", [False, True])
def test_hybrid_refreshes_sharded_masters_and_fp32_cpu_copies(precision_aware):
    model = torch.nn.Parameter(torch.tensor([2.0, 5.0, 7.0, 11.0], dtype=torch.bfloat16))
    outer_master = torch.tensor([-1.0, -1.0])
    # In precision-aware mode the optimizer parameter aliases the model shard;
    # otherwise it is DistributedOptimizer's separate, stale FP32 master.
    optimizer_param = model.detach()[1:3] if precision_aware else outer_master
    inner_param = torch.nn.Parameter(torch.tensor([-2.0, -2.0]))
    fp32_model = torch.nn.Parameter(torch.tensor([13.0]))
    fp32_cpu_copy = torch.nn.Parameter(torch.tensor([-3.0]))
    adam = torch.optim.AdamW([inner_param, fp32_cpu_copy], lr=0.1, weight_decay=0.0)
    for p in (inner_param, fp32_cpu_copy):
        p.grad = torch.ones_like(p)
    adam.step()
    before = copy.deepcopy(adam.state_dict())
    inner = SimpleNamespace(
        param_to_inner_param={optimizer_param: inner_param, fp32_model: fp32_cpu_copy}, state=adam.state
    )
    optimizer = SimpleNamespace(
        reload_model_params=Mock(),
        config=SimpleNamespace(
            use_precision_aware_optimizer_no_fp8_or_ds_fp8=precision_aware,
            optimizer_cpu_offload=True,
            use_distributed_optimizer=True,
        ),
        optimizer=inner,
        model_float16_groups=[[model]],
        shard_fp32_from_float16_groups=[[outer_master]],
        _get_model_param_range_map=lambda param: {"param": SimpleNamespace(start=1, end=3)},
        _is_distopt_quantized_param=lambda param: False,
    )
    reload_optimizer_model_params(optimizer)
    torch.testing.assert_close(inner_param, torch.tensor([5.0, 7.0]))
    torch.testing.assert_close(fp32_cpu_copy, torch.tensor([13.0]))
    if not precision_aware:
        torch.testing.assert_close(outer_master, torch.tensor([5.0, 7.0]))
    for index, p in enumerate((inner_param, fp32_cpu_copy)):
        for name, value in before["state"][index].items():
            torch.testing.assert_close(adam.state[p][name], value)
    # A real Adam update starts from the restored shard, using the old moments.
    oracle_param = torch.nn.Parameter(torch.tensor([5.0, 7.0]))
    oracle_fp32 = torch.nn.Parameter(torch.tensor([13.0]))
    oracle = torch.optim.AdamW([oracle_param, oracle_fp32], lr=0.1, weight_decay=0.0)
    oracle.load_state_dict(copy.deepcopy(before))
    for p in (oracle_param, oracle_fp32):
        p.grad = torch.ones_like(p)
    oracle.step()
    adam.step()
    torch.testing.assert_close(inner_param, oracle_param)
    torch.testing.assert_close(fp32_cpu_copy, oracle_fp32)


def test_non_distributed_hybrid_uses_native_master_reload_then_refreshes_cpu_copy():
    model = torch.tensor([5.0], dtype=torch.bfloat16)
    master = torch.tensor([1.0])
    cpu_copy = torch.tensor([-1.0])
    optimizer = SimpleNamespace(
        # Float16OptimizerWithFloat16Params already refreshes its full masters.
        reload_model_params=Mock(side_effect=lambda: master.copy_(model)),
        config=SimpleNamespace(
            use_precision_aware_optimizer_no_fp8_or_ds_fp8=False,
            optimizer_cpu_offload=True,
            use_distributed_optimizer=False,
        ),
        optimizer=SimpleNamespace(param_to_inner_param={master: cpu_copy}),
    )
    reload_optimizer_model_params(optimizer)
    optimizer.reload_model_params.assert_called_once_with()
    torch.testing.assert_close(master, torch.tensor([5.0]))
    torch.testing.assert_close(cpu_copy, master)

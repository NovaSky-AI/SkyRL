import pytest
import torch

from skyrl.backends.skyrl_train.workers.megatron.fp8_param import (
    initialize_fp8_param_optimizer_masters,
    is_fp8_param_enabled,
)


class _FakeOptimizer:
    def __init__(self):
        self.calls = 0
        self.state_dict = None

    def _copy_model_params_to_main_params(self, state_dict=None):
        self.calls += 1
        self.state_dict = state_dict


class _Range:
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end
        self.size = end - start


class _FakeQuantizedParam:
    def __init__(self, values):
        self._values = torch.tensor(values, dtype=torch.float32)

    def dequantize(self):
        return self._values


class _FakeHybridDeviceOptimizer:
    def __init__(self, cpu_masters, gpu_masters):
        self.cpu_copys_map_gpu_param = dict(zip(cpu_masters, gpu_masters))
        self.param_to_fp32_param = {}


class _FakeHybridMegatronOptimizer:
    def __init__(self):
        self.model = _FakeQuantizedParam([2.0, 4.0, 6.0, 8.0])
        self.unquantized_model = torch.tensor([10.0, 20.0, 30.0, 40.0])
        self.gpu_master = torch.full((2,), -1.0)
        self.gpu_master_unquantized = torch.full((2,), -3.0)
        self.cpu_master = torch.full((2,), -2.0)
        self.cpu_master_unquantized = torch.full((2,), -4.0)
        self.optimizer = _FakeHybridDeviceOptimizer(
            [self.cpu_master, self.cpu_master_unquantized],
            [self.gpu_master, self.gpu_master_unquantized],
        )
        self.model_float16_groups = [[self.model, self.unquantized_model]]
        self.shard_fp32_from_float16_groups = [[self.gpu_master, self.gpu_master_unquantized]]
        self.model_fp32_groups = []
        self.shard_fp32_groups = []
        self.loaded_quantized_model = torch.tensor([100.0, 200.0, 300.0, 400.0])
        self.loaded_unquantized_model = torch.tensor([50.0, 60.0, 70.0, 80.0])
        self.exact_state_dict = {"model": {"ignored": torch.tensor(0.0)}}

    def _is_distopt_quantized_param(self, param):
        return param is self.model

    def _get_model_param_range_map(self, param):
        assert param is self.model or param is self.unquantized_model
        return {"param": _Range(1, 3)}

    def _build_model_param_to_state_dict_param_map(self, state_dict):
        assert state_dict is self.exact_state_dict
        return {
            self.model: self.loaded_quantized_model,
            self.unquantized_model: self.loaded_unquantized_model,
        }

    def _copy_model_params_to_main_params(self):
        raise AssertionError("Hybrid CPU-offload path must seed both master copies directly")


def test_fp8_param_enablement_reads_skyrl_transformer_config_mapping():
    assert is_fp8_param_enabled({"fp8_param": True})
    assert not is_fp8_param_enabled({"fp8_param": False})
    assert not is_fp8_param_enabled({"fp8_param": "false"})
    assert not is_fp8_param_enabled({})


def test_fp8_param_master_initialization_reloads_each_chained_optimizer():
    first = _FakeOptimizer()
    second = _FakeOptimizer()
    chained = type("FakeChainedOptimizer", (), {"chained_optimizers": [first, second]})()

    initialized = initialize_fp8_param_optimizer_masters(
        chained,
        fp8_param=True,
        fp8_param_gather=True,
        state_dict={"model": {}},
    )

    assert initialized == 2
    assert first.calls == 1
    assert second.calls == 1


def test_fp8_param_cpu_offload_uses_exact_checkpoint_state_for_all_masters():
    optimizer = _FakeHybridMegatronOptimizer()

    initialized = initialize_fp8_param_optimizer_masters(
        optimizer,
        fp8_param=True,
        fp8_param_gather=True,
        state_dict=optimizer.exact_state_dict,
    )

    assert initialized == 1
    torch.testing.assert_close(optimizer.gpu_master, torch.tensor([200.0, 300.0]))
    torch.testing.assert_close(optimizer.cpu_master, torch.tensor([200.0, 300.0]))
    torch.testing.assert_close(optimizer.gpu_master_unquantized, torch.tensor([60.0, 70.0]))
    torch.testing.assert_close(optimizer.cpu_master_unquantized, torch.tensor([60.0, 70.0]))


def test_fp8_param_master_initialization_passes_exact_state_to_normal_optimizer():
    optimizer = _FakeOptimizer()
    state_dict = {"model": {"weight": torch.tensor([1.0])}}

    initialized = initialize_fp8_param_optimizer_masters(
        optimizer,
        fp8_param=True,
        fp8_param_gather=True,
        state_dict=state_dict,
    )

    assert initialized == 1
    assert optimizer.state_dict is state_dict


def test_fp8_param_master_initialization_requires_fp8_param_gather():
    optimizer = _FakeOptimizer()

    with pytest.raises(ValueError, match="fp8_param_gather=true"):
        initialize_fp8_param_optimizer_masters(
            optimizer,
            fp8_param=True,
            fp8_param_gather=False,
        )

    assert optimizer.calls == 0


def test_fp8_param_master_initialization_requires_exact_checkpoint_state():
    optimizer = _FakeOptimizer()

    with pytest.raises(ValueError, match="exact unquantized checkpoint state"):
        initialize_fp8_param_optimizer_masters(
            optimizer,
            fp8_param=True,
            fp8_param_gather=True,
        )

    assert optimizer.calls == 0


def test_non_persistent_fp8_does_not_touch_optimizer_masters():
    optimizer = _FakeOptimizer()

    initialized = initialize_fp8_param_optimizer_masters(
        optimizer,
        fp8_param=False,
        fp8_param_gather=False,
    )

    assert initialized == 0
    assert optimizer.calls == 0

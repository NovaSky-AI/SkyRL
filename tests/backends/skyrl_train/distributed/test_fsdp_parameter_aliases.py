"""Keep shared Parameters intact while FSDP initializes from a full state dict."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import skyrl.backends.skyrl_train.distributed.fsdp_strategy as native
from skyrl.train.config import FSDPConfig, ModelConfig, OptimizerConfig


class TiedModel(nn.Module):
    def __init__(self, tied, dtype):
        super().__init__()
        self.embedding = nn.Embedding(10, 4, dtype=dtype)
        self.output = nn.Linear(4, 10, bias=False, dtype=dtype)
        if tied:
            self.output.weight = self.embedding.weight
        self.register_buffer("positions", torch.arange(4), persistent=False)
        self.config = SimpleNamespace(tie_word_embeddings=tied)


@pytest.fixture
def cpu_strategy(monkeypatch):
    """Exercise real meta conversion and assign loading without a process group."""
    monkeypatch.setattr(native.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(native, "apply_fsdp2", lambda *args, **kwargs: None)

    def load_state(model, state, cpu_offload):
        model.load_state_dict(state, assign=True)

    monkeypatch.setattr(native, "fsdp2_load_full_state_dict", load_state)
    return native.FSDPStrategy(FSDPConfig(), OptimizerConfig(), ModelConfig())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tied", [False, True])
def test_initialization_preserves_parameter_sharing(cpu_strategy, tied, dtype):
    model = TiedModel(tied, dtype)
    expected = {name: value.detach().clone() for name, value in model.named_parameters(remove_duplicate=False)}
    positions = model.positions.clone()
    original_swap = torch.__future__.get_swap_module_params_on_conversion()

    result = cpu_strategy._fsdp_init_model(model)

    assert (result.embedding.weight is result.output.weight) is tied
    assert len(list(result.parameters())) == (1 if tied else 2)
    for name, value in result.named_parameters(remove_duplicate=False):
        assert value.dtype == dtype
        assert value.shape == expected[name].shape
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    torch.testing.assert_close(result.positions, positions, rtol=0, atol=0)
    assert torch.__future__.get_swap_module_params_on_conversion() == original_swap


@pytest.mark.parametrize("previous_swap", [False, True])
@pytest.mark.parametrize("failing_stage", ["apply_fsdp2", "fsdp2_load_full_state_dict"])
def test_initialization_restores_swap_setting_on_error(cpu_strategy, monkeypatch, previous_swap, failing_stage):
    def fail(*args, **kwargs):
        raise RuntimeError("initialization failed")

    monkeypatch.setattr(native, failing_stage, fail)
    original_swap = torch.__future__.get_swap_module_params_on_conversion()
    torch.__future__.set_swap_module_params_on_conversion(previous_swap)
    try:
        with pytest.raises(RuntimeError, match="initialization failed"):
            cpu_strategy._fsdp_init_model(TiedModel(True, torch.float32))
        assert torch.__future__.get_swap_module_params_on_conversion() == previous_swap
    finally:
        torch.__future__.set_swap_module_params_on_conversion(original_swap)

import pytest
import torch
import torch.nn as nn

from skyrl.backends.skyrl_train.distributed import fsdp_strategy
from skyrl.train.config import BitsAndBytes4BitConfig, FSDPConfig, ModelConfig


def test_4bit_fsdp_does_not_cast_packed_parameters(monkeypatch):
    captured_kwargs = {}

    def capture_fsdp_kwargs(_model, fsdp_kwargs, _config):
        captured_kwargs.update(fsdp_kwargs)

    monkeypatch.setattr(fsdp_strategy, "apply_fsdp2", capture_fsdp_kwargs)
    strategy = fsdp_strategy.FSDPStrategy(
        fsdp_config=FSDPConfig(),
        model_config=ModelConfig(bitsandbytes_4bit=BitsAndBytes4BitConfig(enabled=True)),
    )
    strategy.world_size = 2

    model = nn.Linear(2, 2)
    assert strategy._fsdp_init_model(model) is model

    policy = captured_kwargs["mp_policy"]
    assert policy.param_dtype is None
    assert policy.reduce_dtype is torch.float32


def test_single_rank_4bit_rejects_native_cpu_offload():
    strategy = fsdp_strategy.FSDPStrategy(
        fsdp_config=FSDPConfig(cpu_offload=True),
        model_config=ModelConfig(bitsandbytes_4bit=BitsAndBytes4BitConfig(enabled=True)),
    )
    strategy.world_size = 1

    with pytest.raises(ValueError, match="single-rank 4-bit training"):
        strategy._fsdp_init_model(nn.Linear(2, 2))

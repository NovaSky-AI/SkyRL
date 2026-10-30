"""CPU unit tests for the LoRA-wrapped ``output_layer`` attribute passthrough.

Run with:
uv run --isolated --extra skyrl-train --extra dev pytest -s tests/backends/skyrl_train/distributed/test_lora_output_layer_passthrough.py

``GPTModel._postprocess`` reads ``output_layer.gather_output`` and toggles
``output_layer.sequence_parallel``; these tests pin that exactly those two go through to
the base linear of an adapter wrapper, while the wrapper's own state stays its own.
"""

import torch
from torch import nn

from skyrl.backends.skyrl_train.distributed.megatron.lora_output_layer import (
    attr_passthrough_class,
    enable_output_layer_attr_passthrough,
)


class _BaseLinear(nn.Linear):
    """Stands in for a vocab-parallel ColumnParallelLinear."""

    def __init__(self):
        super().__init__(4, 8, bias=False)
        self.gather_output = False
        self.sequence_parallel = True


class _Wrapper(nn.Module):
    """Shape of Megatron-Bridge's ``LoRALinear``: ``to_wrap`` + ``adapter`` submodules."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.to_wrap = base
        self.adapter = nn.Linear(4, 8, bias=False)
        self.wrapper_flag = "wrapper"

    def forward(self, x):
        return self.to_wrap(x) + self.adapter(x)


class _Chunk(nn.Module):
    def __init__(self, output_layer: nn.Module):
        super().__init__()
        self.output_layer = output_layer


class _DDPLike(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module


def test_reads_and_writes_reach_the_base_linear():
    base = _BaseLinear()
    wrapper = _Wrapper(base)
    wrapper.__class__ = attr_passthrough_class(_Wrapper)

    assert wrapper.gather_output is False
    assert wrapper.sequence_parallel is True
    wrapper.sequence_parallel = False
    assert base.sequence_parallel is False
    assert "sequence_parallel" not in wrapper.__dict__
    wrapper.sequence_parallel = True
    assert base.sequence_parallel is True

    # The wrapper's own attributes, parameters and submodules are untouched.
    wrapper.wrapper_flag = "changed"
    assert wrapper.__dict__["wrapper_flag"] == "changed"
    assert wrapper.to_wrap is base
    assert isinstance(wrapper.adapter, nn.Linear)
    x = torch.randn(2, 4)
    torch.testing.assert_close(wrapper(x), base(x) + wrapper.adapter(x))


def test_only_the_two_gptmodel_attributes_are_forwarded():
    base = _BaseLinear()
    base.tp_group = "tp"
    wrapper = _Wrapper(base)
    wrapper.__class__ = attr_passthrough_class(_Wrapper)
    # Other base-linear attributes are not proxied, so a caller relying on them fails loudly.
    try:
        wrapper.tp_group
    except AttributeError:
        pass
    else:
        raise AssertionError("expected AttributeError")
    extra = nn.Linear(1, 1)
    wrapper.extra = extra
    assert wrapper._modules["extra"] is extra
    assert "extra" not in wrapper.to_wrap._modules


def test_enable_output_layer_attr_passthrough_targets_only_wrapped_output_layers():
    wrapped = _Chunk(_Wrapper(_BaseLinear()))
    plain = _Chunk(_BaseLinear())
    assert enable_output_layer_attr_passthrough([_DDPLike(wrapped), plain]) == 1
    assert type(wrapped.output_layer).__name__ == "SkyRLOutputLayer_Wrapper"
    assert type(plain.output_layer) is _BaseLinear
    assert wrapped.output_layer.gather_output is False
    # Idempotent; a chunk without an output layer is skipped.
    assert enable_output_layer_attr_passthrough([wrapped, nn.Module()]) == 0
    # One passthrough class per wrapper class.
    assert attr_passthrough_class(_Wrapper) is type(wrapped.output_layer)

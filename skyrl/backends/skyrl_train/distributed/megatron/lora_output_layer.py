"""Attribute forwarding for a LoRA-wrapped Megatron ``output_layer``.

``GPTModel._postprocess`` reads ``self.output_layer.gather_output`` and toggles
``self.output_layer.sequence_parallel`` on whatever module sits at ``output_layer``.
Megatron-Bridge's adapter wrappers (``LoRALinear`` and friends) hold the base linear
as ``to_wrap`` and do not proxy its attributes, so a LoRA target list that includes
``output_layer`` fails with ``'LoRALinear' object has no attribute 'gather_output'``.

Kept free of megatron imports so the class rewrite is unit-testable on CPU.
"""

from functools import cache
from typing import Iterable, Union

import torch


class _OutputLayerAttrs:
    """Forwards the base-linear attributes ``GPTModel`` uses on ``output_layer`` to ``to_wrap``."""

    @property
    def gather_output(self):
        return self.to_wrap.gather_output

    @property
    def sequence_parallel(self):
        return self.to_wrap.sequence_parallel

    @sequence_parallel.setter
    def sequence_parallel(self, value):
        self.to_wrap.sequence_parallel = value


@cache
def attr_passthrough_class(wrapper_cls: type) -> type:
    """``wrapper_cls`` with :class:`_OutputLayerAttrs` mixed in (one class per wrapper class)."""
    return type(
        f"SkyRLOutputLayer{wrapper_cls.__name__}",
        (_OutputLayerAttrs, wrapper_cls),
        {},
    )


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
    while hasattr(module, "module"):
        module = module.module
    return module


def enable_output_layer_attr_passthrough(model_chunks: Union[torch.nn.Module, Iterable[torch.nn.Module]]) -> int:
    """Make a LoRA-wrapped ``output_layer`` expose its base linear's attributes.

    Takes a model chunk or a list of chunks (DDP / Float16Module wrapping is peeled).
    No-op for chunks whose output layer is not adapter-wrapped (no ``to_wrap`` and
    ``adapter`` submodules). Returns the number of wrappers changed.
    """
    chunks = [model_chunks] if isinstance(model_chunks, torch.nn.Module) else list(model_chunks)
    changed = 0
    for chunk in chunks:
        output_layer = getattr(_unwrap(chunk), "output_layer", None)
        if output_layer is None:
            continue
        submodules = output_layer._modules
        if "to_wrap" not in submodules or "adapter" not in submodules:
            continue
        if isinstance(output_layer, _OutputLayerAttrs):
            continue
        output_layer.__class__ = attr_passthrough_class(type(output_layer))
        changed += 1
    return changed

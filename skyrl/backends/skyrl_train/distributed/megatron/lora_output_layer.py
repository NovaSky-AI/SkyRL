"""Attribute passthrough for a LoRA-wrapped Megatron ``output_layer``.

``GPTModel._postprocess`` reads ``self.output_layer.gather_output`` and toggles
``self.output_layer.sequence_parallel`` on whatever module sits at ``output_layer``.
Megatron-Bridge's adapter wrappers (``LoRALinear`` and friends) hold the base linear
as ``to_wrap`` and do not proxy its attributes, so a LoRA target list that includes
``output_layer`` fails with ``'LoRALinear' object has no attribute 'gather_output'``.

Kept free of megatron imports so the class rewrite is unit-testable on CPU.
"""

from typing import Dict, Iterable, Union

import torch

_PASSTHROUGH_CLASSES: Dict[type, type] = {}


def attr_passthrough_class(wrapper_cls: type) -> type:
    """Subclass of an adapter wrapper that forwards unknown attributes to ``to_wrap``.

    Reads of attributes the wrapper does not own resolve on the base linear; writes to
    an attribute the base linear has (and the wrapper does not) land on the base linear
    instead of shadowing it on the wrapper, so ``sequence_parallel = False`` followed by
    a read sees the base linear's state.
    """
    cached = _PASSTHROUGH_CLASSES.get(wrapper_cls)
    if cached is not None:
        return cached

    def __getattr__(self, name):
        try:
            return super(subclass, self).__getattr__(name)
        except AttributeError:
            to_wrap = self.__dict__.get("_modules", {}).get("to_wrap")
            if to_wrap is None or name == "to_wrap":
                raise
            return getattr(to_wrap, name)

    def __setattr__(self, name, value):
        to_wrap = self.__dict__.get("_modules", {}).get("to_wrap")
        owned = (
            name in self.__dict__
            or name in self.__dict__.get("_parameters", {})
            or name in self.__dict__.get("_buffers", {})
            or name in self.__dict__.get("_modules", {})
        )
        if (
            to_wrap is not None
            and not owned
            and not isinstance(value, (torch.nn.Module, torch.nn.Parameter))
            and hasattr(to_wrap, name)
        ):
            setattr(to_wrap, name, value)
            return
        super(subclass, self).__setattr__(name, value)

    subclass = type(
        f"SkyRLAttrPassthrough{wrapper_cls.__name__}",
        (wrapper_cls,),
        {"__getattr__": __getattr__, "__setattr__": __setattr__},
    )
    _PASSTHROUGH_CLASSES[wrapper_cls] = subclass
    return subclass


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
        if type(output_layer) in _PASSTHROUGH_CLASSES.values():
            continue
        output_layer.__class__ = attr_passthrough_class(type(output_layer))
        changed += 1
    return changed

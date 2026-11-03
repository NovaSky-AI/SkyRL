# An accelerator is the non-CPU device torch computes on: "cuda" for NVIDIA,
# "xpu" for Intel, "mps" for Apple, and so on. This helper only adds "xpu" for
# now; other accelerators such as MPS, and hosts without one, fall back to
# "cuda" so they keep their existing behaviour.

import torch

_ENABLED_TRAINING_ACCELERATORS = {"cuda", "xpu"}


def device_type() -> str:
    """Return the device type training runs on. Eg: ``"cuda"`` or ``"xpu"``."""

    if hasattr(torch, "accelerator") and torch.accelerator.is_available():
        accelerator = torch.accelerator.current_accelerator()
        if accelerator.type in _ENABLED_TRAINING_ACCELERATORS:
            return accelerator.type
    return "cuda"


def device_module():
    """Return the torch device module for ``device_type()``. Eg: ``torch.cuda`` or ``torch.xpu``."""

    return torch.get_device_module(device_type())

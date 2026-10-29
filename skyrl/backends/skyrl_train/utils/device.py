# An accelerator is the non-CPU device torch computes on: "cuda" for NVIDIA,
# "xpu" for Intel, "mps" for Apple, and so on. This helper only adds "xpu" for
# now; other accelerators such as MPS, and hosts without one, fall back to
# "cuda" so they keep their existing behaviour.

import torch

_ENABLED_TRAINING_ACCELERATORS = {"cuda", "xpu"}


def device_type() -> str:
    """Return the device type training runs on. Eg: ``"cuda"`` or ``"xpu"``."""

    if torch.accelerator.is_available():
        accelerator = torch.accelerator.current_accelerator()
        if accelerator.type in _ENABLED_TRAINING_ACCELERATORS:
            return accelerator.type
    return "cuda"

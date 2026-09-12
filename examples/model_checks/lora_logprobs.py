"""Deterministic adapter-only perturbation for native GPU checks."""

import hashlib
import math

import torch


@torch.no_grad()
def perturb_adapters(named_parameters, seed=0, multiplier=10):
    """Preserve Bridge's A tensors and give zero-init B a fixed, name-seeded stimulus."""
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("LoRA B multiplier must be positive and finite")
    changed = 0
    tensors = 0
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        assert "adapter" in name or "lora" in name, f"unexpected trainable base parameter: {name}"
        if name.endswith(".linear_in.weight"):
            continue
        assert name.endswith(".linear_out.weight"), f"unexpected adapter tensor: {name}"
        assert torch.count_nonzero(parameter) == 0, f"expected zero-init B: {name}"
        name_seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
        generator = torch.Generator(device=parameter.device).manual_seed((seed + name_seed) % (2**63))
        parameter.add_(
            torch.randn(parameter.shape, generator=generator, device=parameter.device, dtype=parameter.dtype),
            alpha=1e-3,
        )
        parameter.mul_(multiplier)
        changed += parameter.numel()
        tensors += 1
    assert changed > 0
    return {
        "changed_b_tensors": tensors,
        "changed_b_elements": changed,
        "seed": seed,
        "noise_std": 1e-3,
        "multiplier": multiplier,
    }

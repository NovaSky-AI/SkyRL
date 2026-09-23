"""Fixed-token logprob comparisons."""

import hashlib
import math
from itertools import cycle, islice

import torch


def build_probe_sequences(tokenizer):
    return [
        list(islice(cycle(tokenizer.encode(text, add_special_tokens=False)), length))
        for text, length in [
            ("A river flows beneath a bridge. ", 65),
            ("Calculate seven times eight. ", 129),
        ]
    ]


def check_agreement(result, mean_atol, max_atol):
    assert result["mean_abs"] <= mean_atol
    assert result["max_abs"] <= max_atol


def compare_logprobs(reference, actual):
    reference = torch.as_tensor(reference, dtype=torch.float64)
    actual = torch.as_tensor(actual, dtype=torch.float64)
    assert reference.ndim == actual.ndim == 1
    assert reference.shape == actual.shape and reference.numel() > 0
    assert torch.isfinite(reference).all() and torch.isfinite(actual).all()
    error = (reference - actual).abs()
    return {
        "tokens": error.numel(),
        "mean_abs": error.mean().item(),
        "p99_abs": error.quantile(0.99).item(),
        "max_abs": error.max().item(),
    }


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
            torch.randn(
                parameter.shape,
                generator=generator,
                device=parameter.device,
                dtype=parameter.dtype,
            ),
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

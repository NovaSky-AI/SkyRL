"""Numerical checks and adapter-only perturbation for the GPU LoRA test."""

import hashlib

import torch


def check_initial_adapter(report, atol):
    report["base_zero"] = compare_logprobs(report["base"], report["zero"])
    report["zero_parity"] = compare_logprobs(report["trainer_zero"], report["zero"])
    report["repeat_noise"] = compare_logprobs(report["zero"], report["repeat"])
    assert report["base_zero"]["mean_abs"] < atol
    assert report["zero_parity"]["mean_abs"] < atol


def check_withheld_publication(report):
    report["withheld_publication"] = compare_logprobs(report["repeat"], report["stale"])
    noise_budget = max(1e-6, 3 * report["repeat_noise"]["mean_abs"])
    assert report["withheld_publication"]["mean_abs"] <= noise_budget


def check_updated_adapter(report, atol):
    report["updated_parity"] = compare_logprobs(report["trainer_updated"], report["updated"])
    report["sampler_change"] = compare_logprobs(report["zero"], report["updated"])
    report["trainer_change"] = compare_logprobs(report["trainer_zero"], report["trainer_updated"])
    noise_budget = max(1e-6, 3 * report["repeat_noise"]["mean_abs"])
    assert report["updated_parity"]["mean_abs"] < atol
    assert report["sampler_change"]["mean_abs"] > noise_budget
    assert report["trainer_change"]["mean_abs"] > noise_budget


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
def perturb_adapters(named_parameters, seed=0, scale=1e-3):
    """Use name-seeded noise so replicated adapter tensors receive identical updates."""
    changed = 0
    tensors = 0
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        assert "adapter" in name or "lora" in name, f"unexpected trainable base parameter: {name}"
        name_seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
        generator = torch.Generator(device=parameter.device).manual_seed((seed + name_seed) % (2**63))
        parameter.add_(
            torch.randn(parameter.shape, generator=generator, device=parameter.device, dtype=parameter.dtype),
            alpha=scale,
        )
        changed += parameter.numel()
        tensors += 1
    assert changed > 0
    return {"trainable_tensors": tensors, "trainable_elements": changed, "seed": seed, "noise_std": scale}

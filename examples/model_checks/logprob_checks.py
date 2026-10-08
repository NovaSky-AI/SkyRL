"""Fixed-token logprob comparisons and deterministic weight perturbations for model checks."""

import hashlib
from itertools import cycle, islice
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

PROBE_TEXTS: Tuple[Tuple[str, int], ...] = (
    ("A river flows beneath a bridge. ", 65),
    ("Calculate seven times eight. ", 129),
)


def build_probe_sequences(tokenizer, probes: Sequence[Tuple[str, int]] = PROBE_TEXTS) -> List[List[int]]:
    """Token sequences of fixed length, each a repeated encoding of one probe text."""
    return [list(islice(cycle(tokenizer.encode(text, add_special_tokens=False)), length)) for text, length in probes]


def replicate_to_multiple(sequences: List[List[int]], multiple: int) -> List[List[int]]:
    """Repeat ``sequences`` cyclically until their count is divisible by ``multiple``.

    Mesh dispatch splits a batch into one equal chunk per data-parallel rank, so the
    batch size must be a multiple of the data-parallel size.
    """
    if multiple <= 0:
        raise ValueError("multiple must be positive")
    if not sequences:
        raise ValueError("sequences must be non-empty")
    count = -(-len(sequences) // multiple) * multiple
    return [sequences[i % len(sequences)] for i in range(count)]


def compare_logprobs(reference: Sequence[float], actual: Sequence[float]) -> Dict[str, float]:
    """Absolute error statistics between two aligned 1-D logprob lists."""
    reference_t = torch.as_tensor(reference, dtype=torch.float64)
    actual_t = torch.as_tensor(actual, dtype=torch.float64)
    if reference_t.ndim != 1 or actual_t.ndim != 1:
        raise ValueError("logprobs must be 1-D")
    if reference_t.shape != actual_t.shape or reference_t.numel() == 0:
        raise ValueError(f"logprob lists must be aligned and non-empty, got {reference_t.shape} vs {actual_t.shape}")
    if not (torch.isfinite(reference_t).all() and torch.isfinite(actual_t).all()):
        raise ValueError("logprobs must be finite")
    error = (reference_t - actual_t).abs()
    return {
        "tokens": error.numel(),
        "mean_abs": error.mean().item(),
        "p99_abs": error.quantile(0.99).item(),
        "max_abs": error.max().item(),
    }


def check_agreement(result: Dict[str, float], mean_atol: float, max_atol: float, label: str) -> None:
    if result["mean_abs"] > mean_atol:
        raise AssertionError(f"{label}: mean abs logprob error {result['mean_abs']:.6f} exceeds {mean_atol}")
    if result["max_abs"] > max_atol:
        raise AssertionError(f"{label}: max abs logprob error {result['max_abs']:.6f} exceeds {max_atol}")


def _seeded_noise(name: str, like: torch.Tensor, seed: int) -> torch.Tensor:
    name_seed = (seed + int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")) % (2**63)
    generator = torch.Generator(device=like.device).manual_seed(name_seed)
    return torch.randn(like.shape, generator=generator, device=like.device, dtype=like.dtype)


@torch.no_grad()
def perturb_lora_b(
    named_parameters: Iterable[Tuple[str, torch.Tensor]],
    multiplier: float = 10.0,
    seed: int = 0,
    noise_std: float = 1e-3,
) -> Dict[str, float]:
    """Add name-seeded Gaussian noise to every trainable, zero-valued LoRA B tensor.

    Megatron-Bridge names the adapter tensors ``adapter.linear_in`` (A) and
    ``adapter.linear_out`` (B); PEFT names them ``lora_A`` and ``lora_B``. A
    tensors are left as initialized. Every rank produces the same noise for the
    same tensor name, so replicas stay consistent.
    """
    changed_tensors = 0
    changed_elements = 0
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        if "linear_in" in name or "lora_A" in name:
            continue
        if "linear_out" not in name and "lora_B" not in name:
            raise AssertionError(f"unexpected trainable parameter under LoRA: {name}")
        if torch.count_nonzero(param).item() != 0:
            raise AssertionError(f"expected a zero-initialized LoRA B tensor: {name}")
        param.add_(_seeded_noise(name, param, seed), alpha=noise_std * multiplier)
        changed_tensors += 1
        changed_elements += param.numel()
    if changed_tensors == 0:
        raise AssertionError("no trainable LoRA B tensors found")
    return {
        "changed_tensors": changed_tensors,
        "changed_elements": changed_elements,
        "seed": seed,
        "noise_std": noise_std,
        "multiplier": multiplier,
    }


@torch.no_grad()
def perturb_full_weights(
    named_parameters: Iterable[Tuple[str, torch.Tensor]],
    multiplier: float = 1.0,
    seed: int = 0,
    relative_std: float = 1e-3,
) -> Dict[str, float]:
    """Add name-seeded Gaussian noise to every trainable tensor, scaled by its mean magnitude.

    Each tensor receives ``relative_std * multiplier * mean(|w|) * randn``.
    All-zero tensors are left unchanged. Frozen tensors are skipped.
    """
    changed_tensors = 0
    changed_elements = 0
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        scale = param.abs().mean().item()
        if scale == 0.0:
            continue
        param.add_(_seeded_noise(name, param, seed), alpha=relative_std * multiplier * scale)
        changed_tensors += 1
        changed_elements += param.numel()
    if changed_tensors == 0:
        raise AssertionError("no trainable tensors found")
    return {
        "changed_tensors": changed_tensors,
        "changed_elements": changed_elements,
        "seed": seed,
        "relative_std": relative_std,
        "multiplier": multiplier,
    }

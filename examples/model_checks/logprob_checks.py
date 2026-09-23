"""Fixed-token logprob comparisons."""

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

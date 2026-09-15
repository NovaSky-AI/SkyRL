"""Logprob comparisons shared by native and SDK model checks."""

from itertools import cycle, islice

import torch


def check_initial_adapter(report, atol):
    report["base_zero"] = compare_logprobs(report["base"], report["zero"])
    check_policy_snapshot(report, atol)
    if report["base_zero"]["mean_abs"] >= atol:
        raise AssertionError("zero-initialized adapter differs from the base model")


def check_policy_snapshot(report, atol):
    report["zero_parity"] = compare_logprobs(report["trainer_zero"], report["zero"])
    report["repeat_noise"] = compare_logprobs(report["zero"], report["repeat"])
    report["trainer_repeat_noise"] = compare_logprobs(report["trainer_zero"], report["trainer_repeat"])
    if report["zero_parity"]["mean_abs"] >= atol:
        raise AssertionError("trainer and inference logprobs exceed the parity tolerance")


def build_probe_sequences(tokenizer):
    return [
        list(islice(cycle(tokenizer.encode(text, add_special_tokens=False)), length))
        for text, length in [
            ("A river flows beneath a bridge. ", 65),
            ("Calculate seven times eight. ", 129),
        ]
    ]


def check_withheld_publication(report):
    report["withheld_publication"] = compare_logprobs(report["repeat"], report["stale"])
    noise_budget = max(1e-6, 3 * report["repeat_noise"]["mean_abs"])
    if report["withheld_publication"]["mean_abs"] > noise_budget:
        raise AssertionError("unpublished trainer update became visible to inference")


def check_updated_adapter(report, atol):
    report["updated_parity"] = compare_logprobs(report["trainer_updated"], report["updated"])
    report["stale_parity"] = compare_logprobs(report["trainer_updated"], report["stale"])
    report["sampler_change"] = compare_logprobs(report["zero"], report["updated"])
    report["trainer_change"] = compare_logprobs(report["trainer_zero"], report["trainer_updated"])
    noise_budget = max(
        1e-6,
        3 * report["repeat_noise"]["mean_abs"],
        3 * report["trainer_repeat_noise"]["mean_abs"],
    )
    trainer_delta = torch.as_tensor(report["trainer_updated"], dtype=torch.float64) - torch.as_tensor(
        report["trainer_zero"], dtype=torch.float64
    )
    sampler_delta = torch.as_tensor(report["updated"], dtype=torch.float64) - torch.as_tensor(
        report["zero"], dtype=torch.float64
    )
    report["update_delta"] = compare_logprobs(trainer_delta, sampler_delta)
    trainer_norm = trainer_delta.norm().item()
    sampler_norm = sampler_delta.norm().item()
    report["update_delta"].update(
        cosine=(
            (trainer_delta @ sampler_delta).item() / (trainer_norm * sampler_norm)
            if trainer_norm and sampler_norm
            else None
        ),
        scale=sampler_norm / trainer_norm if trainer_norm else None,
        relative_l2=(trainer_delta - sampler_delta).norm().item() / trainer_norm if trainer_norm else None,
    )
    if report["updated_parity"]["mean_abs"] >= atol:
        raise AssertionError("published adapter exceeds the trainer/inference parity tolerance")
    if report["sampler_change"]["mean_abs"] <= noise_budget:
        raise AssertionError("sampler did not measurably change")
    if report["trainer_change"]["mean_abs"] <= noise_budget:
        raise AssertionError("trainer did not measurably change")


def compare_logprobs(reference, actual):
    reference = torch.as_tensor(reference, dtype=torch.float64)
    actual = torch.as_tensor(actual, dtype=torch.float64)
    if reference.ndim != 1 or actual.ndim != 1:
        raise ValueError("logprob inputs must be one-dimensional")
    if reference.shape != actual.shape or reference.numel() == 0:
        raise ValueError("logprob inputs must have the same nonempty shape")
    if not torch.isfinite(reference).all() or not torch.isfinite(actual).all():
        raise ValueError("logprob inputs must be finite")
    error = (reference - actual).abs()
    return {
        "tokens": error.numel(),
        "mean_abs": error.mean().item(),
        "p99_abs": error.quantile(0.99).item(),
        "max_abs": error.max().item(),
    }

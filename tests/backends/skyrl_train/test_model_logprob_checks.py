"""CPU tests for the pure helpers behind examples/model_checks."""

import pytest
import torch

from examples.model_checks.logprob_checks import (
    build_probe_sequences,
    check_agreement,
    compare_logprobs,
    perturb_full_weights,
    perturb_lora_b,
    replicate_to_multiple,
)


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def test_probe_sequences_have_requested_lengths():
    sequences = build_probe_sequences(_Tokenizer())
    assert [len(s) for s in sequences] == [65, 129]


@pytest.mark.parametrize(
    "multiple,expected", [(1, [0, 1]), (2, [0, 1]), (3, [0, 1, 0]), (4, [0, 1, 0, 1]), (5, [0, 1, 0, 1, 0])]
)
def test_replicate_to_multiple_cycles_sequences(multiple, expected):
    sequences = [[0], [1]]
    replicated = replicate_to_multiple(sequences, multiple)
    assert len(replicated) % multiple == 0
    assert [s[0] for s in replicated] == expected


def test_replicate_to_multiple_rejects_bad_inputs():
    with pytest.raises(ValueError):
        replicate_to_multiple([[0]], 0)
    with pytest.raises(ValueError):
        replicate_to_multiple([], 2)


def test_compare_logprobs_reports_error_statistics():
    result = compare_logprobs([-1.0, -2.0, -3.0], [-1.0, -2.0, -3.3])
    assert result["tokens"] == 3
    assert result["mean_abs"] == pytest.approx(0.1)
    assert result["max_abs"] == pytest.approx(0.3)
    assert result["p99_abs"] == pytest.approx(0.3, abs=0.01)


@pytest.mark.parametrize(
    "reference,actual",
    [([], []), ([-1.0], [-1.0, -2.0]), ([-1.0], [float("nan")]), ([float("inf")], [-1.0]), ([[-1.0]], [[-1.0]])],
)
def test_compare_logprobs_rejects_misaligned_or_nonfinite(reference, actual):
    with pytest.raises(ValueError):
        compare_logprobs(reference, actual)


def test_check_agreement_bounds_mean_and_max():
    check_agreement({"mean_abs": 0.05, "max_abs": 0.5}, mean_atol=0.05, max_atol=0.5, label="ok")
    with pytest.raises(AssertionError, match="mean"):
        check_agreement({"mean_abs": 0.06, "max_abs": 0.1}, mean_atol=0.05, max_atol=0.5, label="x")
    with pytest.raises(AssertionError, match="max"):
        check_agreement({"mean_abs": 0.01, "max_abs": 0.6}, mean_atol=0.05, max_atol=0.5, label="x")


def test_perturb_lora_b_touches_only_zero_b_tensors_and_is_deterministic():
    base = torch.nn.Parameter(torch.ones(4), requires_grad=False)
    a = torch.nn.Parameter(torch.ones(4))
    b = torch.nn.Parameter(torch.zeros(4))
    replica = torch.nn.Parameter(torch.zeros(4))
    receipt = perturb_lora_b(
        [("weight", base), ("layer.adapter.linear_in.weight", a), ("layer.adapter.linear_out.weight", b)]
    )
    perturb_lora_b([("layer.adapter.linear_out.weight", replica)])
    assert torch.equal(base, torch.ones(4))
    assert torch.equal(a, torch.ones(4))
    assert torch.equal(b, replica)
    assert b.abs().sum() > 0
    assert receipt["changed_tensors"] == 1 and receipt["changed_elements"] == 4


def test_perturb_lora_b_scales_with_multiplier_and_accepts_peft_names():
    unit = torch.nn.Parameter(torch.zeros(3, 2))
    scaled = torch.nn.Parameter(torch.zeros(3, 2))
    perturb_lora_b([("q_proj.lora_B.default.weight", unit)], multiplier=1.0)
    perturb_lora_b([("q_proj.lora_B.default.weight", scaled)], multiplier=5.0)
    torch.testing.assert_close(scaled, unit * 5.0)


def test_perturb_lora_b_rejects_nonzero_b_and_unexpected_trainables():
    with pytest.raises(AssertionError, match="zero-initialized"):
        perturb_lora_b([("layer.adapter.linear_out.weight", torch.nn.Parameter(torch.ones(4)))])
    with pytest.raises(AssertionError, match="unexpected trainable"):
        perturb_lora_b([("layer.weight", torch.nn.Parameter(torch.zeros(4)))])
    with pytest.raises(AssertionError, match="no trainable"):
        perturb_lora_b([("layer.weight", torch.nn.Parameter(torch.zeros(4), requires_grad=False))])


def test_perturb_full_weights_scales_relative_to_magnitude_and_skips_frozen():
    frozen = torch.nn.Parameter(torch.ones(4), requires_grad=False)
    small = torch.nn.Parameter(torch.full((4,), 0.01))
    large = torch.nn.Parameter(torch.full((4,), 1.0))
    zero = torch.nn.Parameter(torch.zeros(4))
    receipt = perturb_full_weights([("frozen", frozen), ("small", small), ("large", large), ("zero", zero)])
    assert torch.equal(frozen, torch.ones(4))
    assert torch.equal(zero, torch.zeros(4))
    assert receipt["changed_tensors"] == 2
    small_delta = (small - 0.01).abs().mean().item()
    large_delta = (large - 1.0).abs().mean().item()
    assert 0 < small_delta < large_delta
    assert large_delta == pytest.approx(100 * small_delta, rel=0.5)

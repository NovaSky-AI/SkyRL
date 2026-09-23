import pytest
import torch

from examples.model_checks.logprob_checks import check_agreement, compare_logprobs
from examples.model_checks.lora_logprobs import perturb_adapters


def test_logprob_check_preserves_alignment_and_reports_tail_errors():
    result = compare_logprobs([-1, -2, -3], [-1, -2, -3.3])
    assert result["tokens"] == 3
    assert result["mean_abs"] == pytest.approx(0.1)
    assert result["max_abs"] == pytest.approx(0.3)
    assert result["p99_abs"] > 0.29


@pytest.mark.parametrize(
    "reference,actual",
    [
        ([], []),
        ([-1], [-1, -2]),
        ([-1], [float("nan")]),
        ([float("inf")], [-1]),
        ([[-1]], [[-1]]),
    ],
)
def test_logprob_check_rejects_missing_nonfinite_or_misaligned_scores(reference, actual):
    with pytest.raises(AssertionError):
        compare_logprobs(reference, actual)


def test_perturbation_is_adapter_only_and_replica_deterministic():
    base = torch.nn.Parameter(torch.ones(4), requires_grad=False)
    adapter = torch.nn.Parameter(torch.zeros(4))
    replica = torch.nn.Parameter(torch.zeros(4))
    a = torch.nn.Parameter(torch.ones(4))
    receipt = perturb_adapters(
        [
            ("weight", base),
            ("adapter.linear_in.weight", a),
            ("adapter.linear_out.weight", adapter),
        ]
    )
    perturb_adapters([("adapter.linear_out.weight", replica)])
    torch.testing.assert_close(base, torch.ones(4))
    torch.testing.assert_close(adapter, replica, rtol=0, atol=0)
    assert adapter.abs().sum() > 0
    torch.testing.assert_close(a, torch.ones(4), rtol=0, atol=0)
    assert receipt["changed_b_elements"] == 4


def test_perturbation_rejects_an_already_updated_adapter():
    with pytest.raises(AssertionError, match="zero-init B"):
        perturb_adapters([("adapter.linear_out.weight", torch.nn.Parameter(torch.ones(4)))])


@pytest.mark.parametrize("multiplier", [10, 32])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_explicit_stimulus_preserves_a_and_scales_seeded_b(multiplier, dtype):
    a = torch.nn.Parameter(torch.ones(4, dtype=dtype))
    b = torch.nn.Parameter(torch.zeros(4, dtype=dtype))
    direction = torch.nn.Parameter(torch.zeros_like(b))
    perturb_adapters([("adapter.linear_out.weight", direction)], multiplier=1)
    report = perturb_adapters(
        [("adapter.linear_in.weight", a), ("adapter.linear_out.weight", b)],
        multiplier=multiplier,
    )
    assert torch.equal(a, torch.ones_like(a))
    assert torch.equal(b, direction * multiplier)
    assert report["multiplier"] == multiplier


@pytest.mark.parametrize("multiplier", [0, -1, float("nan"), float("inf")])
def test_invalid_stimulus_fails_before_adapter_mutation(multiplier):
    b = torch.nn.Parameter(torch.zeros(4))
    with pytest.raises(ValueError, match="positive and finite"):
        perturb_adapters([("adapter.linear_out.weight", b)], multiplier=multiplier)
    assert torch.equal(b, torch.zeros_like(b))


def test_perturbation_rejects_full_weight_training():
    with pytest.raises(AssertionError, match="trainable base"):
        perturb_adapters([("weight", torch.nn.Parameter(torch.ones(4)))])


def test_perturbation_rejects_missing_trainable_adapters():
    with pytest.raises(AssertionError):
        perturb_adapters([("adapter.weight", torch.nn.Parameter(torch.ones(4), requires_grad=False))])


@pytest.mark.parametrize("mean_error,max_error", [(0.05, 0.5), (0.050001, 0.5), (0.05, 0.500001)])
def test_agreement_accepts_inclusive_limits_and_rejects_excess(mean_error, max_error):
    result = {"mean_abs": mean_error, "max_abs": max_error}
    if mean_error == 0.05 and max_error == 0.5:
        check_agreement(result, 0.05, 0.5)
    else:
        with pytest.raises(AssertionError):
            check_agreement(result, 0.05, 0.5)

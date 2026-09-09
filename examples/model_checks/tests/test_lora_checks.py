import pytest
import torch


from examples.model_checks.lora_logprobs import (
    check_initial_adapter,
    check_updated_adapter,
    check_withheld_publication,
    compare_logprobs,
    perturb_adapters,
)


@pytest.mark.parametrize(
    "fault",
    [None, "base", "stale", "frozen_sampler", "frozen_trainer", "updated_parity", "wrong_direction", "wrong_scale"],
)
def test_publication_checks_reject_broken_phases(fault):
    report = {
        "base": [-2.0, -3.0],
        "zero": [-2.0, -3.0],
        "trainer_zero": [-2.0, -3.0],
        "trainer_repeat": [-2.0, -3.0],
        "repeat": [-2.0, -3.0],
        "stale": [-2.0, -3.0],
        "trainer_updated": [-1.98, -2.98],
        "updated": [-1.98, -2.98],
    }
    if fault == "base":
        report["base"] = [-1.0, -2.0]
    elif fault == "stale":
        report["stale"] = [-1.98, -2.98]
    elif fault == "frozen_sampler":
        report["updated"] = report["zero"]
    elif fault == "frozen_trainer":
        report["trainer_updated"] = report["trainer_zero"]
    elif fault == "updated_parity":
        report["updated"] = [-1.0, -2.0]
    elif fault == "wrong_direction":
        report["updated"] = [-2.02, -3.02]
    elif fault == "wrong_scale":
        report["updated"] = [-1.96, -2.96]

    def check_all():
        check_initial_adapter(report, 0.05)
        check_withheld_publication(report)
        check_updated_adapter(report, 0.05, 0.005)

    if fault is None:
        check_all()
    else:
        with pytest.raises(AssertionError):
            check_all()


def test_update_comparison_cancels_a_fixed_backend_offset():
    report = {
        "base": [-2.0, -3.0],
        "zero": [-2.0, -3.0],
        "repeat": [-2.0, -3.0],
        "trainer_zero": [-1.96, -2.96],
        "trainer_repeat": [-1.96, -2.96],
        "trainer_updated": [-1.94, -2.94],
        "updated": [-1.98, -2.98],
    }
    check_initial_adapter(report, 0.05)
    check_updated_adapter(report, 0.05, 0.005)
    assert report["updated_parity"]["mean_abs"] == pytest.approx(0.04)
    assert report["update_delta"]["max_abs"] < 1e-12


def test_update_smaller_than_the_budget_cannot_qualify_publication():
    report = {
        "base": [-2.0],
        "zero": [-2.0],
        "repeat": [-2.0],
        "trainer_zero": [-2.0],
        "trainer_repeat": [-2.0],
        "trainer_updated": [-1.999],
        "updated": [-1.999],
    }
    check_initial_adapter(report, 0.05)
    with pytest.raises(AssertionError, match="update too small"):
        check_updated_adapter(report, 0.05, 0.005)


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
    receipt = perturb_adapters([("weight", base), ("adapter.weight", adapter)])
    perturb_adapters([("adapter.weight", replica)])
    torch.testing.assert_close(base, torch.ones(4))
    torch.testing.assert_close(adapter, replica, rtol=0, atol=0)
    assert adapter.abs().sum() > 0
    assert receipt["trainable_elements"] == 4


def test_perturbation_rejects_full_weight_training():
    with pytest.raises(AssertionError, match="trainable base"):
        perturb_adapters([("weight", torch.nn.Parameter(torch.ones(4)))])


def test_perturbation_rejects_missing_trainable_adapters():
    with pytest.raises(AssertionError):
        perturb_adapters([("adapter.weight", torch.nn.Parameter(torch.ones(4), requires_grad=False))])

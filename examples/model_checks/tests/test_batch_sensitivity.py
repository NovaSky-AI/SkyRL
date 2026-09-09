import pytest

from examples.model_checks.batch_sensitivity import compare_batches, compare_rows


def test_batch_change_is_reported_separately_from_repeat_noise():
    result = compare_batches(
        {"single": [[-1, -2]], "duplicate": [[-1.25, -2.25], [-1.25, -2.25]], "repeat": [[-1, -2]]}
    )
    assert result["batch_size"] == {"positions": 2, "mean_abs": 0.25, "max_abs": 0.25}
    assert result["repeat"]["max_abs"] == result["duplicate_rows"]["max_abs"] == 0


def test_duplicate_row_disagreement_cannot_be_hidden_by_averaging():
    result = compare_batches({"single": [[-1]], "duplicate": [[-1], [-3]], "repeat": [[-1.5]]})
    assert result["batch_size"]["max_abs"] == 0
    assert result["duplicate_rows"]["max_abs"] == 2
    assert result["repeat"]["max_abs"] == 0.5


@pytest.mark.parametrize("actual", [[], [-1, -2], [float("nan")], [float("inf")]])
def test_misaligned_or_nonfinite_scores_are_rejected(actual):
    with pytest.raises(ValueError):
        compare_rows([-1], actual)


def test_unexpected_row_count_is_rejected():
    with pytest.raises(ValueError):
        compare_batches({"single": [[-1]], "duplicate": [[-1]], "repeat": [[-1]]})

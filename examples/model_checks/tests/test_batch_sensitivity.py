import json

import pytest

from examples.model_checks.batch_sensitivity import (
    compare_batches,
    compare_rows,
    load_fixture,
)


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


@pytest.mark.parametrize("tokens", [[], [1], [1, -1], [1, True], [1, 2.5]])
def test_invalid_fixed_token_inputs_are_rejected(tmp_path, tokens):
    fixture = tmp_path / "tokens.json"
    fixture.write_text(json.dumps({"model_revision": "pinned", "tokens": tokens}))
    with pytest.raises(ValueError, match="token IDs"):
        load_fixture(fixture, tmp_path / "pinned")


def test_snapshot_revision_and_fixed_tokens_are_preserved(tmp_path):
    fixture = tmp_path / "tokens.json"
    fixture.write_text(json.dumps({"model_revision": "pinned", "tokens": [1, 2, 3]}))
    with pytest.raises(ValueError, match="pinned revision"):
        load_fixture(fixture, tmp_path / "other")
    assert load_fixture(fixture, tmp_path / "pinned")["tokens"] == [1, 2, 3]

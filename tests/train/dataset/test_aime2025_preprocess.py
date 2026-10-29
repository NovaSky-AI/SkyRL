"""
uv run --isolated --extra skyrl-train --extra dev pytest tests/train/dataset/test_aime2025_preprocess.py
"""

import pytest

from examples.train.algorithms.dapo.data_preprocess_aime2025 import normalize_aime_label


@pytest.mark.parametrize(
    "label, expected",
    [("70", "70"), ("336^\\circ", "336"), ("336^{\\circ}", "336"), (" 007 ", "7"), ("0", "0")],
)
def test_normalize_aime_label(label, expected):
    assert normalize_aime_label(label) == expected


@pytest.mark.parametrize("label", ["1000", "-1", "\\frac{1}{2}", "3.5", ""])
def test_normalize_aime_label_rejects_non_aime_answers(label):
    with pytest.raises(ValueError):
        normalize_aime_label(label)

"""
Tests for RayPPOTrainer._remove_tail_data.

Covers the batch/DP-shard truncation logic, including the case from
https://github.com/NovaSky-AI/SkyRL/issues/1609 where a batch smaller than
the shard stride was silently truncated to zero prompts and the run crashed
much later inside rollout-metrics aggregation.

uv run --extra dev pytest tests/train/test_remove_tail_data.py -v
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from skyrl.train.trainer import RayPPOTrainer


def _stub_trainer(lcm_dp_size: int, n_samples_per_prompt: int) -> SimpleNamespace:
    """Build a minimal stand-in for the trainer with just the attributes
    _remove_tail_data reads."""
    stub = SimpleNamespace()
    stub.dispatch = MagicMock()
    stub.dispatch.get_lcm_dp_size = MagicMock(return_value=lcm_dp_size)
    stub.cfg = SimpleNamespace(generator=SimpleNamespace(n_samples_per_prompt=n_samples_per_prompt))
    return stub


def test_truncates_to_multiple_of_stride():
    # lcm_dp_size=6, n_samples_per_prompt=4 -> stride=3: 7 prompts keep 6.
    stub = _stub_trainer(lcm_dp_size=6, n_samples_per_prompt=4)
    entries = list(range(7))
    assert RayPPOTrainer._remove_tail_data(stub, entries) == list(range(6))


def test_keeps_all_entries_when_stride_is_one():
    # lcm_dp_size divides n_samples_per_prompt -> stride=1: nothing removed.
    stub = _stub_trainer(lcm_dp_size=4, n_samples_per_prompt=4)
    entries = list(range(5))
    assert RayPPOTrainer._remove_tail_data(stub, entries) == entries


def test_raises_when_batch_is_smaller_than_stride():
    # Repro from issue #1609: lcm_dp_size=6, n_samples_per_prompt=4 ->
    # stride=3, but train_batch_size=2. Previously this returned an empty
    # list and the run died later with an opaque
    # "zero-size array to reduction operation minimum" ValueError.
    stub = _stub_trainer(lcm_dp_size=6, n_samples_per_prompt=4)
    with pytest.raises(ValueError, match="train_batch_size"):
        RayPPOTrainer._remove_tail_data(stub, list(range(2)))

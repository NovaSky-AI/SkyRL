"""Unit tests for SkyRLTrainBackend conversion logic.

Run with:
    pytest tests/tinker/backends/test_skyrl_train_backend.py -v --confcutdir=tests/tinker/backends
"""

import pytest
import torch

from tx.tinker import types

# Skip all tests if skyrl-train not available
skyrl_train = pytest.importorskip("skyrl_train")

from tx.tinker.backends.skyrl_train import SkyRLTrainBackend


def make_prepared_batch(
    all_input_ids: list[list[int]],
    all_targets: list[list[int]],
    all_token_weights: list[list[float]],
    request_batch_slices: list[tuple[str, str, int, int]] | None = None,
) -> types.PreparedModelPassBatch:
    """Helper to create PreparedModelPassBatch with minimal required fields."""
    n = len(all_input_ids)
    if request_batch_slices is None:
        request_batch_slices = [("req_0", "model_0", 0, n)]
    return types.PreparedModelPassBatch(
        all_input_ids=all_input_ids,
        all_targets=all_targets,
        all_token_weights=all_token_weights,
        all_sampling_logprobs=[[] for _ in range(n)],
        all_advantages=[[] for _ in range(n)],
        all_model_ids=["model_0"] * n,
        all_loss_fn_types=[0] * n,
        request_batch_slices=request_batch_slices,
    )


def create_backend_for_unit_test() -> SkyRLTrainBackend:
    """Create backend instance without calling __init__ (for unit testing conversion methods)."""
    return object.__new__(SkyRLTrainBackend)


class TestToTrainingBatchConversion:
    """Unit tests for _to_training_batch() conversion logic."""

    def test_basic_conversion(self):
        """Test basic conversion with single sequence."""
        backend = create_backend_for_unit_test()
        prepared_batch = make_prepared_batch(
            all_input_ids=[[1, 2, 3]],
            all_targets=[[10, 11]],
            all_token_weights=[[1.0, 1.0]],
        )

        batch = backend._to_training_batch(prepared_batch)

        # Sequence should be input_ids + last target token
        # [1, 2, 3] + [11] = [1, 2, 3, 11]
        assert batch["sequences"].shape == (1, 4)
        assert batch["sequences"][0].tolist() == [1, 2, 3, 11]

        # Attention mask should be all 1s (no padding needed)
        assert batch["attention_mask"][0].tolist() == [1, 1, 1, 1]

        # Loss mask should match token_weights
        assert batch["loss_mask"].shape == (1, 2)
        assert batch["loss_mask"][0].tolist() == [1.0, 1.0]

        # Response mask should be 1s for response positions
        assert batch["response_mask"][0].tolist() == [1, 1]

        # Metadata
        assert batch.metadata["response_length"] == 2

    def test_left_padding_alignment(self):
        """Test that shorter sequences are left-padded correctly."""
        backend = create_backend_for_unit_test()
        prepared_batch = make_prepared_batch(
            all_input_ids=[[1, 2, 3], [1, 2, 3, 4, 5]],  # lengths 3 and 5
            all_targets=[[10, 11], [20, 21, 22, 23]],  # lengths 2 and 4
            all_token_weights=[[1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
        )

        batch = backend._to_training_batch(prepared_batch)

        # Max seq len: 5 + 1 = 6 (input_ids + last target)
        # First sequence: [1,2,3,11] padded to [0,0,1,2,3,11]
        # Second sequence: [1,2,3,4,5,23] (no padding)
        assert batch["sequences"].shape == (2, 6)
        assert batch["sequences"][0].tolist() == [0, 0, 1, 2, 3, 11]
        assert batch["sequences"][1].tolist() == [1, 2, 3, 4, 5, 23]

        # Attention masks
        assert batch["attention_mask"][0].tolist() == [0, 0, 1, 1, 1, 1]
        assert batch["attention_mask"][1].tolist() == [1, 1, 1, 1, 1, 1]

        # Loss masks (max response len = 4)
        # First: [1.0, 1.0] padded to [0.0, 0.0, 1.0, 1.0]
        # Second: [1.0, 1.0, 1.0, 1.0] (no padding)
        assert batch["loss_mask"].shape == (2, 4)
        assert batch["loss_mask"][0].tolist() == [0.0, 0.0, 1.0, 1.0]
        assert batch["loss_mask"][1].tolist() == [1.0, 1.0, 1.0, 1.0]

        # Response masks
        assert batch["response_mask"][0].tolist() == [0, 0, 1, 1]
        assert batch["response_mask"][1].tolist() == [1, 1, 1, 1]

    def test_variable_token_weights(self):
        """Test that token weights are preserved correctly."""
        backend = create_backend_for_unit_test()
        prepared_batch = make_prepared_batch(
            all_input_ids=[[1, 2, 3]],
            all_targets=[[10, 11, 12]],
            all_token_weights=[[0.5, 1.0, 0.0]],  # Variable weights
        )

        batch = backend._to_training_batch(prepared_batch)

        # Loss mask should preserve the weights
        assert batch["loss_mask"][0].tolist() == [0.5, 1.0, 0.0]

    def test_empty_targets(self):
        """Test handling of empty targets list."""
        backend = create_backend_for_unit_test()
        prepared_batch = make_prepared_batch(
            all_input_ids=[[1, 2, 3]],
            all_targets=[[]],  # Empty targets
            all_token_weights=[[]],
        )

        batch = backend._to_training_batch(prepared_batch)

        # Sequence should just be input_ids (no target to append)
        assert batch["sequences"][0].tolist() == [1, 2, 3]

    def test_empty_batch(self):
        """Test handling of empty batch returns empty TrainingInputBatch."""
        backend = create_backend_for_unit_test()
        prepared_batch = make_prepared_batch(
            all_input_ids=[],
            all_targets=[],
            all_token_weights=[],
            request_batch_slices=[],
        )

        batch = backend._to_training_batch(prepared_batch)

        # Should return empty TrainingInputBatch (no keys set)
        # TrainingInputBatch is a dict subclass, verify it has no keys
        assert len(batch.keys()) == 0, f"Expected empty batch, got keys: {list(batch.keys())}"
        assert "sequences" not in batch
        assert "attention_mask" not in batch
        assert "loss_mask" not in batch
        assert "response_mask" not in batch

    def test_tensor_dtypes(self):
        """Test that tensors have correct dtypes."""
        backend = create_backend_for_unit_test()
        prepared_batch = make_prepared_batch(
            all_input_ids=[[1, 2, 3]],
            all_targets=[[10, 11]],
            all_token_weights=[[1.0, 1.0]],
        )

        batch = backend._to_training_batch(prepared_batch)

        assert batch["sequences"].dtype == torch.long
        assert batch["attention_mask"].dtype == torch.long
        assert batch["loss_mask"].dtype == torch.float32
        assert batch["response_mask"].dtype == torch.long

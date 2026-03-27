"""
CPU-only tests for VLM (vision-language model) data plumbing.

Verifies that pixel_values / image_grid_thw fields flow correctly through
batch_to_experience and Experience.to_device.
"""

import torch

from skyrl.backends.skyrl_train.training_batch import TensorList, TrainingInputBatch
from skyrl.backends.skyrl_train.workers.worker_utils import BatchIterator


def _make_batch(batch_size=2, seq_len=8, response_length=4, include_vision=True):
    """Build a minimal TrainingInputBatch, optionally with vision fields."""
    data = {
        "sequences": torch.randint(0, 1000, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "response_mask": torch.ones(batch_size, response_length, dtype=torch.long),
    }
    if include_vision:
        # pixel_values: each sample has a different number of patches
        pv_tensors = [torch.randn(3 + i, 1176) for i in range(batch_size)]
        data["pixel_values"] = TensorList(pv_tensors)
        # image_grid_thw: each sample has a different number of images
        thw_tensors = [torch.tensor([[1, 2 + i, 2 + i]]) for i in range(batch_size)]
        data["image_grid_thw"] = TensorList(thw_tensors)

    batch = TrainingInputBatch(data)
    batch.metadata = {"response_length": response_length}
    return batch


class TestBatchToExperienceVisionFields:
    def test_batch_to_experience_extracts_vision_fields(self):
        """Vision fields from TrainingInputBatch should appear on Experience."""
        batch = _make_batch(include_vision=True)
        exp = BatchIterator.batch_to_experience(batch)

        assert exp.pixel_values is not None
        assert exp.image_grid_thw is not None
        assert isinstance(exp.pixel_values, TensorList)
        assert isinstance(exp.image_grid_thw, TensorList)
        assert len(exp.pixel_values) == 2
        assert len(exp.image_grid_thw) == 2

    def test_batch_to_experience_without_vision_fields(self):
        """When batch has no vision keys, Experience fields should be None."""
        batch = _make_batch(include_vision=False)
        exp = BatchIterator.batch_to_experience(batch)

        assert exp.pixel_values is None
        assert exp.image_grid_thw is None


class TestExperienceToDeviceVision:
    def test_experience_to_device_with_vision_fields(self):
        """to_device should transfer vision TensorList fields."""
        batch = _make_batch(include_vision=True)
        exp = BatchIterator.batch_to_experience(batch)

        # Move to CPU (no-op on CPU but exercises the code path)
        exp.to_device(torch.device("cpu"))

        assert exp.pixel_values is not None
        assert exp.image_grid_thw is not None
        assert exp.pixel_values.device == torch.device("cpu")
        assert exp.image_grid_thw.device == torch.device("cpu")

    def test_experience_to_device_without_vision_fields(self):
        """to_device should be safe when vision fields are None."""
        batch = _make_batch(include_vision=False)
        exp = BatchIterator.batch_to_experience(batch)
        # Should not raise
        exp.to_device(torch.device("cpu"))
        assert exp.pixel_values is None
        assert exp.image_grid_thw is None

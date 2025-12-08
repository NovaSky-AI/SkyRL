import pytest
import torch
from skyrl_train.weight_sync import WeightChunk


class TestWeightChunk:
    """Tests for WeightChunk dataclass."""

    def test_initialization_basic(self):
        """Test basic initialization with valid data."""
        names = ["layer1.weight", "layer1.bias"]
        dtypes = ["torch.float32", "torch.float32"]
        shapes = [[4, 3], [4]]
        tensors = [torch.randn(4, 3), torch.randn(4)]

        chunk = WeightChunk(names=names, dtypes=dtypes, shapes=shapes, tensors=tensors)

        assert chunk.names == names
        assert chunk.dtypes == dtypes
        assert chunk.shapes == shapes
        assert len(chunk.tensors) == 2

    def test_validation_length_mismatch(self):
        """Test that validation catches length mismatches."""
        with pytest.raises(ValueError, match="All lists must have the same length"):
            WeightChunk(
                names=["layer1.weight", "layer1.bias"],
                dtypes=["torch.float32"],  # Wrong length
                shapes=[[4, 3], [4]],
                tensors=[torch.randn(4, 3), torch.randn(4)],
            )

        with pytest.raises(ValueError, match="All lists must have the same length"):
            WeightChunk(
                names=["layer1.weight"],
                dtypes=["torch.float32"],
                shapes=[[4, 3], [4]],  # Wrong length
                tensors=[torch.randn(4, 3)],
            )

    def test_len(self):
        """Test __len__ returns number of parameters."""
        chunk = WeightChunk(
            names=["layer1.weight", "layer1.bias", "layer2.weight"],
            dtypes=["torch.float32"] * 3,
            shapes=[[4, 3], [4], [3, 2]],
            tensors=[torch.randn(4, 3), torch.randn(4), torch.randn(3, 2)],
        )

        assert len(chunk) == 3

    def test_total_numel(self):
        """Test total_numel cached property."""
        tensors = [
            torch.randn(4, 3),  # 12 elements
            torch.randn(4),  # 4 elements
            torch.randn(3, 2),  # 6 elements
        ]
        chunk = WeightChunk(
            names=["layer1.weight", "layer1.bias", "layer2.weight"],
            dtypes=["torch.float32"] * 3,
            shapes=[[4, 3], [4], [3, 2]],
            tensors=tensors,
        )

        assert chunk.total_numel == 12 + 4 + 6

    def test_total_size_bytes(self):
        """Test total_size_bytes cached property."""
        tensors = [
            torch.randn(4, 3, dtype=torch.float32),  # 12 * 4 = 48 bytes
            torch.randn(4, dtype=torch.float32),  # 4 * 4 = 16 bytes
        ]
        chunk = WeightChunk(
            names=["layer1.weight", "layer1.bias"],
            dtypes=["torch.float32"] * 2,
            shapes=[[4, 3], [4]],
            tensors=tensors,
        )

        assert chunk.total_size_bytes == 48 + 16

    def test_total_size_bytes_mixed_dtypes(self):
        """Test total_size_bytes with mixed dtypes."""
        tensors = [
            torch.randn(10, dtype=torch.float32),  # 10 * 4 = 40 bytes
            torch.randn(10, dtype=torch.bfloat16),  # 10 * 2 = 20 bytes
        ]
        chunk = WeightChunk(
            names=["layer1.weight", "layer1.bias"],
            dtypes=["torch.float32", "torch.bfloat16"],
            shapes=[[10], [10]],
            tensors=tensors,
        )

        assert chunk.total_size_bytes == 40 + 20

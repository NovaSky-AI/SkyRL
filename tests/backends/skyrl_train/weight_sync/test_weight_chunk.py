import pytest
import torch

from skyrl.backends.skyrl_train.weight_sync import WeightChunk
from skyrl.backends.skyrl_train.weight_sync.base import (
    cuda_uuid_to_str,
    get_weight_chunk_metadata,
    iter_single_dtype_chunks,
)


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


def test_single_dtype_chunk_partition_preserves_first_seen_dtype_order():
    chunk = WeightChunk(
        names=["w0", "s0", "w1", "b0"],
        dtypes=["ignored"] * 4,
        shapes=[[4], [1], [8], [2]],
        tensors=[
            torch.empty((4,), dtype=torch.float8_e4m3fn),
            torch.ones((1,), dtype=torch.float32),
            torch.empty((8,), dtype=torch.float8_e4m3fn),
            torch.ones((2,), dtype=torch.bfloat16),
        ],
    )

    chunks = list(iter_single_dtype_chunks(chunk))

    assert [subchunk.names for subchunk in chunks] == [["w0", "w1"], ["s0"], ["b0"]]
    assert [[tensor.dtype for tensor in subchunk.tensors] for subchunk in chunks] == [
        [torch.float8_e4m3fn, torch.float8_e4m3fn],
        [torch.float32],
        [torch.bfloat16],
    ]


def test_weight_chunk_metadata_uses_actual_tensor_dtypes():
    chunk = WeightChunk(
        names=["w", "scale", "norm"],
        dtypes=["stale"] * 3,
        shapes=[[999], [999], [999]],
        tensors=[
            torch.empty((4,), dtype=torch.float8_e4m3fn),
            torch.empty((1,), dtype=torch.float32),
            torch.empty((2,), dtype=torch.bfloat16),
        ],
    )

    assert get_weight_chunk_metadata(chunk) == {
        "names": ["w", "scale", "norm"],
        "dtype_names": ["float8_e4m3fn", "float32", "bfloat16"],
        "shapes": [[4], [1], [2]],
    }


@pytest.mark.parametrize("uuid", ["GPU-123", b"GPU-123"])
def test_cuda_uuid_to_str_normalizes_string_and_bytes(uuid):
    assert cuda_uuid_to_str(uuid) == "GPU-123"

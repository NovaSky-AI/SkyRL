"""Tests for VLM-related TensorBatch list[Tensor] support and training batch construction."""

import pickle

import pytest
import torch

jaxtyping = pytest.importorskip("jaxtyping")

from skyrl.backends.skyrl_train.training_batch import TensorBatch  # noqa


# ── TensorBatch list[Tensor] support ──────────────────────────────────────


def _make_batch_with_list():
    """Helper to create a TensorBatch with mixed Tensor and list[Tensor] fields."""
    batch_size = 3
    seq_len = 4
    return TensorBatch(
        {
            "sequences": torch.randn(batch_size, seq_len),
            "pixel_values": [torch.randn(2, 8), torch.randn(4, 8), torch.randn(1, 8)],
        }
    )


def test_list_tensor_init():
    data = _make_batch_with_list()
    assert data.batch_size == 3
    assert len(data["pixel_values"]) == 3
    assert data["pixel_values"][1].shape == (4, 8)


def test_list_tensor_batch_size_mismatch():
    with pytest.raises(ValueError, match="Batch size mismatch"):
        TensorBatch(
            {
                "sequences": torch.randn(3, 4),
                "pixel_values": [torch.randn(2, 8), torch.randn(4, 8)],  # only 2, not 3
            }
        )


def test_list_tensor_setitem():
    data = _make_batch_with_list()
    new_pv = [torch.randn(3, 8), torch.randn(5, 8), torch.randn(2, 8)]
    data["pixel_values"] = new_pv
    assert len(data["pixel_values"]) == 3

    with pytest.raises(ValueError, match="Batch size mismatch"):
        data["pixel_values"] = [torch.randn(1, 8)]  # wrong size


def test_list_tensor_chunk():
    data = TensorBatch(
        {
            "sequences": torch.randn(4, 3),
            "pixel_values": [torch.randn(2, 8) for _ in range(4)],
        }
    )
    chunks = data.chunk(2)
    assert len(chunks) == 2
    assert len(chunks[0]["pixel_values"]) == 2
    assert len(chunks[1]["pixel_values"]) == 2
    assert chunks[0]["sequences"].shape == (2, 3)


def test_list_tensor_slice():
    data = TensorBatch(
        {
            "sequences": torch.randn(4, 3),
            "pixel_values": [torch.randn(i + 1, 8) for i in range(4)],
        }
    )
    sliced = data.slice(1, 3)
    assert len(sliced) == 2
    assert len(sliced["pixel_values"]) == 2
    # pixel_values[0] should be the original index 1 (shape (2, 8))
    assert sliced["pixel_values"][0].shape == (2, 8)


def test_list_tensor_cat():
    batch1 = TensorBatch(
        {
            "sequences": torch.randn(2, 3),
            "pixel_values": [torch.randn(1, 8), torch.randn(2, 8)],
        }
    )
    batch2 = TensorBatch(
        {
            "sequences": torch.randn(2, 3),
            "pixel_values": [torch.randn(3, 8), torch.randn(4, 8)],
        }
    )
    catted = TensorBatch.cat([batch1, batch2])
    assert catted.batch_size == 4
    assert len(catted["pixel_values"]) == 4
    assert catted["pixel_values"][2].shape == (3, 8)


def test_list_tensor_repeat():
    data = TensorBatch(
        {
            "sequences": torch.tensor([1, 2, 3]),
            "pixel_values": [torch.randn(1, 8), torch.randn(2, 8), torch.randn(3, 8)],
        }
    )
    repeated = data.repeat(2)
    assert len(repeated) == 6
    assert len(repeated["pixel_values"]) == 6
    # list repeat: [a, b, c, a, b, c]
    assert repeated["pixel_values"][3].shape == repeated["pixel_values"][0].shape


def test_list_tensor_repeat_interleave():
    data = TensorBatch(
        {
            "sequences": torch.tensor([1, 2]),
            "pixel_values": [torch.randn(1, 8), torch.randn(2, 8)],
        }
    )
    repeated = data.repeat_interleave(3)
    assert len(repeated) == 6
    assert len(repeated["pixel_values"]) == 6
    # repeat_interleave: [a, a, a, b, b, b]
    assert repeated["pixel_values"][0].shape == repeated["pixel_values"][1].shape
    assert repeated["pixel_values"][0].shape == repeated["pixel_values"][2].shape


def test_list_tensor_to():
    data = _make_batch_with_list()
    data.to(dtype=torch.float16)
    assert data["sequences"].dtype == torch.float16
    for t in data["pixel_values"]:
        assert t.dtype == torch.float16


def test_list_tensor_contiguous():
    # Make a non-contiguous tensor via transpose
    t = torch.randn(8, 2).T  # (2, 8) non-contiguous
    assert not t.is_contiguous()
    data = TensorBatch(
        {
            "sequences": torch.randn(2, 3),
            "pixel_values": [t, torch.randn(3, 8)],
        }
    )
    data.contiguous()
    assert data["pixel_values"][0].is_contiguous()


def test_list_tensor_eq():
    pv = [torch.randn(2, 8), torch.randn(3, 8)]
    data1 = TensorBatch({"sequences": torch.tensor([1, 2]), "pixel_values": pv})
    data2 = TensorBatch({"sequences": torch.tensor([1, 2]), "pixel_values": pv})
    assert data1 == data2

    data3 = TensorBatch(
        {
            "sequences": torch.tensor([1, 2]),
            "pixel_values": [torch.randn(2, 8), torch.randn(3, 8)],
        }
    )
    assert data1 != data3


def test_list_tensor_pickle_roundtrip():
    data = _make_batch_with_list()
    data.metadata = {"test": True}

    pickled = pickle.dumps(data)
    unpickled = pickle.loads(pickled)

    assert unpickled.batch_size == data.batch_size
    assert unpickled.metadata == data.metadata
    assert len(unpickled["pixel_values"]) == len(data["pixel_values"])
    for orig, restored in zip(data["pixel_values"], unpickled["pixel_values"]):
        assert torch.equal(orig, restored)
    assert torch.equal(unpickled["sequences"], data["sequences"])


def test_list_tensor_getitem_slice():
    """Test __getitem__ with slice for list fields."""
    data = TensorBatch(
        {
            "sequences": torch.tensor([1, 2, 3, 4]),
            "pixel_values": [torch.randn(i + 1, 8) for i in range(4)],
        }
    )
    sliced = data[:2]
    assert len(sliced) == 2
    assert len(sliced["pixel_values"]) == 2


# ── VLM _to_training_batch integration ────────────────────────────────────


def test_to_training_batch_with_image_chunks():
    """Test _to_training_batch produces pixel_values/image_grid_thw when images are present."""
    from unittest.mock import MagicMock

    from skyrl.backends.skyrl_train.utils.vision_utils import ProcessedImage
    from skyrl.tinker.types import (
        EncodedTextChunk,
        ImageChunk,
        ModelInput,
        PreparedModelPassBatch,
    )

    # Build a fake PreparedModelPassBatch with one image sample and one text sample
    image_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 100  # fake JPEG header
    model_input_with_image = ModelInput(
        chunks=[
            EncodedTextChunk(tokens=[1, 2, 3]),
            ImageChunk(data=image_bytes, format="jpeg"),
            EncodedTextChunk(tokens=[4, 5]),
        ]
    )
    model_input_text_only = ModelInput(
        chunks=[EncodedTextChunk(tokens=[10, 11, 12, 13, 14])]
    )

    prepared = PreparedModelPassBatch(
        all_input_ids=[[1, 2, 3, 4, 5], [10, 11, 12, 13, 14]],
        all_targets=[[100, 101], [200, 201]],
        all_token_weights=[[1.0, 1.0], [1.0, 1.0]],
        all_sampling_logprobs=[[], []],
        all_advantages=[[], []],
        all_model_ids=["m1", "m1"],
        all_loss_fns=["cross_entropy", "cross_entropy"],
        all_loss_fn_configs=[None, None],
        all_model_inputs=[model_input_with_image, model_input_text_only],
        request_batch_slices=[("r1", "m1", 0, 2)],
    )

    # Create a mock backend to test _to_training_batch
    from skyrl.backends.skyrl_train_backend import SkyRLTrainBackend

    # Mock the vision processor
    mock_processor = MagicMock()
    fake_processed = ProcessedImage(
        pixel_values=torch.randn(4, 16),
        image_grid_thw=torch.tensor([[1, 2, 2]]),
        num_tokens=4,
    )
    mock_processor.process_image.return_value = fake_processed

    # We can't easily construct a full backend, so we'll test the method directly
    # by creating a minimal mock
    backend = object.__new__(SkyRLTrainBackend)
    backend._vision_processor = mock_processor
    backend._image_token_id = 999
    backend._tokenizer = MagicMock()
    backend._tokenizer.pad_token_id = 0

    batch = backend._to_training_batch(prepared)

    assert "pixel_values" in batch
    assert "image_grid_thw" in batch
    assert isinstance(batch["pixel_values"], list)
    assert len(batch["pixel_values"]) == 2
    # First sample has image data
    assert batch["pixel_values"][0].shape == (4, 16)
    # Second sample is text-only (empty)
    assert batch["pixel_values"][1].numel() == 0
    # Verify image token was inserted: the first sample's sequence should contain 999 tokens
    seq = batch["sequences"][0].tolist()
    assert seq.count(999) == 4, f"Expected 4 image tokens, got {seq.count(999)}"

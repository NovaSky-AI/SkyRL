import pickle

import numpy as np
import pytest
import ray
import torch

from skyrl.backends.skyrl_train.training_batch import (
    TensorBatch,
    TensorList,
    TrainingInput,
    TrainingInputBatch,
    pad_training_input_batch,
)


def test_train_batch_initialization():
    # Test basic initialization
    batch_size = 3
    seq_len = 4
    sequences = torch.randn(batch_size, seq_len)
    attention_mask = torch.ones(batch_size, seq_len)
    loss_mask = torch.ones(batch_size, seq_len)
    response_mask = torch.ones(batch_size, seq_len)

    data = TensorBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "response_mask": response_mask,
        }
    )
    assert isinstance(data, TensorBatch)
    assert data.batch_size == batch_size
    assert torch.equal(data["sequences"], sequences)
    assert torch.equal(data["attention_mask"], attention_mask)


def test_train_batch_validation():
    # Test validation of batch sizes
    batch_size = 3
    seq_len = 4
    sequences = torch.randn(batch_size, seq_len)
    attention_mask = torch.ones(batch_size - 1, seq_len)  # Different size

    with pytest.raises(ValueError, match="Batch size mismatch"):
        batch = TensorBatch(
            {
                "sequences": sequences,
                "attention_mask": attention_mask,
            }
        )
        TensorBatch(batch=batch, metadata={})


def test_train_batch_chunk():
    batch_size = 4
    seq_len = 3
    sequences = torch.randn(batch_size, seq_len)
    attention_mask = torch.ones(batch_size, seq_len)
    data = TensorBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
        }
    )

    chunks = data.chunk(2)
    assert len(chunks) == 2
    assert chunks[0].batch_size == 2
    assert chunks[1].batch_size == 2
    assert chunks[0]["sequences"].shape == (2, seq_len)
    assert chunks[1]["sequences"].shape == (2, seq_len)


def test_train_batch_slice():
    batch_size = 4
    seq_len = 3
    sequences = torch.randn(batch_size, seq_len)
    attention_mask = torch.ones(batch_size, seq_len)
    data = TensorBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
        }
    )

    sliced = data.slice(1, 3)
    assert len(sliced) == 2
    assert sliced["sequences"].shape == (2, seq_len)
    assert sliced["attention_mask"].shape == (2, seq_len)


def test_train_batch_to_dtype():
    batch_size = 3
    seq_len = 4
    sequences = torch.randn(batch_size, seq_len)
    attention_mask = None
    data = TensorBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
        }
    )

    data.to(dtype=torch.float16)
    assert data["sequences"].dtype == torch.float16
    assert data["attention_mask"] is None


def test_train_batch_select():
    batch_size = 3
    seq_len = 4
    sequences = torch.randn(batch_size, seq_len)
    attention_mask = torch.ones(batch_size, seq_len)
    loss_mask = torch.ones(batch_size, seq_len)
    metadata = {"info": "test", "extra": "data"}

    data = TensorBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
        }
    )
    data.metadata = metadata

    selected = data.select(["sequences", "attention_mask"], ["info"])
    assert "sequences" in selected
    assert "attention_mask" in selected
    assert "loss_mask" not in selected
    assert "info" in selected.metadata
    assert "extra" not in selected.metadata


def test_train_batch_cat():
    batch_size = 3
    seq_len = 4
    sequences1 = torch.randn(batch_size, seq_len)
    attention_mask1 = torch.ones(batch_size, seq_len)
    data1 = TensorBatch(
        {
            "sequences": sequences1,
            "attention_mask": attention_mask1,
        }
    )
    sequences2 = torch.randn(batch_size, seq_len)
    attention_mask2 = torch.ones(batch_size, seq_len)
    data2 = TensorBatch(
        {
            "sequences": sequences2,
            "attention_mask": attention_mask2,
        }
    )

    concatenated = data1.cat([data1, data2])
    assert len(concatenated) == 2 * batch_size
    assert concatenated["sequences"].shape == (2 * batch_size, seq_len)
    assert concatenated["attention_mask"].shape == (2 * batch_size, seq_len)


def test_train_batch_pickle():
    # Test pickle serialization
    batch_size = 3
    seq_len = 4
    sequences = torch.randn(batch_size, seq_len)
    attention_mask = torch.ones(batch_size, seq_len)
    data = TensorBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
        }
    )
    metadata = {"info": "test"}
    data.metadata = metadata

    # Serialize
    pickled = pickle.dumps(data)

    # Deserialize
    unpickled = pickle.loads(pickled)

    # Verify all components are preserved
    assert len(unpickled) == len(data)
    assert all(torch.equal(unpickled[k], data[k]) for k in data.keys())
    assert unpickled.metadata == data.metadata


def test_train_batch_pickle_bfloat16():
    """Test pickle serialization with bfloat16 tensors (uses torch fallback path)."""
    batch_size = 3
    seq_len = 4
    # bfloat16 is not supported by numpy, so this tests the torch.save fallback
    sequences_bf16 = torch.randn(batch_size, seq_len, dtype=torch.bfloat16)
    # Mix with regular float32 to test both paths in same batch
    attention_mask_f32 = torch.ones(batch_size, seq_len, dtype=torch.float32)
    # Also test int64 (numpy-compatible)
    indices = torch.arange(batch_size, dtype=torch.int64)

    data = TensorBatch(
        {
            "sequences": sequences_bf16,
            "attention_mask": attention_mask_f32,
            "indices": indices,
        }
    )
    metadata = {"dtype_test": "bfloat16"}
    data.metadata = metadata

    # Serialize
    pickled = pickle.dumps(data)

    # Deserialize
    unpickled = pickle.loads(pickled)

    # Verify dtypes are preserved
    assert unpickled["sequences"].dtype == torch.bfloat16, "bfloat16 dtype not preserved"
    assert unpickled["attention_mask"].dtype == torch.float32, "float32 dtype not preserved"
    assert unpickled["indices"].dtype == torch.int64, "int64 dtype not preserved"

    # Verify data is preserved (use float() for bfloat16 comparison due to precision)
    assert torch.allclose(unpickled["sequences"].float(), data["sequences"].float()), "bfloat16 data mismatch"
    assert torch.equal(unpickled["attention_mask"], data["attention_mask"]), "float32 data mismatch"
    assert torch.equal(unpickled["indices"], data["indices"]), "int64 data mismatch"

    # Verify metadata preserved
    assert unpickled.metadata == data.metadata


def test_train_batch_pickle_uint_dtypes():
    """Test pickle serialization with uint8/uint16 tensors.

    Without the __reduce__ patch in training_batch.py, pickle's default dict-subclass protocol
    separately serializes raw dict items through torch's storage-level pickle,
    which fails for uint16 (UntypedStorage has no .dtype attribute). This tests
    that we are correctly skipping the default dict-subclass protocol and
    using the custom __reduce__ method to serialize the TensorBatch object.
    """
    batch_size = 3
    seq_len = 4
    num_layers = 2
    topk = 2

    data = TensorBatch(
        {
            "sequences": torch.randn(batch_size, seq_len),
            "rollout_expert_indices_u16": torch.randint(
                0, 256, (batch_size, seq_len, num_layers, topk), dtype=torch.uint16
            ),
            "rollout_expert_indices_u8": torch.randint(
                0, 128, (batch_size, seq_len, num_layers, topk), dtype=torch.uint8
            ),
        }
    )
    data.metadata = {"dtype_test": "uint"}

    pickled = pickle.dumps(data)
    unpickled = pickle.loads(pickled)

    assert unpickled["rollout_expert_indices_u16"].dtype == torch.uint16
    assert unpickled["rollout_expert_indices_u8"].dtype == torch.uint8
    assert torch.equal(unpickled["rollout_expert_indices_u16"], data["rollout_expert_indices_u16"])
    assert torch.equal(unpickled["rollout_expert_indices_u8"], data["rollout_expert_indices_u8"])
    assert torch.equal(unpickled["sequences"], data["sequences"])
    assert unpickled.metadata == data.metadata


def test_train_batch_setitem():
    batch_size = 3
    seq_len = 4
    sequences = torch.randn(batch_size, seq_len)
    attention_mask = torch.ones(batch_size, seq_len)
    data = TensorBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
        }
    )

    # Test setting tensor
    new_sequences = torch.randn(batch_size, seq_len)
    data["sequences"] = new_sequences
    assert torch.equal(data["sequences"], new_sequences)

    # Test invalid tensor shape
    with pytest.raises(ValueError, match="Batch size mismatch"):
        data["sequences"] = torch.randn(batch_size + 1, seq_len)

    # Test invalid types
    # 1. string
    with pytest.raises(ValueError, match="must be a tensor"):
        data["sequences"] = "invalid"
    # 2. numpy array
    with pytest.raises(ValueError, match="must be a tensor"):
        data["sequences"] = np.zeros((batch_size, seq_len))


def test_train_batch_ray_serialization():
    data = TensorBatch(
        **{"a": torch.tensor([1.2, 2.4, 3.6, 4.8]), "b": torch.tensor([4, 5, 6, 7])},
    )
    data.metadata = {"hello": "world"}

    def _task(inp: TensorBatch):
        assert inp == data

    _inp_ray = ray.put(data)
    ray.remote(_task).remote(_inp_ray)


def test_train_batch_repeat():
    batch = {"a": torch.tensor([1, 2, 3]), "b": torch.tensor([4, 5, 6])}
    data = TensorBatch(**batch)
    data.metadata = {"d": 1, "e": "test"}
    repeated = data.repeat(2)
    assert len(repeated) == 6
    assert torch.equal(repeated["a"], torch.tensor([1, 2, 3, 1, 2, 3]))
    assert torch.equal(repeated["b"], torch.tensor([4, 5, 6, 4, 5, 6]))
    assert repeated.metadata == {"d": 1, "e": "test"}


def test_train_batch_repeat_interleave():
    batch = {"a": torch.tensor([1, 2, 3]), "b": torch.tensor([4, 5, 6])}
    data = TensorBatch(**batch)
    data.metadata = {"c": "test"}
    repeated = data.repeat_interleave(2)
    assert len(repeated) == 6
    assert torch.equal(repeated["a"], torch.tensor([1, 1, 2, 2, 3, 3]))
    assert torch.equal(repeated["b"], torch.tensor([4, 4, 5, 5, 6, 6]))
    assert repeated.metadata == {"c": "test"}


def test_train_batch_get_item():
    batch = {"a": torch.tensor([1, 2, 3, 4]), "b": torch.tensor([4, 5, 6, 7])}
    data = TensorBatch(**batch)
    data.metadata = {"c": "test"}
    new_data = data[:2]
    assert torch.equal(new_data["a"], torch.tensor([1, 2]))
    assert torch.equal(new_data["b"], torch.tensor([4, 5]))


# ── TensorList unit tests ────────────────────────────────────────────────────


def test_tensor_list_basic():
    """Create TensorList, verify len, indexing, slicing."""
    t0 = torch.randn(3, 4)
    t1 = torch.randn(5, 4)
    t2 = torch.randn(2, 4)
    tl = TensorList([t0, t1, t2])

    assert len(tl) == 3
    assert torch.equal(tl[0], t0)
    assert torch.equal(tl[1], t1)
    assert torch.equal(tl[2], t2)

    sliced = tl[0:2]
    assert isinstance(sliced, TensorList)
    assert len(sliced) == 2
    assert torch.equal(sliced[0], t0)
    assert torch.equal(sliced[1], t1)


def test_tensor_list_device_transfer():
    """.to(device) moves all tensors (CPU-only test)."""
    tl = TensorList([torch.randn(2, 3), torch.randn(4, 3)])
    moved = tl.to(device=torch.device("cpu"))
    assert len(moved) == 2
    assert moved.device == torch.device("cpu")
    # Verify data unchanged
    assert torch.equal(moved[0], tl[0])


def test_tensor_list_repeat():
    """.repeat(n) tiles the list."""
    tl = TensorList([torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])])
    repeated = tl.repeat(3)
    assert len(repeated) == 9
    assert torch.equal(repeated[0], torch.tensor([1.0]))
    assert torch.equal(repeated[3], torch.tensor([1.0]))
    assert torch.equal(repeated[6], torch.tensor([1.0]))


def test_tensor_list_repeat_interleave():
    """Each element repeated n times."""
    tl = TensorList([torch.tensor([1.0]), torch.tensor([2.0])])
    ri = tl.repeat_interleave(3)
    assert len(ri) == 6
    assert torch.equal(ri[0], torch.tensor([1.0]))
    assert torch.equal(ri[1], torch.tensor([1.0]))
    assert torch.equal(ri[2], torch.tensor([1.0]))
    assert torch.equal(ri[3], torch.tensor([2.0]))
    assert torch.equal(ri[4], torch.tensor([2.0]))
    assert torch.equal(ri[5], torch.tensor([2.0]))


def test_tensor_list_cat():
    """Concatenation of multiple TensorLists."""
    tl1 = TensorList([torch.tensor([1.0]), torch.tensor([2.0])])
    tl2 = TensorList([torch.tensor([3.0])])
    tl3 = TensorList([torch.tensor([4.0]), torch.tensor([5.0])])
    result = TensorList.cat([tl1, tl2, tl3])
    assert len(result) == 5
    for i, expected in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
        assert torch.equal(result[i], torch.tensor([expected]))


# ── TensorBatch + TensorList integration tests ───────────────────────────────


def _make_mixed_batch(batch_size=4, seq_len=3):
    """Helper: create a TensorBatch with both Tensor and TensorList fields."""
    sequences = torch.randn(batch_size)
    pixel_values = TensorList([torch.randn(i + 1, 8) for i in range(batch_size)])
    return TensorBatch({"sequences": sequences, "pixel_values": pixel_values})


def test_tensor_batch_with_tensor_list():
    """TensorBatch containing both Tensor and TensorList fields."""
    batch = _make_mixed_batch(batch_size=4)
    assert batch.batch_size == 4
    assert isinstance(batch["sequences"], torch.Tensor)
    assert isinstance(batch["pixel_values"], TensorList)
    assert len(batch["pixel_values"]) == 4


def test_tensor_batch_chunk_with_tensor_list():
    """chunk splits TensorList correctly."""
    batch = _make_mixed_batch(batch_size=4)
    chunks = batch.chunk(2)
    assert len(chunks) == 2
    assert chunks[0].batch_size == 2
    assert chunks[1].batch_size == 2
    assert isinstance(chunks[0]["pixel_values"], TensorList)
    assert len(chunks[0]["pixel_values"]) == 2
    assert len(chunks[1]["pixel_values"]) == 2
    # Verify correct elements are in each chunk
    assert chunks[0]["pixel_values"][0].shape[0] == 1  # i=0 → 1 patch
    assert chunks[1]["pixel_values"][0].shape[0] == 3  # i=2 → 3 patches


def test_tensor_batch_slice_with_tensor_list():
    """slice works for TensorList."""
    batch = _make_mixed_batch(batch_size=4)
    sliced = batch.slice(1, 3)
    assert sliced.batch_size == 2
    assert isinstance(sliced["pixel_values"], TensorList)
    assert len(sliced["pixel_values"]) == 2
    assert sliced["pixel_values"][0].shape[0] == 2  # i=1 → 2 patches
    assert sliced["pixel_values"][1].shape[0] == 3  # i=2 → 3 patches


def test_tensor_batch_cat_with_tensor_list():
    """cat merges TensorList across shards."""
    b1 = _make_mixed_batch(batch_size=2, seq_len=3)
    b2 = _make_mixed_batch(batch_size=3, seq_len=3)
    result = TensorBatch.cat([b1, b2])
    assert result.batch_size == 5
    assert isinstance(result["pixel_values"], TensorList)
    assert len(result["pixel_values"]) == 5


def test_tensor_batch_pickle_with_tensor_list():
    """Serialization roundtrip with TensorList."""
    batch = _make_mixed_batch(batch_size=3)
    batch.metadata = {"test": True}

    pickled = pickle.dumps(batch)
    unpickled = pickle.loads(pickled)

    assert unpickled.batch_size == 3
    assert unpickled.metadata == {"test": True}
    assert isinstance(unpickled["pixel_values"], TensorList)
    assert len(unpickled["pixel_values"]) == 3
    for i in range(3):
        assert torch.equal(unpickled["sequences"][i], batch["sequences"][i])
        assert torch.equal(unpickled["pixel_values"][i], batch["pixel_values"][i])


def test_tensor_batch_repeat_with_tensor_list():
    """repeat with mixed Tensor + TensorList fields."""
    batch = _make_mixed_batch(batch_size=2)
    repeated = batch.repeat(3)
    assert repeated.batch_size == 6
    assert len(repeated["pixel_values"]) == 6
    # Tiled: [t0, t1, t0, t1, t0, t1]
    assert torch.equal(repeated["pixel_values"][0], batch["pixel_values"][0])
    assert torch.equal(repeated["pixel_values"][2], batch["pixel_values"][0])
    assert torch.equal(repeated["pixel_values"][4], batch["pixel_values"][0])


def test_tensor_batch_setitem_tensor_list():
    """Setting TensorList field validates batch size."""
    batch = _make_mixed_batch(batch_size=3)

    # Valid: same batch size
    new_pv = TensorList([torch.randn(2, 8) for _ in range(3)])
    batch["pixel_values"] = new_pv
    assert len(batch["pixel_values"]) == 3

    # Invalid: wrong batch size
    with pytest.raises(ValueError, match="Batch size mismatch"):
        batch["pixel_values"] = TensorList([torch.randn(2, 8) for _ in range(5)])


def test_tensor_batch_none_tensor_list():
    """None TensorList fields handled correctly alongside real ones."""
    batch = TensorBatch(
        {
            "sequences": torch.randn(3, 4),
            "pixel_values": None,
        }
    )
    assert batch.batch_size == 3
    assert batch["pixel_values"] is None

    # Operations still work
    sliced = batch.slice(0, 2)
    assert sliced.batch_size == 2
    assert sliced["pixel_values"] is None

    chunks = batch.chunk(2)
    assert chunks[0]["pixel_values"] is None

    pickled = pickle.dumps(batch)
    unpickled = pickle.loads(pickled)
    assert unpickled["pixel_values"] is None


# ---------------------------------------------------------------------------
# Tests for pad_training_input_batch
# ---------------------------------------------------------------------------


# The canonical set of TrainingInput fields. Adding a new field to the TrainingInput TypedDict
# without updating `pad_training_input_batch()` and this set will make
# `test_pad_batch_typeddict_matches_expected_fields` fail, forcing the author to decide how the
# new field should be padded.
EXPECTED_TRAINING_INPUT_FIELDS = {
    "sequences",
    "attention_mask",
    "loss_mask",
    "response_mask",
    "action_log_probs",
    "base_action_log_probs",
    "values",
    "returns",
    "advantages",
    "kl",
    "rewards",
    "rollout_logprobs",
    "rollout_expert_indices",
    "pixel_values",
    "image_grid_thw",
}


def _make_full_training_batch(batch_size: int = 4, seq_len: int = 5) -> TrainingInputBatch:
    """Build a TrainingInputBatch populated with every TrainingInput field.

    ``pixel_values`` and ``image_grid_thw`` use variable per-row shapes to exercise TensorList.
    """
    torch.manual_seed(0)
    data = {
        "sequences": torch.arange(batch_size * seq_len).reshape(batch_size, seq_len).long(),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "loss_mask": torch.ones(batch_size, seq_len, dtype=torch.float),
        "response_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "action_log_probs": torch.randn(batch_size, seq_len),
        "base_action_log_probs": torch.randn(batch_size, seq_len),
        "values": torch.randn(batch_size, seq_len),
        "returns": torch.randn(batch_size, seq_len),
        "advantages": torch.randn(batch_size, seq_len),
        "kl": torch.randn(batch_size, seq_len),
        "rewards": torch.randn(batch_size, seq_len),
        "rollout_logprobs": torch.randn(batch_size, seq_len),
        "rollout_expert_indices": torch.randint(0, 8, (batch_size, seq_len, 2, 3), dtype=torch.long),
        "pixel_values": TensorList([torch.randn(i + 1, 3) for i in range(batch_size)]),  # batch_size * (i + 1) * 3
        "image_grid_thw": TensorList([torch.tensor([[1, 2, 3]]) for _ in range(batch_size)]),  # batch_size * 1 * 3
    }
    batch = TrainingInputBatch(data)
    batch.metadata = {
        "uids": [f"u{i}" for i in range(batch_size)],
        "is_last_step": [i % 2 == 1 for i in range(batch_size)],  # [False, True, False, True, ...]
        "response_length": seq_len,
    }
    return batch


def test_pad_batch_typeddict_matches_expected_fields():
    """
    Guard: if TrainingInput gains a new field, bump EXPECTED_TRAINING_INPUT_FIELDS and make sure it is
    well handled by pad_training_input_batch().
    """
    typed_dict_fields = set(TrainingInput.__annotations__.keys())
    assert typed_dict_fields == EXPECTED_TRAINING_INPUT_FIELDS, (
        "TrainingInput fields changed. Update EXPECTED_TRAINING_INPUT_FIELDS AND make sure "
        "pad_training_input_batch() handles the new field."
    )


def test_pad_batch_all_fields():
    """Comprehensive test: pad_training_input_batch pads every field correctly.

    Verifies: all tensor fields have correct batch dim, original rows are untouched,
    loss_mask padding is zero, other tensor padding is row-0 clones, TensorList padding
    is row-0 clones, metadata (uids, is_last_step) is extended correctly, and
    the input batch is not mutated.
    """
    batch_size, seq_len, pad_size = 4, 5, 3
    batch = _make_full_training_batch(batch_size=batch_size, seq_len=seq_len)

    # Sanity: the test fixture must exercise every TrainingInput field.
    assert (
        set(batch.keys()) == EXPECTED_TRAINING_INPUT_FIELDS
    ), "Test fixture is missing TrainingInput fields; update _make_full_training_batch."

    # Snapshot input metadata before padding to verify immutability.
    original_uids = list(batch.metadata["uids"])
    original_is_last_step = list(batch.metadata["is_last_step"])

    padded = pad_training_input_batch(batch, pad_size=pad_size)
    assert padded.batch_size == batch_size + pad_size
    assert set(padded.keys()) == EXPECTED_TRAINING_INPUT_FIELDS, "Padded batch is missing TrainingInput fields"

    # --- Tensor fields ---
    for key in EXPECTED_TRAINING_INPUT_FIELDS:
        value = padded[key]
        assert value is not None, f"Field {key!r} became None after padding"
        assert len(value) == batch_size + pad_size, f"Field {key!r} has wrong batch dim"

    # loss_mask: original rows untouched, padding rows all-zero.
    assert torch.equal(padded["loss_mask"][:batch_size], batch["loss_mask"])
    assert torch.all(padded["loss_mask"][batch_size:] == 0)

    # Regular tensor fields (not loss_mask, not TensorList): original rows untouched,
    # padding rows are copies of row 0.
    regular_tensor_keys = EXPECTED_TRAINING_INPUT_FIELDS - {"loss_mask", "pixel_values", "image_grid_thw"}
    for key in regular_tensor_keys:
        assert torch.equal(padded[key][:batch_size], batch[key]), f"Original rows changed for {key!r}"
        for i in range(batch_size, batch_size + pad_size):
            assert torch.equal(padded[key][i], batch[key][0]), f"Padding row {i} of {key!r} is not row 0"

    # TensorList fields (pixel_values, image_grid_thw): original rows untouched,
    # padding rows are clones of row 0.
    for key in ("pixel_values", "image_grid_thw"):
        tl = padded[key]
        assert isinstance(tl, TensorList)
        for i in range(batch_size):
            assert torch.equal(tl[i], batch[key][i]), f"Original TensorList row {i} changed for {key!r}"
        for i in range(batch_size, batch_size + pad_size):
            assert torch.equal(tl[i], batch[key][0]), f"TensorList padding row {i} of {key!r} is not row 0"

    # --- Metadata ---
    assert padded.metadata["uids"] == [f"u{i}" for i in range(batch_size)] + [f"pad{i}" for i in range(pad_size)]
    assert padded.metadata["is_last_step"] == [i % 2 == 1 for i in range(batch_size)] + [True] * pad_size
    assert padded.metadata["pad_size"] == pad_size
    assert padded.metadata["response_length"] == seq_len

    # --- Input immutability ---
    assert batch.metadata["uids"] == original_uids, "pad_training_input_batch mutated input uids"
    assert (
        batch.metadata["is_last_step"] == original_is_last_step
    ), "pad_training_input_batch mutated input is_last_step"


def test_pad_batch_zero_pad_size_returns_same_batch():
    batch = _make_full_training_batch(batch_size=3, seq_len=5)
    padded = pad_training_input_batch(batch, pad_size=0)
    assert padded is batch


def test_pad_batch_rejects_non_cpu_device():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA available")
    batch = _make_full_training_batch(batch_size=2, seq_len=5)
    cuda_data = {
        "sequences": batch["sequences"].cuda(),
        "loss_mask": batch["loss_mask"].cuda(),
    }
    cuda_batch = TrainingInputBatch(cuda_data)
    cuda_batch.metadata = {"uids": ["u0", "u1"]}
    with pytest.raises(AssertionError, match="expects batch on CPU"):
        pad_training_input_batch(cuda_batch, pad_size=1)


def test_pad_batch_preserves_none_fields():
    batch = TrainingInputBatch(
        {
            "sequences": torch.arange(12).reshape(3, 4).long(),
            "loss_mask": torch.ones(3, 4, dtype=torch.float),
            "values": None,
        }
    )
    batch.metadata = {"uids": ["a", "b", "c"]}
    padded = pad_training_input_batch(batch, pad_size=1)
    assert padded["values"] is None
    assert padded.batch_size == 4

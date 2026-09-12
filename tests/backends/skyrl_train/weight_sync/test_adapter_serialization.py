import torch
from safetensors.torch import load_file

from skyrl.backends.skyrl_train.weight_sync.adapter_serialization import (
    compact_adapter_state,
    save_adapter_state,
)


def test_compaction_preserves_exact_tensors_and_serialized_aliases(tmp_path):
    repeated = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    state = {
        "adapter_a": repeated.clone(),
        "adapter_b": repeated.clone(),
        "adapter_c": repeated.clone(),
        "adapter_d": repeated.clone(),
        "different_shape": repeated.reshape(4, 3).clone(),
        "positive_zero": torch.tensor([0.0]),
        "negative_zero": torch.tensor([-0.0]),
        "nan_a": torch.tensor([float("nan")]),
        "nan_b": torch.tensor([float("nan")]),
    }

    compact = compact_adapter_state(state)

    assert compact is not None
    assert compact["adapter_a"].data_ptr() == compact["adapter_b"].data_ptr()
    assert compact["nan_a"].data_ptr() == compact["nan_b"].data_ptr()
    assert compact["positive_zero"].data_ptr() != compact["negative_zero"].data_ptr()

    save_adapter_state(state, str(tmp_path))
    restored = torch.load(tmp_path / "adapter_model.bin", weights_only=True)

    assert restored["adapter_a"].data_ptr() == restored["adapter_b"].data_ptr()
    for name, tensor in state.items():
        assert restored[name].dtype == tensor.dtype
        assert restored[name].shape == tensor.shape
        assert torch.equal(restored[name].view(torch.uint8), tensor.view(torch.uint8))

    converted = {name: tensor.to(torch.bfloat16) for name, tensor in restored.items()}
    assert converted["adapter_a"].data_ptr() != converted["adapter_b"].data_ptr()


def test_serialization_replaces_stale_format_when_compaction_changes(tmp_path):
    repeated = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    duplicate_state = {
        "first": repeated.clone(),
        "second": repeated.clone(),
        "third": repeated.clone(),
    }
    unique_state = {
        "first": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "second": torch.arange(12, 24, dtype=torch.float32).reshape(3, 4),
    }

    save_adapter_state(duplicate_state, str(tmp_path))
    assert (tmp_path / "adapter_model.bin").is_file()
    assert not (tmp_path / "adapter_model.safetensors").exists()

    save_adapter_state(unique_state, str(tmp_path))
    assert not (tmp_path / "adapter_model.bin").exists()
    restored = load_file(tmp_path / "adapter_model.safetensors")
    assert restored.keys() == unique_state.keys()
    for name, tensor in unique_state.items():
        assert torch.equal(restored[name], tensor)

    save_adapter_state(duplicate_state, str(tmp_path))
    assert (tmp_path / "adapter_model.bin").is_file()
    assert not (tmp_path / "adapter_model.safetensors").exists()


def test_serialization_uses_writer_specific_temporary_path(tmp_path):
    repeated = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    state = {
        "first": repeated.clone(),
        "second": repeated.clone(),
        "third": repeated.clone(),
    }
    unrelated_temporary = tmp_path / "adapter_model.bin.tmp"
    unrelated_temporary.write_bytes(b"another writer")

    save_adapter_state(state, str(tmp_path), temporary_suffix="7")

    assert unrelated_temporary.read_bytes() == b"another writer"
    assert not (tmp_path / "adapter_model.bin.tmp7").exists()
    restored = torch.load(tmp_path / "adapter_model.bin", weights_only=True)
    assert restored["first"].dtype == torch.bfloat16
    assert torch.equal(restored["first"], repeated)

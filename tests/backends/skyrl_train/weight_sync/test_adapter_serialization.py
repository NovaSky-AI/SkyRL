import pytest
import torch
from safetensors.torch import load_file

from skyrl.backends.skyrl_train.weight_sync import adapter_serialization
from skyrl.backends.skyrl_train.weight_sync.adapter_serialization import (
    compact_adapter_state,
    save_adapter_state,
)


def _expert_state(
    tensor: torch.Tensor,
    *,
    projections: tuple[str, ...] = ("down_proj",),
    experts: int = 3,
) -> dict[str, torch.Tensor]:
    prefix = "base_model.model.model.layers.0.mlp.experts"
    return {
        f"{prefix}.{expert}.{projection}.lora_B.weight": tensor.clone()
        for expert in range(experts)
        for projection in projections
    }


def test_compaction_preserves_exact_tensors_and_serialized_aliases(tmp_path):
    repeated = torch.tensor([0.0, -0.0, float("nan"), 3.0], requires_grad=True)
    state = _expert_state(repeated)

    compact = compact_adapter_state(state)

    assert compact is not None
    names = list(state)
    assert compact[names[0]].data_ptr() == compact[names[1]].data_ptr()
    assert not compact[names[0]].requires_grad

    save_adapter_state(state, str(tmp_path))
    restored = torch.load(tmp_path / "adapter_model.bin", weights_only=True)

    assert restored[names[0]].data_ptr() == restored[names[1]].data_ptr()
    for name, tensor in state.items():
        assert restored[name].dtype == tensor.dtype
        assert restored[name].shape == tensor.shape
        assert torch.equal(restored[name].view(torch.uint8), tensor.detach().view(torch.uint8))


def test_compaction_keeps_nonexpert_tensors_independent():
    repeated = torch.ones((4, 2), dtype=torch.bfloat16)
    state = {
        f"base_model.model.model.layers.{layer}.self_attn.q_proj.lora_B.weight": repeated.clone() for layer in range(3)
    }

    assert compact_adapter_state(state) is None


@pytest.mark.vllm
def test_compacted_expert_aliases_are_copied_before_vllm_scaling(tmp_path):
    weights = pytest.importorskip("vllm.lora.lora_weights")
    LoRALayerWeights = weights.LoRALayerWeights
    PackedLoRALayerWeights = weights.PackedLoRALayerWeights

    state = _expert_state(
        torch.ones((4, 2), dtype=torch.bfloat16),
        projections=("gate_proj", "down_proj", "up_proj"),
    )
    save_adapter_state(state, str(tmp_path))
    restored = torch.load(tmp_path / "adapter_model.bin", weights_only=True)

    loras = []
    prefix = "base_model.model.model.layers.0.mlp.experts"
    for expert in range(3):
        for projection in ("gate_proj", "down_proj", "up_proj"):
            loras.append(
                LoRALayerWeights(
                    f"{expert}.{projection}",
                    rank=2,
                    lora_alpha=1,
                    lora_a=torch.ones((2, 4), dtype=torch.bfloat16),
                    lora_b=restored[f"{prefix}.{expert}.{projection}.lora_B.weight"],
                )
            )

    packed = PackedLoRALayerWeights.pack_moe(loras, "experts")
    packed.optimize()

    assert all(torch.equal(tensor, torch.ones_like(tensor)) for tensor in restored.values())
    assert all(torch.equal(tensor, torch.full_like(tensor, 0.5)) for tensor in packed.lora_b)


def test_empty_adapter_state_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="adapter_state cannot be empty"):
        save_adapter_state({}, str(tmp_path))


def test_serialization_replaces_stale_format_when_compaction_changes(tmp_path):
    repeated = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    duplicate_state = _expert_state(repeated)
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


@pytest.mark.parametrize("start_compact", [False, True])
def test_format_change_installs_new_artifact_before_cleanup(tmp_path, monkeypatch, start_compact):
    repeated = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    duplicate_state = _expert_state(repeated)
    unique_state = {
        "first": repeated.clone(),
        "second": (repeated + 1).clone(),
    }
    old_state, new_state = (duplicate_state, unique_state) if start_compact else (unique_state, duplicate_state)
    save_adapter_state(old_state, str(tmp_path))

    def fail_cleanup(_path):
        raise OSError("injected cleanup failure")

    monkeypatch.setattr(adapter_serialization.os, "remove", fail_cleanup)
    with pytest.raises(OSError, match="injected cleanup failure"):
        save_adapter_state(new_state, str(tmp_path), temporary_suffix="7")

    assert (tmp_path / "adapter_model.bin").is_file()
    assert (tmp_path / "adapter_model.safetensors").is_file()


def test_serialization_uses_writer_specific_temporary_path(tmp_path):
    repeated = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    state = _expert_state(repeated)
    unrelated_temporary = tmp_path / "adapter_model.bin.tmp"
    unrelated_temporary.write_bytes(b"another writer")

    save_adapter_state(state, str(tmp_path), temporary_suffix="7")

    assert unrelated_temporary.read_bytes() == b"another writer"
    assert not (tmp_path / "adapter_model.bin.tmp7").exists()
    restored = torch.load(tmp_path / "adapter_model.bin", weights_only=True)
    first = next(iter(state))
    assert restored[first].dtype == torch.bfloat16
    assert torch.equal(restored[first], repeated)

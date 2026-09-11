import hashlib
import os
from collections.abc import Mapping

import torch
from safetensors.torch import save_file


def compact_adapter_state(
    adapter_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor] | None:
    canonical_tensors = {}
    identities = {}
    logical_bytes = 0
    unique_bytes = 0
    for name, tensor in adapter_state.items():
        tensor = tensor.contiguous()
        logical_bytes += tensor.nbytes
        identity = (
            tensor.dtype,
            tuple(tensor.shape),
            hashlib.sha256(tensor.view(torch.uint8).numpy()).digest(),
        )
        if identity not in canonical_tensors:
            canonical_tensors[identity] = tensor
            unique_bytes += tensor.nbytes
        identities[name] = identity
    if unique_bytes * 2 >= logical_bytes:
        return None
    canonical_tensors = {identity: tensor.clone() for identity, tensor in canonical_tensors.items()}
    return {name: canonical_tensors[identity] for name, identity in identities.items()}


def save_adapter_state(
    adapter_state: Mapping[str, torch.Tensor],
    output_directory: str,
    temporary_suffix: str = "",
) -> None:
    compact_state = compact_adapter_state(adapter_state)
    safetensors_path = os.path.join(output_directory, "adapter_model.safetensors")
    compact_path = os.path.join(output_directory, "adapter_model.bin")
    if compact_state is None:
        temporary_path = f"{safetensors_path}.tmp{temporary_suffix}"
        save_file(adapter_state, temporary_path)
        if os.path.exists(compact_path):
            os.remove(compact_path)
        os.replace(temporary_path, safetensors_path)
        return

    temporary_path = f"{compact_path}.tmp{temporary_suffix}"
    torch.save(compact_state, temporary_path)
    if os.path.exists(safetensors_path):
        os.remove(safetensors_path)
    os.replace(temporary_path, compact_path)

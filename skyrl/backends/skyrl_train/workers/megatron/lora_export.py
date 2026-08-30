from collections.abc import Iterable

import torch


def build_lora_adapter_state(
    adapter_weights: Iterable[tuple[str, torch.Tensor]], *, preserve_dtype: bool
) -> dict[str, torch.Tensor]:
    """Clone exported Megatron adapter weights into PEFT's key layout."""
    adapter_state = {}
    for name, tensor in adapter_weights:
        exported_tensor = tensor.clone()
        if not preserve_dtype:
            exported_tensor = exported_tensor.float()
        adapter_state[f"base_model.model.{name}"] = exported_tensor
    return adapter_state

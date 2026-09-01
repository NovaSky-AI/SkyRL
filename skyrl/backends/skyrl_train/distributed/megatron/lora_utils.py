from typing import TypeVar

_Tensor = TypeVar("_Tensor")


def remap_qwen_megatron_lora_state_for_vllm(
    adapter_state: dict[str, _Tensor], *, include_mtp: bool
) -> dict[str, _Tensor]:
    """Map Megatron Bridge Qwen adapter keys to vLLM's PEFT layout."""
    qwen_decoder_prefix = "base_model.model.model.language_model."
    if not any(key.startswith(qwen_decoder_prefix) for key in adapter_state):
        return dict(adapter_state)

    remapped = {}
    for key, tensor in adapter_state.items():
        if ".visual." in key:
            continue
        if ".mtp." in key or ".multi_token" in key:
            if include_mtp:
                remapped[key] = tensor
            continue
        key = key.replace(qwen_decoder_prefix, "base_model.model.language_model.model.")
        remapped[key] = tensor
    return remapped

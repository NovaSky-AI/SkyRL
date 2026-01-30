"""Weight loading and saving utilities for stacked layer models."""

from __future__ import annotations

from enum import Enum
import os
from pathlib import Path
from typing import Callable, TYPE_CHECKING

from cloudpathlib import CloudPath
from flax import nnx
from huggingface_hub import snapshot_download
import jax
import jax.numpy as jnp
import numpy as np
import optax
import safetensors.numpy
from transformers import PretrainedConfig
import peft

from tx.models.configs import ModelConfig
from tx.utils.log import logger
from tx.utils.storage import download_and_unpack, pack_and_upload
from tx.tinker.types import LoraConfig

if TYPE_CHECKING:
    import torch


def resolve_model_path(model_name_or_path: str) -> str:
    """Resolve a model name or path to a local directory path.

    If the model_name_or_path points to an existing local directory, it will be
    used directly. Otherwise, the model will be downloaded from HuggingFace Hub.

    Args:
        model_name_or_path: Either a local path to a model directory or a
            HuggingFace model identifier (e.g., "Qwen/Qwen3-0.6B").

    Returns:
        Path to the local directory containing model config and weights.
    """
    local_path = Path(model_name_or_path).expanduser()
    if local_path.is_dir():
        logger.info(f"Using local model at {local_path}")
        return str(local_path)
    return snapshot_download(model_name_or_path, allow_patterns=["*.safetensors", "*.json"])


def get_dtype(dtype: str | torch.dtype) -> jnp.dtype:
    "Convert torch dtype to jax dtype."

    match str(dtype):
        case "torch.float32" | "float32":
            return jnp.float32
        case "torch.bfloat16" | "bfloat16":
            return jnp.bfloat16
        case "torch.float16" | "float16":
            return jnp.float16
        case _:
            raise ValueError(f"Unsupported torch dtype: {dtype}")


def get_model_class(config: PretrainedConfig) -> Callable[..., nnx.Module]:
    "Get the correct model class based on the config."
    import tx.models.llama3
    import tx.models.qwen3
    import tx.models.deepseekv3

    for architecture in config.architectures or []:
        if hasattr(tx.models.llama3, architecture):
            return getattr(tx.models.llama3, architecture)
        if hasattr(tx.models.qwen3, architecture):
            return getattr(tx.models.qwen3, architecture)
        if hasattr(tx.models.deepseekv3, architecture):
            return getattr(tx.models.deepseekv3, architecture)

    raise ValueError(f"None of the architectures {config.architectures} is currently supported.")


def is_stacked_lora_path(path: tuple) -> bool:
    """Check if a parameter path is for stacked layer weights (for LoRA indexing).

    Stacked layer params have the adapter dimension at axis 1: (num_layers, num_adapters, ...).
    Non-stacked params (e.g., embed_tokens) have adapter dimension at axis 0: (num_adapters, ...).

    Args:
        path: Parameter path tuple (can be nnx path objects or strings).

    Returns:
        True if the path contains 'layers', 'dense_layers', or 'moe_layers'.
    """
    path_strs = [p.key if hasattr(p, "key") else str(p) for p in path]
    return any(name in path_strs for name in ("layers", "dense_layers", "moe_layers"))


def _is_stacked_layer_param(path: tuple) -> bool:
    """Check if a parameter path corresponds to a STACKED decoder layer weight.

    Stacked layers have paths like:
    - Qwen3/Llama3: ('model', 'layers', 'self_attn', ...)
    - DeepSeekV3 dense: ('model', 'dense_layers', 'self_attn', ...)
    - DeepSeekV3 MoE: ('model', 'moe_layers', 'self_attn', ...)

    Non-stacked layers have paths like: ('model', 'layers', '0', 'self_attn', ...)
    """
    path_strs = [p.key if hasattr(p, "key") else str(p) for p in path]
    # Check for split stacked layer names (DeepSeekV3)
    if "dense_layers" in path_strs or "moe_layers" in path_strs:
        return True
    # Check for regular stacked layers (Qwen3/Llama3)
    if "layers" not in path_strs:
        return False
    layers_idx = path_strs.index("layers")
    if layers_idx + 1 < len(path_strs) and path_strs[layers_idx + 1].isdigit():
        return False  # Non-stacked: path already contains layer index
    return True  # Stacked: no layer index in path


def _get_layer_group_info(path: tuple, config: ModelConfig) -> tuple[str, int]:
    """Get layer group name and starting layer index for a stacked param path.

    Returns:
        Tuple of (layer_name_for_hf_key, layer_offset) where:
        - layer_name_for_hf_key is 'layers' (HF always uses 'layers')
        - layer_offset is the starting layer index for this group
    """
    path_strs = [p.key if hasattr(p, "key") else str(p) for p in path]
    if "dense_layers" in path_strs:
        return "layers", 0
    elif "moe_layers" in path_strs:
        return "layers", getattr(config, "first_k_dense_replace", 0)
    else:
        return "layers", 0


def _path_to_hf_key(path: tuple, layer_idx: int | None = None) -> str:
    """Convert param path to HuggingFace key. If layer_idx provided, insert it after 'layers'.

    Handles split stacked layer names (dense_layers, moe_layers) by converting them to 'layers'.
    """
    parts = []
    for p in path:
        key = p.key if hasattr(p, "key") else str(p)
        # Handle split stacked layer names - convert to 'layers' with index
        if key in ("layers", "dense_layers", "moe_layers") and layer_idx is not None:
            parts.append(f"layers.{layer_idx}")
        elif key in ("kernel", "embedding"):
            parts.append("weight")
        elif key in ("lora_A", "lora_B"):
            parts.extend([key, "weight"])
        else:
            parts.append(key)
    return ".".join(parts)


def _load_hf_tensor(tensors: dict, key: str, target_shape: tuple, num_experts: int | None) -> np.ndarray:
    """Load tensor from HF format, handling experts, transpose, and reshape."""
    # Handle MoE expert weights (HF stores each expert separately)
    if ".experts." in key and num_experts:
        tensor = np.stack([
            tensors[key.replace(".experts.", f".experts.{i}.")].T
            for i in range(num_experts)
        ], axis=0)
    else:
        tensor = tensors[key]
        if "embed_tokens" not in key:
            tensor = tensor.T

    # Reshape attention projections to match model's grouped head format
    if any(p in key for p in ("q_proj", "k_proj", "v_proj", "o_proj")):
        tensor = tensor.reshape(target_shape)

    return tensor


def _save_hf_tensor(tensors: dict, key: str, param: np.ndarray, num_experts: int | None) -> None:
    """Save tensor to HF format, handling experts, transpose, and reshape."""
    # Handle MoE expert weights
    if ".experts." in key and num_experts:
        for i in range(num_experts):
            tensors[key.replace(".experts.", f".experts.{i}.")] = param[i].T
        return

    # Reshape attention projections back to 2D
    if any(p in key for p in ("q_proj", "k_proj", "v_proj")):
        param = param.reshape(param.shape[0], -1)
    elif "o_proj" in key:
        param = param.reshape(-1, param.shape[-1])

    # Transpose to HF format
    tensors[key] = param if "embed_tokens" in key else param.T


def load_safetensors(
    checkpoint_dir: str | os.PathLike,
    config: ModelConfig,
    model: nnx.Module,
    skip_lora: bool = True,
    prefix: str = "",
    filter_fn: Callable[[tuple], bool] | None = None,
) -> None:
    """Load safetensors weights into a model with stacked layers."""
    tensors = {}
    for file in Path(checkpoint_dir).glob("*.safetensors"):
        tensors.update(safetensors.numpy.load_file(file))
    tensors = {k.removeprefix(prefix): v for k, v in tensors.items()}

    num_experts = config.get_num_experts()
    model_params = nnx.to_flat_state(nnx.state(model))
    updates = []

    for path, param in model_params:
        if filter_fn is not None and not filter_fn(path):
            continue

        path_keys = [p.key if hasattr(p, "key") else str(p) for p in path]
        if skip_lora and any(k in path_keys for k in ("lora_A", "lora_B", "lora_scaling", "lora_ranks")):
            continue

        if _is_stacked_layer_param(path):
            # Stack per-layer weights from HF format
            # Infer layer count from param shape and get offset for split stacked layers
            stacked_layer_count = param.shape[0]
            _, layer_offset = _get_layer_group_info(path, config)
            stacked_tensor = np.empty(param.shape, dtype=param.dtype)
            for i in range(stacked_layer_count):
                key = _path_to_hf_key(path, layer_idx=layer_offset + i)
                stacked_tensor[i] = _load_hf_tensor(tensors, key, param.shape[1:], num_experts)
        else:
            # Non-stacked layers or non-layer params
            key = _path_to_hf_key(path)
            stacked_tensor = _load_hf_tensor(tensors, key, param.shape, num_experts)

        assert param.shape == stacked_tensor.shape, f"Shape mismatch for {path}"
        updates.append((path, jax.device_put(stacked_tensor.astype(param.dtype), param.sharding)))

    nnx.update(model, nnx.from_flat_state(updates))


def save_safetensors(
    config: ModelConfig,
    model: nnx.Module,
    filename: Path,
    prefix: str = "",
    filter_fn: Callable[[tuple], bool] | None = None,
) -> None:
    """Save model weights to safetensors, unstacking layer weights for HF compatibility."""
    num_experts = config.get_num_experts()
    model_params = nnx.to_flat_state(nnx.state(model))
    tensors = {}

    for path, param in model_params:
        path_keys = [p.key if hasattr(p, "key") else str(p) for p in path]
        if "rngs" in path_keys:
            continue
        if filter_fn is not None and not filter_fn(path):
            continue

        if _is_stacked_layer_param(path):
            # Unstack and save as individual layer weights
            # Infer layer count from param shape and get offset for split stacked layers
            stacked_layer_count = param.shape[0]
            _, layer_offset = _get_layer_group_info(path, config)
            for i in range(stacked_layer_count):
                key = prefix + _path_to_hf_key(path, layer_idx=layer_offset + i)
                _save_hf_tensor(tensors, key, param[i], num_experts)
        else:
            # Non-stacked layers or non-layer params
            key = prefix + _path_to_hf_key(path)
            _save_hf_tensor(tensors, key, param, num_experts)

    # In multi-host mode, gather all shards and only save from rank 0
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils
        tensors = {k: multihost_utils.process_allgather(v, tiled=True) for k, v in tensors.items()}

    if jax.process_index() == 0:
        safetensors.numpy.save_file({k: np.asarray(v) for k, v in tensors.items()}, filename)


def filter_lora(adapter_config: LoraConfig, path: tuple[str, ...]) -> bool:
    if not adapter_config.train_attn and "self_attn" in path:
        return False
    if not adapter_config.train_mlp and ("mlp" in path or "experts" in path):
        return False
    if not adapter_config.train_unembed and ("embed_tokens" in path or "lm_head" in path):
        return False
    return True


def load_lora_checkpoint(
    model: nnx.Module, adapter_config: LoraConfig, adapter_index: int, checkpoint_path: Path | CloudPath
) -> None:
    """Load LoRA adapter weights from a sampling checkpoint into the model.

    Args:
        model: The Qwen3ForCausalLM model to load the adapter into
        adapter_config: LoRA adapter configuration
        adapter_index: Index of the adapter to load into
        checkpoint_path: Path to the checkpoint tar.gz file
    """
    _, lora_params, _ = nnx.split(model, model.is_lora_param, ...)

    adapter_lora_params = extract_adapter_state(adapter_index, lora_params, adapter_config.rank)

    with download_and_unpack(checkpoint_path) as temp_dir:
        load_safetensors(
            temp_dir,
            model.config,
            adapter_lora_params,
            skip_lora=False,
            prefix="base_model.model.",
            filter_fn=lambda path: filter_lora(adapter_config, path),
        )
    insert_adapter_state(adapter_index, lora_params, adapter_lora_params, adapter_config.rank)


def save_lora_checkpoint(
    model: nnx.Module,
    base_model_name: str,
    adapter_config: LoraConfig,
    adapter_index: int,
    output_path: Path | CloudPath,
):
    """Save a LoRA checkpoint as a tar.gz archive.

    Args:
        model: The Qwen3ForCausalLM model to extract LoRA parameters from
        adapter_config: LoRA adapter configuration
        adapter_index: Index of the adapter to save
        output_path: Path to save the checkpoint tar.gz file
    """
    _, lora_params, _ = nnx.split(model, model.is_lora_param, ...)

    adapter_lora_params = extract_adapter_state(adapter_index, lora_params, adapter_config.rank)

    peft_config = peft.LoraConfig(
        base_model_name_or_path=base_model_name, r=adapter_config.rank, lora_alpha=adapter_config.alpha
    )

    with pack_and_upload(output_path) as temp_dir:
        save_safetensors(
            model.config,
            adapter_lora_params,
            temp_dir / "adapter_model.safetensors",
            prefix="base_model.model.",
            filter_fn=lambda path: filter_lora(adapter_config, path),
        )
        peft_config.save_pretrained(temp_dir)


class OptimizerName(str, Enum):
    adamw = "adamw"


def get_optimizer(optimizer_name: OptimizerName, optimizer_args: dict) -> optax.GradientTransformation:
    match (optimizer_name, optimizer_args):
        case (OptimizerName.adamw, {"learning_rate": lr, **kwargs}):
            return optax.adamw(lr, **kwargs)
        case (_, {"learning_rate": _}):
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        case _:
            raise ValueError("The 'learning_rate' key must be provided in optimizer_args.")


@nnx.jit(static_argnames=("adapter_index", "rank"))
def extract_adapter_state(adapter_index: int, lora_params: nnx.GraphState, rank: int) -> nnx.GraphState:
    "Helper function to extract the adapter parameters for a specific adapter index."

    def extract_state(path: tuple, p: jnp.ndarray):
        if path[-2].key not in {"lora_A", "lora_B"}:
            return p
        # LoRA param shapes:
        # - 3D: Non-stacked linear/embed (A, in, R) or (A, R, out)
        # - 4D: Stacked linear/embed (L, A, in, R) or non-stacked expert (A, E, in, R)
        # - 5D: Stacked expert (L, A, E, in, R)
        # We extract adapter_index from the adapter dimension (axis 1 for stacked, axis 0 otherwise)
        assert p.ndim in {3, 4, 5}, f"LoRA parameters must have 3-5 dimensions, got shape {p.shape}"
        is_stacked = is_stacked_lora_path(path)
        if path[-2].key == "lora_A":
            if is_stacked:  # (L, A, ..., R)
                return p[:, adapter_index, ..., :rank]
            else:  # (A, ..., R)
                return p[adapter_index, ..., :rank]
        if path[-2].key == "lora_B":
            if is_stacked:  # (L, A, ..., out)
                return p[:, adapter_index, ..., :rank, :]
            else:  # (A, ..., out)
                return p[adapter_index, ..., :rank, :]
        return p  # Defensive fallback (should not be reached)

    return jax.tree.map_with_path(extract_state, lora_params)


# We need to use nnx.jit here instead of jax.jit so the nnx.update will be handled correctly
@nnx.jit(static_argnames=("adapter_index", "rank"))
def insert_adapter_state(
    adapter_index: int, lora_params: nnx.GraphState, new_params: nnx.GraphState, rank: int
) -> None:
    "Helper function to insert the adapter parameters for a specific adapter index (inverse of extract_adapter_state)."

    def insert_state(path: tuple, p: jax.Array, new: jax.Array):
        if path[-2].key not in {"lora_A", "lora_B"}:
            return new
        # See extract_adapter_state for shape documentation
        assert p.ndim in {3, 4, 5}, f"LoRA parameters must have 3-5 dimensions, got shape {p.shape}"
        is_stacked = is_stacked_lora_path(path)
        if path[-2].key == "lora_A":
            if is_stacked:  # (L, A, ..., R)
                return p.at[:, adapter_index, ..., :rank].set(new)
            else:  # (A, ..., R)
                return p.at[adapter_index, ..., :rank].set(new)
        elif path[-2].key == "lora_B":
            if is_stacked:  # (L, A, ..., out)
                return p.at[:, adapter_index, ..., :rank, :].set(new)
            else:  # (A, ..., out)
                return p.at[adapter_index, ..., :rank, :].set(new)
        return new  # Defensive fallback (should not be reached)

    updated = jax.tree.map_with_path(insert_state, lora_params, new_params)
    nnx.update(lora_params, updated)


def round_up_seq_len(seq_len: int) -> int:
    """
    Rounds a sequence length up to roughly two significant binary digits.
    We do this to pad sequences, so the Jax JIT compiler needs to
    compile fewer different shapes.
    """
    if seq_len <= 32:
        return 32

    # Find the position of the most significant bit.
    msb_pos = seq_len.bit_length() - 1
    # Create a mask for the two most significant bits.
    mask = (1 << msb_pos) | (1 << (msb_pos - 1))
    # Round down to the nearest value with at most two significant bits.
    result = seq_len & mask

    # If we rounded down, round up to the next bucket boundary.
    if result < seq_len:
        result += 1 << (msb_pos - 1)

    return result

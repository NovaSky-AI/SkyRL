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


def _is_layer_param(path: tuple) -> bool:
    """Check if a parameter path corresponds to a stacked decoder layer weight."""
    path_strs = [p.key if hasattr(p, "key") else str(p) for p in path]
    # Layer params have 'layers' in their path but not as part of another word
    return "layers" in path_strs


def _get_hf_key_for_layer(path: tuple, layer_idx: int) -> str:
    """Convert a stacked layer param path to a per-layer HuggingFace key."""
    parts = []
    for p in path:
        key = p.key if hasattr(p, "key") else str(p)
        if key == "layers":
            parts.append(f"layers.{layer_idx}")
        elif key in ("kernel", "embedding"):
            parts.append("weight")
        elif key in ("lora_A", "lora_B"):
            parts.append(key)
            parts.append("weight")
        else:
            parts.append(key)
    return ".".join(parts)


def _get_hf_key(path: tuple) -> str:
    """Convert a non-layer param path to a HuggingFace key."""
    parts = []
    for p in path:
        key = p.key if hasattr(p, "key") else str(p)
        if key in ("kernel", "embedding"):
            parts.append("weight")
        elif key in ("lora_A", "lora_B"):
            parts.append(key)
            parts.append("weight")
        else:
            parts.append(key)
    return ".".join(parts)


def load_safetensors(
    checkpoint_dir: str | os.PathLike,
    config: ModelConfig,
    model: nnx.Module,
    num_layers: int,
    skip_lora: bool = True,
    prefix: str = "",
    filter_fn: Callable[[tuple], bool] | None = None,
) -> None:
    """Load safetensors weights into a model with stacked layers.

    For layer parameters, loads individual layer weights and stacks them.
    For non-layer parameters, loads directly.

    Args:
        checkpoint_dir: Directory containing safetensors files.
        config: Model configuration.
        model: Model with stacked layer weights (created with create_stacked_layers).
        num_layers: Number of decoder layers.
        skip_lora: Whether to skip LoRA parameters.
        prefix: Prefix to remove from tensor keys.
        filter_fn: Optional filter for which parameters to load.
    """
    tensors = {}
    for file in Path(checkpoint_dir).glob("*.safetensors"):
        tensors.update(safetensors.numpy.load_file(file))
    tensors = {k.removeprefix(prefix): v for k, v in tensors.items()}

    model_params = nnx.to_flat_state(nnx.state(model))
    updates = []

    for path, param in model_params:
        if filter_fn is not None and not filter_fn(path):
            continue

        path_keys = [p.key if hasattr(p, "key") else str(p) for p in path]

        # Skip LoRA parameters if requested
        if skip_lora and any(k in path_keys for k in ("lora_A", "lora_B", "lora_scaling", "lora_ranks")):
            continue

        if _is_layer_param(path):
            # Stack layer weights from individual layer tensors
            layer_tensors = []
            for layer_idx in range(num_layers):
                key = _get_hf_key_for_layer(path, layer_idx)

                # Handle expert weights (MoE) - HF stores each expert separately
                # Our model has shape (num_experts, in, out), HF has experts.{idx}.*.weight
                if ".experts." in key and hasattr(config, "num_experts"):
                    num_experts = config.num_experts
                    expert_tensors = []
                    for expert_idx in range(num_experts):
                        # Insert expert index: experts.gate_proj -> experts.0.gate_proj
                        expert_key = key.replace(".experts.", f".experts.{expert_idx}.")
                        if expert_key in tensors:
                            expert_tensors.append(tensors[expert_key].T)
                    if expert_tensors:
                        tensor = np.stack(expert_tensors, axis=0)
                    else:
                        raise KeyError(f"Expert weights not found for {key}")
                else:
                    tensor = tensors[key]
                    # Transpose linear weights (HF uses [out, in], we use [in, out])
                    if "embed_tokens" not in key:
                        tensor = tensor.T

                # Reshape attention projections if needed
                if any(proj in key for proj in ("q_proj", "k_proj", "v_proj", "o_proj")):
                    # param.shape[1:] gives the target shape without the layer axis
                    target_shape = param.shape[1:]
                    tensor = tensor.reshape(target_shape)

                layer_tensors.append(tensor)

            stacked_tensor = np.stack(layer_tensors, axis=0)
        else:
            # Non-layer parameter - load directly
            key = _get_hf_key(path)

            if ".experts." in key and hasattr(config, "num_experts"):
                num_experts = config.num_experts
                expert_tensors = []
                for expert_idx in range(num_experts):
                    expert_key = key.replace(".experts.", f".experts.{expert_idx}.")
                    if expert_key in tensors:
                        expert_tensors.append(tensors[expert_key].T)
                if expert_tensors:
                    stacked_tensor = np.stack(expert_tensors, axis=0)
                else:
                    raise KeyError(f"Expert weights not found for {key}")
            else:
                stacked_tensor = tensors[key]
                if "embed_tokens" not in key:
                    stacked_tensor = stacked_tensor.T

        assert param.shape == stacked_tensor.shape, (
            f"Shape mismatch for {path}: expected {param.shape}, got {stacked_tensor.shape}"
        )
        sharded_tensor = jax.device_put(stacked_tensor.astype(param.dtype), param.sharding)
        updates.append((path, sharded_tensor))

    nnx.update(model, nnx.from_flat_state(updates))


def save_safetensors(
    config: ModelConfig,
    model: nnx.Module,
    filename: Path,
    num_layers: int,
    prefix: str = "",
    filter_fn: Callable[[tuple], bool] | None = None,
) -> None:
    """Save model weights to safetensors, unstacking layer weights for HF compatibility.

    Args:
        config: Model configuration.
        model: Model with stacked layer weights.
        filename: Output safetensors file path.
        num_layers: Number of decoder layers.
        prefix: Prefix to add to tensor keys.
        filter_fn: Optional filter for which parameters to save.
    """
    model_params = nnx.to_flat_state(nnx.state(model))
    tensors = {}

    for path, param in model_params:
        path_keys = [p.key if hasattr(p, "key") else str(p) for p in path]
        if "rngs" in path_keys:
            continue
        if filter_fn is not None and not filter_fn(path):
            continue

        if _is_layer_param(path):
            # Unstack and save as individual layer weights
            for layer_idx in range(num_layers):
                key = prefix + _get_hf_key_for_layer(path, layer_idx)
                layer_param = param[layer_idx]

                # Handle expert weights (MoE) - save each expert separately for HF compatibility
                if ".experts." in key and hasattr(config, "num_experts"):
                    for expert_idx in range(config.num_experts):
                        expert_key = key.replace(".experts.", f".experts.{expert_idx}.")
                        tensors[expert_key] = layer_param[expert_idx].T
                else:
                    # Reshape attention projections back to 2D
                    if "q_proj" in key or "k_proj" in key or "v_proj" in key:
                        layer_param = layer_param.reshape(layer_param.shape[0], -1)
                    elif "o_proj" in key:
                        layer_param = layer_param.reshape(-1, layer_param.shape[-1])

                    # Transpose back to HF format
                    if "embed_tokens" not in key:
                        layer_param = layer_param.T
                    tensors[key] = layer_param
        else:
            # Non-layer parameter - save directly
            key = prefix + _get_hf_key(path)

            if ".experts." in key and hasattr(config, "num_experts"):
                for expert_idx in range(config.num_experts):
                    expert_key = key.replace(".experts.", f".experts.{expert_idx}.")
                    tensors[expert_key] = param[expert_idx].T
            else:
                tensor = param
                if "embed_tokens" not in key:
                    tensor = tensor.T
                tensors[key] = tensor

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
            model.model.num_layers,
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
            model.model.num_layers,
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
        # For stacked layers, LoRA params have shape (num_layers, num_adapters, ...)
        # We extract adapter_index from the adapter dimension
        assert p.ndim in {3, 4, 5}, f"LoRA parameters must have 3-5 dimensions, got shape {p.shape}"
        if path[-2].key == "lora_A":
            # Shape: (L, A, in, R) or (A, in, R) -> extract [..., :, :rank]
            if p.ndim == 4:  # Stacked: (L, A, in, R)
                return p[:, adapter_index, :, :rank]
            else:  # Non-stacked: (A, in, R)
                return p[adapter_index, :, :rank]
        if path[-2].key == "lora_B":
            # Shape: (L, A, R, out) or (A, R, out) -> extract [..., :rank, :]
            if p.ndim == 4:  # Stacked: (L, A, R, out)
                return p[:, adapter_index, :rank, :]
            else:  # Non-stacked: (A, R, out)
                return p[adapter_index, :rank, :]

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
        assert p.ndim in {3, 4, 5}, f"LoRA parameters must have 3-5 dimensions, got shape {p.shape}"
        if path[-2].key == "lora_A":
            if p.ndim == 4:  # Stacked: (L, A, in, R)
                return p.at[:, adapter_index, :, :rank].set(new)
            else:  # Non-stacked: (A, in, R)
                return p.at[adapter_index, :, :rank].set(new)
        elif path[-2].key == "lora_B":
            if p.ndim == 4:  # Stacked: (L, A, R, out)
                return p.at[:, adapter_index, :rank, :].set(new)
            else:  # Non-stacked: (A, R, out)
                return p.at[adapter_index, :rank, :].set(new)

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

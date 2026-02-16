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


def is_stacked_path(path: tuple) -> bool:
    """Check if a parameter path is for stacked layer weights.

    Stacked layer params have an extra leading dimension: (num_layers, ...).

    Args:
        path: Parameter path tuple (can be nnx path objects or strings).

    Returns:
        True if the path contains '_stacked' (from StackedDecoderLayers).
    """
    path_strs = [p.key if hasattr(p, "key") else str(p) for p in path]
    return "_stacked" in path_strs


def get_adapter_idx(path: tuple, adapter_index: int) -> tuple:
    """Return index tuple for accessing an adapter at the given path.

    Stacked layer params have shape (num_layers, num_adapters, ...) -> index as [:, adapter_index].
    Non-stacked params (embed_tokens) have shape (num_adapters, ...) -> index as [adapter_index].
    """
    if is_stacked_path(path):
        return (slice(None), adapter_index)
    return (adapter_index,)


def get_lora_adapter_slice(path: tuple, adapter_index: int, rank: int) -> tuple | None:
    """Return index tuple for accessing a single adapter's LoRA weight in an unstacked param.

    After unstack_state, LoRA params have shape (num_adapters, ..., max_rank, ...).
    Returns the slice to extract/insert one adapter's trimmed-rank weights, or None
    for non-LoRA params.
    """
    if "lora_A" in path:
        return (adapter_index, slice(None), slice(None, rank))
    if "lora_B" in path:
        return (adapter_index, slice(None, rank), slice(None))
    return None


def get_param_key(path: tuple, prefix: str = "") -> str:
    "Get the safetensors key for a given model path."
    if path[-1] in {"embedding", "kernel"}:
        path = (*path[:-1], "weight")
    elif path[-1] in {"lora_A", "lora_B"}:
        path = (*path, "weight")
    return prefix + ".".join(map(str, path))


def get_expert_key(path: tuple, expert_idx: int) -> str:
    "Get the safetensors key for an expert weight model path."
    path = tuple(s if s != "experts" else f"experts.{expert_idx}" for s in path)
    return ".".join(map(str, path))


def load_safetensors(
    checkpoint_dir: str | os.PathLike,
    config: ModelConfig,
    model: nnx.Module,
    skip_lora: bool = True,
    prefix: str = "",
    filter_fn: Callable[[tuple], bool] | None = None,
    adapter_index: int | None = None,
    rank: int | None = None,
) -> None:
    """Load safetensors weights into a model with stacked layers.

    When adapter_index and rank are provided, loads LoRA weights into a specific
    adapter slot instead of replacing the full parameter.
    """
    from tx.layers.stacked import unstack_state

    tensors = {}
    for file in Path(checkpoint_dir).glob("*.safetensors"):
        tensors.update(safetensors.numpy.load_file(file))
    tensors = {k.removeprefix(prefix): v for k, v in tensors.items()}

    # unstack_state converts stacked paths (layers._stacked.xxx) to per-layer paths
    # (layers.0.xxx) with ArrayRef write-through, matching checkpoint key format
    for path, param in nnx.to_flat_state(unstack_state(model)):
        if filter_fn is not None and not filter_fn(path):
            continue
        key = get_param_key(path)
        # Skip LoRA parameters if requested
        if skip_lora and ("lora_A" in path or "lora_B" in path or "lora_scaling" in path or "lora_ranks" in path):
            continue
        if key not in tensors:
            continue
        if "experts" in path:
            tensor = np.stack([tensors[get_expert_key(path, i)].T for i in range(config.get_num_experts())], axis=0)
        else:
            tensor = tensors[key] if "embed_tokens" in key else tensors[key].T
        lora_idx = get_lora_adapter_slice(path, adapter_index, rank) if adapter_index is not None else None
        if lora_idx is not None:
            # Load into specific adapter slot via ArrayRef write-through
            arr = param[...]
            param[...] = arr.at[lora_idx].set(jnp.array(tensor, dtype=arr.dtype))
        else:
            if path[-2] in {"q_proj", "k_proj", "v_proj", "o_proj"}:
                tensor = tensor.reshape(param.shape)
            assert param.shape == tensor.shape, f"shape mismatch for {key}"
            # ArrayRef.set_raw_value writes through to the stacked parent variable
            param.set_raw_value(jax.device_put(tensor.astype(param.dtype), param.sharding))


def save_safetensors(
    config: ModelConfig,
    model: nnx.Module,
    filename: Path,
    prefix: str = "",
    filter_fn: Callable[[tuple], bool] | None = None,
    adapter_index: int | None = None,
    rank: int | None = None,
) -> None:
    """Save model weights to safetensors, unstacking layer weights for HF compatibility.

    When adapter_index and rank are provided, extracts a single adapter's LoRA
    weights instead of saving the full parameter.
    """
    from tx.layers.stacked import unstack_state

    # unstack_state converts stacked paths (layers._stacked.xxx) to per-layer paths
    # (layers.0.xxx) matching the checkpoint key format used by HuggingFace
    tensors = {}
    for path, param in nnx.to_flat_state(unstack_state(model)):
        if "rngs" in path:
            continue
        if filter_fn is not None and not filter_fn(path):
            continue
        key = get_param_key(path, prefix=prefix)
        # Extract specific adapter's LoRA weights when adapter_index is provided
        lora_idx = get_lora_adapter_slice(path, adapter_index, rank) if adapter_index is not None else None
        if lora_idx is not None:
            param = param[lora_idx]
        if "experts" in path:
            for i in range(config.get_num_experts()):
                tensors[get_expert_key(path, i)] = param[i, :, :].T
            continue
        if "q_proj" in path or "k_proj" in path or "v_proj" in path:
            param = param.reshape(param.shape[0], -1)
        elif "o_proj" in path:
            param = param.reshape(-1, param.shape[-1])
        tensors[key] = param if "embed_tokens" in path else param.T

    # In multi-host mode, gather all shards and only save from rank 0
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils

        tensors = {k: multihost_utils.process_allgather(v, tiled=True) for k, v in tensors.items()}

    safetensors.numpy.save_file({k: np.asarray(v) for k, v in tensors.items()}, filename)


def filter_lora(adapter_config: LoraConfig, path: tuple[str, ...]) -> bool:
    """Check if a path's module matches the adapter config's training targets."""
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
    with download_and_unpack(checkpoint_path) as temp_dir:
        load_safetensors(
            temp_dir,
            model.config,
            model,
            skip_lora=False,
            prefix="base_model.model.",
            filter_fn=lambda path: ("lora_A" in path or "lora_B" in path) and filter_lora(adapter_config, path),
            adapter_index=adapter_index,
            rank=adapter_config.rank,
        )


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
    peft_config = peft.LoraConfig(
        base_model_name_or_path=base_model_name, r=adapter_config.rank, lora_alpha=adapter_config.alpha
    )

    with pack_and_upload(output_path, rank=jax.process_index()) as temp_dir:

        save_safetensors(
            model.config,
            model,
            temp_dir / "adapter_model.safetensors",
            prefix="base_model.model.",
            filter_fn=lambda path: ("lora_A" in path or "lora_B" in path) and filter_lora(adapter_config, path),
            adapter_index=adapter_index,
            rank=adapter_config.rank,
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
        key = path[-2].key
        if key not in {"lora_A", "lora_B"}:
            return p
        assert p.ndim in {3, 4, 5}, f"LoRA parameters must have 3-5 dimensions, got shape {p.shape}"
        idx = get_adapter_idx(path, adapter_index)
        if key == "lora_A":
            return p[*idx, ..., :rank]
        return p[*idx, ..., :rank, :]

    return jax.tree.map_with_path(extract_state, lora_params)


# We need to use nnx.jit here instead of jax.jit so the nnx.update will be handled correctly
@nnx.jit(static_argnames=("adapter_index", "rank"))
def insert_adapter_state(
    adapter_index: int, lora_params: nnx.GraphState, new_params: nnx.GraphState, rank: int
) -> None:
    "Helper function to insert the adapter parameters for a specific adapter index (inverse of extract_adapter_state)."

    def insert_state(path: tuple, p: jax.Array, new: jax.Array):
        key = path[-2].key
        if key not in {"lora_A", "lora_B"}:
            return new
        assert p.ndim in {3, 4, 5}, f"LoRA parameters must have 3-5 dimensions, got shape {p.shape}"
        idx = get_adapter_idx(path, adapter_index)
        if key == "lora_A":
            return p.at[*idx, ..., :rank].set(new)
        return p.at[*idx, ..., :rank, :].set(new)

    updated = jax.tree.map_with_path(insert_state, lora_params, new_params)
    nnx.update(lora_params, updated)


def round_up_seq_len(seq_len: int, cp: int | None = None) -> int:
    """
    Rounds a sequence length up to roughly two significant binary digits.
    We do this to pad sequences, so the Jax JIT compiler needs to
    compile fewer different shapes.

    If Context parallelism is enabled (cp > 1), the sequence length is
    padded to be a multiple of shard count for even distribution.
    """
    if seq_len <= 32:
        rounded = 32
    else:
        # Find the position of the most significant bit.
        msb_pos = seq_len.bit_length() - 1
        # Create a mask for the two most significant bits.
        mask = (1 << msb_pos) | (1 << (msb_pos - 1))
        # Round down to the nearest value with at most two significant bits.
        rounded = seq_len & mask

        # If we rounded down, round up to the next bucket boundary.
        if rounded < seq_len:
            rounded += 1 << (msb_pos - 1)

    if cp is None:
        # Try to infer from the mesh if not explicitly provided
        mesh = jax.sharding.get_abstract_mesh()
        cp = mesh.shape.get("cp", 1) if mesh is not None else 1

    if cp > 1:
        rounded = ((rounded + cp - 1) // cp) * cp

    return rounded

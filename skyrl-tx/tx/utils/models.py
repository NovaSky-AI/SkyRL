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
    import tx.models.qwen3

    for architecture in config.architectures or []:
        if hasattr(tx.models.qwen3, architecture):
            return getattr(tx.models.qwen3, architecture)

    raise ValueError(f"None of the architectures {config.architectures} is currently supported.")


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
    config: PretrainedConfig,
    model: nnx.Module,
    skip_lora: bool = True,
    prefix: str = "",
    filter_fn: Callable[[tuple], bool] | None = None,
) -> None:
    tensors = {}
    for file in Path(checkpoint_dir).glob("*.safetensors"):
        tensors.update(safetensors.numpy.load_file(file))
    tensors = {k.removeprefix(prefix): v for k, v in tensors.items()}

    model_params = nnx.to_flat_state(nnx.state(model))
    updates = []
    for path, param in model_params:
        if filter_fn is not None and not filter_fn(path):
            continue
        key = get_param_key(path)
        # Skip LoRA parameters if requested
        if skip_lora and ("lora_A" in path or "lora_B" in path or "lora_scaling" in path or "lora_ranks" in path):
            continue
        if "experts" in path:
            tensors[key] = np.stack([tensors[get_expert_key(path, i)].T for i in range(config.num_experts)], axis=0)
        else:
            tensors[key] = tensors[key] if "embed_tokens" in path else tensors[key].T
        if path[-2] in {"q_proj", "k_proj", "v_proj", "o_proj"}:
            tensors[key] = tensors[key].reshape(param.shape)
        assert param.shape == tensors[key].shape, f"shape mismatch for {key}"
        sharded_tensor = jax.device_put(tensors[key].astype(param.dtype), param.sharding)
        updates.append((path, sharded_tensor))
    nnx.update(model, nnx.from_flat_state(updates))


def save_safetensors(
    config: PretrainedConfig,
    model: nnx.Module,
    filename: Path,
    prefix: str = "",
    filter_fn: Callable[[tuple], bool] | None = None,
) -> None:
    model_params = nnx.to_flat_state(nnx.state(model))
    tensors = {}
    for path, param in model_params:
        if "rngs" in path:
            continue
        if filter_fn is not None and not filter_fn(path):
            continue
        key = get_param_key(path, prefix=prefix)
        if "experts" in path:
            for i in range(config.num_experts):
                tensors[get_expert_key(path, i)] = param[i, :, :].T
            continue
        if "q_proj" in path or "k_proj" in path or "v_proj" in path:
            param = param.reshape(param.shape[0], -1)
        elif "o_proj" in path:
            param = param.reshape(-1, param.shape[-1])
        tensors[key] = param if "embed_tokens" in path else param.T
    safetensors.numpy.save_file(tensors, filename)


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
        assert p.ndim in {3, 4}, f"LoRA parameters must have 3 or 4 dimensions, got shape {p.shape}"
        if path[-2].key == "lora_A":
            return p[adapter_index, ..., :, :rank]
        if path[-2].key == "lora_B":
            return p[adapter_index, ..., :rank, :]

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
        assert p.ndim in {3, 4}, f"LoRA parameters must have 3 or 4 dimensions, got shape {p.shape}"
        if path[-2].key == "lora_A":
            return p.at[adapter_index, ..., :, :rank].set(new)
        elif path[-2].key == "lora_B":
            return p.at[adapter_index, ..., :rank, :].set(new)

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

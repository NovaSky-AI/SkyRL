from __future__ import annotations

from enum import Enum
import os
from pathlib import Path
from typing import Callable, TYPE_CHECKING

from flax import nnx
import jax.numpy as jnp
import optax
import safetensors.numpy
from transformers import PretrainedConfig

from tx import models

if TYPE_CHECKING:
    import torch


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

    for architecture in config.architectures or []:
        if hasattr(models, architecture):
            return getattr(models, architecture)

    raise ValueError(f"None of the architectures {config.architectures} is currently supported.")


def get_param_key(path: tuple) -> str:
    "Get the safetensors key for a given model path."
    if path[-1] in {"embedding", "kernel"}:
        path = (*path[:-1], "weight")
    elif path[-1] in {"lora_A", "lora_B"}:
        path = (*path, "weight")
    return ".".join(map(str, path))


def get_expert_key(path: tuple, expert_idx: int) -> str:
    "Get the safetensors key for an expert weight model path."
    path = tuple(s if s != "experts" else f"experts.{expert_idx}" for s in path)
    return ".".join(map(str, path)) + ".weight"


def load_checkpoint(checkpoint_dir: str | os.PathLike, config: PretrainedConfig, model: nnx.Module) -> None:
    tensors = {}
    for file in Path(checkpoint_dir).glob("*.safetensors"):
        tensors.update(safetensors.numpy.load_file(file))
    model_params = nnx.to_flat_state(nnx.state(model))
    updates = []
    for path, param in model_params:
        key = get_param_key(path)
        # Skip LoRA parameters that are not in the checkpoint
        if "lora_A" in path or "lora_B" in path or "lora_scaling" in path or "lora_ranks" in path:
            continue
        if "experts" in path:
            # In order to load the expert weights, we concatenate the relevant tensors
            expert_tensors = [tensors[get_expert_key(path, i)].T for i in range(config.num_experts)]
            tensors[key] = jnp.stack(expert_tensors, axis=0)
        else:
            tensors[key] = tensors[key] if "embed_tokens" in path else tensors[key].T
        if path[-2] in {"q_proj", "k_proj", "v_proj", "o_proj"}:
            tensors[key] = tensors[key].reshape(param.shape)
        assert param.shape == tensors[key].shape, f"shape mismatch for {key}"
        updates.append((path, tensors[key]))
    nnx.update(model, nnx.from_flat_state(updates))


def save_checkpoint(config: PretrainedConfig, model: nnx.Module, filename: str | os.PathLike) -> None:
    model_params = nnx.to_flat_state(nnx.state(model))
    tensors = {}
    for path, param in model_params:
        if "rngs" in path:
            continue
        key = get_param_key(path)
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

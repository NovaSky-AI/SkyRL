from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from flax import nnx
import jax.numpy as jnp
import safetensors.numpy
from transformers import AutoConfig, PretrainedConfig

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


def get_model_class(config: PretrainedConfig) -> type[nnx.Module]:
    "Get the correct model class based on the config."

    for architecture in config.architectures or []:
        if hasattr(models, architecture):
            return getattr(models, architecture)

    raise ValueError(
        f"None of the architectures {config.architectures} is currently supported."
    )


def get_param_key(path: tuple) -> str:
    "Get the safetensors key for a given model path."

    if path[-1] in {"embedding", "kernel"}:
        path = (*path[:-1], "weight")
    return ".".join(map(str, path))


def load_checkpoint(path: str | os.PathLike, config: PretrainedConfig, model: nnx.Module) -> None:
    tensors = {}
    for file in Path(path).glob("*.safetensors"):
        tensors.update(safetensors.numpy.load_file(file))
    model_params = nnx.to_flat_state(nnx.state(model))
    updates = []
    for path, param in model_params:
        key = get_param_key(path)
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
        if "q_proj" in path or "k_proj" in path or "v_proj" in path:
            param = param.reshape(param.shape[0], -1)
        elif "o_proj" in path:
            param = param.reshape(-1, param.shape[-1])
        tensors[key] = param if "embed_tokens" in path else param.T
    safetensors.numpy.save_file(tensors, filename)


class FrozenModelConfig:
    "Frozen version of PretrainedConfig so it is hashable and can be passed to jax.jit."

    def __init__(self, config: PretrainedConfig) -> None:
        self.data = json.dumps(config.to_dict(), sort_keys=True)

    def unfreeze(self) -> PretrainedConfig:
        return AutoConfig.for_model(**json.loads(self.data))

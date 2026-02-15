from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
from transformers import AutoConfig

from skyrl.tx.models.configs import ModelConfig
from skyrl.tx.models.types import ModelForCausalLM
from skyrl.tx.utils.models import load_safetensors, resolve_model_path


def create_model(
    model_name: str,
    config_cls: type[ModelConfig],
    model_cls: type[ModelForCausalLM],
    mesh_axes: tuple[str, ...],
    *,
    mesh_shape: tuple[int, ...] | None = None,
    mesh_axis_types: tuple[jax.sharding.AxisType, ...] | None = None,
    seed: int = 0,
    **config_kwargs: Any,
) -> tuple[ModelConfig, ModelForCausalLM]:
    """Create a JAX model with initialized weights."""
    base_config = AutoConfig.from_pretrained(model_name)
    config = config_cls(base_config, shard_attention_heads=True, **config_kwargs)
    if mesh_shape is None:
        mesh_shape = (1,) * len(mesh_axes)
    if mesh_axis_types is None:
        mesh_axis_types = (jax.sharding.AxisType.Auto,) * len(mesh_axes)
    mesh = jax.make_mesh(mesh_shape, mesh_axes, axis_types=mesh_axis_types)
    with jax.set_mesh(mesh):
        model = model_cls(config, dtype=jnp.float32, rngs=nnx.Rngs(seed))
    return config, model


def load_model(
    model_name: str,
    config_cls: type[ModelConfig],
    model_cls: type[ModelForCausalLM],
    mesh_axes: tuple[str, ...],
    *,
    mesh_shape: tuple[int, ...] | None = None,
    mesh_axis_types: tuple[jax.sharding.AxisType, ...] | None = None,
    seed: int = 0,
    **config_kwargs: Any,
) -> tuple[ModelConfig, ModelForCausalLM]:
    """Create a JAX model and load weights from the HuggingFace cache."""
    config, model = create_model(
        model_name,
        config_cls,
        model_cls,
        mesh_axes,
        mesh_shape=mesh_shape,
        mesh_axis_types=mesh_axis_types,
        seed=seed,
        **config_kwargs,
    )
    weights_dir = resolve_model_path(model_name)
    load_safetensors(weights_dir, config, model)
    return config, model

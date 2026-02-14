from flax import nnx
import jax
import jax.numpy as jnp
from transformers import AutoConfig

from tx.models.configs import ModelConfig
from tx.models.types import ModelForCausalLM
from tx.utils.models import load_safetensors, resolve_model_path


def load_model(
    model_name: str,
    config_cls: type[ModelConfig],
    model_cls: type[ModelForCausalLM],
    mesh_axes: tuple[str, ...],
    *,
    mesh_shape: tuple[int, ...] | None = None,
    **config_kwargs,
) -> tuple[ModelConfig, ModelForCausalLM]:
    """Create a JAX model and load weights from the HuggingFace cache."""
    weights_dir = resolve_model_path(model_name)
    base_config = AutoConfig.from_pretrained(model_name)
    config = config_cls(base_config, shard_attention_heads=True, **config_kwargs)
    if mesh_shape is None:
        mesh_shape = (1,) * len(mesh_axes)
    mesh = jax.make_mesh(mesh_shape, mesh_axes, axis_types=(jax.sharding.AxisType.Auto,) * len(mesh_axes))
    with jax.set_mesh(mesh):
        model = model_cls(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
    load_safetensors(weights_dir, config, model)
    return config, model

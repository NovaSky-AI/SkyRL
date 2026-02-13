import os
import tempfile

from filelock import FileLock
from flax import nnx
import jax
import jax.numpy as jnp
import pytest
from transformers import AutoConfig, AutoModelForCausalLM

from tx.models.configs import ModelConfig
from tx.models.types import ModelForCausalLM
from tx.utils.models import load_safetensors

_WEIGHTS_CACHE = os.path.join(tempfile.gettempdir(), "skyrl-tx-test-weights")


def _get_or_save_hf_weights(model_name: str) -> str:
    """Return path to saved HF weights, downloading and saving only on first call.

    Uses a filesystem-based cache so that multiple tests (including across
    pytest --forked or pytest-xdist subprocesses) share the same saved weights
    instead of each calling save_pretrained() separately. A file lock ensures
    only one process performs the download at a time.
    """
    safe_name = model_name.replace("/", "--")
    weights_dir = os.path.join(_WEIGHTS_CACHE, safe_name)
    marker = os.path.join(weights_dir, ".done")
    lock = FileLock(weights_dir + ".lock")
    with lock:
        if not os.path.exists(marker):
            os.makedirs(weights_dir, exist_ok=True)
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_name, attn_implementation="eager", use_safetensors=True
            )
            hf_model.save_pretrained(weights_dir, safe_serialization=True)
            del hf_model
            open(marker, "w").close()
    return weights_dir


def _load_model(
    weights_dir: str,
    model_name: str,
    config_cls: type[ModelConfig],
    model_cls: type[ModelForCausalLM],
    mesh_axes: tuple[str, ...],
    *,
    mesh_shape: tuple[int, ...] | None = None,
    **config_kwargs,
) -> tuple[ModelConfig, ModelForCausalLM]:
    """Create a JAX model and load saved HF weights into it."""
    base_config = AutoConfig.from_pretrained(model_name)
    config = config_cls(base_config, shard_attention_heads=True, **config_kwargs)
    if mesh_shape is None:
        mesh_shape = (1,) * len(mesh_axes)
    mesh = jax.make_mesh(mesh_shape, mesh_axes, axis_types=(jax.sharding.AxisType.Auto,) * len(mesh_axes))
    with jax.set_mesh(mesh):
        model = model_cls(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
    load_safetensors(weights_dir, config, model)
    return config, model


@pytest.fixture
def hf_weights_dir():
    """Fixture providing a function to get cached HF model weights directories."""
    return _get_or_save_hf_weights


@pytest.fixture
def load_model():
    """Fixture providing a function to create a JAX model from cached HF weights."""
    return _load_model

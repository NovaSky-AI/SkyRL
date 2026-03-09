import jax
import pytest


@pytest.fixture(scope="session", autouse=True)
def configure_jax_cpu_devices():
    """Configure JAX to use 2 CPU devices for testing parallelism."""
    if not jax._src.xla_bridge.backends_are_initialized():
        jax.config.update("jax_num_cpu_devices", 2)

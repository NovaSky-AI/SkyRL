import pytest
import ray

from tests.backends.skyrl_train.gpu.utils import ray_init_for_tests

# Hardware the default CI runners do not have. Each marker is auto-skipped unless its own
# name is passed via `-m`, so `-m megatron_models` never picks these up.
_OPT_IN_GPU_MARKERS = {
    "h100": "requires H100 GPUs",
    "b300": "requires a B300 node",
}


def pytest_configure(config):
    for marker, requirement in _OPT_IN_GPU_MARKERS.items():
        config.addinivalue_line(
            "markers",
            f"{marker}: opt-in tests that {requirement}; auto-skipped unless `-m {marker}` is passed.",
        )
    config.addinivalue_line("markers", "megatron: tests that require the Megatron backend extra.")


def pytest_collection_modifyitems(config, items):
    markexpr = config.getoption("markexpr", default="") or ""
    for marker, requirement in _OPT_IN_GPU_MARKERS.items():
        if marker in markexpr:
            continue
        skip = pytest.mark.skip(reason=f"{marker} test ({requirement}) — run explicitly with `-m {marker}`")
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)


@pytest.fixture
def ray_init_fixture():
    if ray.is_initialized():
        ray.shutdown()
    ray_init_for_tests()
    yield
    # call ray shutdown after a test regardless
    ray.shutdown()


@pytest.fixture(scope="module")
def module_scoped_ray_init_fixture():
    if ray.is_initialized():
        ray.shutdown()
    ray_init_for_tests()
    yield
    # call ray shutdown after a test regardless
    ray.shutdown()

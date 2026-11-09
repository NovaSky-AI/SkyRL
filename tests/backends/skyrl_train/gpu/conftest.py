import pytest
import ray

from tests.backends.skyrl_train.gpu.utils import ray_init_for_tests


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "h100: opt-in tests that require H100 GPUs; auto-skipped unless `-m h100` is passed.",
    )
    config.addinivalue_line(
        "markers",
        "b200: opt-in tests that require B200 (SM100+) GPUs; auto-skipped unless `-m b200` is passed.",
    )
    config.addinivalue_line("markers", "megatron: tests that require the Megatron backend extra.")


def pytest_collection_modifyitems(config, items):
    markexpr = config.getoption("markexpr", default="") or ""
    for mark in ("h100", "b200"):
        if mark in markexpr:
            continue
        skip = pytest.mark.skip(reason=f"{mark.upper()} test — run explicitly with `-m {mark}`")
        for item in items:
            if mark in item.keywords:
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

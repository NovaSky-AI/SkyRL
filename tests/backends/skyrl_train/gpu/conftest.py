import pytest
import ray

from tests.backends.skyrl_train.gpu.utils import ray_init_for_tests


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "h100: opt-in tests that require H100 GPUs; auto-skipped unless `-m h100` is passed.",
    )


def pytest_collection_modifyitems(config, items):
    markexpr = config.getoption("markexpr", default="") or ""
    if "h100" in markexpr:
        return
    skip_h100 = pytest.mark.skip(reason="H100 test — run explicitly with `-m h100`")
    for item in items:
        if "h100" in item.keywords:
            item.add_marker(skip_h100)


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

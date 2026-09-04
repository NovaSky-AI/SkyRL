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
        "b300: opt-in tests that require B300 GPUs; auto-skipped unless `-m b300` is passed.",
    )
    config.addinivalue_line(
        "markers", "megatron: tests that require the Megatron backend extra."
    )


def pytest_collection_modifyitems(config, items):
    markexpr = config.getoption("markexpr", default="") or ""
    opt_in_markers = {
        "h100": "H100 test — run explicitly with `-m h100`",
        "b300": "B300 test — run explicitly with `-m b300`",
    }
    for item in items:
        for marker, reason in opt_in_markers.items():
            if marker not in markexpr and marker in item.keywords:
                item.add_marker(pytest.mark.skip(reason=reason))


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

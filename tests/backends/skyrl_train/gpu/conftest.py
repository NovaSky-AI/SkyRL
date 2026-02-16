import pytest
import ray
from tests.backends.skyrl_train.gpu.utils import ray_init_for_tests


@pytest.fixture
def ray_init_fixture():
    if ray.is_initialized():
        ray.shutdown()
    ray_init_for_tests()
    yield
    # call ray shutdown after a test regardless
    ray.shutdown()

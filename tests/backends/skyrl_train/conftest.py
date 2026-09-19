import pickle
from typing import Any

import pytest
import ray

OUT_OF_BAND_PICKLE_PROTOCOL = 5


@pytest.fixture(scope="session", autouse=True)
def ray_init():
    """Initialize Ray once for the entire test session."""
    if not ray.is_initialized():
        ray.init()
    yield
    if ray.is_initialized():
        ray.shutdown()


@pytest.fixture
def oob_round_trip():
    """Round trip through protocol-5 buffers, optionally as read-only views."""

    def round_trip(obj: Any, read_only: bool = False) -> tuple[Any, bytes, list[memoryview]]:
        buffers: list[pickle.PickleBuffer] = []
        payload = pickle.dumps(obj, protocol=OUT_OF_BAND_PICKLE_PROTOCOL, buffer_callback=buffers.append)
        views = [memoryview(bytes(buffer.raw())) if read_only else buffer.raw() for buffer in buffers]
        return pickle.loads(payload, buffers=views), payload, views

    return round_trip

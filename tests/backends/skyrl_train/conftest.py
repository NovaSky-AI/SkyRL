import pickle
from typing import Any

import pytest
import ray

# The protocol that carries buffers out of band, i.e. the one Ray pickles with.
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
    """Round trip an object through protocol-5 out-of-band buffers, the way Ray does.

    Returns the rebuilt object, the in-band payload, and the out-of-band buffer views.
    With ``read_only`` the buffers are re-wrapped as immutable memoryviews, to stand in
    for the plasma memory Ray maps read-only in the reader.
    """

    def round_trip(obj: Any, read_only: bool = False) -> tuple[Any, bytes, list[memoryview]]:
        buffers: list[pickle.PickleBuffer] = []
        payload = pickle.dumps(obj, protocol=OUT_OF_BAND_PICKLE_PROTOCOL, buffer_callback=buffers.append)
        views = [memoryview(bytes(buffer.raw())) if read_only else buffer.raw() for buffer in buffers]
        return pickle.loads(payload, buffers=views), payload, views

    return round_trip

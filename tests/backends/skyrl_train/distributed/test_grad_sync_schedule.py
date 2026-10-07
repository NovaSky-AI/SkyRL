from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from skyrl.backends.skyrl_train.workers.megatron.grad_sync import configure_no_sync


class Chunk:
    def __init__(self, overlap=True):
        self.ddp_config = SimpleNamespace(overlap_grad_reduce=overlap)
        self.is_last_microbatch = True
        self.entries = 0

    @contextmanager
    def no_sync(self):
        self.is_last_microbatch = False
        self.entries += 1
        try:
            yield
        finally:
            self.is_last_microbatch = True


def test_single_chunk_last_microbatch_and_exception_cleanup():
    chunk = Chunk()
    config = SimpleNamespace(no_sync_func=None)
    assert configure_no_sync([chunk], config)
    with config.no_sync_func():
        assert not chunk.is_last_microbatch
    assert chunk.is_last_microbatch
    with pytest.raises(ValueError):
        with config.no_sync_func():
            raise ValueError("forward failed")
    assert chunk.is_last_microbatch


def test_virtual_pipeline_chunks_share_one_callable():
    # Non-interleaved schedules call no_sync_func directly, so a list would raise.
    chunks = [Chunk(), Chunk()]
    config = SimpleNamespace(no_sync_func=None)
    assert configure_no_sync(chunks, config)
    assert callable(config.no_sync_func) and not isinstance(config.no_sync_func, list)
    with pytest.raises(RuntimeError):
        with config.no_sync_func():
            assert not any(c.is_last_microbatch for c in chunks)
            raise RuntimeError("backward failed")
    assert all(c.is_last_microbatch for c in chunks)


def test_reentrant_across_steps():
    chunk = Chunk()
    config = SimpleNamespace(no_sync_func=None)
    configure_no_sync([chunk], config)
    for _ in range(3):
        with config.no_sync_func():
            assert not chunk.is_last_microbatch
    assert chunk.entries == 3


def test_custom_schedule_context_preserved():
    callback = object()
    config = SimpleNamespace(no_sync_func=callback)
    assert not configure_no_sync([Chunk()], config)
    assert config.no_sync_func is callback


@pytest.mark.parametrize(
    "chunks",
    [[], [Chunk(False)], [SimpleNamespace()], [Chunk(), Chunk(False)], [Chunk(), SimpleNamespace()]],
    ids=["empty", "overlap_off", "unwrapped", "mixed_overlap", "partially_wrapped"],
)
def test_non_overlap_and_unwrapped_models_unchanged(chunks):
    config = SimpleNamespace(no_sync_func=None)
    assert not configure_no_sync(chunks, config)
    assert config.no_sync_func is None

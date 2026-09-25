from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from skyrl.backends.skyrl_train.workers.megatron.grad_sync import configure_no_sync


class Chunk:
    def __init__(self, overlap=True):
        self.ddp_config = SimpleNamespace(overlap_grad_reduce=overlap)
        self.is_last_microbatch = True

    @contextmanager
    def no_sync(self):
        self.is_last_microbatch = False
        try:
            yield
        finally:
            self.is_last_microbatch = True


def test_single_chunk_last_microbatch_and_exception_cleanup():
    chunk = Chunk()
    config = SimpleNamespace(no_sync_func=None)
    configure_no_sync([chunk], config)
    with config.no_sync_func():
        assert not chunk.is_last_microbatch
    assert chunk.is_last_microbatch
    with pytest.raises(ValueError):
        with config.no_sync_func():
            raise ValueError("forward failed")
    assert chunk.is_last_microbatch


def test_virtual_pipeline_chunk_callbacks_are_independent():
    chunks = [Chunk(), Chunk()]
    config = SimpleNamespace(no_sync_func=None)
    configure_no_sync(chunks, config)
    assert len(config.no_sync_func) == 2
    with config.no_sync_func[0]():
        assert not chunks[0].is_last_microbatch
        assert chunks[1].is_last_microbatch
    with config.no_sync_func[1]():
        assert chunks[0].is_last_microbatch
        assert not chunks[1].is_last_microbatch


def test_custom_schedule_context_preserved():
    callback = object()
    config = SimpleNamespace(no_sync_func=callback)
    configure_no_sync([Chunk()], config)
    assert config.no_sync_func is callback


@pytest.mark.parametrize("chunks", [[], [Chunk(False)], [SimpleNamespace()]])
def test_non_overlap_and_unwrapped_models_unchanged(chunks):
    config = SimpleNamespace(no_sync_func=None)
    configure_no_sync(chunks, config)
    assert config.no_sync_func is None

"""Optimizer-window ownership of Megatron gradient synchronization."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from skyrl.backends.skyrl_train.distributed.megatron.grad_sync import (
    defer_grad_sync,
    start_deferred_grad_sync,
)


class DDPChunk:
    """DDP's public no-sync contract with observable synchronization calls."""

    def __init__(self, overlap=True):
        self.hooks_enabled = True
        self.ddp_config = SimpleNamespace(overlap_grad_reduce=overlap)
        self.start_grad_sync = Mock()
        self.config = SimpleNamespace(no_sync_func=self.no_sync, grad_sync_func=self.start_grad_sync)

    @contextmanager
    def no_sync(self):
        self.hooks_enabled = False
        try:
            yield
        finally:
            self.hooks_enabled = True


@pytest.mark.parametrize("request_sizes", [(2,), (1,), (3,), (2, 1), (1, 2, 1)])
@pytest.mark.parametrize("shared_config", [False, True])
def test_gradients_stay_local_until_optimizer_step(request_sizes, shared_config):
    chunks = [DDPChunk(), DDPChunk()]
    if shared_config:
        chunks[1].config = chunks[0].config
    original_callbacks = [(c.config.no_sync_func, c.config.grad_sync_func) for c in chunks]

    for microbatches in request_sizes:
        with defer_grad_sync(chunks):
            for chunk in chunks:
                # A pipeline schedule exits its no-sync context before its last
                # backward. That exit must not re-enable the chunk's DDP hooks.
                callback = chunk.config.no_sync_func or nullcontext
                with callback():
                    for _ in range(microbatches - 1):
                        assert not chunk.hooks_enabled
                assert not chunk.hooks_enabled
                assert chunk.config.grad_sync_func is None
        for chunk, callbacks in zip(chunks, original_callbacks):
            assert chunk.hooks_enabled
            assert (chunk.config.no_sync_func, chunk.config.grad_sync_func) == callbacks
            chunk.start_grad_sync.assert_not_called()

    start_deferred_grad_sync(chunks)
    for chunk in chunks:
        chunk.start_grad_sync.assert_called_once_with()


def test_no_sync_and_callbacks_are_restored_after_exception():
    chunks = [DDPChunk(), DDPChunk()]
    original_callbacks = [(c.config.no_sync_func, c.config.grad_sync_func) for c in chunks]
    with pytest.raises(RuntimeError, match="backward failed"):
        with defer_grad_sync(chunks):
            raise RuntimeError("backward failed")
    for chunk, callbacks in zip(chunks, original_callbacks):
        assert chunk.hooks_enabled
        assert (chunk.config.no_sync_func, chunk.config.grad_sync_func) == callbacks
        chunk.start_grad_sync.assert_not_called()


def test_non_overlap_reduction_is_left_to_finalize():
    chunks = [DDPChunk(overlap=False), DDPChunk(overlap=True)]
    start_deferred_grad_sync(chunks)
    chunks[0].start_grad_sync.assert_not_called()
    chunks[1].start_grad_sync.assert_called_once_with()

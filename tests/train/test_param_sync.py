"""Export must wait for every updated shard, with FP8 staging before collectives."""

from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.skyrl_train.workers.megatron.param_sync import (
    sync_params_for_export,
)


class Chunk:
    def __init__(self, events, name, overlap=True):
        self.events, self.name = events, name
        self.ddp_config = SimpleNamespace(overlap_param_gather=overlap)

    def start_param_sync(self, *, force_sync):
        assert force_sync and not torch.is_grad_enabled()
        self.events.append(self.name)


def test_all_chunks_wait_after_optimizer_stages_shared_buffers():
    events = []
    optimizer = SimpleNamespace(prepare_model_params_for_param_sync=lambda: events.append("stage"))
    sync_params_for_export([Chunk(events, "a"), Chunk(events, "b")], optimizer)
    assert events == ["stage", "a", "b"]


def test_non_overlap_and_unwrapped_models_do_not_restage_optimizer():
    events = []
    optimizer = SimpleNamespace(prepare_model_params_for_param_sync=lambda: events.append("stage"))
    sync_params_for_export([Chunk(events, "a", overlap=False), object()], optimizer)
    assert events == []


@pytest.mark.parametrize("optimizer", [None, object()])
def test_initial_export_and_older_optimizer_without_staging(optimizer):
    events = []
    sync_params_for_export([Chunk(events, "a")], optimizer)
    assert events == ["a"]


def test_staging_failure_prevents_collective_and_restores_grad_mode():
    events = []

    def fail():
        raise RuntimeError("staging failed")

    with torch.enable_grad(), pytest.raises(RuntimeError, match="staging failed"):
        sync_params_for_export([Chunk(events, "a")], SimpleNamespace(prepare_model_params_for_param_sync=fail))
    assert events == []
    assert torch.is_grad_enabled()


def test_chained_optimizer_stages_once_through_its_own_api():
    # Core's ChainedOptimizer zeroes each shared chunk buffer once, then stages
    # every sub-optimizer; staging the sub-optimizers one by one would let a
    # later one zero what an earlier one staged.
    events = []
    sub = SimpleNamespace(prepare_model_params_for_param_sync=lambda: events.append("sub"))
    chained = SimpleNamespace(
        chained_optimizers=[sub, sub],
        prepare_model_params_for_param_sync=lambda: events.append("chained"),
    )
    sync_params_for_export([Chunk(events, "a")], chained)
    assert events == ["chained", "a"]


def test_no_model_chunks_is_a_no_op():
    sync_params_for_export(None, SimpleNamespace(prepare_model_params_for_param_sync=pytest.fail))


def _mxfp8_reuse_optimizer(events):
    return SimpleNamespace(
        config=SimpleNamespace(reuse_grad_buf_for_mxfp8_param_ag=True, overlap_param_gather=True),
        prepare_model_params_for_param_sync=lambda: events.append("stage"),
    )


def _grad_buffer(freed):
    grad_data = torch.zeros(4)
    if freed:
        grad_data.untyped_storage().resize_(0)
    return SimpleNamespace(grad_data=grad_data)


def test_mxfp8_staging_refuses_freed_grad_buffers():
    events = []
    chunk = Chunk(events, "a")
    chunk.buffers, chunk.expert_parallel_buffers = [_grad_buffer(freed=False)], [_grad_buffer(freed=True)]
    with pytest.raises(RuntimeError, match="optimizer is offloaded"):
        sync_params_for_export([chunk], _mxfp8_reuse_optimizer(events))
    assert events == []


def test_mxfp8_staging_with_resident_grad_buffers():
    events = []
    chunk = Chunk(events, "a")
    chunk.buffers, chunk.expert_parallel_buffers = [_grad_buffer(freed=False)], []
    sync_params_for_export([chunk], _mxfp8_reuse_optimizer(events))
    assert events == ["stage", "a"]

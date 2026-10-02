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

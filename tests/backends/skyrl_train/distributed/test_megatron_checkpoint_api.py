"""CPU checks against the pinned native checkpoint API, without distributed I/O."""

from types import SimpleNamespace
from unittest.mock import Mock, create_autospec

import pytest

pytest.importorskip("megatron.core")

from nvidia_resiliency_ext.checkpointing.async_ckpt.core import AsyncRequest

from skyrl.backends.skyrl_train.distributed.megatron import (
    megatron_strategy as strategy,
)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_save_uses_native_checkpoint_signature_and_queue(tmp_path, monkeypatch, asynchronous):
    queue = Mock(spec=strategy.AsyncCallsQueue)
    monkeypatch.setattr(strategy, "_async_calls", queue)
    monkeypatch.setattr(strategy, "AsyncCallsQueue", Mock(return_value=queue))
    monkeypatch.setattr(strategy.dist, "barrier", Mock())
    monkeypatch.setattr(strategy.mpu, "get_data_parallel_group", Mock(return_value=None))
    monkeypatch.setattr(strategy, "get_default_save_sharded_strategy", Mock())
    monkeypatch.setattr(strategy, "FullyParallelSaveStrategyWrapper", Mock())
    monkeypatch.setattr(strategy.MegatronStrategy, "_release_grad_buffers", Mock(return_value=False))

    request = AsyncRequest(None, (), []) if asynchronous else None
    # Autospec uses the installed Core signature: a removed keyword must fail here.
    native_save = create_autospec(strategy.dist_checkpointing.save, return_value=request)
    monkeypatch.setattr(strategy.dist_checkpointing, "save", native_save)
    owner = strategy.MegatronStrategy.__new__(strategy.MegatronStrategy)
    owner.megatron_config = SimpleNamespace(async_dist_ckpt_save=asynchronous, async_save_prestage_to_cpu=False)
    owner.is_lora = False
    owner.is_rank_0 = Mock(return_value=False)
    owner.get_rng_state = Mock(return_value={})
    owner.print = Mock()
    model = SimpleNamespace(actor_module=[SimpleNamespace(sharded_state_dict=lambda: {})])

    owner.save_checkpoint(model, str(tmp_path / "checkpoint"), node_local_rank=0)

    assert native_save.call_args.kwargs["async_sharded_save"] is asynchronous
    if asynchronous:
        queue.maybe_finalize_async_calls.assert_called_once_with(blocking=True)
        queue.schedule_async_request.assert_called_once_with(request)
        queue.close.assert_not_called()
    else:
        queue.schedule_async_request.assert_not_called()
        queue.close.assert_called_once_with()


def test_grad_buffers_resident_tracks_released_storage():
    import torch

    from skyrl.backends.skyrl_train.distributed.megatron import megatron_utils

    ddp = megatron_utils.DDP.__new__(megatron_utils.DDP)
    released = torch.zeros(4)
    released.untyped_storage().resize_(0)  # what DDP.offload_grad_buffers does to each buffer
    ddp.buffers = [SimpleNamespace(grad_data=torch.zeros(4))]
    ddp.expert_parallel_buffers = []
    assert megatron_utils.megatron_grad_buffers_resident([ddp])
    ddp.expert_parallel_buffers = [SimpleNamespace(grad_data=released)]
    assert not megatron_utils.megatron_grad_buffers_resident([ddp])


@pytest.mark.parametrize(
    ("pending", "resident", "released"),
    [(False, True, True), (True, True, False), (False, False, False)],
)
def test_grad_buffers_are_released_only_when_that_loses_nothing(monkeypatch, pending, resident, released):
    offload = Mock()
    monkeypatch.setattr(strategy, "offload_megatron_grads_to_cpu", offload)
    monkeypatch.setattr(strategy, "megatron_grad_buffers_resident", Mock(return_value=resident))
    monkeypatch.setattr(strategy.torch.cuda, "empty_cache", Mock())
    monkeypatch.setattr(strategy.torch.cuda, "synchronize", Mock())
    owner = strategy.MegatronStrategy.__new__(strategy.MegatronStrategy)
    owner.print = Mock()
    # A forward_backward leaves a pending sync until optim_step; the Tinker server can checkpoint in between.
    wrapper = SimpleNamespace(_pending_grad_sync={"num_tokens": None} if pending else None, actor_module=[object()])

    assert owner._release_grad_buffers(wrapper) is released
    assert offload.call_count == int(released)


@pytest.mark.parametrize("released", [False, True])
def test_save_restores_only_the_grad_buffers_it_released(tmp_path, monkeypatch, released):
    restore = Mock()
    monkeypatch.setattr(strategy, "load_megatron_grads_to_gpu", restore)
    monkeypatch.setattr(strategy.MegatronStrategy, "_release_grad_buffers", Mock(return_value=released))
    monkeypatch.setattr(strategy, "_async_calls", Mock(spec=strategy.AsyncCallsQueue))
    monkeypatch.setattr(strategy, "AsyncCallsQueue", Mock())
    monkeypatch.setattr(strategy.dist, "barrier", Mock())
    monkeypatch.setattr(strategy.mpu, "get_data_parallel_group", Mock(return_value=None))
    monkeypatch.setattr(strategy, "get_default_save_sharded_strategy", Mock())
    monkeypatch.setattr(strategy, "FullyParallelSaveStrategyWrapper", Mock())
    monkeypatch.setattr(strategy.dist_checkpointing, "save", Mock(return_value=None))
    monkeypatch.setattr(strategy.torch.cuda, "empty_cache", Mock())
    owner = strategy.MegatronStrategy.__new__(strategy.MegatronStrategy)
    owner.megatron_config = SimpleNamespace(async_dist_ckpt_save=False, async_save_prestage_to_cpu=False)
    owner.is_lora = False
    owner.is_rank_0 = Mock(return_value=False)
    owner.get_rng_state = Mock(return_value={})
    owner.print = Mock()
    model = SimpleNamespace(actor_module=[SimpleNamespace(sharded_state_dict=dict)])

    owner.save_checkpoint(model, str(tmp_path / "checkpoint"), node_local_rank=0)

    assert restore.call_count == int(released)


@pytest.mark.parametrize("released", [False, True])
def test_load_restores_only_the_grad_buffers_it_released(tmp_path, monkeypatch, released):
    # Under managed colocation the buffers can already be offloaded on entry: restoring them would leave
    # them on the GPU while the offload bookkeeping says they are not.
    restore = Mock()
    monkeypatch.setattr(strategy, "load_megatron_grads_to_gpu", restore)
    monkeypatch.setattr(strategy.MegatronStrategy, "_release_grad_buffers", Mock(return_value=released))
    monkeypatch.setattr(strategy.io, "exists", Mock(return_value=True))
    monkeypatch.setattr(strategy.io, "is_cloud_path", Mock(return_value=False))
    monkeypatch.setattr(strategy.mpu, "get_data_parallel_group", Mock(return_value=None))
    monkeypatch.setattr(strategy, "get_default_load_sharded_strategy", Mock())
    monkeypatch.setattr(strategy, "FullyParallelLoadStrategyWrapper", Mock())
    monkeypatch.setattr(strategy.dist_checkpointing, "load", Mock(return_value={"model": {}}))
    monkeypatch.setattr(strategy.torch.cuda, "empty_cache", Mock())
    owner = strategy.MegatronStrategy.__new__(strategy.MegatronStrategy)
    owner.is_lora = False
    owner.print = Mock()
    owner.finalize_pending_saves = Mock()
    chunk = SimpleNamespace(sharded_state_dict=dict, load_state_dict=Mock())

    owner.load_checkpoint(
        SimpleNamespace(actor_module=[chunk]),
        str(tmp_path),
        load_optimizer_states=False,
        load_lr_scheduler_states=False,
    )

    chunk.load_state_dict.assert_called_once()
    assert restore.call_count == int(released)


def test_prestage_preserves_native_request_callbacks():
    staged = object()
    preload = Mock(return_value=staged)
    finalize = Mock()
    request = AsyncRequest(None, ("path", object(), "results"), [finalize], preload_fn=preload)

    result = strategy._stage_async_request_to_host(request)

    preload.assert_called_once_with()
    assert result.async_fn_args == ("path", staged, "results")
    assert result.preload_fn is None
    assert result.finalize_fns == [finalize]
    assert request.preload_fn is preload

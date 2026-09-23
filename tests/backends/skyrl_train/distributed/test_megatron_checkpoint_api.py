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

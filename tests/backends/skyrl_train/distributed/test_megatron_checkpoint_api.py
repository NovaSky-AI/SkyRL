"""CPU checks against the pinned native checkpoint API, without distributed I/O."""

from types import SimpleNamespace
from unittest.mock import Mock, create_autospec

import pytest
import torch

pytest.importorskip("megatron.core")

from nvidia_resiliency_ext.checkpointing.async_ckpt.core import AsyncRequest

from skyrl.backends.skyrl_train.distributed.megatron import (
    megatron_strategy as strategy,
)
from skyrl.backends.skyrl_train.distributed.megatron import megatron_utils
from skyrl.backends.skyrl_train.workers.worker import Worker


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


@pytest.fixture
def checkpoint_loader(monkeypatch):
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
    owner._release_grad_buffers = Mock(return_value=False)
    return owner


@pytest.mark.parametrize("released", [False, True])
def test_load_restores_only_the_grad_buffers_it_released(tmp_path, monkeypatch, checkpoint_loader, released):
    # Under managed colocation the buffers can already be offloaded on entry: restoring them would leave
    # them on the GPU while the offload bookkeeping says they are not.
    restore = Mock()
    monkeypatch.setattr(strategy, "load_megatron_grads_to_gpu", restore)
    checkpoint_loader._release_grad_buffers.return_value = released
    chunk = torch.nn.Module()
    chunk.sharded_state_dict = dict
    chunk.load_state_dict = Mock()

    checkpoint_loader.load_checkpoint(
        SimpleNamespace(actor_module=[chunk]),
        str(tmp_path),
        load_optimizer_states=False,
        load_lr_scheduler_states=False,
    )

    chunk.load_state_dict.assert_called_once()
    assert restore.call_count == int(released)


@pytest.mark.megatron
@pytest.mark.parametrize("is_lora", [False, True], ids=["full_model", "lora"])
def test_weights_only_resume_updates_from_loaded_shard(tmp_path, checkpoint_loader, is_lora):
    chunk = torch.nn.Linear(1, 2, bias=False, dtype=torch.bfloat16)
    chunk.sharded_state_dict = chunk.state_dict
    loaded_weights = torch.tensor([[5.0], [7.0]], dtype=torch.bfloat16)
    checkpoint_loader.is_lora = is_lora
    if is_lora:
        checkpoint_loader._get_rank_path = lambda directory: tmp_path / "adapter.pt"
        torch.save({"model_state_dict": {"weight": loaded_weights}}, tmp_path / "adapter.pt")
    else:
        strategy.dist_checkpointing.load.return_value = {"model": {"weight": loaded_weights}}

    master = torch.nn.Parameter(torch.ones(1))
    adam = torch.optim.AdamW([master], lr=0.25, weight_decay=0)
    master.grad = torch.ones_like(master)
    adam.step()
    optimizer = strategy.DistributedOptimizer.__new__(strategy.DistributedOptimizer)
    optimizer.optimizer = adam
    optimizer.config = SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=False)
    optimizer.ddp_config = SimpleNamespace(use_megatron_fsdp=False, fp8_param_gather=False, fp4_param_gather=False)
    optimizer.is_stub_optimizer = False
    optimizer.model_float16_groups, optimizer.shard_fp32_from_float16_groups = [[chunk.weight]], [[master]]
    optimizer.model_fp32_groups, optimizer.shard_fp32_groups = [], []
    shard = SimpleNamespace(start=1, end=2, size=1)
    optimizer._get_model_param_range_map = lambda param: {"param": shard, "gbuf_world_in_bucket": shard}
    optimizer.model_param_gbuf_map = {chunk.weight: (0, torch.bfloat16, 0)}
    optimizer.buffers = [SimpleNamespace(buckets=[SimpleNamespace(param_data=chunk.weight.detach())])]
    worker = Worker.__new__(Worker)
    worker.model = SimpleNamespace(actor_module=[chunk])
    worker.optimizer = optimizer
    worker.strategy = checkpoint_loader

    worker.load_checkpoint(str(tmp_path), load_optimizer_states=False, load_lr_scheduler_states=False)

    torch.testing.assert_close(chunk.weight, loaded_weights)
    assert "optimizer" not in strategy.dist_checkpointing.load.call_args.kwargs["sharded_state_dict"]
    adam.step()
    optimizer._copy_main_params_to_model_params()
    torch.testing.assert_close(chunk.weight, torch.tensor([[5.0], [6.75]], dtype=torch.bfloat16))
    assert adam.state[master]["step"].item() == 2


@pytest.mark.megatron
def test_weights_only_resume_preserves_frozen_weights_across_offload(tmp_path, monkeypatch, checkpoint_loader):
    """Exercise file-backed offload with separate CPU storage standing in for device memory."""

    class EmulatedCudaParameter(torch.nn.Parameter):
        @property
        def is_cuda(self):
            return True

    chunk = torch.nn.Module()
    chunk.weight = EmulatedCudaParameter(torch.ones(1), requires_grad=False)
    chunk.buffers, chunk.expert_parallel_buffers = [], []
    chunk.sharded_state_dict = chunk.state_dict
    monkeypatch.setattr(megatron_utils, "DDP", torch.nn.Module)
    monkeypatch.setattr(megatron_utils, "_FROZEN_OFFLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(megatron_utils, "_frozen_offload_failed", False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))

    def offload_and_backload():
        megatron_utils.offload_megatron_model_to_cpu([chunk])
        megatron_utils.load_megatron_model_to_gpu([chunk])
        chunk.weight.data = chunk.weight.data.clone()

    offload_and_backload()
    loaded_weight = torch.tensor([7.0])
    strategy.dist_checkpointing.load.return_value = {"model": {"weight": loaded_weight}}
    checkpoint_loader.load_checkpoint(SimpleNamespace(actor_module=[chunk]), str(tmp_path), load_optimizer_states=False)
    offload_and_backload()

    torch.testing.assert_close(chunk.weight, loaded_weight)


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

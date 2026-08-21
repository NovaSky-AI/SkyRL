"""Tests for Megatron backend correctness fixes.

Tests that require megatron-core (GPU dependency) are skipped when it is not
installed.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


# NOTE: this duplicates the config helper in test_megatron_correctness.py, but that is
# intentional to keep the two tests independent.
def _fft_dispatch_cfg(weight_sync_backend: str = "nccl") -> SimpleNamespace:
    """Build the minimal ``self.cfg`` view that ``save_weights_for_sampler``
    inspects on the non-colocated path. Defaults to FFT (lora.rank=0) so
    the pause/resume branch is taken.

    ``weight_sync_backend`` defaults to ``"nccl"`` so the caller-pauses branch is
    exercised; pass ``"delta"`` for the branch where the sender pauses internally.
    """
    return SimpleNamespace(
        trainer=SimpleNamespace(
            strategy="fsdp",
            policy=SimpleNamespace(
                model=SimpleNamespace(lora=SimpleNamespace(rank=0)),
                megatron_config=SimpleNamespace(lora_config=SimpleNamespace(merge_lora=False)),
            ),
        ),
        generator=SimpleNamespace(
            inference_engine=SimpleNamespace(weight_sync_backend=weight_sync_backend, offload_kv_for_weight_sync=False),
        ),
    )


class TestSaveWeights:
    """Tests for `WorkerDispatch.save_weights_for_sampler`"""

    @pytest.mark.asyncio
    async def test_non_colocated_calls_pause_and_resume(self):
        from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch

        dispatch = WorkerDispatch.__new__(WorkerDispatch)
        dispatch.colocate_all = False
        dispatch.cfg = _fft_dispatch_cfg()
        dispatch._inference_engine_client = AsyncMock()
        dispatch._broadcast_to_inference_engines = MagicMock()
        dispatch._prepare_for_weight_sync = MagicMock()
        dispatch._finish_weight_sync = MagicMock()
        dispatch.ensure_active_adapter = MagicMock()

        await dispatch.save_weights_for_sampler()

        dispatch._inference_engine_client.pause_generation.assert_awaited_once()
        dispatch._broadcast_to_inference_engines.assert_called_once()
        dispatch._inference_engine_client.resume_generation.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_non_colocated_delta_does_not_pause(self):
        """Delta sync owns pause/resume itself.

        ``DeltaWeightTransferSender._apply_receiver_update`` fetches before pausing and
        pauses only around the final reload, so the dispatcher must not pause as well --
        doing so would hold generation down across the whole publish+upload+fetch window
        instead of just the reload.
        """
        from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch

        dispatch = WorkerDispatch.__new__(WorkerDispatch)
        dispatch.colocate_all = False
        dispatch.cfg = _fft_dispatch_cfg(weight_sync_backend="delta")
        dispatch._inference_engine_client = AsyncMock()
        dispatch._broadcast_to_inference_engines = MagicMock()
        dispatch._prepare_for_weight_sync = MagicMock()
        dispatch._finish_weight_sync = MagicMock()
        dispatch.ensure_active_adapter = MagicMock()

        await dispatch.save_weights_for_sampler()

        dispatch._inference_engine_client.pause_generation.assert_not_awaited()
        dispatch._inference_engine_client.resume_generation.assert_not_awaited()
        # The sync itself must still happen, and still be finalized.
        dispatch._broadcast_to_inference_engines.assert_called_once()
        dispatch._finish_weight_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_colocated_uses_wake_up(self):
        from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch

        dispatch = WorkerDispatch.__new__(WorkerDispatch)
        dispatch.colocate_all = True
        dispatch.cfg = _fft_dispatch_cfg()
        dispatch._inference_engine_client = AsyncMock()
        dispatch._broadcast_to_inference_engines = MagicMock()
        dispatch._prepare_for_weight_sync = MagicMock()
        dispatch._finish_weight_sync = MagicMock()
        dispatch.ensure_active_adapter = MagicMock()

        await dispatch.save_weights_for_sampler()

        dispatch._inference_engine_client.wake_up.assert_awaited()
        dispatch._inference_engine_client.pause_generation.assert_not_awaited()
        dispatch._inference_engine_client.resume_generation.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_colocated_pause_before_broadcast(self):
        from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch

        call_order = []

        dispatch = WorkerDispatch.__new__(WorkerDispatch)
        dispatch.colocate_all = False
        dispatch.cfg = _fft_dispatch_cfg()
        dispatch._inference_engine_client = AsyncMock()
        dispatch._inference_engine_client.pause_generation = AsyncMock(side_effect=lambda: call_order.append("pause"))
        dispatch._inference_engine_client.resume_generation = AsyncMock(side_effect=lambda: call_order.append("resume"))
        dispatch._broadcast_to_inference_engines = MagicMock(
            side_effect=lambda *args, **kwargs: call_order.append("broadcast")
        )
        dispatch._prepare_for_weight_sync = MagicMock()
        dispatch._finish_weight_sync = MagicMock()
        dispatch.ensure_active_adapter = MagicMock()

        await dispatch.save_weights_for_sampler()

        assert call_order == ["pause", "broadcast", "resume"]

    @pytest.mark.asyncio
    async def test_non_colocated_resumes_on_broadcast_failure(self):
        """resume_generation must be called even if broadcast raises."""
        from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch

        dispatch = WorkerDispatch.__new__(WorkerDispatch)
        dispatch.colocate_all = False
        dispatch.cfg = _fft_dispatch_cfg()
        dispatch._inference_engine_client = AsyncMock()
        dispatch._broadcast_to_inference_engines = MagicMock(side_effect=RuntimeError("broadcast failed"))
        dispatch._prepare_for_weight_sync = MagicMock()
        dispatch._finish_weight_sync = MagicMock()
        dispatch.ensure_active_adapter = MagicMock()

        with pytest.raises(RuntimeError, match="broadcast failed"):
            await dispatch.save_weights_for_sampler()

        dispatch._inference_engine_client.pause_generation.assert_awaited_once()
        dispatch._inference_engine_client.resume_generation.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_non_colocated_inplace_lora_skips_pause_and_resume(self):
        """In-place LoRA (lora.rank>0, no merge_lora) must NOT pause/resume.

        Mirrors the multi-tenant branch in
        ``save_weights_for_sampler``: when the engine's LoRA tensors are
        swapped in place via ``load_lora_adapter``, the weight sync is
        dispatched without any pause — load_lora_adapter is the engine-
        side primitive that's expected to be safe under in-flight
        requests on its own.
        """
        from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch

        cfg = _fft_dispatch_cfg()
        cfg.trainer.policy.model.lora.rank = 32  # in-place LoRA path
        cfg.trainer.policy.megatron_config.lora_config.merge_lora = False

        dispatch = WorkerDispatch.__new__(WorkerDispatch)
        dispatch.colocate_all = False
        dispatch.cfg = cfg
        dispatch._inference_engine_client = AsyncMock()
        dispatch._broadcast_to_inference_engines = MagicMock()
        dispatch._prepare_for_weight_sync = MagicMock()
        dispatch._finish_weight_sync = MagicMock()
        dispatch.ensure_active_adapter = MagicMock()

        await dispatch.save_weights_for_sampler(model_id="lora-target")

        dispatch._broadcast_to_inference_engines.assert_called_once()
        dispatch._inference_engine_client.pause_generation.assert_not_awaited()
        dispatch._inference_engine_client.resume_generation.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_colocated_megatron_merge_lora_still_pauses(self):
        """Megatron + merge_lora keeps the pause/resume path (LoRA merged
        into the base weights → tensors flow over NCCL, not load_lora_adapter)."""
        from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch

        cfg = _fft_dispatch_cfg()
        cfg.trainer.strategy = "megatron"
        cfg.trainer.policy.model.lora.rank = 32
        cfg.trainer.policy.megatron_config.lora_config.merge_lora = True

        dispatch = WorkerDispatch.__new__(WorkerDispatch)
        dispatch.colocate_all = False
        dispatch.cfg = cfg
        dispatch._inference_engine_client = AsyncMock()
        dispatch._broadcast_to_inference_engines = MagicMock()
        dispatch._prepare_for_weight_sync = MagicMock()
        dispatch._finish_weight_sync = MagicMock()
        dispatch.ensure_active_adapter = MagicMock()

        await dispatch.save_weights_for_sampler()

        dispatch._inference_engine_client.pause_generation.assert_awaited_once()
        dispatch._inference_engine_client.resume_generation.assert_awaited_once()


def _adapter_sync_dispatch(*, model_on_gpu: bool, optimizer_on_gpu: bool = False):
    """Dispatch wired for the colocated adapter-only sync path
    (megatron + lora.rank>0 + merge_lora=False + colocate_all)."""
    from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch

    cfg = _fft_dispatch_cfg()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.policy.model.lora.rank = 32
    cfg.trainer.policy.megatron_config.lora_config.merge_lora = False

    dispatch = WorkerDispatch.__new__(WorkerDispatch)
    dispatch.colocate_all = True
    dispatch.cfg = cfg
    dispatch._inference_engine_client = AsyncMock()
    dispatch.ensure_active_adapter = MagicMock()
    dispatch._ensure_on_gpu = MagicMock()
    dispatch._offload = MagicMock()
    dispatch.empty_cache = MagicMock()
    dispatch._load_lora_on_engines = AsyncMock()
    dispatch._gpu_state = {
        "policy": SimpleNamespace(model_on_gpu=model_on_gpu, optimizer_on_gpu=optimizer_on_gpu)
    }
    group = MagicMock()
    group.async_run_ray_method.return_value = "export-future"
    dispatch._actor_groups = {"policy": group}
    return dispatch


class TestAdapterOnlyColocatedSync:
    """Tests for `WorkerDispatch._sync_lora_adapters_colocated`.

    The adapter export streams from the GPU-resident LoRA DDP buffers
    (exempt from offload), so the TB-scale frozen masters must never be
    backloaded for a sampler sync — a cold-trainer sync that re-faulted
    them through an evicted page cache stalled run starts for 15+ minutes.
    """

    @pytest.mark.asyncio
    async def test_cold_sync_skips_sleep_backload_and_offload(self, monkeypatch):
        """Fully offloaded trainer + awake engines: the sync must not sleep
        the engines, not backload the masters, and not offload anything —
        just export the adapter and load it on the live engines."""
        from skyrl.backends.skyrl_train.workers import worker_dispatch as wd

        dispatch = _adapter_sync_dispatch(model_on_gpu=False)
        monkeypatch.setattr(wd.ray, "get", lambda _: [("adapter-m1", "/tmp/lora/m1")])

        await dispatch.save_weights_for_sampler(model_id="m1", engines_asleep=False)

        dispatch._inference_engine_client.sleep.assert_not_awaited()
        dispatch._ensure_on_gpu.assert_not_called()
        dispatch._offload.assert_not_called()
        dispatch._inference_engine_client.wake_up.assert_not_awaited()
        dispatch._load_lora_on_engines.assert_awaited_once_with("adapter-m1", "/tmp/lora/m1")

    @pytest.mark.asyncio
    async def test_cold_sync_wakes_asleep_engines_without_backload(self, monkeypatch):
        """Asleep engines still get the weights+kv wake sequence, but the
        masters stay offloaded."""
        from skyrl.backends.skyrl_train.workers import worker_dispatch as wd

        dispatch = _adapter_sync_dispatch(model_on_gpu=False)
        monkeypatch.setattr(wd.ray, "get", lambda _: [("adapter-m1", "/tmp/lora/m1")])

        await dispatch.save_weights_for_sampler(model_id="m1", engines_asleep=True)

        dispatch._ensure_on_gpu.assert_not_called()
        dispatch._inference_engine_client.sleep.assert_not_awaited()
        wake_tags = [c.kwargs["tags"] for c in dispatch._inference_engine_client.wake_up.await_args_list]
        assert wake_tags == [["weights"], ["kv_cache"]]
        dispatch._load_lora_on_engines.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_hot_sync_offloads_resident_masters_before_wake(self, monkeypatch):
        """Post-optim sync (masters resident, engines asleep): masters and
        optimizer must be offloaded so the engine wake fits, but still no
        backload."""
        from skyrl.backends.skyrl_train.workers import worker_dispatch as wd

        dispatch = _adapter_sync_dispatch(model_on_gpu=True, optimizer_on_gpu=True)
        monkeypatch.setattr(wd.ray, "get", lambda _: [("adapter-m1", "/tmp/lora/m1")])

        await dispatch.save_weights_for_sampler(model_id="m1", engines_asleep=True)

        dispatch._ensure_on_gpu.assert_not_called()
        dispatch._offload.assert_called_once_with("policy", offload_optimizer=True, offload_model=True)
        dispatch.empty_cache.assert_called_once_with("policy")
        wake_tags = [c.kwargs["tags"] for c in dispatch._inference_engine_client.wake_up.await_args_list]
        assert wake_tags == [["weights"], ["kv_cache"]]
        dispatch._load_lora_on_engines.assert_awaited_once()

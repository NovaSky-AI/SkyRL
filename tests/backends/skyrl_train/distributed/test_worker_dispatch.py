"""Tests for ``WorkerDispatch``'s weight-sync orchestration.

Covers which pause/sleep/wake bracket ``save_weights_for_sampler`` puts around
the broadcast for each placement, LoRA mode and weight-sync backend, plus the
optimizer-offload policy applied before and after the sync. CPU-only: the
dispatch is built via ``__new__`` with mocked actor groups and client.
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
    exercised; pass ``"delta"`` for the branch where the trainer engine pauses internally.
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
        dispatch._prepare_for_weight_sync = AsyncMock()
        dispatch._finish_weight_sync = MagicMock()
        dispatch.ensure_active_adapter = MagicMock()

        await dispatch.save_weights_for_sampler()

        dispatch._inference_engine_client.pause_generation.assert_awaited_once()
        dispatch._broadcast_to_inference_engines.assert_called_once()
        dispatch._inference_engine_client.resume_generation.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_non_colocated_delta_does_not_pause(self):
        """Delta sync owns pause/resume itself.

        ``DeltaTrainerWeightTransferEngine._apply_receiver_update`` fetches before pausing
        and pauses only around the final reload, so the dispatcher must not pause as well.
        """
        from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch

        dispatch = WorkerDispatch.__new__(WorkerDispatch)
        dispatch.colocate_all = False
        dispatch.cfg = _fft_dispatch_cfg(weight_sync_backend="delta")
        dispatch._inference_engine_client = AsyncMock()
        dispatch._broadcast_to_inference_engines = MagicMock()
        dispatch._prepare_for_weight_sync = AsyncMock()
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
        dispatch._prepare_for_weight_sync = AsyncMock()
        dispatch._finish_weight_sync = MagicMock()
        dispatch.ensure_active_adapter = MagicMock()

        await dispatch.save_weights_for_sampler()

        dispatch._prepare_for_weight_sync.assert_awaited_once_with(adapter_only_sync=False)
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
        dispatch._prepare_for_weight_sync = AsyncMock()
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
        dispatch._prepare_for_weight_sync = AsyncMock()
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
        dispatch._prepare_for_weight_sync = AsyncMock()
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
        dispatch._prepare_for_weight_sync = AsyncMock()
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
    dispatch._inference_engine_client.increment_weight_version = MagicMock()
    dispatch._broadcast_to_inference_engines = MagicMock()
    dispatch._ensure_on_gpu = MagicMock()
    dispatch._offload = MagicMock()
    dispatch.empty_cache = MagicMock()
    dispatch._gpu_state = {"policy": SimpleNamespace(model_on_gpu=model_on_gpu, optimizer_on_gpu=optimizer_on_gpu)}
    group = MagicMock()
    group.async_run_ray_method.return_value = "swap-future"
    dispatch._actor_groups = {"policy": group}
    return dispatch


class _FakeActorGroup:
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def backload_to_gpu(self, backload_optimizer=True, backload_model=True):
        self.calls.append(("backload", self.name, backload_optimizer, backload_model))

    def offload_to_cpu(self, offload_optimizer=True, offload_model=True):
        self.calls.append(("offload", self.name, offload_optimizer, offload_model))


def _prepare_sync_dispatch(initial_state, *, offload_after_step=True):
    from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch

    cfg = _fft_dispatch_cfg()
    cfg.trainer.policy.optimizer_config = SimpleNamespace(offload_after_step=offload_after_step)

    calls = []
    dispatch = WorkerDispatch.__new__(WorkerDispatch)
    dispatch.colocate_all = True
    dispatch.colocate_policy_ref = False
    dispatch.cfg = cfg
    dispatch._inference_engine_client = AsyncMock()
    dispatch._inference_engine_client.is_sleeping.return_value = False
    dispatch.empty_cache = MagicMock()
    dispatch._gpu_state = {
        name: SimpleNamespace(model_on_gpu=model_on_gpu, optimizer_on_gpu=optimizer_on_gpu)
        for name, (model_on_gpu, optimizer_on_gpu) in initial_state.items()
    }
    dispatch._actor_groups = {name: _FakeActorGroup(name, calls) for name in initial_state}
    return dispatch, calls


def _gpu_state_snapshot(dispatch):
    return {name: (state.model_on_gpu, state.optimizer_on_gpu) for name, state in dispatch._gpu_state.items()}


class TestAdapterOnlyColocatedSync:
    """Tests for adapter-only colocated LoRA sync.

    The common broadcast path exports from GPU-resident LoRA DDP buffers
    (exempt from offload), so the TB-scale frozen masters must never be
    backloaded for a sampler sync.
    """

    @pytest.mark.asyncio
    async def test_cold_sync_skips_sleep_backload_and_offload(self, monkeypatch):
        """Fully offloaded trainer: do not sleep engines or backload masters."""
        from skyrl.backends.skyrl_train.workers import worker_dispatch as wd

        dispatch = _adapter_sync_dispatch(model_on_gpu=False)
        monkeypatch.setattr(wd.ray, "get", lambda _: None)

        await dispatch.save_weights_for_sampler(model_id="m1")

        dispatch._inference_engine_client.sleep.assert_not_awaited()
        dispatch._ensure_on_gpu.assert_not_called()
        dispatch._offload.assert_not_called()
        wake_tags = [c.kwargs["tags"] for c in dispatch._inference_engine_client.wake_up.await_args_list]
        assert wake_tags == [["weights"], ["kv_cache"]]
        dispatch._broadcast_to_inference_engines.assert_called_once_with(
            dispatch._inference_engine_client, model_id="m1"
        )
        dispatch._inference_engine_client.increment_weight_version.assert_called_once()

    @pytest.mark.asyncio
    async def test_swap_does_not_require_model_resident(self, monkeypatch):
        """Adapter-only sync swaps the live adapter without forcing a base-model backload."""
        from skyrl.backends.skyrl_train.workers import worker_dispatch as wd

        dispatch = _adapter_sync_dispatch(model_on_gpu=False)
        monkeypatch.setattr(wd.ray, "get", lambda _: None)

        await dispatch.save_weights_for_sampler(model_id="m1")

        dispatch._ensure_on_gpu.assert_not_called()
        dispatch._actor_groups["policy"].async_run_ray_method.assert_called_once_with(
            "pass_through", "swap_to_adapter", "m1"
        )

    @pytest.mark.asyncio
    async def test_hot_sync_offloads_resident_masters_before_wake(self, monkeypatch):
        """Post-optim sync offloads trainer state so the engine wake fits."""
        from skyrl.backends.skyrl_train.workers import worker_dispatch as wd

        dispatch = _adapter_sync_dispatch(model_on_gpu=True, optimizer_on_gpu=True)
        monkeypatch.setattr(wd.ray, "get", lambda _: None)

        await dispatch.save_weights_for_sampler(model_id="m1")

        dispatch._ensure_on_gpu.assert_not_called()
        dispatch._offload.assert_called_once_with("policy", offload_optimizer=True, offload_model=True)
        dispatch.empty_cache.assert_called_once_with("policy")
        wake_tags = [c.kwargs["tags"] for c in dispatch._inference_engine_client.wake_up.await_args_list]
        assert wake_tags == [["weights"], ["kv_cache"]]
        dispatch._broadcast_to_inference_engines.assert_called_once_with(
            dispatch._inference_engine_client, model_id="m1"
        )


@pytest.mark.parametrize(
    ("initial_state", "expected_calls"),
    [
        ({"policy": (False, False)}, []),
        ({"policy": (True, True)}, [("offload", "policy", True, True)]),
        (
            {"policy": (False, False), "critic": (True, True)},
            [("offload", "critic", True, True)],
        ),
    ],
)
@pytest.mark.asyncio
async def test_prepare_for_adapter_only_sync_offloads_all_tracked_trainer_state(initial_state, expected_calls):
    dispatch, calls = _prepare_sync_dispatch(initial_state)

    await dispatch._prepare_for_weight_sync(adapter_only_sync=True)

    assert _gpu_state_snapshot(dispatch) == {name: (False, False) for name in initial_state}
    assert calls == expected_calls
    # adapter only sync doesn't issue any sleep/ wake up calls to the inference engine
    dispatch._inference_engine_client.sleep.assert_not_awaited()
    dispatch._inference_engine_client.wake_up.assert_not_awaited()
    dispatch.empty_cache.assert_called_once_with("policy")


@pytest.mark.parametrize(
    ("offload_after_step", "initial_state", "expected_state", "expected_calls"),
    [
        (
            True,
            {"policy": (False, False)},
            {"policy": (True, False)},
            [("backload", "policy", False, True)],
        ),
        (
            True,
            {"policy": (True, True)},
            {"policy": (True, False)},
            [("offload", "policy", True, False)],
        ),
        (
            False,
            {"policy": (True, True)},
            {"policy": (True, True)},
            [],
        ),
        (
            True,
            {"policy": (False, False), "critic": (True, True)},
            {"policy": (True, False), "critic": (False, False)},
            [("offload", "critic", True, True), ("backload", "policy", False, True)],
        ),
    ],
)
@pytest.mark.asyncio
async def test_prepare_for_full_sync_leaves_policy_weights_on_gpu(
    offload_after_step, initial_state, expected_state, expected_calls
):
    dispatch, calls = _prepare_sync_dispatch(initial_state, offload_after_step=offload_after_step)

    await dispatch._prepare_for_weight_sync(adapter_only_sync=False)

    assert _gpu_state_snapshot(dispatch) == expected_state
    assert calls == expected_calls
    dispatch._inference_engine_client.sleep.assert_awaited_once()
    dispatch._inference_engine_client.wake_up.assert_not_awaited()
    dispatch.empty_cache.assert_called_once_with("policy")


@pytest.mark.parametrize("offload_after_step", [False, True])
@pytest.mark.asyncio
async def test_weight_sync_honors_optimizer_offload_policy(offload_after_step):
    from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch

    cfg = _fft_dispatch_cfg()
    cfg.trainer.policy.optimizer_config = SimpleNamespace(offload_after_step=offload_after_step)

    dispatch = WorkerDispatch.__new__(WorkerDispatch)
    dispatch.colocate_all = True
    dispatch.cfg = cfg
    dispatch._inference_engine_client = AsyncMock()
    dispatch._inference_engine_client.is_sleeping.return_value = False
    dispatch.empty_cache = MagicMock()

    dispatch._gpu_state = {
        "policy": SimpleNamespace(model_on_gpu=True, optimizer_on_gpu=True),
    }
    dispatch._ensure_on_gpu = MagicMock()
    dispatch._offload = MagicMock()

    await dispatch._prepare_for_weight_sync()

    dispatch._inference_engine_client.sleep.assert_awaited_once()
    dispatch._ensure_on_gpu.assert_called_once_with(
        "policy",
        need_optimizer=False,
        need_model=True,
    )
    if offload_after_step:
        dispatch._offload.assert_called_once_with("policy", offload_optimizer=True, offload_model=False)
    else:
        dispatch._offload.assert_not_called()

    dispatch._offload.reset_mock()
    dispatch._finish_weight_sync()
    dispatch._offload.assert_called_once_with("policy", offload_optimizer=offload_after_step, offload_model=True)


def test_offload_inactive_model_records_offloaded_state():
    """
    ``_offload_inactive_model`` performs a real ``offload_to_cpu()``, so it must
    record the model as not resident. Recording "resident" would make the next
    ``_ensure_on_gpu`` skip the backload and run the model from CPU.
    """
    from skyrl.backends.skyrl_train.workers.worker_dispatch import (
        GPUState,
        WorkerDispatch,
    )

    calls = []
    group = SimpleNamespace(offload_to_cpu=lambda *a, **k: calls.append("offload"))
    stub = SimpleNamespace(
        _actor_groups={"policy": group},
        _gpu_state={"policy": GPUState(model_on_gpu=True, optimizer_on_gpu=True)},
    )

    WorkerDispatch._offload_inactive_model(stub, "policy")

    assert calls == ["offload"]
    assert stub._gpu_state["policy"] == GPUState(model_on_gpu=False, optimizer_on_gpu=False)


def test_gpu_state_requires_explicit_intent():
    """GPUState must not be constructible without stating both fields."""
    from skyrl.backends.skyrl_train.workers.worker_dispatch import GPUState

    with pytest.raises(TypeError):
        GPUState()

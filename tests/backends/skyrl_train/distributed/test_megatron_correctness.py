"""Tests for Megatron backend correctness fixes.

Tests that require megatron-core (GPU dependency) are skipped when it is not
installed.
"""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _fft_dispatch_cfg() -> SimpleNamespace:
    """Build the minimal ``self.cfg`` view that ``save_weights_for_sampler``
    inspects on the non-colocated path. Defaults to FFT (lora.rank=0) so
    the pause/resume branch is taken.
    """
    return SimpleNamespace(
        trainer=SimpleNamespace(
            strategy="fsdp",
            policy=SimpleNamespace(
                model=SimpleNamespace(lora=SimpleNamespace(rank=0)),
                megatron_config=SimpleNamespace(lora_config=SimpleNamespace(merge_lora=False)),
            ),
        )
    )


_has_megatron = "megatron" in sys.modules or __import__("importlib").util.find_spec("megatron") is not None


# ---------------------------------------------------------------------------
# C1: grad_scale_func fix
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_megatron, reason="megatron-core not installed")
class TestGradScaleFunc:
    """Verify MegatronModelWrapper sets grad_scale_func when optimizer is provided."""

    def test_grad_scale_func_set_with_optimizer(self):
        """When optimizer is provided, grad_scale_func should be set."""
        from skyrl.backends.skyrl_train.workers.megatron.megatron_model_wrapper import (
            MegatronModelWrapper,
        )

        mock_module = MagicMock()
        mock_config_obj = MagicMock()
        mock_config_obj.finalize_model_grads_func = None
        mock_config_obj.grad_scale_func = None

        mock_optimizer = MagicMock()
        mock_optimizer.scale_loss = MagicMock(return_value=1.0)

        with patch(
            "skyrl.backends.skyrl_train.workers.megatron.megatron_model_wrapper.get_model_config",
            return_value=mock_config_obj,
        ):
            mock_skyrl_config = MagicMock()
            mock_skyrl_config.trainer.use_sample_packing = False

            MegatronModelWrapper(
                config=mock_skyrl_config,
                actor_module=[mock_module],
                actor_optimizer=mock_optimizer,
            )

        assert mock_config_obj.grad_scale_func is mock_optimizer.scale_loss

    def test_grad_scale_func_not_set_without_optimizer(self):
        """When optimizer is None (ref model), grad_scale_func stays None."""
        from skyrl.backends.skyrl_train.workers.megatron.megatron_model_wrapper import (
            MegatronModelWrapper,
        )

        mock_module = MagicMock()
        mock_config_obj = MagicMock()
        mock_config_obj.finalize_model_grads_func = None
        mock_config_obj.grad_scale_func = None

        with patch(
            "skyrl.backends.skyrl_train.workers.megatron.megatron_model_wrapper.get_model_config",
            return_value=mock_config_obj,
        ):
            mock_skyrl_config = MagicMock()
            mock_skyrl_config.trainer.use_sample_packing = False

            MegatronModelWrapper(
                config=mock_skyrl_config,
                actor_module=[mock_module],
                actor_optimizer=None,
            )

        assert mock_config_obj.grad_scale_func is None


# ---------------------------------------------------------------------------
# C4: Seed variation by PP rank
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_megatron, reason="megatron-core not installed")
class TestSeedVariation:
    """Verify set_seed varies the seed by PP rank."""

    @pytest.mark.parametrize(
        "pp_rank, expected_seed",
        [
            (0, 42),  # PP=1: seed unchanged
            (1, 142),  # 42 + 100*1
            (3, 342),  # 42 + 100*3
        ],
    )
    def test_seed_offset_by_pp_rank(self, pp_rank, expected_seed):
        from skyrl.backends.skyrl_train.distributed.megatron.megatron_strategy import (
            MegatronStrategy,
        )
        from skyrl.train.config.config import MegatronConfig

        strategy = MegatronStrategy(megatron_config=MegatronConfig(), seed=42)

        with patch("skyrl.backends.skyrl_train.distributed.megatron.megatron_strategy.mpu") as mock_mpu:
            mock_mpu.get_pipeline_model_parallel_rank.return_value = pp_rank
            captured = []
            with patch("random.seed", side_effect=lambda s: captured.append(s)):
                strategy.set_seed(42)
            assert captured[0] == expected_seed


# ---------------------------------------------------------------------------
# C5: Pause/flush for non-colocated weight sync
# ---------------------------------------------------------------------------


class TestWeightSyncPauseFlush:
    """Verify save_weights_for_sampler pauses/resumes in non-colocated mode."""

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

"""Tests for Megatron backend correctness fixes.

Tests that require megatron-core (GPU dependency) are skipped when it is not
installed.
"""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
# C6: VPP checkpoint layout
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_megatron, reason="megatron-core not installed")
class TestVPPCheckpointLayout:
    """Verify VPP checkpoint helpers preserve legacy single-chunk layout."""

    def test_single_chunk_keeps_legacy_model_key(self):
        from skyrl.backends.skyrl_train.distributed.megatron.megatron_strategy import (
            MegatronStrategy,
        )
        from skyrl.train.config.config import MegatronConfig

        class _Chunk:
            def sharded_state_dict(self):
                return {"chunk0": "chunk0_state"}

        strategy = MegatronStrategy(megatron_config=MegatronConfig(), seed=42)
        _, model_sharded_state_dicts, sharded_state_dict = strategy._get_sharded_model_state_dicts([_Chunk()])

        assert model_sharded_state_dicts == [{"chunk0": "chunk0_state"}]
        assert sharded_state_dict == {"model": {"chunk0": "chunk0_state"}}

    def test_multi_chunk_uses_per_chunk_keys(self):
        from skyrl.backends.skyrl_train.distributed.megatron.megatron_strategy import (
            MegatronStrategy,
        )
        from skyrl.train.config.config import MegatronConfig

        class _Chunk:
            def __init__(self, name):
                self.name = name

            def sharded_state_dict(self):
                return {self.name: f"{self.name}_state"}

        chunk0 = _Chunk("chunk0")
        chunk1 = _Chunk("chunk1")
        wrapper1 = type("_Wrapper", (), {"module": chunk1})()

        strategy = MegatronStrategy(megatron_config=MegatronConfig(), seed=42)
        unwrapped, model_sharded_state_dicts, sharded_state_dict = strategy._get_sharded_model_state_dicts(
            [chunk0, wrapper1]
        )

        assert unwrapped == [chunk0, chunk1]
        assert model_sharded_state_dicts == [{"chunk0": "chunk0_state"}, {"chunk1": "chunk1_state"}]
        assert sharded_state_dict == {"model0": {"chunk0": "chunk0_state"}, "model1": {"chunk1": "chunk1_state"}}

    def test_vpp_lora_checkpointing_not_supported(self):
        from skyrl.backends.skyrl_train.distributed.megatron.megatron_strategy import (
            MegatronStrategy,
        )
        from skyrl.train.config.config import MegatronConfig

        strategy = MegatronStrategy(megatron_config=MegatronConfig(), seed=42, is_lora=True)

        with pytest.raises(NotImplementedError, match="VPP \\+ LoRA"):
            strategy._get_sharded_model_state_dicts([object(), object()])


@pytest.mark.skipif(not _has_megatron, reason="megatron-core not installed")
class TestVPPValidation:
    def test_rejects_vpp_sizes_lte_one(self):
        from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
            MegatronWorker,
        )
        from skyrl.train.config.config import MegatronConfig

        worker = MegatronWorker.__new__(MegatronWorker)
        worker.strategy = SimpleNamespace(hf_config=SimpleNamespace(num_hidden_layers=32))

        for vpp_size in (0, 1):
            megatron_config = MegatronConfig(
                pipeline_model_parallel_size=4,
                virtual_pipeline_model_parallel_size=vpp_size,
            )
            with pytest.raises(AssertionError, match="virtual_pipeline_model_parallel_size must be greater than 1"):
                worker._validate_virtual_pipeline_parallelism(megatron_config)


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
        dispatch._inference_engine_client = AsyncMock()
        dispatch.broadcast_to_inference_engines = MagicMock()
        dispatch.prepare_for_weight_sync = MagicMock()
        dispatch.finish_weight_sync = MagicMock()

        await dispatch.save_weights_for_sampler()

        dispatch._inference_engine_client.pause_generation.assert_awaited_once()
        dispatch.broadcast_to_inference_engines.assert_called_once()
        dispatch._inference_engine_client.resume_generation.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_colocated_uses_wake_up(self):
        from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch

        dispatch = WorkerDispatch.__new__(WorkerDispatch)
        dispatch.colocate_all = True
        dispatch._inference_engine_client = AsyncMock()
        dispatch.broadcast_to_inference_engines = MagicMock()
        dispatch.prepare_for_weight_sync = MagicMock()
        dispatch.finish_weight_sync = MagicMock()

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
        dispatch._inference_engine_client = AsyncMock()
        dispatch._inference_engine_client.pause_generation = AsyncMock(side_effect=lambda: call_order.append("pause"))
        dispatch._inference_engine_client.resume_generation = AsyncMock(side_effect=lambda: call_order.append("resume"))
        dispatch.broadcast_to_inference_engines = MagicMock(side_effect=lambda _: call_order.append("broadcast"))
        dispatch.prepare_for_weight_sync = MagicMock()
        dispatch.finish_weight_sync = MagicMock()

        await dispatch.save_weights_for_sampler()

        assert call_order == ["pause", "broadcast", "resume"]

    @pytest.mark.asyncio
    async def test_non_colocated_resumes_on_broadcast_failure(self):
        """resume_generation must be called even if broadcast raises."""
        from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch

        dispatch = WorkerDispatch.__new__(WorkerDispatch)
        dispatch.colocate_all = False
        dispatch._inference_engine_client = AsyncMock()
        dispatch.broadcast_to_inference_engines = MagicMock(side_effect=RuntimeError("broadcast failed"))
        dispatch.prepare_for_weight_sync = MagicMock()
        dispatch.finish_weight_sync = MagicMock()

        with pytest.raises(RuntimeError, match="broadcast failed"):
            await dispatch.save_weights_for_sampler()

        dispatch._inference_engine_client.pause_generation.assert_awaited_once()
        dispatch._inference_engine_client.resume_generation.assert_awaited_once()

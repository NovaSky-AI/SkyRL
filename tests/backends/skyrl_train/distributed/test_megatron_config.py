"""Tests for MoE config fields and Megatron correctness fixes.

Tests that require megatron-core (GPU dependency) are skipped when it is not
installed.  Config-level tests run on CPU without any extra dependencies.
"""

import dataclasses
import sys
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from skyrl.train.config.config import MegatronConfig, build_nested_dataclass

# megatron-core is a GPU-only dependency; guard imports that need it
_has_megatron = "megatron" in sys.modules or not (
    # Try a lightweight probe
    __import__("importlib").util.find_spec("megatron") is None
)


# ---------------------------------------------------------------------------
# TODO-1: MoE config fields on MegatronConfig dataclass
# ---------------------------------------------------------------------------


class TestMegatronConfigMoEFields:
    """Verify the 5 new MoE config fields exist with correct types and defaults."""

    def test_moe_fields_exist(self):
        cfg = MegatronConfig()
        assert hasattr(cfg, "moe_token_dispatcher_type")
        assert hasattr(cfg, "moe_router_load_balancing_type")
        assert hasattr(cfg, "moe_grouped_gemm")
        assert hasattr(cfg, "moe_router_score_function")
        assert hasattr(cfg, "moe_router_enable_expert_bias")

    def test_moe_field_defaults(self):
        cfg = MegatronConfig()
        assert cfg.moe_token_dispatcher_type == "alltoall"
        assert cfg.moe_router_load_balancing_type == "none"
        assert cfg.moe_grouped_gemm is False
        assert cfg.moe_router_score_function is None
        assert cfg.moe_router_enable_expert_bias is None

    def test_moe_fields_override(self):
        cfg = MegatronConfig(
            moe_token_dispatcher_type="allgather",
            moe_router_load_balancing_type="aux_loss",
            moe_grouped_gemm=True,
            moe_router_score_function="sigmoid",
            moe_router_enable_expert_bias=True,
        )
        assert cfg.moe_token_dispatcher_type == "allgather"
        assert cfg.moe_router_load_balancing_type == "aux_loss"
        assert cfg.moe_grouped_gemm is True
        assert cfg.moe_router_score_function == "sigmoid"
        assert cfg.moe_router_enable_expert_bias is True

    def test_moe_config_from_dict(self):
        """MoE fields should survive dict -> dataclass round-trip."""
        d = {
            "moe_token_dispatcher_type": "alltoall",
            "moe_router_load_balancing_type": "none",
            "moe_grouped_gemm": True,
            "moe_router_score_function": "sigmoid",
            "moe_router_enable_expert_bias": True,
        }
        cfg = build_nested_dataclass(MegatronConfig, d)
        assert cfg.moe_grouped_gemm is True
        assert cfg.moe_router_score_function == "sigmoid"
        assert cfg.moe_router_enable_expert_bias is True

    def test_backward_compatible_defaults(self):
        """Default values must match the old hardcoded values for backward compat."""
        cfg = MegatronConfig()
        # These were previously hardcoded in megatron_worker.py
        assert cfg.moe_token_dispatcher_type == "alltoall"
        assert cfg.moe_router_load_balancing_type == "none"

    def test_parallelism_fields_unchanged(self):
        """Existing parallelism fields should still work."""
        cfg = MegatronConfig(
            tensor_model_parallel_size=4,
            pipeline_model_parallel_size=2,
            expert_model_parallel_size=8,
        )
        assert cfg.tensor_model_parallel_size == 4
        assert cfg.pipeline_model_parallel_size == 2
        assert cfg.expert_model_parallel_size == 8


# ---------------------------------------------------------------------------
# TODO-3a: grad_scale_func fix
# Requires megatron-core to import MegatronModelWrapper
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
# TODO-3b: Seed variation by PP rank
# Also requires megatron-core for MegatronStrategy imports
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_megatron, reason="megatron-core not installed")
class TestSeedVariation:
    """Verify set_seed varies the seed by PP rank."""

    def test_seed_offset_by_pp_rank(self):
        """Seeds should differ by 100 * pp_rank."""
        from skyrl.backends.skyrl_train.distributed.megatron.megatron_strategy import (
            MegatronStrategy,
        )

        strategy = MegatronStrategy(megatron_config=MegatronConfig(), seed=42)

        with patch("skyrl.backends.skyrl_train.distributed.megatron.megatron_strategy.mpu") as mock_mpu:
            seeds_seen = []

            def capture_seed(seed):
                seeds_seen.append(seed)

            # PP rank 0
            mock_mpu.get_pipeline_model_parallel_rank.return_value = 0
            with patch("random.seed", side_effect=capture_seed):
                strategy.set_seed(42)
            assert seeds_seen[-1] == 42  # 42 + 100*0

            # PP rank 1
            mock_mpu.get_pipeline_model_parallel_rank.return_value = 1
            with patch("random.seed", side_effect=capture_seed):
                strategy.set_seed(42)
            assert seeds_seen[-1] == 142  # 42 + 100*1

            # PP rank 3
            mock_mpu.get_pipeline_model_parallel_rank.return_value = 3
            with patch("random.seed", side_effect=capture_seed):
                strategy.set_seed(42)
            assert seeds_seen[-1] == 342  # 42 + 100*3

    def test_pp1_seed_unchanged(self):
        """With PP=1, the seed should be identical to the input."""
        from skyrl.backends.skyrl_train.distributed.megatron.megatron_strategy import (
            MegatronStrategy,
        )

        strategy = MegatronStrategy(megatron_config=MegatronConfig(), seed=42)

        with patch("skyrl.backends.skyrl_train.distributed.megatron.megatron_strategy.mpu") as mock_mpu:
            mock_mpu.get_pipeline_model_parallel_rank.return_value = 0

            captured = []
            with patch("random.seed", side_effect=lambda s: captured.append(s)):
                strategy.set_seed(42)

            assert captured[0] == 42


# ---------------------------------------------------------------------------
# TODO-3c: Pause/flush for non-colocated weight sync
# ---------------------------------------------------------------------------


class TestWeightSyncPauseFlush:
    """Verify save_weights_for_sampler pauses/resumes in non-colocated mode."""

    @pytest.mark.asyncio
    async def test_non_colocated_calls_pause_and_resume(self):
        """Non-colocated path should call pause_generation and resume_generation."""
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
        """Colocated path should use wake_up, not pause/resume."""
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
        """pause_generation must happen before broadcast_to_inference_engines."""
        from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch

        call_order = []

        dispatch = WorkerDispatch.__new__(WorkerDispatch)
        dispatch.colocate_all = False
        dispatch._inference_engine_client = AsyncMock()
        dispatch._inference_engine_client.pause_generation = AsyncMock(
            side_effect=lambda: call_order.append("pause")
        )
        dispatch._inference_engine_client.resume_generation = AsyncMock(
            side_effect=lambda: call_order.append("resume")
        )
        dispatch.broadcast_to_inference_engines = MagicMock(
            side_effect=lambda _: call_order.append("broadcast")
        )
        dispatch.prepare_for_weight_sync = MagicMock()
        dispatch.finish_weight_sync = MagicMock()

        await dispatch.save_weights_for_sampler()

        assert call_order == ["pause", "broadcast", "resume"]

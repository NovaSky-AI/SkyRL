"""Tests for Megatron backend correctness fixes.

Tests that require megatron-core (GPU dependency) are skipped when it is not
installed.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

pytestmark = pytest.mark.megatron


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
            inference_engine=SimpleNamespace(offload_kv_for_weight_sync=False, weight_sync_backend=weight_sync_backend)
        ),
    )


_has_megatron = "megatron" in sys.modules or __import__("importlib").util.find_spec("megatron") is not None


# ---------------------------------------------------------------------------
# Packed aligned tensors
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_megatron, reason="megatron-core not installed")
@pytest.mark.parametrize("targets_rolled", [False, True])
def test_packed_fused_active_mask_is_cp_local_and_two_dimensional(targets_rolled):
    from skyrl.backends.skyrl_train.distributed.megatron import model_utils

    hidden = torch.zeros((1, 4, 128))
    weight = torch.zeros((16, 128))
    cu_seqlens = torch.tensor([0, 8])
    if targets_rolled:
        # Already this CP rank's next-token targets and mask, as the host builder lays them out.
        target = torch.arange(4).unsqueeze(0)
        active_mask = torch.tensor([[True, False, True, False]])
    else:
        target = torch.arange(8).unsqueeze(0)
        active_mask = torch.tensor([[True, False, True, False, True, False, True, False]])
    tp_group = object()
    cp_group = object()

    def fused_apply(*args, **kwargs):
        # No final logit softcap here; a softcapped model (Beam) passes it by keyword.
        assert kwargs.get("softcap") is None
        local_mask = args[-3]
        assert local_mask.shape == (1, 4)
        torch.testing.assert_close(local_mask, torch.tensor([[True, False, True, False]]))
        return torch.zeros((1, 4))

    with (
        patch.object(model_utils.torch.distributed, "get_world_size", return_value=2),
        patch.object(model_utils.torch.distributed, "get_rank", return_value=0),
        patch.object(model_utils, "_fused_lm_head_logprob_apply", side_effect=fused_apply),
        patch.object(model_utils, "allgather_cp_sharded_packed_tensor", return_value=torch.zeros(8)),
    ):
        result = model_utils.from_parallel_hidden_to_logprobs_packed_sequences(
            hidden,
            weight,
            target,
            cu_seqlens,
            unpacked_seqlen=8,
            vocab_start_index=0,
            vocab_end_index=16,
            group=tp_group,
            cp_group=cp_group,
            active_mask=active_mask,
            targets_rolled=targets_rolled,
        )

    assert result.shape == (1, 7)


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
        mock_config_obj.calculate_per_token_loss = False
        mock_config_obj.mlp_chunks_for_training = 1
        mock_config_obj.sequence_parallel = True

        mock_optimizer = MagicMock()
        mock_optimizer.scale_loss = MagicMock(return_value=1.0)

        with patch(
            "skyrl.backends.skyrl_train.workers.megatron.megatron_model_wrapper.get_model_config",
            return_value=mock_config_obj,
        ):
            mock_skyrl_config = MagicMock()
            mock_skyrl_config.trainer.remove_microbatch_padding = False

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
        mock_config_obj.calculate_per_token_loss = False
        mock_config_obj.mlp_chunks_for_training = 1
        mock_config_obj.sequence_parallel = False

        with patch(
            "skyrl.backends.skyrl_train.workers.megatron.megatron_model_wrapper.get_model_config",
            return_value=mock_config_obj,
        ):
            mock_skyrl_config = MagicMock()
            mock_skyrl_config.trainer.remove_microbatch_padding = False

            MegatronModelWrapper(
                config=mock_skyrl_config,
                actor_module=[mock_module],
                actor_optimizer=None,
            )

        assert mock_config_obj.grad_scale_func is None


@pytest.mark.skipif(not _has_megatron, reason="megatron-core not installed")
@pytest.mark.parametrize("overlap_grad_reduce", [False, True])
def test_per_token_loss_requires_no_overlap_grad_reduce(overlap_grad_reduce):
    """Under calculate_per_token_loss the wrapper rescales grad buffers between calls, which would
    race with overlapped grad reductions, so overlap_grad_reduce=True is refused at init."""
    from skyrl.backends.skyrl_train.workers.megatron.megatron_model_wrapper import (
        MegatronModelWrapper,
    )

    mock_module = MagicMock()
    mock_module.ddp_config.overlap_grad_reduce = overlap_grad_reduce
    mock_config_obj = MagicMock()
    mock_config_obj.calculate_per_token_loss = True
    mock_config_obj.mlp_chunks_for_training = 1
    mock_config_obj.sequence_parallel = False
    mock_skyrl_config = MagicMock()
    mock_skyrl_config.trainer.remove_microbatch_padding = True

    with patch(
        "skyrl.backends.skyrl_train.workers.megatron.megatron_model_wrapper.get_model_config",
        return_value=mock_config_obj,
    ):
        if overlap_grad_reduce:
            with pytest.raises(ValueError, match="overlap_grad_reduce=false"):
                MegatronModelWrapper(config=mock_skyrl_config, actor_module=[mock_module], actor_optimizer=None)
        else:
            wrapper = MegatronModelWrapper(config=mock_skyrl_config, actor_module=[mock_module], actor_optimizer=None)
            assert wrapper._per_token_loss


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

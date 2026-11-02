"""GPU coverage for score centering over the recorded sample support.

Run with::

    uv run --isolated --extra dev --extra megatron pytest -s -vvv \
        tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_score_centering.py
"""

import math

import pytest
import ray
import torch

from skyrl.backends.skyrl_train.utils.off_policy_correction_utils import (
    DPPOTrustRegion,
    DPPOType,
    RatioBand,
)
from skyrl.backends.skyrl_train.utils.packed_ragged_tensor import PackedRaggedTensor
from skyrl.backends.skyrl_train.utils.packed_tensor import cu_seqlens_from_lengths
from skyrl.backends.skyrl_train.utils.sample_support_replay import (
    ScoreCenteringMetric,
    sample_support_scores,
)
from skyrl.backends.skyrl_train.workers.megatron import (
    megatron_worker as megatron_worker_module,
)
from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
    MegatronPolicyWorkerBase,
)
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.utils import validate_cfg
from tests.backends.skyrl_train.gpu.score_centering_utils import (
    MODEL_NAME,
    SUPPORT_WIDTH,
    training_batch,
)
from tests.backends.skyrl_train.gpu.utils import (
    init_worker_with_type,
    ray_init_for_tests,
)


def _score_centering_cfg(*, tensor_parallel_size: int, context_parallel_size: int, packed: bool) -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.policy.model.path = MODEL_NAME
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.colocate_policy_ref = False
    cfg.trainer.placement.policy_num_gpus_per_node = tensor_parallel_size * context_parallel_size
    cfg.trainer.train_batch_size = cfg.trainer.policy_mini_batch_size = 2
    cfg.trainer.micro_train_batch_size_per_gpu = 2
    cfg.trainer.remove_microbatch_padding = packed
    cfg.trainer.logger = "console"
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = tensor_parallel_size
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.context_parallel_size = context_parallel_size
    cfg.generator.n_samples_per_prompt = 1
    cfg.generator.inference_engine.enable_return_sample_support_set = True
    cfg.generator.sampling_params.top_k = SUPPORT_WIDTH
    cfg.trainer.algorithm.policy_loss_type = "rollout_is"
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.use_entropy_loss = False
    cfg.trainer.algorithm.enable_sample_support_replay = True
    cfg.trainer.algorithm.enable_score_centering = True
    validate_cfg(cfg)
    return cfg


def _train_step_metrics(cfg: SkyRLTrainConfig) -> dict:
    actor_group = init_worker_with_type(
        "policy",
        shared_pg=None,
        colocate_all=False,
        num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
        cfg=cfg,
    )
    results = ray.get(actor_group.async_run_ray_method("mesh", "forward_backward", training_batch()))
    grad_norm = ray.get(actor_group.async_run_ray_method("pass_through", "optim_step"))[0]
    metrics = dict(results[0].metrics)
    metrics["grad_norm"] = grad_norm
    return metrics


class _ScoreCenteringTPProbeWorker(MegatronPolicyWorkerBase):
    """Exposes a TP=2 score-centering probe without production instrumentation."""

    def probe_tp2_centering_matches_unsharded(self) -> dict:
        """Compare TP=2 centering outputs and logit gradients with the unsharded calculation."""
        import megatron.core.parallel_state as mpu

        tp_group = mpu.get_tensor_model_parallel_group()
        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()
        if tp_size != 2:
            raise ValueError(f"expected TP=2, got TP={tp_size}")

        local_vocab_size = 8
        total_vocab_size = local_vocab_size * tp_size
        device = torch.device("cuda", torch.cuda.current_device())
        generator = torch.Generator(device="cpu").manual_seed(0)
        full_logits = torch.randn(2, 3, total_vocab_size, generator=generator).to(device)
        position_weights = torch.randn(2, 3, generator=generator).to(device)
        sampled_ids = torch.tensor([[1, 10, 2], [11, 3, 12]], dtype=torch.long, device=device)
        # Every row spans both shards: the sampled ID, its opposite-shard twin, and a third member.
        support_ids = torch.stack(
            (
                sampled_ids,
                (sampled_ids + total_vocab_size // 2) % total_vocab_size,
                (sampled_ids + 3) % total_vocab_size,
            ),
            dim=-1,
        )
        member_logits = full_logits.gather(-1, support_ids)
        # A sampler near the trainer, so some members fall inside the rollout_is band and some outside.
        sampler_logprobs = (
            member_logits + 0.3 * torch.randn(member_logits.shape, generator=generator).to(device)
        ).log_softmax(-1)
        support = PackedRaggedTensor(
            [support_ids.reshape(-1).to(torch.int32), sampler_logprobs.reshape(-1).float()],
            cu_seqlens_from_lengths([support_ids.shape[-1]] * sampled_ids.numel(), device=device),
            cu_seqlens_from_lengths([sampled_ids.numel()], device=device),
        )
        row_ids = torch.arange(sampled_ids.numel(), device=device).reshape(sampled_ids.shape)
        band = RatioBand.from_clip(0.2, 0.2)

        def score(logits, *, vocab_start_index, vocab_end_index, group):
            scores = sample_support_scores(
                logits,
                sampled_ids,
                support,
                row_ids,
                vocab_start_index=vocab_start_index,
                vocab_end_index=vocab_end_index,
                tp_group=group,
                centering_weight=band,
            )
            (position_weights * (scores.logprobs + scores.centering_term)).sum().backward()
            return scores

        vocab_start = tp_rank * local_vocab_size
        local_logits = full_logits[..., vocab_start : vocab_start + local_vocab_size].clone().requires_grad_()
        sharded = score(
            local_logits,
            vocab_start_index=vocab_start,
            vocab_end_index=vocab_start + local_vocab_size,
            group=tp_group,
        )
        unsharded_logits = full_logits.clone().requires_grad_()
        unsharded = score(unsharded_logits, vocab_start_index=0, vocab_end_index=total_vocab_size, group=None)

        # A sign-dependent weight reduces each sign's term across the shards too.
        trust_region = DPPOTrustRegion(DPPOType.BINARY_TV, delta_low=0.05, delta_high=0.05)
        sign_split = [
            sample_support_scores(
                logits,
                sampled_ids,
                support,
                row_ids,
                vocab_start_index=start,
                vocab_end_index=end,
                tp_group=group,
                centering_weight=trust_region,
            )
            for logits, start, end, group in (
                (local_logits.detach(), vocab_start, vocab_start + local_vocab_size, tp_group),
                (full_logits, 0, total_vocab_size, None),
            )
        ]

        pairs = {
            "logprobs": (sharded.logprobs, unsharded.logprobs),
            "centering_term": (sharded.centering_term, unsharded.centering_term),
            "sampler_logprobs": (sharded.sampler_logprobs, unsharded.sampler_logprobs),
            "support_tv": (sharded.support_tv, unsharded.support_tv),
            "dppo_positive_term": (sign_split[0].centering_term, sign_split[1].centering_term),
            "dppo_negative_term": (sign_split[0].centering_term_negative, sign_split[1].centering_term_negative),
            "logit_grad": (
                local_logits.grad,
                unsharded_logits.grad[..., vocab_start : vocab_start + local_vocab_size],
            ),
        }
        return {
            "rank": tp_rank,
            "max_abs_diff": {name: (a.detach() - b.detach()).abs().max().item() for name, (a, b) in pairs.items()},
            "centering_term_abs_max": unsharded.centering_term.abs().max().item(),
        }


_ScoreCenteringTPProbeWorkerRemote = ray.remote(num_gpus=1)(_ScoreCenteringTPProbeWorker)


@pytest.mark.megatron
def test_score_centering_tp2_matches_unsharded(ray_init_fixture):
    """TP=2 vocabulary reductions give the unsharded centering term, sampler logprobs and gradients."""
    cfg = _score_centering_cfg(tensor_parallel_size=2, context_parallel_size=1, packed=False)
    original_worker = megatron_worker_module.PolicyWorker
    megatron_worker_module.PolicyWorker = _ScoreCenteringTPProbeWorkerRemote
    try:
        actor_group = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=2,
            cfg=cfg,
        )
        results = ray.get(actor_group.async_run_ray_method("pass_through", "probe_tp2_centering_matches_unsharded"))
    finally:
        megatron_worker_module.PolicyWorker = original_worker

    assert {result["rank"] for result in results} == {0, 1}
    assert all(result["centering_term_abs_max"] > 1e-3 for result in results)
    for result in results:
        for name, diff in result["max_abs_diff"].items():
            assert diff <= 1e-5, (result["rank"], name, diff)


LAYOUTS = [
    pytest.param(dict(tensor_parallel_size=2, context_parallel_size=1, packed=False), id="tp2"),
    pytest.param(dict(tensor_parallel_size=2, context_parallel_size=2, packed=True), id="tp2-cp2-packed"),
]


@pytest.mark.megatron
@pytest.mark.parametrize("layout", LAYOUTS)
def test_score_centering_trains_identically_across_parallel_layouts(ray_init_fixture, layout):
    reference = _train_step_metrics(_score_centering_cfg(tensor_parallel_size=1, context_parallel_size=1, packed=False))
    # Release the reference actors' GPUs before placing the sharded layout.
    ray.shutdown()
    ray_init_for_tests()
    sharded = _train_step_metrics(_score_centering_cfg(**layout))

    for metrics in (reference, sharded):
        assert math.isfinite(metrics["policy_loss"]) and math.isfinite(metrics["grad_norm"])
        assert metrics[ScoreCenteringMetric.TERM_ABS_MEAN.value] > 0
        assert metrics[ScoreCenteringMetric.SUPPORT_TV_MEAN.value] > 0
        assert metrics[ScoreCenteringMetric.SAMPLER_ROW_MASS_MIN.value] == pytest.approx(1.0, abs=1e-5)
    # Exact sharding is covered by the probe above. Across layouts the bf16 logits differ, and rows
    # near the band edge can flip in or out, so this only rules out gross errors like a TP-size factor.
    for key in ("policy_loss", "grad_norm", *(metric.value for metric in ScoreCenteringMetric)):
        assert sharded[key] == pytest.approx(reference[key], rel=0.15, abs=1e-4), key

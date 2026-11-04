"""GPU coverage for the support logit term (TT-OPD's top-K term) on the Megatron backend.

Run with::

    uv run --isolated --extra dev --extra megatron pytest -s -vvv \
        tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_support_logit_term.py
"""

import math

import pytest
import ray
import torch

from skyrl.backends.skyrl_train.utils.packed_ragged_tensor import PackedRaggedTensor
from skyrl.backends.skyrl_train.utils.packed_tensor import cu_seqlens_from_lengths
from skyrl.backends.skyrl_train.utils.sample_support_replay import support_logit_term
from skyrl.backends.skyrl_train.workers.megatron import (
    megatron_worker as megatron_worker_module,
)
from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
    MegatronPolicyWorkerBase,
)
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.utils import validate_cfg
from tests.backends.skyrl_train.gpu.score_centering_utils import MODEL_NAME
from tests.backends.skyrl_train.gpu.support_logit_term_utils import (
    expected_support_loss,
    weighted_training_batch,
)
from tests.backends.skyrl_train.gpu.utils import (
    init_worker_with_type,
    ray_init_for_tests,
)

SUPPORT_LOSS_KEY = "loss_metrics/topk_policy_loss"


def _support_term_cfg(
    *, tensor_parallel_size: int, context_parallel_size: int, packed: bool, fused_lm_head: bool = False
) -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.policy.model.path = MODEL_NAME
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.colocate_policy_ref = False
    cfg.trainer.placement.policy_num_gpus_per_node = tensor_parallel_size * context_parallel_size
    cfg.trainer.train_batch_size = cfg.trainer.policy_mini_batch_size = 2
    cfg.trainer.micro_train_batch_size_per_gpu = 2
    cfg.trainer.remove_microbatch_padding = packed
    cfg.trainer.fused_lm_head_logprob = fused_lm_head
    cfg.trainer.logger = "console"
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = tensor_parallel_size
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.context_parallel_size = context_parallel_size
    cfg.generator.n_samples_per_prompt = 1
    cfg.trainer.algorithm.policy_loss_type = "importance_sampling"
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.use_entropy_loss = False
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
    results = ray.get(actor_group.async_run_ray_method("mesh", "forward_backward", weighted_training_batch()))
    grad_norm = ray.get(actor_group.async_run_ray_method("pass_through", "optim_step"))[0]
    metrics = dict(results[0].metrics)
    metrics["grad_norm"] = grad_norm
    return metrics


class _SupportTermTPProbeWorker(MegatronPolicyWorkerBase):
    """Exposes a TP=2 support-logit-term probe without production instrumentation."""

    def probe_tp2_term_matches_unsharded(self) -> dict:
        """Compare the TP=2 term and its logit gradient with the unsharded calculation."""
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
        sampled_ids = torch.tensor([[1, 10, 2], [11, 3, 12]], dtype=torch.long, device=device)
        # Rows [sampled, its opposite-shard twin, the sampled token again]: every row spans both shards and holds
        # a duplicate, as a TT-OPD row does when the sampled token is also a candidate.
        support_ids = torch.stack(
            (sampled_ids, (sampled_ids + total_vocab_size // 2) % total_vocab_size, sampled_ids), dim=-1
        )
        candidate_weights = torch.randn(2, 3, 2, generator=generator).to(device)
        weights = torch.cat([-candidate_weights.sum(-1, keepdim=True), candidate_weights], dim=-1)
        support = PackedRaggedTensor(
            [
                support_ids.reshape(-1).to(torch.int32),
                torch.zeros(support_ids.numel(), device=device),
                weights.reshape(-1).float(),
            ],
            cu_seqlens_from_lengths([support_ids.shape[-1]] * sampled_ids.numel(), device=device),
            cu_seqlens_from_lengths([sampled_ids.numel()], device=device),
        )
        row_ids = torch.arange(sampled_ids.numel(), device=device).reshape(sampled_ids.shape)

        def term(logits, *, vocab_start_index, vocab_end_index, group):
            values = support_logit_term(
                logits,
                sampled_ids,
                support,
                row_ids,
                vocab_start_index=vocab_start_index,
                vocab_end_index=vocab_end_index,
                tp_group=group,
            )
            values.sum().backward()
            return values

        vocab_start = tp_rank * local_vocab_size
        local_logits = full_logits[..., vocab_start : vocab_start + local_vocab_size].clone().requires_grad_()
        sharded = term(
            local_logits,
            vocab_start_index=vocab_start,
            vocab_end_index=vocab_start + local_vocab_size,
            group=tp_group,
        )
        unsharded_logits = full_logits.clone().requires_grad_()
        unsharded = term(unsharded_logits, vocab_start_index=0, vocab_end_index=total_vocab_size, group=None)
        return {
            "rank": tp_rank,
            "term_diff": (sharded.detach() - unsharded.detach()).abs().max().item(),
            "grad_diff": (local_logits.grad - unsharded_logits.grad[..., vocab_start : vocab_start + local_vocab_size])
            .abs()
            .max()
            .item(),
            "term_abs_max": unsharded.abs().max().item(),
        }


_SupportTermTPProbeWorkerRemote = ray.remote(num_gpus=1)(_SupportTermTPProbeWorker)


@pytest.mark.megatron
def test_support_logit_term_tp2_matches_unsharded(ray_init_fixture):
    cfg = _support_term_cfg(tensor_parallel_size=2, context_parallel_size=1, packed=False)
    original_worker = megatron_worker_module.PolicyWorker
    megatron_worker_module.PolicyWorker = _SupportTermTPProbeWorkerRemote
    try:
        actor_group = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=2,
            cfg=cfg,
        )
        results = ray.get(actor_group.async_run_ray_method("pass_through", "probe_tp2_term_matches_unsharded"))
    finally:
        megatron_worker_module.PolicyWorker = original_worker

    assert {result["rank"] for result in results} == {0, 1}
    for result in results:
        assert result["term_abs_max"] > 1e-3
        assert result["term_diff"] <= 1e-5 and result["grad_diff"] <= 1e-5, result


LAYOUTS = [
    pytest.param(dict(tensor_parallel_size=1, context_parallel_size=1, packed=True), id="packed"),
    pytest.param(dict(tensor_parallel_size=2, context_parallel_size=1, packed=False), id="tp2"),
    pytest.param(
        dict(tensor_parallel_size=2, context_parallel_size=1, packed=True, fused_lm_head=True), id="tp2-fused"
    ),
    pytest.param(dict(tensor_parallel_size=2, context_parallel_size=2, packed=True), id="tp2-cp2-packed"),
]


@pytest.mark.megatron
@pytest.mark.parametrize("layout", LAYOUTS)
def test_support_logit_term_trains_identically_across_parallel_layouts(ray_init_fixture, layout):
    reference = _train_step_metrics(_support_term_cfg(tensor_parallel_size=1, context_parallel_size=1, packed=False))
    # Release the reference actors' GPUs before placing the sharded layout.
    ray.shutdown()
    ray_init_for_tests()
    sharded = _train_step_metrics(_support_term_cfg(**layout))

    for metrics in (reference, sharded):
        assert math.isfinite(metrics["policy_loss"]) and math.isfinite(metrics["grad_norm"])
    # The term reads each row at the logit that predicts its token; a shifted position reads other logits.
    assert reference[SUPPORT_LOSS_KEY] == pytest.approx(expected_support_loss(), rel=0.1, abs=1e-3)
    # The layouts' bf16 logits differ, so this rules out gross errors such as a TP-size factor or a lost shard.
    for key in ("policy_loss", "grad_norm", SUPPORT_LOSS_KEY):
        assert sharded[key] == pytest.approx(reference[key], rel=0.15, abs=1e-4), key

"""GPU coverage for the support logit term (TT-OPD's top-K term) on the FSDP backend.

Run with::

    uv run --isolated --extra dev --extra fsdp pytest -s -vvv \
        tests/backends/skyrl_train/gpu/gpu_ci/test_support_logit_term_fsdp.py
"""

import math

import pytest
import ray

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


def _support_term_cfg(*, num_gpus: int, packed: bool, ulysses: int) -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = "fsdp"
    cfg.trainer.policy.model.path = MODEL_NAME
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.colocate_policy_ref = False
    cfg.trainer.placement.policy_num_gpus_per_node = num_gpus
    cfg.trainer.policy.sequence_parallel_size = ulysses
    cfg.trainer.train_batch_size = cfg.trainer.policy_mini_batch_size = 2
    cfg.trainer.micro_train_batch_size_per_gpu = 2
    cfg.trainer.remove_microbatch_padding = packed
    cfg.trainer.logger = "console"
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


LAYOUTS = [
    pytest.param(dict(num_gpus=1, packed=True, ulysses=1), id="packed"),
    pytest.param(dict(num_gpus=2, packed=True, ulysses=2), id="packed-ulysses2"),
]


@pytest.mark.parametrize("layout", LAYOUTS)
def test_support_logit_term_trains_identically_across_fsdp_layouts(ray_init_fixture, layout):
    """Also the GPU cover for the term sharing the logits with flash-attn's in-place cross-entropy backward."""
    reference = _train_step_metrics(_support_term_cfg(num_gpus=1, packed=False, ulysses=1))
    # Release the reference actors' GPUs before placing the other layout.
    ray.shutdown()
    ray_init_for_tests()
    other = _train_step_metrics(_support_term_cfg(**layout))

    for metrics in (reference, other):
        assert math.isfinite(metrics["policy_loss"]) and math.isfinite(metrics["grad_norm"])
    # The term reads each row at the logit that predicts its token; a shifted position reads other logits.
    assert reference[SUPPORT_LOSS_KEY] == pytest.approx(expected_support_loss(), rel=0.1, abs=1e-3)
    # The layouts' bf16 logits differ, so this only rules out gross errors such as a position shift.
    for key in ("policy_loss", "grad_norm", SUPPORT_LOSS_KEY):
        assert other[key] == pytest.approx(reference[key], rel=0.15, abs=1e-4), key

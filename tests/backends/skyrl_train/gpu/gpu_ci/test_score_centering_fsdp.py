"""GPU coverage for score centering on the FSDP backend.

Run with::

    uv run --isolated --extra dev --extra fsdp pytest -s -vvv \
        tests/backends/skyrl_train/gpu/gpu_ci/test_score_centering_fsdp.py
"""

import math

import pytest
import ray

from skyrl.backends.skyrl_train.utils.sample_support_replay import ScoreCenteringMetric
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


def _score_centering_cfg(*, policy_loss_type: str, num_gpus: int, packed: bool, ulysses: int) -> SkyRLTrainConfig:
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
    cfg.generator.inference_engine.enable_return_sample_support_set = True
    cfg.generator.sampling_params.top_k = SUPPORT_WIDTH
    cfg.trainer.algorithm.policy_loss_type = policy_loss_type
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


LAYOUTS = [
    pytest.param(dict(num_gpus=1, packed=True, ulysses=1), id="packed"),
    pytest.param(dict(num_gpus=2, packed=True, ulysses=2), id="packed-ulysses2"),
]


@pytest.mark.parametrize("policy_loss_type", ["reinforce", "rollout_is", "dppo"])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_score_centering_trains_identically_across_fsdp_layouts(ray_init_fixture, layout, policy_loss_type):
    reference = _train_step_metrics(
        _score_centering_cfg(policy_loss_type=policy_loss_type, num_gpus=1, packed=False, ulysses=1)
    )
    # Release the reference actors' GPUs before placing the other layout.
    ray.shutdown()
    ray_init_for_tests()
    other = _train_step_metrics(_score_centering_cfg(policy_loss_type=policy_loss_type, **layout))

    for metrics in (reference, other):
        assert math.isfinite(metrics["policy_loss"]) and math.isfinite(metrics["grad_norm"])
        assert metrics[ScoreCenteringMetric.TERM_ABS_MEAN.value] > 0
        assert metrics[ScoreCenteringMetric.SUPPORT_TV_MEAN.value] > 0
        assert metrics[ScoreCenteringMetric.SAMPLER_ROW_MASS_MIN.value] == pytest.approx(1.0, abs=1e-5)
    # The layouts' bf16 logits differ, and rows near a weight's edge can flip, so this only rules out
    # gross errors such as a position shift or an unnormalized term.
    for key in ("policy_loss", "grad_norm", *(metric.value for metric in ScoreCenteringMetric)):
        assert other[key] == pytest.approx(reference[key], rel=0.15, abs=1e-4), key

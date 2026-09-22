"""Configuration for on-policy distillation (``skyrl.train.entrypoints.main_opd``).

Adds two blocks to the standard config and changes two algorithm defaults:

- ``trainer.teacher``: the frozen teacher, a sibling of ``trainer.ref`` (which stays independent:
  a reference-KL penalty can still be turned on alongside the teacher).
- ``trainer.algorithm.opd``: the distillation knobs.
- ``trainer.algorithm.use_kl_loss`` defaults to ``False`` (the core default would instantiate a
  reference model for nothing) and ``policy_loss_type`` to ``"importance_sampling"`` (the
  Thinking Machines / tinker recipe). Every other core default is already right for OPD.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from skyrl.backends.skyrl_train.utils.ppo_utils import LOSSES_WITHOUT_OLD_LOGPROBS
from skyrl.train.config import AlgorithmConfig, TrainerConfig, make_config
from skyrl.train.config.config import BaseConfig

TEACHER_BACKENDS = ("fireworks",)


@dataclass
class OPDConfig(BaseConfig):
    kl_coef: float = 1.0
    """Coefficient of the per-token teacher term: ``advantages -= kl_coef * (log pi_student - log pi_teacher)``.
    ``1.0`` is the Thinking Machines / tinker recipe."""
    use_task_reward: bool = False
    """``False``: pure on-policy distillation; the environment reward is zeroed after its metrics are
    logged, so the advantage estimator contributes nothing. ``True``: the reward flows through the
    configured estimator and the teacher term is added on top (Miles mixed mode / AReaL KDRL)."""
    self_test_samples: int = 8
    """Number of identical scoring requests sent to the teacher before training starts."""
    self_test_max_abs_diff: float = 0.05
    """Largest disagreement (nats) tolerated between those requests. A teacher whose logprobs depend on
    which replica answers is refused; the reverse-KL signal is itself only a few hundredths of a nat."""


@dataclass
class TeacherConfig(BaseConfig):
    """The frozen teacher, served by an inference engine."""

    backend: str = "fireworks"
    """``"fireworks"``: a Fireworks model or dedicated deployment id. vLLM-served teachers come next."""
    model: str = ""
    """``accounts/fireworks/models/<id>`` or a dedicated ``accounts/<account>/deployments/<id>``."""
    base_url: Optional[str] = None
    """Fireworks server root without ``/v1``; defaults to the Fireworks data plane."""
    api_key_var: str = "FIREWORKS_API_KEY"
    """Environment variable holding the Fireworks API key."""
    max_concurrency: int = 32
    """Maximum teacher requests in flight."""
    request_timeout_s: float = 120.0
    """Per-request timeout (Fireworks backend)."""
    max_retries: int = 3
    """Retries with exponential backoff on timeouts and retryable HTTP statuses (Fireworks backend)."""


@dataclass
class OPDAlgorithmConfig(AlgorithmConfig):
    opd: OPDConfig = field(default_factory=OPDConfig)
    """On-policy distillation knobs."""
    use_kl_loss: bool = False
    """Off by default: the reference model is not needed for distillation. Turn it on to add a
    reference-KL loss alongside the teacher term."""
    policy_loss_type: str = "importance_sampling"
    """The Thinking Machines / tinker recipe. ``"regular"`` (PPO clip) is also valid."""


@dataclass
class OPDTrainerConfig(TrainerConfig):
    algorithm: OPDAlgorithmConfig = field(default_factory=OPDAlgorithmConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    """The frozen teacher; a sibling of ``trainer.ref``."""


OPDExpConfig = make_config(trainer_cls=OPDTrainerConfig)


def validate_opd_cfg(cfg) -> None:
    """Rules the standard ``validate_cfg`` does not know about. Call it after ``validate_cfg``."""
    teacher: TeacherConfig = cfg.trainer.teacher
    algorithm = cfg.trainer.algorithm
    opd: OPDConfig = algorithm.opd

    if teacher.backend not in TEACHER_BACKENDS:
        raise ValueError(f"trainer.teacher.backend must be one of {TEACHER_BACKENDS}, got {teacher.backend!r}")
    if not teacher.model:
        raise ValueError("trainer.teacher.model must be set")
    if teacher.backend == "fireworks" and not os.environ.get(teacher.api_key_var):
        raise ValueError(f"trainer.teacher.api_key_var={teacher.api_key_var!r} is not set in the environment")
    if teacher.max_concurrency <= 0:
        raise ValueError("trainer.teacher.max_concurrency must be positive")

    if opd.kl_coef < 0:
        raise ValueError("trainer.algorithm.opd.kl_coef must be >= 0")
    if opd.self_test_samples < 2:
        raise ValueError("trainer.algorithm.opd.self_test_samples must be >= 2")

    if algorithm.zero_variance_filter:
        raise ValueError(
            "trainer.algorithm.zero_variance_filter must be False for OPD: pure distillation makes every "
            "prompt group zero-variance, and the filter would loss-mask the whole batch"
        )
    if algorithm.dynamic_sampling.type is not None:
        raise ValueError(
            "trainer.algorithm.dynamic_sampling is not supported with OPD: zero task rewards never satisfy "
            "a resampling criterion"
        )
    if algorithm.advantage_batch_normalize:
        raise ValueError(
            "trainer.algorithm.advantage_batch_normalize must be False for OPD: it would z-score the "
            "teacher term away"
        )
    if algorithm.policy_loss_type in {loss.value for loss in LOSSES_WITHOUT_OLD_LOGPROBS}:
        raise ValueError(
            f"policy_loss_type={algorithm.policy_loss_type!r} skips the old-logprob forward pass, which the "
            "OPD term reads (action_log_probs)"
        )
    if cfg.generator.step_wise_trajectories:
        raise ValueError("step_wise_trajectories are not yet supported with OPD")

    sampling = cfg.generator.sampling_params
    if sampling.temperature != 1.0 or sampling.top_p != 1.0 or sampling.top_k != -1:
        logger.warning(
            "OPD expects untruncated sampling (temperature=1.0, top_p=1.0, top_k=-1): the sampled-token "
            f"reverse-KL estimate is biased otherwise. Got temperature={sampling.temperature}, "
            f"top_p={sampling.top_p}, top_k={sampling.top_k}."
        )
    if algorithm.use_kl_in_reward:
        logger.warning(
            "use_kl_in_reward=True adds a reference-model KL to the rewards on top of the teacher term; "
            "this costs a reference forward pass per step"
        )

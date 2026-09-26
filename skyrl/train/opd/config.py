"""Configuration for on-policy distillation (``skyrl.train.entrypoints.main_opd``).

Adds two blocks to the standard config and changes two algorithm defaults:

- ``trainer.teacher``: the frozen teacher, a sibling of ``trainer.ref`` (which stays independent:
  a reference-KL penalty can still be turned on alongside the teacher). ``backend`` picks one of three
  exclusive homes: launched by this job (``skyrl``, the default), vLLM servers you run (``vllm``), or
  Fireworks (``fireworks``).
- ``trainer.algorithm.opd``: the distillation knobs.
- ``trainer.algorithm.use_kl_loss`` defaults to ``False`` (the core default would instantiate a
  reference model for nothing) and ``policy_loss_type`` to ``"importance_sampling"`` (the
  Thinking Machines / tinker recipe). Every other core default is already right for OPD.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

from loguru import logger

from skyrl.backends.skyrl_train.utils.ppo_utils import (
    LOSSES_WITHOUT_OLD_LOGPROBS,
    PolicyLossType,
)
from skyrl.train.config import (
    AlgorithmConfig,
    InferenceEngineConfig,
    TrainerConfig,
    make_config,
)
from skyrl.train.config.config import BaseConfig

TEACHER_BACKENDS = ("skyrl", "vllm", "fireworks")


@dataclass
class TeacherEngineConfig(InferenceEngineConfig):
    """The teacher's engine block: ``InferenceEngineConfig`` with the defaults a frozen prefill-only server wants.

    The changed defaults live on the class, not in a ``default_factory`` instance, because
    ``from_cli_overrides`` rebuilds a nested block from its field type whenever any key inside it is
    overridden; a factory's values would silently revert to the student's the moment a user sets
    ``trainer.teacher.inference_engine.num_engines``.
    """

    enable_prefix_caching: bool = False
    """Off: vLLM never reads the prefix cache for ``prompt_logprobs`` requests, so caching would only cost block hashing on write."""
    gpu_memory_utilization: float = 0.9
    """Higher than the student's 0.8: nothing else lives on the teacher's GPUs."""
    enable_ray_prometheus_stats: bool = False
    """Off: the trainer's metrics scraper would fold the teacher's servers into the ``vllm/train/*`` metrics."""


@dataclass
class OPDConfig(BaseConfig):
    kl_coef: float = 1.0
    """Coefficient of the per-token teacher term: ``advantages -= kl_coef * (log pi_student - log pi_teacher)``.
    ``1.0`` is the Thinking Machines / tinker recipe."""
    use_task_reward: bool = False
    """Pure distillation when ``False``; task reward plus the teacher term when ``True``.
    ``False`` zeroes the environment reward after its metrics are logged, so the advantage estimator
    contributes nothing. ``True`` sends the reward through the configured estimator and adds the
    teacher term on top."""


@dataclass
class TeacherConfig(BaseConfig):
    """The frozen teacher, served by an inference engine."""

    backend: str = "skyrl"
    """Which service scores the teacher: ``"skyrl"``, ``"vllm"`` or ``"fireworks"``.
    ``"skyrl"`` (default): this job launches a vLLM deployment from ``inference_engine`` on its own GPUs and
    drives it through a ``RemoteInferenceClient``, as it does the student's engines. ``"vllm"``:
    OpenAI-compatible vLLM servers you run (a stock ``vllm serve`` or SkyRL's ``serve`` entrypoint), named
    by ``server_urls``. ``"fireworks"``: the Fireworks completions API at ``base_url``. Each backend reads
    only its own fields; setting another backend's fields is a config error."""
    model: str = ""
    """The teacher model; what it names depends on the backend.
    ``skyrl``: an HF id or local path the job loads, e.g. ``Qwen/Qwen3-32B``. ``vllm``: the served model
    name of your servers. ``fireworks``: ``accounts/fireworks/models/<id>`` or a dedicated
    ``accounts/<account>/deployments/<id>``."""
    inference_engine: TeacherEngineConfig = field(default_factory=TeacherEngineConfig)
    """The launched deployment for ``backend="skyrl"``, the same block as ``generator.inference_engine``.
    Weight-sync, sleep, LoRA, PD, speculative-decoding, routed-expert and external-URL fields do not apply
    to a frozen model and are rejected when set. Defaults that differ from the student's: prefix caching
    off, ``gpu_memory_utilization`` 0.9, Ray Prometheus stats off. ``engine_init_kwargs.max_model_len``
    defaults to the longest input plus the longest response plus one."""
    server_urls: Optional[List[str]] = None
    """Base URLs of the vLLM servers for ``backend="vllm"``, e.g. ``["http://host:8000"]``.
    Requests round-robin across them, and a retry moves to the next one."""
    base_url: Optional[str] = None
    """Fireworks server root without ``/v1``; defaults to the Fireworks data plane."""
    api_key_var: str = "FIREWORKS_API_KEY"
    """Environment variable holding the Fireworks API key (Fireworks only)."""
    max_concurrency: int = 32
    """Maximum teacher requests in flight, for every backend.
    A launched teacher is also capped by its ``RemoteInferenceClient``'s per-engine limit
    (``SKYRL_GENERATE_CONCURRENCY_PER_ENGINE`` per server), like the student's rollouts."""
    request_timeout_s: float = 120.0
    """Per-request timeout of the ``vllm`` and ``fireworks`` clients.
    The ``skyrl`` backend uses its ``RemoteInferenceClient``'s policy, the one the student's rollouts get."""
    max_retries: int = 3
    """Retries with exponential backoff for the ``vllm`` and ``fireworks`` clients.
    The ``skyrl`` backend uses its ``RemoteInferenceClient``'s policy, the one the student's rollouts get."""


@dataclass
class OPDAlgorithmConfig(AlgorithmConfig):
    opd: OPDConfig = field(default_factory=OPDConfig)
    """On-policy distillation knobs."""
    use_kl_loss: bool = False
    """Off by default; the reference model is not needed for distillation.
    Turn it on to add a reference-KL loss alongside the teacher term."""
    policy_loss_type: str = "importance_sampling"
    """The Thinking Machines / tinker recipe. ``"regular"`` (PPO clip) is also valid."""


@dataclass
class OPDTrainerConfig(TrainerConfig):
    algorithm: OPDAlgorithmConfig = field(default_factory=OPDAlgorithmConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    """The frozen teacher; a sibling of ``trainer.ref``."""


OPDExpConfig = make_config(trainer_cls=OPDTrainerConfig)


def teacher_max_model_len(cfg) -> int:
    """The context a teacher server needs: the longest input, the longest response, and vLLM's one generated token.

    ``generator.max_input_length`` bounds what the engine sees per turn (multi-turn), else
    ``trainer.max_prompt_length`` does; the scoring request is that plus the response, and vLLM
    refuses ``max_tokens=0``, so one token is generated and discarded.
    """
    max_input = cfg.generator.max_input_length or cfg.trainer.max_prompt_length
    return max_input + cfg.generator.sampling_params.max_generate_length + 1


# Fields of the teacher's engine block that would change a launched server's behaviour if honoured.
# Refusing beats silently ignoring; weight_sync_backend and delta_weight_sync are never read on the
# frozen path and need no rule.
_FROZEN_ENGINE_FIELDS_MUST_BE_OFF = (
    "enable_pd",
    "enable_return_routed_experts",
    "enable_return_sample_support_set",
    "offload_kv_for_weight_sync",
)
_FROZEN_ENGINE_FIELDS_MUST_BE_NONE = (
    "speculative_config",
    "fp8_weight_sync_mode",
    "external_server_urls",
    "external_proxy_url",
)


def _validate_launched_teacher(cfg, teacher: TeacherConfig) -> None:
    """``backend="skyrl"``: the job launches the teacher, so nothing may point elsewhere."""
    ie_cfg: TeacherEngineConfig = teacher.inference_engine
    if teacher.server_urls is not None:
        raise ValueError(
            "trainer.teacher.server_urls is for backend='vllm' (servers you run); backend='skyrl' launches the "
            "teacher in this job from trainer.teacher.inference_engine"
        )
    if teacher.base_url is not None:
        raise ValueError("trainer.teacher.base_url is for backend='fireworks'")
    if ie_cfg.backend != "vllm":
        raise ValueError(f"trainer.teacher.inference_engine.backend must be 'vllm', got {ie_cfg.backend!r}")
    if not ie_cfg.run_engines_locally:
        raise ValueError(
            "trainer.teacher.inference_engine.run_engines_locally must be true for backend='skyrl'; for servers "
            "you run use backend='vllm' with trainer.teacher.server_urls"
        )
    for name in _FROZEN_ENGINE_FIELDS_MUST_BE_OFF:
        if getattr(ie_cfg, name):
            raise ValueError(f"trainer.teacher.inference_engine.{name} does not apply to a frozen teacher")
    for name in _FROZEN_ENGINE_FIELDS_MUST_BE_NONE:
        if getattr(ie_cfg, name) is not None:
            raise ValueError(f"trainer.teacher.inference_engine.{name} does not apply to a frozen teacher")
    num_gpus = (
        ie_cfg.num_engines * ie_cfg.tensor_parallel_size * ie_cfg.pipeline_parallel_size * ie_cfg.data_parallel_size
    )
    if num_gpus < 1:
        raise ValueError("trainer.teacher.inference_engine must describe at least one GPU")
    max_model_len = ie_cfg.engine_init_kwargs.get("max_model_len")
    needed = teacher_max_model_len(cfg)
    if max_model_len is not None and max_model_len < needed:
        raise ValueError(
            f"trainer.teacher.inference_engine.engine_init_kwargs.max_model_len={max_model_len} is shorter than the "
            f"longest input + longest response + 1 = {needed}; the teacher would reject the scoring request"
        )


def _reject_launch_settings(teacher: TeacherConfig) -> None:
    """``backend`` names servers elsewhere, so a launch block is a mistake, not a no-op."""
    if teacher.inference_engine != TeacherEngineConfig():
        raise ValueError(
            "trainer.teacher.inference_engine is for backend='skyrl' (a teacher this job launches); "
            f"backend={teacher.backend!r} names servers elsewhere, so leave the block at its defaults"
        )


def validate_opd_cfg(cfg) -> None:
    """Rules the standard ``validate_cfg`` does not know about. Call it after ``validate_cfg``."""
    teacher: TeacherConfig = cfg.trainer.teacher
    algorithm = cfg.trainer.algorithm
    opd: OPDConfig = algorithm.opd

    if teacher.backend not in TEACHER_BACKENDS:
        raise ValueError(f"trainer.teacher.backend must be one of {TEACHER_BACKENDS}, got {teacher.backend!r}")
    if not teacher.model:
        raise ValueError("trainer.teacher.model must be set")
    if teacher.backend == "skyrl":
        _validate_launched_teacher(cfg, teacher)
    elif teacher.backend == "vllm":
        if not teacher.server_urls:
            raise ValueError("trainer.teacher.backend='vllm' requires trainer.teacher.server_urls")
        if teacher.base_url is not None:
            raise ValueError("trainer.teacher.base_url is for backend='fireworks'")
        _reject_launch_settings(teacher)
    elif teacher.backend == "fireworks":
        if not os.environ.get(teacher.api_key_var):
            raise ValueError(f"trainer.teacher.api_key_var={teacher.api_key_var!r} is not set in the environment")
        if teacher.server_urls is not None:
            raise ValueError("trainer.teacher.server_urls is for backend='vllm'")
        _reject_launch_settings(teacher)
    if teacher.max_concurrency <= 0:
        raise ValueError("trainer.teacher.max_concurrency must be positive")

    if opd.kl_coef < 0:
        raise ValueError("trainer.algorithm.opd.kl_coef must be >= 0")

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
    if algorithm.policy_loss_type == PolicyLossType.CISPO and algorithm.cispo.cispo_anchor == "rollout":
        # Same reason: the trainer skips the forward pass for the rollout anchor (_skip_policy_forward),
        # and the anchor is not visible through LOSSES_WITHOUT_OLD_LOGPROBS.
        raise ValueError(
            "policy_loss_type='cispo' with cispo.cispo_anchor='rollout' skips the old-logprob forward pass, "
            "which the OPD term reads (action_log_probs); use cispo_anchor='old'"
        )
    if algorithm.advantage_estimator == "gae" and not opd.use_task_reward:
        raise ValueError(
            "advantage_estimator='gae' needs a task reward (opd.use_task_reward=true): with zero rewards GAE's "
            "advantages are the critic's whitened value residuals, not zeros, and they would replace the teacher "
            "signal. Pure distillation needs a reward-only estimator such as grpo."
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

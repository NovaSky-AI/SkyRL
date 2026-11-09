"""Configuration for on-policy distillation (``skyrl.train.entrypoints.main_opd``).

Adds two blocks to the standard config and changes two algorithm defaults:

- ``trainer.teacher`` or ``trainer.teachers``, the frozen teacher(s), siblings of ``trainer.ref`` (which stays
  independent: a reference-KL penalty can still be turned on alongside them). ``trainer.teacher`` is one teacher
  that scores every row; ``trainer.teachers`` names several and routes each prompt group to the one its
  ``opd.route_key`` label names (multi-teacher OPD, MOPD). Set one of the two. Once built, the config holds the
  teachers in ``trainer.teachers`` only: a single ``trainer.teacher`` becomes ``{"default": teacher}`` and
  ``trainer.teacher`` is reset to ``None``, so the rest of the code has one path. Each teacher's ``backend`` picks one
  of two exclusive homes: launched by this job (``skyrl``, the default) or vLLM servers you run (``vllm``).
- ``trainer.algorithm.opd``: the distillation knobs, ``opd.topk_metrics`` (the top-K overlap metrics) and
  ``opd.route_key`` (which row label picks the teacher under ``trainer.teachers``).
- ``trainer.algorithm.use_kl_loss`` defaults to ``False`` and
  ``policy_loss_type`` to ``"importance_sampling"``.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

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
from skyrl.train.config.config import BaseConfig, build_nested_dataclass
from skyrl.train.opd.teacher_client import normalize_server_url

TEACHER_BACKENDS = ("skyrl", "vllm")
DEFAULT_TEACHER = "default"
"""``trainer.teacher``'s name once normalized into ``trainer.teachers``: it scores every row, with no route label.
Reserved: ``trainer.teachers`` may use it only alone, as the normalized form does."""
# Teacher names appear in metric keys.
_TEACHER_NAME = re.compile(r"[A-Za-z0-9_-]+")


@dataclass
class TeacherEngineConfig(InferenceEngineConfig):
    """The teacher's engine block: ``InferenceEngineConfig`` with the defaults a frozen prefill-only server wants.

    The changed defaults live on the class, not in a ``default_factory`` instance, because
    ``from_cli_overrides`` rebuilds a nested block from its field type whenever any key inside it is
    overridden; a factory's values would silently revert to the student's the moment a user sets
    ``trainer.teacher.inference_engine.num_engines``.
    """

    enable_prefix_caching: bool = False
    """Off: vLLM never reads the prefix cache for prompt-scoring requests, so caching would only cost block hashing on write."""
    gpu_memory_utilization: float = 0.9
    """Higher than the student's 0.8: nothing else lives on the teacher's GPUs."""
    enable_ray_prometheus_stats: bool = False
    """Off: the trainer's metrics scraper would fold the teacher's servers into the ``vllm/train/*`` metrics."""


@dataclass
class OPDTopKMetricsConfig(BaseConfig):
    """The overlap ratio and overlap-token advantage metrics, which compare the
    student's and the teacher's top ``top_k`` tokens at each response position.

    The student's top-K comes from its sampler, so turning them on requires untruncated sampling at temperature 1.
    Nothing here changes the loss.
    """

    num_rows: int = 0
    """Rows per step and teacher, evenly spaced over each teacher's rows, on which the overlap metrics are measured;
    ``0`` is off. Each asks its teacher for its own top ``top_k``, which vLLM returns as JSON over the whole
    sequence."""
    top_k: int = 16
    """K, the number of top tokens of each distribution compared; at least 8. 16 is the paper's."""


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
    divergence_clip: Optional[float] = None
    """Ceiling on each token's pointwise reverse KL, ``p log(p / q)``, in nats; ``None`` is off.
    ``p`` is the student's and ``q`` the teacher's probability of the sampled token. OPSD's ``min(l, tau)``: a token
    above the ceiling gets no teacher signal, so the clip is one-sided."""
    advantage_clip_min: Optional[float] = None
    """Lower bound on each token's ``log q - log p``, in nats, before ``kl_coef``; ``None`` is unbounded.
    At most 0, so it limits a token's signal without forcing its sign. An alternative to ``advantage_log_compress``."""
    advantage_clip_max: Optional[float] = None
    """Upper bound on each token's ``log q - log p``, in nats, before ``kl_coef``; ``None`` is unbounded.
    At least 0. An alternative to ``advantage_log_compress``."""
    advantage_log_compress: bool = False
    """Compress each token's ``log q - log p`` to ``sign(a) log(1 + |a|)``, before ``kl_coef``.
    An alternative to the advantage clip bounds; setting both is a config error."""
    topk_metrics: OPDTopKMetricsConfig = field(default_factory=OPDTopKMetricsConfig)
    """Top-K overlap metrics: overlap ratio and overlap-token advantage. Off by default."""
    route_key: str = "skyrl_mopd_domain"
    """Which row label names the teacher of each prompt group under ``trainer.teachers``.
    It names an ``env_extras`` field: any dataset column besides the prompt and ``env_class``."""


@dataclass
class TeacherConfig(BaseConfig):
    """The frozen teacher, served by an inference engine."""

    backend: str = "skyrl"
    """Which service scores the teacher: ``"skyrl"`` or ``"vllm"``.
    ``"skyrl"`` (default): this job launches a vLLM deployment from ``inference_engine`` on its own GPUs and
    drives it through a ``RemoteInferenceClient``, as it does the student's engines. ``"vllm"``:
    one vLLM endpoint you run for the teacher model, at ``server_url``. Each backend reads only its own fields; setting the other's is a config error."""
    model: str = ""
    """The teacher model; what it names depends on the backend.
    ``skyrl``: an HF id or local path the job loads, e.g. ``Qwen/Qwen3-32B``. ``vllm``: the served model
    name of your servers."""
    inference_engine: TeacherEngineConfig = field(default_factory=TeacherEngineConfig)
    """The launched deployment for ``backend="skyrl"``, the same block as ``generator.inference_engine``.
    Weight-sync, sleep, LoRA, PD, speculative-decoding, routed-expert and external-URL fields do not apply
    to a frozen model and are rejected when set. Defaults that differ from the student's: prefix caching
    off, ``gpu_memory_utilization`` 0.9, Ray Prometheus stats off. As for the student, vLLM uses the model's
    native context unless ``engine_init_kwargs.max_model_len`` is set."""
    server_url: Optional[str] = None
    """Root URL of the one vLLM endpoint serving the teacher for ``backend="vllm"``, e.g. ``"http://host:8000"``.
    A single server, a data-parallel one, or a router in front of several (SkyRL's ``serve`` entrypoint
    starts one and logs it as ``proxy_url``); spreading requests over replicas is the endpoint's job. The
    client appends ``/inference/v1/generate``, so leave off a trailing ``/v1``. The server must run vLLM >= 0.32
    with the V2 model runner, and ``--max-logprobs`` of at least ``opd.topk_metrics.top_k`` when the overlap
    metrics are on; the job checks this at startup."""
    max_concurrency: int = 32
    """Maximum teacher requests in flight, for every backend.
    A launched teacher is also capped by its ``RemoteInferenceClient``'s per-engine limit
    (``SKYRL_GENERATE_CONCURRENCY_PER_ENGINE`` per server), like the student's rollouts."""
    request_timeout_s: float = 120.0
    """Per-request timeout of the ``vllm`` client.
    The ``skyrl`` backend uses its ``RemoteInferenceClient``'s policy, the one the student's rollouts get."""
    max_retries: int = 3
    """Retries with exponential backoff for the ``vllm`` client.
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


def teacher_path(name: str) -> str:
    """A teacher's config path, for messages: ``trainer.teacher`` or ``trainer.teachers.<name>``."""
    return "trainer.teacher" if name == DEFAULT_TEACHER else f"trainer.teachers.{name}"


def _as_teacher_config(name: str, entry: Any) -> TeacherConfig:
    """One teacher as a ``TeacherConfig``; the loader passes a map's values through as dicts."""
    if isinstance(entry, TeacherConfig):
        return entry
    if not isinstance(entry, dict):
        raise ValueError(f"{teacher_path(name)} must be a teacher block, got {entry!r}")
    try:
        return build_nested_dataclass(TeacherConfig, entry)
    except ValueError as e:
        raise ValueError(f"{teacher_path(name)}: {e}") from e


@dataclass
class OPDTrainerConfig(TrainerConfig):
    algorithm: OPDAlgorithmConfig = field(default_factory=OPDAlgorithmConfig)
    teacher: Optional[TeacherConfig] = None
    """The frozen teacher, a sibling of ``trainer.ref``; it scores every row. For several teachers, set
    ``trainer.teachers`` instead. Once built, the config moves it to ``trainer.teachers`` as ``default`` and
    resets this field to ``None``."""
    teachers: Optional[Dict[str, TeacherConfig]] = None
    """Several frozen teachers by name, for multi-teacher OPD (MOPD); each entry is a ``trainer.teacher`` block.
    Each prompt group is scored by the teacher its ``opd.route_key`` label names. Exclusive with ``trainer.teacher``."""

    def __post_init__(self):
        if self.teacher is not None and self.teachers is not None:
            raise ValueError(
                "trainer.teacher and trainer.teachers are alternatives; set one: trainer.teacher is a single teacher "
                "that scores every row, trainer.teachers names several and routes each row by its label"
            )
        if self.teacher is not None:
            # One teacher takes the same form as several, under the reserved name, so the rest of the code has one
            # path; rerunning this on the normalized config leaves it as it is.
            self.teachers = {DEFAULT_TEACHER: _as_teacher_config(DEFAULT_TEACHER, self.teacher)}
            self.teacher = None
        if not self.teachers:
            raise ValueError("set trainer.teacher for one teacher, or trainer.teachers for several")
        if DEFAULT_TEACHER in self.teachers and len(self.teachers) > 1:
            raise ValueError(f"trainer.teachers.{DEFAULT_TEACHER}: the name is reserved for trainer.teacher")
        self.teachers = {name: _as_teacher_config(name, entry) for name, entry in self.teachers.items()}


OPDExpConfig = make_config(trainer_cls=OPDTrainerConfig)


# Fields of the teacher's engine block that would change a launched server's behaviour if honoured.
# Refusing beats silently ignoring; weight_sync_backend and delta_weight_sync are never read on the
# frozen path and need no rule.
_FROZEN_ENGINE_FIELDS_MUST_BE_OFF = (
    "enable_pd",
    "enable_return_routed_experts",
    "enable_return_sample_support_set",
    "enable_return_sample_support_logprobs",
    "offload_kv_for_weight_sync",
)
_FROZEN_ENGINE_FIELDS_MUST_BE_NONE = (
    "speculative_config",
    "fp8_weight_sync_mode",
    "candidate_top_k",
    "external_server_urls",
    "external_proxy_url",
)


def _validate_launched_teacher(path: str, teacher: TeacherConfig) -> None:
    """``backend="skyrl"``: the job launches the teacher, so nothing may point elsewhere. ``path`` is the teacher's
    config path, ``trainer.teachers.<name>``."""
    ie_cfg: TeacherEngineConfig = teacher.inference_engine
    if teacher.server_url is not None:
        raise ValueError(
            f"{path}.server_url is for backend='vllm' (a server you run); backend='skyrl' launches the "
            f"teacher in this job from {path}.inference_engine"
        )
    if ie_cfg.backend != "vllm":
        raise ValueError(f"{path}.inference_engine.backend must be 'vllm', got {ie_cfg.backend!r}")
    if not ie_cfg.run_engines_locally:
        raise ValueError(
            f"{path}.inference_engine.run_engines_locally must be true for backend='skyrl'; for a server "
            f"you run use backend='vllm' with {path}.server_url"
        )
    for name in _FROZEN_ENGINE_FIELDS_MUST_BE_OFF:
        if getattr(ie_cfg, name):
            raise ValueError(f"{path}.inference_engine.{name} does not apply to a frozen teacher")
    for name in _FROZEN_ENGINE_FIELDS_MUST_BE_NONE:
        if getattr(ie_cfg, name) is not None:
            raise ValueError(f"{path}.inference_engine.{name} does not apply to a frozen teacher")
    num_gpus = (
        ie_cfg.num_engines * ie_cfg.tensor_parallel_size * ie_cfg.pipeline_parallel_size * ie_cfg.data_parallel_size
    )
    if num_gpus < 1:
        raise ValueError(f"{path}.inference_engine must describe at least one GPU")
    if ie_cfg.engine_init_kwargs.get("max_model_len") is None:
        logger.warning(
            f"{path}.inference_engine.engine_init_kwargs.max_model_len is not set, so vLLM uses the teacher "
            "model's native context, whose KV cache may not fit on the teacher's GPUs. Set it to cover the longest "
            "prompt + response + 1."
        )


def _reject_launch_settings(path: str, teacher: TeacherConfig) -> None:
    """``backend`` names servers elsewhere, so a launch block is a mistake, not a no-op."""
    if teacher.inference_engine != TeacherEngineConfig():
        raise ValueError(
            f"{path}.inference_engine is for backend='skyrl' (a teacher this job launches); "
            f"backend={teacher.backend!r} names servers elsewhere, so leave the block at its defaults"
        )


def _validate_teacher(path: str, teacher: TeacherConfig) -> None:
    """One teacher's backend rules; ``path`` is its config path, named in every message."""
    if teacher.backend not in TEACHER_BACKENDS:
        raise ValueError(f"{path}.backend must be one of {TEACHER_BACKENDS}, got {teacher.backend!r}")
    if not teacher.model:
        raise ValueError(f"{path}.model must be set")
    if teacher.backend == "skyrl":
        _validate_launched_teacher(path, teacher)
    elif teacher.backend == "vllm":
        if not teacher.server_url:
            raise ValueError(f"{path}.backend='vllm' requires {path}.server_url")
        normalize_server_url(teacher.server_url)  # refuses a trailing /v1 before anything launches
        _reject_launch_settings(path, teacher)
    if teacher.max_concurrency <= 0:
        raise ValueError(f"{path}.max_concurrency must be positive")


def _validate_teachers(cfg) -> None:
    """The names, the route key, then each teacher's backend rules. ``OPDTrainerConfig.__post_init__`` has already
    checked that one of ``trainer.teacher`` and ``trainer.teachers`` is set and moved a single teacher into the map."""
    if cfg.trainer.teacher is not None:
        raise ValueError(
            "trainer.teacher was set after the config was built, so it was never moved into trainer.teachers; set it "
            "when building the config"
        )
    teachers = cfg.trainer.teachers
    for name in list(teachers):
        if not isinstance(name, str) or not _TEACHER_NAME.fullmatch(name):
            raise ValueError(f"teacher name {name!r} must match [A-Za-z0-9_-]+: it appears in metric keys")
        teachers[name] = _as_teacher_config(name, teachers[name])  # entries assigned after the config was built
        _validate_teacher(teacher_path(name), teachers[name])
    if DEFAULT_TEACHER not in teachers and not cfg.trainer.algorithm.opd.route_key:
        raise ValueError("trainer.algorithm.opd.route_key must be set")


def teacher_logprob_width(cfg) -> int:
    """Ids the teacher scores per response position: the sampled token."""
    return 1


def teacher_topk(cfg) -> int:
    """How many of its own top tokens the teacher returns on the rows the overlap metrics measure; ``0`` is none."""
    topk_metrics = cfg.trainer.algorithm.opd.topk_metrics
    return topk_metrics.top_k if topk_metrics.num_rows else 0


def _enable_topk_capture(cfg) -> None:
    """Turn on the student's top-K capture for the overlap metrics, after checking what capture needs.

    The standard config's own capture checks have already run, so the ones that apply are repeated here.
    """
    algorithm = cfg.trainer.algorithm
    topk_metrics: OPDTopKMetricsConfig = algorithm.opd.topk_metrics
    if topk_metrics.top_k < 8:
        raise ValueError("trainer.algorithm.opd.topk_metrics.top_k must be at least 8")
    if algorithm.enable_sample_support_replay:
        raise ValueError(
            "trainer.algorithm.opd.topk_metrics captures the student's plain top-K, which sample-support replay "
            "cannot use: replay needs the sampler's own support"
        )
    sampling = cfg.generator.sampling_params
    untruncated = sampling.temperature == 1.0 and sampling.top_p == 1.0 and sampling.top_k == -1
    if not untruncated or sampling.min_p > 0.0 or sampling.repetition_penalty != 1.0:
        raise ValueError(
            "trainer.algorithm.opd.topk_metrics reads the student's distribution from its sampler, so the sampler "
            "must be the policy: temperature=1, top_p=1, top_k=-1, min_p=0 and repetition_penalty=1"
        )
    if sampling.additional_kwargs or cfg.generator.vision_language_generator:
        raise ValueError(
            "trainer.algorithm.opd.topk_metrics captures the student's top-K, which supports neither "
            "sampling_params.additional_kwargs nor vision_language_generator"
        )
    ie_cfg = cfg.generator.inference_engine
    ie_cfg.enable_return_sample_support_set = True
    ie_cfg.candidate_top_k = topk_metrics.top_k
    ie_cfg.enable_return_sample_support_logprobs = True


def validate_opd_cfg(cfg) -> None:
    """Rules the standard ``validate_cfg`` does not know about. Call it after ``validate_cfg``.

    With ``opd.topk_metrics.num_rows > 0`` it also turns on the student's top-K capture, so the user sets one field.
    """
    algorithm = cfg.trainer.algorithm
    opd: OPDConfig = algorithm.opd

    _validate_teachers(cfg)

    if opd.kl_coef < 0:
        raise ValueError("trainer.algorithm.opd.kl_coef must be >= 0")
    if opd.divergence_clip is not None and opd.divergence_clip <= 0:
        raise ValueError("trainer.algorithm.opd.divergence_clip must be positive")
    if opd.advantage_clip_min is not None and opd.advantage_clip_min > 0:
        raise ValueError("trainer.algorithm.opd.advantage_clip_min must be <= 0")
    if opd.advantage_clip_max is not None and opd.advantage_clip_max < 0:
        raise ValueError("trainer.algorithm.opd.advantage_clip_max must be >= 0")
    if opd.advantage_log_compress and (opd.advantage_clip_min is not None or opd.advantage_clip_max is not None):
        raise ValueError(
            "trainer.algorithm.opd.advantage_log_compress and the advantage clip bounds are alternatives; set one"
        )

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
    if opd.topk_metrics.num_rows < 0:
        raise ValueError("trainer.algorithm.opd.topk_metrics.num_rows must be >= 0")
    if opd.topk_metrics.num_rows:
        _enable_topk_capture(cfg)
    elif sampling.temperature != 1.0 or sampling.top_p != 1.0 or sampling.top_k != -1:
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

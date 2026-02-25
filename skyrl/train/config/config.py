"""
Typed configuration dataclasses for SkyRL.

These mirror the YAML configuration structure 1:1. The top-level SkyRLTrainConfig
can be constructed from a Hydra DictConfig via SkyRLTrainConfig.from_dict_config().
"""

import os
from abc import ABC
import dataclasses
from dataclasses import dataclass, field, asdict
import typing
from typing import Any, Dict, List, Optional, Union, Type, TypeVar, Annotated
import yaml
import copy

from omegaconf import DictConfig, OmegaConf

from skyrl_gym.envs.search.env import SearchEnvConfig
from skyrl_gym.envs.sql.env import Text2SQLEnvConfig

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class BaseConfig(ABC):
    """
    Base configuration class for SkyRL-Train
    """

    @classmethod
    def from_dict_config(cls, cfg: DictConfig) -> "BaseConfig":
        """Construct a typed BaseConfig from a Hydra DictConfig."""
        raw = OmegaConf.to_container(cfg, resolve=True)
        return build_nested_dataclass(cls, raw)


@dataclass
class DataConfig(BaseConfig):
    train_data: List[str] = field(default_factory=lambda: [os.path.expanduser("~/data/gsm8k/train.parquet")])
    val_data: List[str] = field(default_factory=lambda: [os.path.expanduser("~/data/gsm8k/validation.parquet")])


# ---------------------------------------------------------------------------
# Model / LoRA
# ---------------------------------------------------------------------------


# added prefix SkyRL to avoid conflict with peft.LoraConfig
@dataclass
class SkyRLLoraConfig(BaseConfig):
    rank: int = 0
    alpha: int = 16
    dropout: float = 0.0
    lora_sync_path: str = "/tmp/skyrl_lora_sync"
    target_modules: str = "all-linear"
    exclude_modules: Optional[str] = None
    init_method: str = "kaiming"
    """For FSDP, corresponds to ``init_lora_weights`` in PEFT.
    For Megatron, used for ``lora_A_init_method``; supports "xavier", "normal", "kaiming", "zero"."""


@dataclass
class ModelConfig(BaseConfig):
    path: Optional[str] = None
    lora: SkyRLLoraConfig = field(default_factory=SkyRLLoraConfig)


# ---------------------------------------------------------------------------
# Optimizer / FSDP
# ---------------------------------------------------------------------------


@dataclass
class OptimizerConfig(BaseConfig):
    lr: float = 1e-6
    adam_betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    offload_after_step: bool = True
    """Offload optimizer state to CPU after each full training step. Only applicable when ``colocate_all=True``."""
    num_warmup_steps: int = 0
    """Number of mini-batch steps to warmup the optimizer."""
    scheduler: str = "constant_with_warmup"


@dataclass
class MixedPrecisionConfig(BaseConfig):
    param_dtype: str = "bf16"
    reduce_dtype: str = "fp32"
    buffer_dtype: str = "fp32"


@dataclass
class FSDPConfig(BaseConfig):
    cpu_offload: bool = False
    """Offload params and optimizer state to CPU during the forward pass."""
    reshard_after_forward: Union[bool, int] = True
    """FSDP2 only. Accepts True, False, or an int between 1 and ``fsdp_size``."""
    fsdp_size: int = -1
    mixed_precision: Optional[MixedPrecisionConfig] = None
    # specify wrap policy as a dict with `transformer_layer_cls_to_wrap` key for custom module based wrapping
    wrap_policy: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Megatron
# ---------------------------------------------------------------------------


@dataclass
class MegatronDDPConfig(BaseConfig):
    grad_reduce_in_fp32: bool = True
    overlap_grad_reduce: bool = False
    overlap_param_gather: bool = False
    average_in_collective: bool = True


@dataclass
class MegatronTorchProfilerConfig(BaseConfig):
    enable: bool = False
    ranks: List[int] = field(default_factory=list)
    save_path: Optional[str] = None


@dataclass
class MegatronLoraConfig(BaseConfig):
    lora_type: str = "lora"


DEFAULT_MEGATRON_OPTIMIZER_KWARGS = {
    "overlap_cpu_optimizer_d2h_h2d": False,
    "use_precision_aware_optimizer": False,
    "optimizer_cpu_offload": False,
    "optimizer_offload_fraction": 0.0,
}

DEFAULT_TRANSFORMER_CONFIG_KWARGS = {
    "recompute_granularity": "full",
    "recompute_modules": ["core_attn"],
    "recompute_method": "uniform",
    "recompute_num_layers": 1,
}


@dataclass
class MegatronConfig(BaseConfig):
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: Optional[int] = None
    ddp_config: MegatronDDPConfig = field(default_factory=MegatronDDPConfig)
    torch_profiler_config: MegatronTorchProfilerConfig = field(default_factory=MegatronTorchProfilerConfig)
    lora_config: MegatronLoraConfig = field(default_factory=MegatronLoraConfig)
    optimizer_config_kwargs: Dict[str, Any] = field(
        default_factory=lambda: copy.deepcopy(DEFAULT_MEGATRON_OPTIMIZER_KWARGS)
    )
    transformer_config_kwargs: Dict[str, Any] = field(
        default_factory=lambda: copy.deepcopy(DEFAULT_TRANSFORMER_CONFIG_KWARGS)
    )
    empty_cuda_cache: Optional[bool] = None
    model_config_kwargs: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------


@dataclass
class PlacementConfig(BaseConfig):
    colocate_all: bool = True
    """When True, training and inference share the same GPUs."""
    colocate_policy_ref: bool = True
    policy_num_nodes: int = 1
    policy_num_gpus_per_node: int = 1
    critic_num_nodes: int = 1
    critic_num_gpus_per_node: int = 1
    ref_num_nodes: int = 1
    ref_num_gpus_per_node: int = 1


# ---------------------------------------------------------------------------
# Policy / Critic / Ref
# ---------------------------------------------------------------------------


@dataclass
class PolicyConfig(BaseConfig):
    model: ModelConfig = field(default_factory=lambda: copy.deepcopy(ModelConfig(path="Qwen/Qwen2.5-1.5B-Instruct")))
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)
    sequence_parallel_size: int = 1
    use_torch_compile: bool = False
    """Apply torch.compile to logits calculation."""
    record_memory: bool = False
    """Save memory snapshots to ``{ckpt_path}/memory_snapshots/``.
    Visualize by dragging pickle files to https://docs.pytorch.org/memory_viz."""
    megatron_config: MegatronConfig = field(default_factory=MegatronConfig)
    model_config_kwargs: dict = field(default_factory=dict)
    """Pass-through kwargs for the HuggingFace model config (FSDP backends).
    For Megatron, use ``policy.megatron_config.transformer_config_kwargs`` instead."""


@dataclass
class CriticConfig(BaseConfig):
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer_config: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(lr=5e-6))
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)
    sequence_parallel_size: int = 1
    model_config_kwargs: dict = field(default_factory=dict)


# TODO: Have global config init so that the default value for the ref model path is the policy model path
@dataclass
class RefConfig(BaseConfig):
    model: ModelConfig = field(default_factory=ModelConfig)
    sequence_parallel_size: int = 1
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)
    megatron_config: MegatronConfig = field(default_factory=MegatronConfig)
    model_config_kwargs: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------


@dataclass
class KLCtrlConfig(BaseConfig):
    """Only used when ``use_kl_in_reward=True`` (not applied when ``use_kl_loss=True``).
    Uses ``kl_loss_coef`` as the initial KL coefficient."""

    type: str = "fixed"
    """``"fixed"`` or ``"adaptive"``."""
    kl_target: float = 0.1
    """Target KL divergence for the adaptive KL controller."""
    horizon: int = 10000
    """Controls the update rate of the adaptive KL controller."""


@dataclass
class SAPOConfig(BaseConfig):
    tau_pos: float = 1.0
    tau_neg: float = 1.05


@dataclass
class DynamicSamplingConfig(BaseConfig):
    type: Optional[str] = None
    """``"filter"``, ``"replace"``, or ``None``."""
    max_sample_batches: int = 30
    """Sample at most this many batches before stopping. ``-1`` to sample forever."""
    min_replace_ratio: float = 0.3
    """Minimum proportion of good samples to replace bad samples. Only used with ``"replace"`` strategy."""


@dataclass
class ClipCovConfig(BaseConfig):
    """Only used when ``policy_loss_type="clip_cov"``."""

    clip_ratio: float = 0.0002
    """Fraction of tokens to clip based on covariance."""
    clip_cov_lb: float = 1.0
    clip_cov_ub: float = 5.0


@dataclass
class KLCovConfig(BaseConfig):
    """Only used when ``policy_loss_type="kl_cov"``."""

    kl_cov_frac: float = 0.2
    """Fraction of tokens to apply KL regularization to."""
    ppo_kl_coef: float = 1.0


@dataclass
class CISPOConfig(BaseConfig):
    """Only used when ``policy_loss_type="cispo"``."""

    cispo_eps_clip_low: float = 0.0
    """Offset for lower bound of importance sampling ratio clipping (as opposed to PPO token update clipping)."""
    cispo_eps_clip_high: float = 5.0
    """Offset for upper bound of importance sampling ratio clipping (as opposed to PPO token update clipping)."""


# see https://docs.skyrl.ai/docs/algorithms/off_policy_correction for more details
@dataclass
class OffPolicyCorrectionConfig(BaseConfig):
    tis_ratio_type: Optional[str] = None
    """Importance sampling ratio type for PPO loss correction: ``None``, ``"token"``, or ``"sequence"``.
    The ratio is ``exp(logprobs_policy_old - logprobs_rollout_policy)``."""
    token_tis_ratio_clip_high: float = 2.0
    """Used when ``tis_ratio_type="token"``. Recommended range: 1.5--5.0."""
    sequence_tis_ratio_clip_high: float = 5.0
    """Used when ``tis_ratio_type="sequence"``. Recommended range: 2.0--10.0."""
    sequence_mask_metric: Optional[str] = None
    """Method for masking sequences with cumulative IS ratios outside cap: ``None``, ``"product"``, or ``"geometric"``."""
    geo_mask_high: float = 1.01
    """Used when ``sequence_mask_metric="geometric"``. Recommended ~0.99--1.01; MoE models may need a wider range."""
    geo_mask_low: float = 0.99
    """Used when ``sequence_mask_metric="geometric"``."""
    product_mask_high: float = 2.0
    """Used when ``sequence_mask_metric="product"``. Recommended ~0.5--2.0."""
    product_mask_low: float = 0.5
    """Used when ``sequence_mask_metric="product"``."""
    outlier_token_is_threshold_low: Optional[float] = None
    """Mask sequences with any token IS ratio below this threshold. Suggested: 1e-4. ``None`` to disable."""
    outlier_token_is_threshold_high: Optional[float] = None
    """Mask sequences with any token IS ratio above this threshold. Suggested: 100. ``None`` to disable."""


@dataclass
class AlgorithmConfig(BaseConfig):
    advantage_estimator: str = "grpo"
    """``"grpo"``, ``"gae"``, ``"rloo"``, ``"reinforce++"``, or custom via ``AdvantageEstimatorRegistry``."""
    kl_ctrl: KLCtrlConfig = field(default_factory=KLCtrlConfig)
    kl_estimator_type: str = "k3"
    """``"k1"``, ``"k2"``, ``"k3"``, ``"abs"``. See http://joschu.net/blog/kl-approx.html."""
    use_kl_in_reward: bool = False
    """Apply KL penalty to rewards. Mutually exclusive with ``use_kl_loss``."""
    use_kl_loss: bool = True
    """Apply KL loss in the policy model. Mutually exclusive with ``use_kl_in_reward``."""
    kl_loss_coef: float = 0.001
    use_entropy_loss: bool = False
    entropy_loss_coef: float = 0.01
    temperature: float = 1.0
    """Temperature for scaling logits in policy loss computation.
    Typically set from ``generator.sampling_params.temperature`` during config validation.
    
    NOTE: When using HTTP endpoints directly, make sure to set ``generator.sampling_params.temperature`` to the temperature used during generation - the algorithm config temperature inherits this value.
    """
    advantage_batch_normalize: bool = False
    value_head_prefix: str = "value_head"
    policy_loss_type: str = "regular"
    """``"regular"``, ``"dual_clip"``, ``"gspo"``, ``"clip_cov"``, ``"kl_cov"``, or custom via ``PolicyLossRegistry``."""
    loss_reduction: str = "token_mean"
    """``"token_mean"``, ``"sequence_mean"``, or ``"seq_mean_token_sum_norm"``."""
    grpo_norm_by_std: bool = True
    zero_variance_filter: bool = False
    """Loss-mask prompts with zero-variance rewards. Only applicable when rewards are response-level."""
    lambd: float = 1.0
    gamma: float = 1.0
    eps_clip_low: float = 0.2
    eps_clip_high: float = 0.2
    clip_ratio_c: float = 3.0
    """Dual-clip parameter."""
    tis_imp_ratio_cap: float = -1.0
    """Deprecated: use ``off_policy_correction.tis_ratio_type="token"`` and ``token_tis_ratio_clip_high`` instead."""
    use_tis: bool = False
    """Deprecated: use ``off_policy_correction`` instead."""
    off_policy_correction: OffPolicyCorrectionConfig = field(default_factory=OffPolicyCorrectionConfig)
    sapo: SAPOConfig = field(default_factory=SAPOConfig)
    value_clip: float = 0.2
    dynamic_sampling: DynamicSamplingConfig = field(default_factory=DynamicSamplingConfig)
    clip_cov: ClipCovConfig = field(default_factory=ClipCovConfig)
    kl_cov: KLCovConfig = field(default_factory=KLCovConfig)
    cispo: CISPOConfig = field(default_factory=CISPOConfig)
    max_seq_len: Optional[int] = None
    """Used for ``seq_mean_token_sum_norm`` loss reduction; set explicitly for multi-turn.
    If ``None``, calculated as ``generator.max_input_length + generator.sampling_params.max_generate_length``."""


# ---------------------------------------------------------------------------
# Fully Async
# ---------------------------------------------------------------------------


@dataclass
class FullyAsyncConfig(BaseConfig):
    """Knobs for fully async training.
    See https://docs.skyrl.ai/docs/tutorials/fully_async#step-2-config-knobs-to-tune-for-fully-async-training."""

    max_staleness_steps: int = 4
    """Maximum off-policy steps allowed. If a trajectory group is scheduled at step *i* and trained at step *j*,
    then ``j - i <= max_staleness_steps``. Larger values increase throughput but also off-policy-ness."""
    num_parallel_generation_workers: int = 768
    """Number of generation workers to spawn. Should be >= ``policy_mini_batch_size`` and
    <= ``policy_mini_batch_size * (max_staleness_steps + 1)``."""


# ---------------------------------------------------------------------------
# Sampling / Chat Template
# ---------------------------------------------------------------------------


@dataclass
class SamplingParams(BaseConfig):
    max_generate_length: int = 1024
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    min_p: float = 0.0
    top_k: int = -1
    logprobs: Optional[int] = 1
    stop: Optional[List[str]] = None
    additional_kwargs: Optional[Dict[str, Any]] = None


@dataclass
class ChatTemplateConfig(BaseConfig):
    source: str = "name"
    name_or_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Inference Engine
# ---------------------------------------------------------------------------


@dataclass
class InferenceEngineConfig(BaseConfig):
    """Configuration for inference engine instantiation and management."""

    model_dtype: str = "bfloat16"
    """Should match the dtype used by the inference engine."""
    run_engines_locally: bool = True
    num_engines: int = 1
    backend: str = "vllm"
    """``"vllm"`` or ``"sglang"``."""
    weight_sync_backend: str = "nccl"
    weight_transfer_threshold_cuda_ipc_GB: float = 1.0
    """When using ``cuda_ipc``, send weights in batches of this size (GB)."""
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    expert_parallel_size: int = 1
    data_parallel_size: int = 1
    async_engine: bool = True
    vllm_v1_disable_multiproc: bool = True
    """Sets ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` for reproducibility."""
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    max_num_batched_tokens: int = 8192
    enforce_eager: bool = True
    """Disable CUDA graphs for stability. Set to ``False`` for higher performance,
    but this may affect convergence for long-running or long-context training jobs."""
    fully_sharded_loras: bool = False
    enable_ray_prometheus_stats: bool = False
    """Enable Ray Prometheus stats logger for inference engine metrics (vLLM v1 only)."""
    gpu_memory_utilization: float = 0.8
    max_num_seqs: int = 1024
    remote_urls: List[str] = field(default_factory=lambda: [])
    enable_http_endpoint: bool = False
    """When ``True``, launch an OpenAI-compatible HTTP endpoint for the inference engine client so that generators can send requests to this server instead of using ``.generate()`` Python calls.
    
    NOTE: When using HTTP endpoints directly, make sure to set ``generator.sampling_params.temperature`` to the temperature used during generation - the algorithm config temperature (used for logits scaling) inherits this value.
    """
    http_endpoint_host: str = "127.0.0.1"
    http_endpoint_port: int = 8000
    served_model_name: Optional[str] = None
    """Model name for HTTP endpoint validation. If set, must be used in the ``model`` field of
    ``/chat/completions`` requests instead of the model path. If ``None``, the model path is used."""
    engine_init_kwargs: Dict[str, Any] = field(default_factory=dict)
    """Pass-through kwargs for the vLLM/SGLang engine. Names must match the engine's args."""
    override_existing_update_group: str = "auto"
    """``"auto"``, ``"enable"``, or ``"disable"``."""
    external_proxy_url: Optional[str] = None
    """Data-plane URL (load-balanced router) for the new inference layer."""
    external_server_urls: Optional[List[str]] = None
    """Control-plane URLs (direct backend access) for the new inference layer."""


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


@dataclass
class GeneratorConfig(BaseConfig):
    """Configuration for generation behavior."""

    inference_engine: InferenceEngineConfig = field(default_factory=InferenceEngineConfig)
    n_samples_per_prompt: int = 5
    batched: bool = False
    max_turns: int = 1
    max_input_length: Optional[int] = None
    """Max generator input length for multi-turn conversations. For single-turn, set equal to ``max_prompt_length``."""
    chat_template: ChatTemplateConfig = field(default_factory=ChatTemplateConfig)
    chat_template_kwargs: Dict[str, Any] = field(default_factory=dict)
    """Kwargs passed to ``tokenizer.apply_chat_template``."""
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    use_conversation_multi_turn: bool = True
    """If ``True``, each multi-turn model response and env observation is stored in a separate
    assistant/user message. If ``False``, they are appended to the original assistant response."""
    append_eos_token_after_stop_str_in_multi_turn: bool = True
    """When ``use_conversation_multi_turn=True`` and ``sampling_params.stop`` is set, append
    ``eos_token_id`` to generations that end with a matched stop string."""
    eval_sampling_params: Optional[SamplingParams] = None
    """Separate sampling params for evaluation. If ``None``, then it defaults to ``SamplingParams(temperature=0.0, max_generate_length=generator.sampling_params.max_generate_length)``."""
    eval_n_samples_per_prompt: int = 1
    zero_reward_on_non_stop: bool = False
    """Set reward to 0 when ``stop_reason`` is not ``"stop"`` (i.e., generation was truncated or aborted)."""
    apply_overlong_filtering: bool = False
    """Apply DAPO Overlong Filtering: mask out all tokens in the loss mask for trajectories that
    exceed max length (truncated, no EOS token)."""
    rope_scaling: Optional[Dict[str, Any]] = None
    """Can differ from the trainer's ``rope_scaling``, useful for thinking models."""
    rope_theta: Optional[float] = None
    step_wise_trajectories: bool = False

    def __post_init__(self):

        if self.eval_sampling_params is None:
            self.eval_sampling_params = SamplingParams(
                temperature=0.0, max_generate_length=self.sampling_params.max_generate_length
            )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


# NOTE: Redefinition of Judge Env configuration because this is currently only available in examples/
@dataclass
class GSM8kLLMJudgeEnvConfig(BaseConfig):
    model: str = "gpt-4o-mini"
    base_url: Optional[str] = None


@dataclass
class SkyRLGymConfig(BaseConfig):
    max_env_workers: int = 32
    text2sql: Text2SQLEnvConfig = field(default_factory=Text2SQLEnvConfig)
    llm_as_a_judge: GSM8kLLMJudgeEnvConfig = field(default_factory=GSM8kLLMJudgeEnvConfig)
    search: SearchEnvConfig = field(default_factory=SearchEnvConfig)


@dataclass
class EnvironmentConfig(BaseConfig):
    env_class: str = "gsm8k"
    skyrl_gym: SkyRLGymConfig = field(default_factory=SkyRLGymConfig)


# ---------------------------------------------------------------------------
# Trainer (top-level)
# ---------------------------------------------------------------------------


@dataclass
class TrainerConfig(BaseConfig):
    placement: PlacementConfig = field(default_factory=PlacementConfig)
    sequence_parallel_backend: str = "ulysses"
    strategy: str = "fsdp2"
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    ref: RefConfig = field(default_factory=RefConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    fully_async: FullyAsyncConfig = field(default_factory=FullyAsyncConfig)
    gradient_checkpointing: bool = True
    gradient_checkpointing_use_reentrant: bool = False
    seed: int = 42
    resume_mode: Optional[str] = "latest"
    """``None``/``"none"``, ``"latest"``, or ``"from_path"``."""
    resume_path: Optional[str] = None
    log_path: str = "/tmp/skyrl-logs"
    """Path for infrastructure log files. For multi-node, use a shared filesystem path to consolidate logs."""
    ckpt_path: str = field(default_factory=lambda: os.path.expanduser("~/ckpts/"))
    max_ckpts_to_keep: int = -1
    """``-1`` to keep all checkpoints, ``N`` to keep only the last N."""
    ckpt_interval: int = 10
    hf_save_interval: int = -1
    """Save HuggingFace-format model every N steps. ``-1`` to disable."""
    export_path: str = field(default_factory=lambda: os.path.expanduser("~/exports/"))
    """Path for exported artifacts (HF models, debug dumps, etc.)."""
    bf16: bool = True
    epochs: int = 1
    update_epochs_per_batch: int = 1
    """Number of gradient update passes over each training batch."""
    train_batch_size: int = 1024
    """See ``utils/utils.py::validate_batch_sizes`` for train, mini, and micro batch size constraints."""
    policy_mini_batch_size: int = 256
    critic_mini_batch_size: int = 256
    micro_train_batch_size_per_gpu: int = 1
    micro_forward_batch_size_per_gpu: int = 1
    update_ref_every_epoch: bool = False
    use_sample_packing: bool = True
    eval_batch_size: int = 1024
    eval_before_train: bool = True
    eval_interval: int = 5
    """``-1`` to disable evaluation."""
    max_prompt_length: int = 512
    flash_attn: bool = True
    disable_fast_tokenizer: bool = False
    project_name: str = "skyrl"
    run_name: str = "test_run"
    logger: str = "wandb"
    dump_data_batch: bool = False
    dump_eval_results: bool = True
    rope_scaling: Optional[Dict[str, Any]] = None
    rope_theta: Optional[float] = None

    def __post_init__(self):
        # ref model defaults to the policy model
        if self.ref.model.path is None:
            self.ref.model.path = self.policy.model.path


def validate_dict_keys_against_dataclass(datacls: Type[Any], d: dict):
    """
    Validate the keys of a dict against fields of a dataclass.

    Args:
        datacls: The dataclass class to validate
    """
    valid_fields = {f.name for f in dataclasses.fields(datacls)}
    if invalid_keys := set(d.keys() - valid_fields):
        raise ValueError(f"Invalid fields {invalid_keys} for {datacls.__name__}. Valid fields are {valid_fields}.")


def _resolve_dataclass_type(type_annotation: Any) -> Optional[Type]:
    """Extract the concrete dataclass type from a type annotation.

    Handles plain types, Optional[T], Union[T, None], and Annotated[T, ...].
    Returns None if no dataclass type can be resolved.
    """
    origin = typing.get_origin(type_annotation)

    if origin is Union:
        # Optional[X] is Union[X, None]. Find the non-None dataclass arg.
        for arg in typing.get_args(type_annotation):
            if arg is type(None):
                continue
            resolved = _resolve_dataclass_type(arg)
            if resolved is not None:
                return resolved
        return None

    if origin is Annotated:
        return _resolve_dataclass_type(typing.get_args(type_annotation)[0])

    # Plain class check
    if isinstance(type_annotation, type) and dataclasses.is_dataclass(type_annotation):
        return type_annotation

    return None


T = TypeVar("T")


def build_nested_dataclass(datacls: Type[T], d: dict) -> T:
    """Recursively build a dataclass from a dict, handling nested dataclasses.

    Supports fields typed as standard python types, plain dataclasses, Optional[DataclassType],
    Union[DataclassType, None], and Annotated[...] wrappers. Non-dataclass
    fields (primitives, dicts, lists, etc.) are passed through as-is.

    Args:
        datacls: The dataclass class to build.
        d: The dict to build the dataclass from.

    Returns:
        An instance of the dataclass.
    """
    validate_dict_keys_against_dataclass(datacls, d)
    kwargs = {}
    for f in dataclasses.fields(datacls):
        if f.name not in d:
            continue
        value = d[f.name]
        nested_cls = _resolve_dataclass_type(f.type)
        if nested_cls is not None and isinstance(value, dict):
            kwargs[f.name] = build_nested_dataclass(nested_cls, value)
        else:
            # Primitives, None, lists, raw dicts, already-constructed objects
            kwargs[f.name] = value
    return datacls(**kwargs)


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass
class SkyRLTrainConfig(BaseConfig):
    data: DataConfig = field(default_factory=DataConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)

    def __post_init__(self):

        # generator.max_input_length defaults to trainer.max_prompt_length
        if self.generator.max_input_length is None:
            self.generator.max_input_length = self.trainer.max_prompt_length

        # generator rope params default to trainer rope params
        if self.generator.rope_scaling is None and self.trainer.rope_scaling is not None:
            self.generator.rope_scaling = self.trainer.rope_scaling
        if self.generator.rope_theta is None and self.trainer.rope_theta is not None:
            self.generator.rope_theta = self.trainer.rope_theta
        # Copy temperature from generator sampling params to algorithm config
        # so workers can access it without needing the generator config
        self.trainer.algorithm.temperature = self.generator.sampling_params.temperature

    @classmethod
    def from_cli_overrides(cls, args: Union[List[str], dict]) -> "SkyRLTrainConfig":
        """Construct a SkyRLTrainConfig from CLI arguments or a dict of overrides.

        Parses CLI arguments and builds a typed config. Dataclass field defaults
        are used for any values not specified on the command line.

        Supports both new-style config paths (e.g., generator.inference_engine.backend)
        and legacy YAML-style paths (e.g., generator.backend) for backward compatibility.

        Args:
            args: Either a list of CLI arguments in 'key.path=value' format, or a dict
                  mapping dot-notation keys to values.
                  Example list: ['trainer.policy.model.path=Qwen/Qwen2.5-1.5B-Instruct', 'trainer.seed=123']
                  Example dict: {'trainer.policy.model.path': 'Qwen/Qwen2.5-1.5B-Instruct', 'trainer.seed': 123}

        Returns:
            A fully constructed SkyRLTrainConfig with CLI overrides applied.

        Raises:
            ValueError: If an argument uses the unsupported '+' prefix.
        """
        if isinstance(args, dict):
            args = [f"{k}={v}" for k, v in args.items()]

        from skyrl.train.config.legacy import (
            is_legacy_config,
            translate_legacy_config,
            warn_legacy_config,
        )
        from skyrl.train.config.utils import get_legacy_config

        # Check for unsupported '+' prefix
        for arg in args:
            if arg.startswith("+"):
                raise ValueError(
                    f"The '+' prefix for adding new config fields is not supported: '{arg}'. "
                    "To add custom config fields, subclass the relevant config dataclass."
                )
        overrides = OmegaConf.from_cli(args)

        # Try new format first
        try:
            return cls.from_dict_config(overrides)
        except ValueError:
            # Fall back to legacy format: load base YAML, merge overrides, translate
            try:
                base_cfg = get_legacy_config()
                merged = OmegaConf.merge(base_cfg, overrides)
                merged_dict = OmegaConf.to_container(merged, resolve=True)

                if is_legacy_config(merged_dict):
                    warn_legacy_config()
                    translated = translate_legacy_config(merged_dict)
                    return build_nested_dataclass(cls, translated)
            except Exception:
                pass  # Legacy translation failed, re-raise original error

            # Re-raise original error if not a legacy config issue
            raise


def make_config(
    algorithm_cls: Optional[Type[AlgorithmConfig]] = None,
    trainer_cls: Optional[Type[TrainerConfig]] = None,
    generator_cls: Optional[Type[GeneratorConfig]] = None,
) -> Type[SkyRLTrainConfig]:
    """Create a SkyRLTrainConfig subclass with custom nested config classes.

    Convenience helper to avoid boilerplate when extending configs for custom
    algorithms or generators. For full IDE autocomplete on custom fields, use
    explicit subclassing instead (see examples/algorithms/dapo/main_dapo.py).

    Args:
        algorithm_cls: Custom AlgorithmConfig subclass. If provided without
            trainer_cls, a TrainerConfig subclass is automatically created.
        trainer_cls: Custom TrainerConfig subclass. Takes precedence over
            algorithm_cls for the trainer config.
        generator_cls: Custom GeneratorConfig subclass.

    Returns:
        A SkyRLTrainConfig subclass wired up with the custom config classes.

    Example::

        @dataclass
        class MyAlgorithmConfig(AlgorithmConfig):
            my_param: int = 42

        MyConfig = make_config(algorithm_cls=MyAlgorithmConfig)
        cfg = MyConfig.from_cli_overrides(sys.argv[1:])
    """
    effective_trainer_cls = trainer_cls

    if algorithm_cls is not None and trainer_cls is None:
        effective_trainer_cls = dataclass(
            type(
                f"_{algorithm_cls.__name__}TrainerConfig",
                (TrainerConfig,),
                {
                    "__annotations__": {"algorithm": algorithm_cls},
                    "algorithm": field(default_factory=algorithm_cls),
                },
            )
        )

    ns: Dict[str, Any] = {}
    annotations: Dict[str, Any] = {}

    if effective_trainer_cls is not None:
        annotations["trainer"] = effective_trainer_cls
        ns["trainer"] = field(default_factory=effective_trainer_cls)

    if generator_cls is not None:
        annotations["generator"] = generator_cls
        ns["generator"] = field(default_factory=generator_cls)

    ns["__annotations__"] = annotations

    return dataclass(type("_CustomSkyRLTrainConfig", (SkyRLTrainConfig,), ns))


def get_config_as_dict(cfg: Union[dict, BaseConfig]) -> dict:
    if isinstance(cfg, dict):
        return cfg
    return asdict(cfg)


def get_config_as_yaml_str(cfg: BaseConfig) -> str:
    return yaml.dump(asdict(cfg))

"""
Typed configuration dataclasses for SkyRL.

These mirror the YAML configuration structure 1:1. The top-level SkyRLConfig
can be constructed from a Hydra DictConfig via SkyRLConfig.from_dict_config().
"""

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
    train_data: List[str] = field(default_factory=list)
    val_data: List[str] = field(default_factory=list)


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
    num_warmup_steps: int = 0
    scheduler: str = "constant_with_warmup"


@dataclass
class MixedPrecisionConfig(BaseConfig):
    param_dtype: str = "bf16"
    reduce_dtype: str = "fp32"
    buffer_dtype: str = "fp32"


@dataclass
class FSDPConfig(BaseConfig):
    cpu_offload: bool = False
    reshard_after_forward: Union[bool, int] = True
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
    colocate_policy_ref: bool = True
    policy_num_nodes: int = 1
    policy_num_gpus_per_node: int = 4
    critic_num_nodes: int = 1
    critic_num_gpus_per_node: int = 4
    ref_num_nodes: int = 1
    ref_num_gpus_per_node: int = 4


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
    record_memory: bool = False
    megatron_config: MegatronConfig = field(default_factory=MegatronConfig)
    model_config_kwargs: dict = field(default_factory=dict)


@dataclass
class CriticConfig(BaseConfig):
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)
    sequence_parallel_size: int = 1
    model_config_kwargs: dict = field(default_factory=dict)


# TODO: Have global config init so that the default value for the ref model path is the policy model path
@dataclass
class RefConfig(BaseConfig):
    model: ModelConfig = field(default_factory=lambda: copy.deepcopy(ModelConfig(path="Qwen/Qwen2.5-1.5B-Instruct")))
    sequence_parallel_size: int = 1
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)
    megatron_config: MegatronConfig = field(default_factory=MegatronConfig)
    model_config_kwargs: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------


@dataclass
class KLCtrlConfig(BaseConfig):
    type: str = "fixed"
    kl_target: float = 0.1
    horizon: int = 10000


@dataclass
class SAPOConfig(BaseConfig):
    tau_pos: float = 1.0
    tau_neg: float = 1.05


@dataclass
class DynamicSamplingConfig(BaseConfig):
    type: Optional[str] = None
    max_sample_batches: int = 30
    min_replace_ratio: float = 0.3


@dataclass
class ClipCovConfig(BaseConfig):
    clip_ratio: float = 0.0002
    clip_cov_lb: float = 1.0
    clip_cov_ub: float = 5.0


@dataclass
class KLCovConfig(BaseConfig):
    kl_cov_frac: float = 0.2
    ppo_kl_coef: float = 1.0


@dataclass
class CISPOConfig(BaseConfig):
    cispo_eps_clip_low: float = 0.0
    cispo_eps_clip_high: float = 5.0


# see https://docs.skyrl.ai/docs/algorithms/off_policy_correction for more details
@dataclass
class OffPolicyCorrectionConfig(BaseConfig):
    tis_ratio_type: Optional[str] = None
    token_tis_ratio_clip_high: float = 2.0
    sequence_tis_ratio_clip_high: float = 5.0
    sequence_mask_metric: Optional[str] = None
    geo_mask_high: float = 1.01
    geo_mask_low: float = 0.99
    product_mask_high: float = 2.0
    product_mask_low: float = 0.5
    outlier_token_is_threshold_low: Optional[float] = None
    outlier_token_is_threshold_high: Optional[float] = None


@dataclass
class AlgorithmConfig(BaseConfig):
    advantage_estimator: str = "grpo"
    kl_ctrl: KLCtrlConfig = field(default_factory=KLCtrlConfig)
    kl_estimator_type: str = "k3"
    use_kl_in_reward: bool = False
    use_kl_loss: bool = True
    kl_loss_coef: float = 0.001
    use_entropy_loss: bool = False
    entropy_loss_coef: float = 0.01
    advantage_batch_normalize: bool = False
    value_head_prefix: str = "value_head"
    policy_loss_type: str = "regular"
    loss_reduction: str = "token_mean"
    grpo_norm_by_std: bool = True
    zero_variance_filter: bool = False
    lambd: float = 1.0
    gamma: float = 1.0
    eps_clip_low: float = 0.2
    eps_clip_high: float = 0.2
    clip_ratio_c: float = 3.0
    tis_imp_ratio_cap: float = -1.0
    use_tis: bool = False
    off_policy_correction: OffPolicyCorrectionConfig = field(default_factory=OffPolicyCorrectionConfig)
    sapo: SAPOConfig = field(default_factory=SAPOConfig)
    value_clip: float = 0.2
    dynamic_sampling: DynamicSamplingConfig = field(default_factory=DynamicSamplingConfig)
    clip_cov: ClipCovConfig = field(default_factory=ClipCovConfig)
    kl_cov: KLCovConfig = field(default_factory=KLCovConfig)
    cispo: CISPOConfig = field(default_factory=CISPOConfig)
    max_seq_len: Optional[int] = None


# ---------------------------------------------------------------------------
# Fully Async
# ---------------------------------------------------------------------------


@dataclass
class FullyAsyncConfig(BaseConfig):
    max_staleness_steps: int = 4
    num_parallel_generation_workers: int = 768


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
    logprobs: Optional[int] = 0
    stop: Optional[List[str]] = None
    additional_kwargs: Optional[Dict[str, Any]] = None


@dataclass
class ChatTemplateConfig(BaseConfig):
    source: str = "name"
    name_or_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


@dataclass
class GeneratorConfig(BaseConfig):
    model_name: str = ""
    model_dtype: str = "bfloat16"
    run_engines_locally: bool = True
    num_inference_engines: int = 1
    backend: str = "vllm"
    weight_sync_backend: str = "nccl"
    weight_transfer_threshold_cuda_ipc_GB: float = 1.0
    inference_engine_tensor_parallel_size: int = 4
    inference_engine_pipeline_parallel_size: int = 1
    inference_engine_expert_parallel_size: int = 1
    inference_engine_data_parallel_size: int = 1
    n_samples_per_prompt: int = 5
    async_engine: bool = True
    batched: bool = False
    max_input_length: int = 512
    vllm_v1_disable_multiproc: bool = True
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    max_num_batched_tokens: int = 8192
    enforce_eager: bool = True
    fully_sharded_loras: bool = False
    enable_ray_prometheus_stats: bool = False
    gpu_memory_utilization: float = 0.8
    max_num_seqs: int = 1024
    remote_inference_engine_urls: List[str] = field(default_factory=lambda: ["127.0.0.1:8001"])
    enable_http_endpoint: bool = False
    http_endpoint_host: str = "127.0.0.1"
    http_endpoint_port: int = 8000
    served_model_name: Optional[str] = None
    max_turns: int = 1
    chat_template: ChatTemplateConfig = field(default_factory=ChatTemplateConfig)
    chat_template_kwargs: Dict[str, Any] = field(default_factory=dict)
    engine_init_kwargs: Dict[str, Any] = field(default_factory=dict)
    override_existing_update_group: str = "auto"
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    use_conversation_multi_turn: bool = True
    append_eos_token_after_stop_str_in_multi_turn: bool = True
    eval_sampling_params: SamplingParams = field(default_factory=lambda: SamplingParams(temperature=0.0))
    eval_n_samples_per_prompt: int = 1
    zero_reward_on_non_stop: bool = False
    apply_overlong_filtering: bool = False
    rope_scaling: Optional[Dict[str, Any]] = None
    rope_theta: Optional[float] = None
    step_wise_trajectories: bool = False

    external_proxy_url: Optional[str] = None
    external_server_urls: Optional[List[str]] = None


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
    resume_path: Optional[str] = None
    ckpt_path: str = ""
    max_ckpts_to_keep: int = -1
    ckpt_interval: int = 10
    hf_save_interval: int = -1
    export_path: str = ""
    bf16: bool = True
    epochs: int = 1
    update_epochs_per_batch: int = 1
    train_batch_size: int = 1024
    policy_mini_batch_size: int = 256
    critic_mini_batch_size: int = 256
    micro_train_batch_size_per_gpu: int = 1
    micro_forward_batch_size_per_gpu: int = 1
    update_ref_every_epoch: bool = False
    use_sample_packing: bool = True
    eval_batch_size: int = 1024
    eval_before_train: bool = True
    eval_interval: int = 5
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


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass
class SkyRLConfig(BaseConfig):
    data: DataConfig = field(default_factory=DataConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)


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


def get_config_as_dict(cfg: Union[dict, BaseConfig, DictConfig]) -> dict:
    if isinstance(cfg, dict):
        return cfg
    elif isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)
    return asdict(cfg)


def get_config_as_yaml_str(cfg: Union[BaseConfig, DictConfig]) -> str:
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_yaml(cfg)
    return yaml.dump(asdict(cfg))

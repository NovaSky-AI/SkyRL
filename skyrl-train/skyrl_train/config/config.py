"""
Typed configuration dataclasses for SkyRL.

These mirror the YAML configuration structure 1:1. The top-level SkyRLConfig
can be constructed from a Hydra DictConfig via SkyRLConfig.from_dict_config().
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from omegaconf import DictConfig, OmegaConf

from skyrl_gym.envs.search.env import SearchEnvConfig
from skyrl_gym.envs.sql.env import Text2SQLEnvConfig

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    train_data: List[str] = field(default_factory=list)
    val_data: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Model / LoRA
# ---------------------------------------------------------------------------


@dataclass
class LoraConfig:
    rank: int = 0
    alpha: int = 16
    dropout: float = 0.0
    lora_sync_path: str = "/tmp/skyrl_lora_sync"
    target_modules: str = "all-linear"
    exclude_modules: Optional[str] = None
    init_method: str = "kaiming"


@dataclass
class ModelConfig:
    path: str = "Qwen/Qwen2.5-1.5B-Instruct"
    lora: Optional[LoraConfig] = None


# ---------------------------------------------------------------------------
# Optimizer / FSDP
# ---------------------------------------------------------------------------


@dataclass
class OptimizerConfig:
    lr: float = 1e-6
    adam_betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    offload_after_step: bool = True
    num_warmup_steps: int = 0
    scheduler: str = "constant_with_warmup"


@dataclass
class MixedPrecisionConfig:
    param_dtype: str = "bf16"
    reduce_dtype: str = "fp32"
    buffer_dtype: str = "fp32"


@dataclass
class FSDPConfig:
    cpu_offload: bool = False
    reshard_after_forward: Union[bool, int] = True
    fsdp_size: int = -1
    mixed_precision: Optional[MixedPrecisionConfig] = None
    wrap_policy: Optional[str] = None


# ---------------------------------------------------------------------------
# Megatron
# ---------------------------------------------------------------------------


@dataclass
class MegatronDDPConfig:
    grad_reduce_in_fp32: bool = True
    overlap_grad_reduce: bool = False
    overlap_param_gather: bool = False
    average_in_collective: bool = True


@dataclass
class MegatronTorchProfilerConfig:
    enable: bool = False
    ranks: List[int] = field(default_factory=list)
    save_path: Optional[str] = None


@dataclass
class MegatronLoraConfig:
    lora_type: str = "lora"


@dataclass
class MegatronOptimizerKwargs:
    overlap_cpu_optimizer_d2h_h2d: bool = False
    use_precision_aware_optimizer: bool = False
    optimizer_cpu_offload: bool = False
    optimizer_offload_fraction: float = 0.0


@dataclass
class MegatronTransformerKwargs:
    recompute_granularity: Optional[str] = "full"
    recompute_modules: Optional[List[str]] = field(default_factory=lambda: ["core_attn"])
    recompute_method: Optional[str] = "uniform"
    recompute_num_layers: Optional[int] = 1
    num_layers: Optional[int] = None
    num_layers_in_last_pipeline_stage: Optional[int] = None


@dataclass
class MegatronConfig:
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: Optional[int] = None
    ddp_config: Optional[MegatronDDPConfig] = None
    torch_profiler_config: Optional[MegatronTorchProfilerConfig] = None
    lora_config: Optional[MegatronLoraConfig] = None
    optimizer_config_kwargs: Optional[MegatronOptimizerKwargs] = None
    transformer_config_kwargs: Optional[MegatronTransformerKwargs] = None
    empty_cuda_cache: Optional[bool] = None
    model_config_kwargs: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------


@dataclass
class PlacementConfig:
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
class PolicyConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)
    sequence_parallel_size: int = 1
    use_torch_compile: bool = False
    record_memory: bool = False
    megatron_config: MegatronConfig = field(default_factory=MegatronConfig)
    model_config_kwargs: dict = field(default_factory=dict)


@dataclass
class CriticConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)
    sequence_parallel_size: int = 1
    model_config_kwargs: dict = field(default_factory=dict)


@dataclass
class RefConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    sequence_parallel_size: int = 1
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)
    megatron_config: MegatronConfig = field(default_factory=MegatronConfig)
    model_config_kwargs: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------


@dataclass
class KLCtrlConfig:
    type: str = "fixed"
    kl_target: float = 0.1
    horizon: int = 10000


@dataclass
class SAPOConfig:
    tau_pos: float = 1.0
    tau_neg: float = 1.05


@dataclass
class DynamicSamplingConfig:
    type: Optional[str] = None
    max_sample_batches: int = 30
    min_replace_ratio: float = 0.3


@dataclass
class ClipCovConfig:
    clip_ratio: float = 0.0002
    clip_cov_lb: float = 1.0
    clip_cov_ub: float = 5.0


@dataclass
class KLCovConfig:
    kl_cov_frac: float = 0.2
    ppo_kl_coef: float = 1.0


@dataclass
class CISPOConfig:
    cispo_eps_clip_low: float = 0.0
    cispo_eps_clip_high: float = 5.0


@dataclass
class AlgorithmConfig:
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
class FullyAsyncConfig:
    max_staleness_steps: int = 4
    num_parallel_generation_workers: int = 768


# ---------------------------------------------------------------------------
# Sampling / Chat Template
# ---------------------------------------------------------------------------


@dataclass
class SamplingParams:
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
class ChatTemplateConfig:
    source: str = "name"
    name_or_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


@dataclass
class GeneratorConfig:
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


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


# redefinition of Judge Env configuration because this is currently only available in examples/
@dataclass
class GSM8kLLMJudgeEnvConfig:
    model: str = "gpt-4o-mini"
    base_url: Optional[str] = None


@dataclass
class SkyRLGymConfig:
    max_env_workers: int = 32
    text2sql: Text2SQLEnvConfig = field(default_factory=Text2SQLEnvConfig)
    llm_as_a_judge: GSM8kLLMJudgeEnvConfig = field(default_factory=GSM8kLLMJudgeEnvConfig)
    search: SearchEnvConfig = field(default_factory=SearchEnvConfig)


@dataclass
class EnvironmentConfig:
    env_class: str = "gsm8k"
    skyrl_gym: SkyRLGymConfig = field(default_factory=SkyRLGymConfig)


# ---------------------------------------------------------------------------
# Trainer (top-level)
# ---------------------------------------------------------------------------


@dataclass
class TrainerConfig:
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
class SkyRLConfig:
    data: DataConfig = field(default_factory=DataConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)

    @classmethod
    def from_dict_config(cls, cfg: DictConfig) -> "SkyRLConfig":
        """Construct a typed SkyRLConfig from a Hydra DictConfig."""
        raw = OmegaConf.to_container(cfg, resolve=True)
        return _build_skyrl_config(raw)


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------


def _build_flat(datacls, d: dict):
    """Build a flat dataclass from a dict, filtering to valid fields only."""
    import dataclasses

    valid_fields = {f.name for f in dataclasses.fields(datacls)}
    filtered = {k: v for k, v in d.items() if k in valid_fields}
    return datacls(**filtered)


def _build_model_config(d: dict) -> ModelConfig:
    lora_d = d.get("lora")
    return ModelConfig(
        path=d.get("path"),
        lora=_build_flat(LoraConfig, lora_d) if lora_d else None,
    )


def _build_megatron_config(d: dict) -> MegatronConfig:
    return MegatronConfig(
        tensor_model_parallel_size=d["tensor_model_parallel_size"],
        pipeline_model_parallel_size=d["pipeline_model_parallel_size"],
        context_parallel_size=d["context_parallel_size"],
        expert_model_parallel_size=d["expert_model_parallel_size"],
        expert_tensor_parallel_size=d.get("expert_tensor_parallel_size"),
        ddp_config=_build_flat(MegatronDDPConfig, d["ddp_config"]) if "ddp_config" in d else None,
        torch_profiler_config=(
            _build_flat(MegatronTorchProfilerConfig, d["torch_profiler_config"])
            if "torch_profiler_config" in d
            else None
        ),
        lora_config=_build_flat(MegatronLoraConfig, d["lora_config"]) if "lora_config" in d else None,
        optimizer_config_kwargs=(
            _build_flat(MegatronOptimizerKwargs, d["optimizer_config_kwargs"])
            if "optimizer_config_kwargs" in d
            else None
        ),
        transformer_config_kwargs=(
            _build_flat(MegatronTransformerKwargs, d["transformer_config_kwargs"])
            if "transformer_config_kwargs" in d and d["transformer_config_kwargs"]
            else None
        ),
        empty_cuda_cache=d.get("empty_cuda_cache"),
        model_config_kwargs=d.get("model_config_kwargs", {}),
    )


def _build_algorithm_config(d: dict) -> AlgorithmConfig:
    nested = {
        "kl_ctrl": _build_flat(KLCtrlConfig, d["kl_ctrl"]),
        "sapo": _build_flat(SAPOConfig, d["sapo"]),
        "dynamic_sampling": _build_flat(DynamicSamplingConfig, d["dynamic_sampling"]),
        "clip_cov": _build_flat(ClipCovConfig, d["clip_cov"]),
        "kl_cov": _build_flat(KLCovConfig, d["kl_cov"]),
        "cispo": _build_flat(CISPOConfig, d["cispo"]),
    }
    flat_keys = {f.name for f in AlgorithmConfig.__dataclass_fields__.values()} - set(nested.keys())
    flat = {k: d[k] for k in flat_keys if k in d}
    return AlgorithmConfig(**flat, **nested)


def _build_generator_config(d: dict) -> GeneratorConfig:
    nested = {
        "chat_template": _build_flat(ChatTemplateConfig, d["chat_template"]),
        "sampling_params": _build_flat(SamplingParams, d["sampling_params"]),
        "eval_sampling_params": _build_flat(SamplingParams, d["eval_sampling_params"]),
    }
    flat_keys = {f.name for f in GeneratorConfig.__dataclass_fields__.values()} - set(nested.keys())
    flat = {k: d[k] for k in flat_keys if k in d}
    return GeneratorConfig(**flat, **nested)


def _build_policy_config(d: dict) -> PolicyConfig:
    return PolicyConfig(
        model=_build_model_config(d["model"]),
        optimizer_config=_build_flat(OptimizerConfig, d["optimizer_config"]),
        fsdp_config=_build_flat(FSDPConfig, d["fsdp_config"]),
        sequence_parallel_size=d["sequence_parallel_size"],
        use_torch_compile=d["use_torch_compile"],
        record_memory=d["record_memory"],
        megatron_config=_build_megatron_config(d["megatron_config"]),
    )


def _build_critic_config(d: dict) -> CriticConfig:
    return CriticConfig(
        model=_build_model_config(d["model"]),
        optimizer_config=_build_flat(OptimizerConfig, d["optimizer_config"]),
        fsdp_config=_build_flat(FSDPConfig, d["fsdp_config"]),
        sequence_parallel_size=d["sequence_parallel_size"],
    )


def _build_ref_config(d: dict) -> RefConfig:
    return RefConfig(
        model=_build_model_config(d["model"]),
        sequence_parallel_size=d["sequence_parallel_size"],
        fsdp_config=_build_flat(FSDPConfig, d["fsdp_config"]),
        megatron_config=_build_megatron_config(d["megatron_config"]),
    )


def _build_environment_config(d: dict) -> EnvironmentConfig:
    skyrl_gym_d = d["skyrl_gym"]
    skyrl_gym = SkyRLGymConfig(
        max_env_workers=skyrl_gym_d["max_env_workers"],
        text2sql=_build_flat(Text2SQLEnvConfig, skyrl_gym_d["text2sql"]),
        llm_as_a_judge=_build_flat(GSM8kLLMJudgeEnvConfig, skyrl_gym_d["llm_as_a_judge"]),
        search=_build_flat(SearchEnvConfig, skyrl_gym_d["search"]),
    )
    return EnvironmentConfig(
        env_class=d["env_class"],
        skyrl_gym=skyrl_gym,
    )


def _build_trainer_config(d: dict) -> TrainerConfig:
    nested = {
        "placement": _build_flat(PlacementConfig, d["placement"]),
        "policy": _build_policy_config(d["policy"]),
        "ref": _build_ref_config(d["ref"]),
        "critic": _build_critic_config(d["critic"]),
        "algorithm": _build_algorithm_config(d["algorithm"]),
        "fully_async": _build_flat(FullyAsyncConfig, d["fully_async"]),
    }
    flat_keys = {f.name for f in TrainerConfig.__dataclass_fields__.values()} - set(nested.keys())
    flat = {k: d[k] for k in flat_keys if k in d}
    return TrainerConfig(**flat, **nested)


def _build_skyrl_config(raw: dict) -> SkyRLConfig:
    return SkyRLConfig(
        data=_build_flat(DataConfig, raw["data"]),
        trainer=_build_trainer_config(raw["trainer"]),
        generator=_build_generator_config(raw["generator"]),
        environment=_build_environment_config(raw["environment"]),
    )

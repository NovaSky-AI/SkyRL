"""
Pydantic configuration models for SkyRL training.

This module provides type-safe Pydantic models that mirror the YAML configuration
structure in ppo_base_config.yaml. These models enable:
1. Type checking and IDE autocomplete for configuration
2. Python-based configuration alongside YAML
3. Future validation logic migration from validate_cfg()

The hierarchy matches the YAML structure exactly for 1:1 mapping.
"""

from typing import Optional, List, Dict, Any, Literal, Union
from pydantic import BaseModel, Field


# ============================================================================
# LoRA Configuration
# ============================================================================

class LoRAConfig(BaseModel):
    """LoRA (Low-Rank Adaptation) configuration."""
    rank: int = Field(default=0, description="LoRA rank, 0 to disable")
    alpha: int = Field(default=16, description="LoRA alpha parameter")
    dropout: float = Field(default=0.0, description="LoRA dropout rate")
    lora_sync_path: Optional[str] = Field(default="/tmp/skyrl_lora_sync", description="Path for LoRA sync")
    target_modules: Optional[str] = Field(default="all-linear", description="Target modules for LoRA")
    exclude_modules: Optional[str] = Field(default=None, description="Modules to exclude from LoRA")
    init_method: str = Field(default="kaiming", description="Initialization method")


# ============================================================================
# Model Configuration
# ============================================================================

class ModelConfig(BaseModel):
    """Model configuration for policy, reference, or critic models."""
    path: str = Field(description="HuggingFace model path or local path")
    lora: LoRAConfig = Field(default_factory=LoRAConfig, description="LoRA configuration")


class CriticModelConfig(BaseModel):
    """Critic model configuration (no lora_sync_path)."""
    path: Optional[str] = Field(default=None, description="HuggingFace model path or local path")
    lora: LoRAConfig = Field(default_factory=LoRAConfig, description="LoRA configuration")


# ============================================================================
# Optimizer Configuration
# ============================================================================

class OptimizerConfig(BaseModel):
    """Optimizer configuration for policy or critic."""
    lr: float = Field(description="Learning rate")
    adam_betas: List[float] = Field(default=[0.9, 0.999], description="Adam beta parameters")
    weight_decay: float = Field(default=1e-2, description="Weight decay coefficient")
    max_grad_norm: float = Field(default=1.0, description="Max gradient norm for clipping")
    offload_after_step: bool = Field(default=True, description="Offload optimizer state to CPU after step")
    num_warmup_steps: int = Field(default=0, description="Number of warmup steps")
    scheduler: str = Field(default="constant_with_warmup", description="Learning rate scheduler type")


# ============================================================================
# FSDP Configuration
# ============================================================================

class FSDPConfig(BaseModel):
    """FSDP (Fully Sharded Data Parallel) configuration."""
    cpu_offload: bool = Field(default=False, description="Offload params + optimizer state to CPU")
    reshard_after_forward: Union[bool, int] = Field(default=True, description="Reshard after forward pass")
    fsdp_size: int = Field(default=-1, description="FSDP sharding size, -1 for auto")


# ============================================================================
# Placement Configuration
# ============================================================================

class PlacementConfig(BaseModel):
    """Resource placement configuration for distributed training."""
    colocate_all: bool = Field(default=True, description="Colocate all components")
    colocate_policy_ref: bool = Field(default=True, description="Colocate policy and reference models")
    policy_num_nodes: int = Field(default=1, description="Number of nodes for policy")
    policy_num_gpus_per_node: int = Field(default=4, description="GPUs per node for policy")
    critic_num_nodes: int = Field(default=1, description="Number of nodes for critic")
    critic_num_gpus_per_node: int = Field(default=4, description="GPUs per node for critic")
    ref_num_nodes: int = Field(default=1, description="Number of nodes for reference")
    ref_num_gpus_per_node: int = Field(default=4, description="GPUs per node for reference")


# ============================================================================
# Policy Configuration
# ============================================================================

class PolicyConfig(BaseModel):
    """Policy model configuration."""
    model: ModelConfig = Field(description="Policy model configuration")
    optimizer_config: OptimizerConfig = Field(description="Optimizer configuration")
    fsdp_config: FSDPConfig = Field(default_factory=FSDPConfig, description="FSDP configuration")
    sequence_parallel_size: int = Field(default=1, description="Sequence parallelism size")
    use_torch_compile: bool = Field(default=False, description="Use torch.compile for logits")
    record_memory: bool = Field(default=False, description="Record memory snapshots")
    model_config_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Model config kwargs")


# ============================================================================
# Reference Model Configuration
# ============================================================================

class RefConfig(BaseModel):
    """Reference model configuration."""
    model: ModelConfig = Field(description="Reference model configuration")
    sequence_parallel_size: int = Field(default=1, description="Sequence parallelism size")
    fsdp_config: FSDPConfig = Field(default_factory=FSDPConfig, description="FSDP configuration")
    model_config_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Model config kwargs")


# ============================================================================
# Critic Configuration
# ============================================================================

class CriticConfig(BaseModel):
    """Critic model configuration."""
    model: CriticModelConfig = Field(description="Critic model configuration")
    optimizer_config: OptimizerConfig = Field(description="Optimizer configuration")
    fsdp_config: FSDPConfig = Field(default_factory=FSDPConfig, description="FSDP configuration")
    sequence_parallel_size: int = Field(default=1, description="Sequence parallelism size")
    model_config_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Model config kwargs")


# ============================================================================
# Algorithm Configuration
# ============================================================================

class KLCtrlConfig(BaseModel):
    """KL divergence controller configuration."""
    type: Literal["fixed", "adaptive"] = Field(default="fixed", description="KL controller type")
    kl_target: float = Field(default=0.1, description="Target KL divergence")
    horizon: int = Field(default=10000, description="Horizon for adaptive controller")


class SAPOConfig(BaseModel):
    """SAPO algorithm specific parameters."""
    tau_pos: float = Field(default=1.0, description="Positive tau parameter")
    tau_neg: float = Field(default=1.05, description="Negative tau parameter")


class DynamicSamplingConfig(BaseModel):
    """Dynamic sampling configuration."""
    type: Optional[Literal["filter", "replace"]] = Field(default=None, description="Dynamic sampling type")
    max_sample_batches: int = Field(default=30, description="Max batches to sample")
    min_replace_ratio: float = Field(default=0.3, description="Min proportion for replacement")


class ClipCovConfig(BaseModel):
    """Clip-Cov algorithm specific parameters."""
    clip_ratio: float = Field(default=0.0002, description="Fraction of tokens to clip")
    clip_cov_lb: float = Field(default=1.0, description="Lower bound for covariance clipping")
    clip_cov_ub: float = Field(default=5.0, description="Upper bound for covariance clipping")


class KLCovConfig(BaseModel):
    """KL-Cov algorithm specific parameters."""
    kl_cov_frac: float = Field(default=0.2, description="Percentage of tokens for KL regularization")
    ppo_kl_coef: float = Field(default=1.0, description="PPO KL coefficient")


class CISPOConfig(BaseModel):
    """CISPO algorithm specific parameters."""
    cispo_eps_clip_low: float = Field(default=0, description="Lower bound offset for IS ratio clipping")
    cispo_eps_clip_high: float = Field(default=5, description="Upper bound offset for IS ratio clipping")


class AlgorithmConfig(BaseModel):
    """RL algorithm configuration."""
    advantage_estimator: str = Field(default="grpo", description="Advantage estimator type")
    kl_ctrl: KLCtrlConfig = Field(default_factory=KLCtrlConfig, description="KL controller config")
    kl_estimator_type: Literal["k1", "k2", "k3", "abs"] = Field(default="k3", description="KL estimator type")
    use_kl_in_reward: bool = Field(default=False, description="Apply KL loss to rewards")
    use_kl_loss: bool = Field(default=True, description="Use KL loss in policy")
    kl_loss_coef: float = Field(default=0.001, description="KL loss coefficient")
    use_entropy_loss: bool = Field(default=False, description="Use entropy loss")
    entropy_loss_coef: float = Field(default=0.01, description="Entropy loss coefficient")
    advantage_batch_normalize: bool = Field(default=False, description="Normalize advantages per batch")
    value_head_prefix: str = Field(default="value_head", description="Value head prefix")
    policy_loss_type: str = Field(default="regular", description="Policy loss type")
    loss_reduction: Literal["token_mean", "sequence_mean", "seq_mean_token_sum_norm"] = Field(
        default="token_mean", description="Loss reduction method"
    )
    grpo_norm_by_std: bool = Field(default=True, description="Normalize by std in GRPO")
    zero_variance_filter: bool = Field(default=False, description="Filter zero variance prompts")
    lambd: float = Field(default=1.0, description="GAE lambda parameter")
    gamma: float = Field(default=1.0, description="Discount factor")
    eps_clip_low: float = Field(default=0.2, description="PPO epsilon clip lower bound")
    eps_clip_high: float = Field(default=0.2, description="PPO epsilon clip upper bound")
    clip_ratio_c: float = Field(default=3.0, description="Dual clip ratio C")
    tis_imp_ratio_cap: float = Field(default=-1.0, description="TIS importance ratio cap")
    use_tis: bool = Field(default=False, description="Use Truncated Importance Sampling")
    sapo: SAPOConfig = Field(default_factory=SAPOConfig, description="SAPO parameters")
    value_clip: float = Field(default=0.2, description="Value function clipping")
    dynamic_sampling: DynamicSamplingConfig = Field(
        default_factory=DynamicSamplingConfig, description="Dynamic sampling config"
    )
    clip_cov: ClipCovConfig = Field(default_factory=ClipCovConfig, description="Clip-Cov parameters")
    kl_cov: KLCovConfig = Field(default_factory=KLCovConfig, description="KL-Cov parameters")
    cispo: CISPOConfig = Field(default_factory=CISPOConfig, description="CISPO parameters")


# ============================================================================
# Fully Async Configuration
# ============================================================================

class FullyAsyncConfig(BaseModel):
    """Fully asynchronous training configuration."""
    max_staleness_steps: int = Field(default=4, description="Maximum off-policy staleness steps")
    num_parallel_generation_workers: int = Field(
        default=768, description="Number of parallel generation workers"
    )


# ============================================================================
# Trainer Configuration
# ============================================================================

class TrainerConfig(BaseModel):
    """Main trainer configuration."""
    placement: PlacementConfig = Field(description="Resource placement configuration")
    sequence_parallel_backend: str = Field(default="ulysses", description="Sequence parallel backend")
    strategy: Literal["fsdp", "fsdp2", "megatron"] = Field(default="fsdp2", description="Training strategy")
    policy: PolicyConfig = Field(description="Policy configuration")
    ref: RefConfig = Field(description="Reference model configuration")
    critic: CriticConfig = Field(description="Critic configuration")
    algorithm: AlgorithmConfig = Field(description="RL algorithm configuration")
    fully_async: FullyAsyncConfig = Field(default_factory=FullyAsyncConfig, description="Fully async config")
    gradient_checkpointing: bool = Field(default=True, description="Use gradient checkpointing")
    gradient_checkpointing_use_reentrant: bool = Field(default=False, description="Use reentrant checkpointing")
    seed: int = Field(default=42, description="Random seed")
    resume_mode: Optional[Literal["latest", "from_path"]] = Field(default=None, description="Resume mode")
    resume_path: Optional[str] = Field(default=None, description="Resume from this path")
    ckpt_path: str = Field(description="Checkpoint save path")
    max_ckpts_to_keep: int = Field(default=-1, description="Max checkpoints to keep, -1 for all")
    ckpt_interval: int = Field(default=10, description="Checkpoint save interval")
    hf_save_interval: int = Field(default=-1, description="HuggingFace format save interval")
    export_path: str = Field(description="Export path for artifacts")
    bf16: bool = Field(default=True, description="Use bfloat16")
    epochs: int = Field(default=1, description="Number of training epochs")
    update_epochs_per_batch: int = Field(default=1, description="Update epochs per batch")
    train_batch_size: int = Field(default=1024, description="Training batch size")
    policy_mini_batch_size: int = Field(default=256, description="Policy mini-batch size")
    critic_mini_batch_size: int = Field(default=256, description="Critic mini-batch size")
    micro_train_batch_size_per_gpu: int = Field(default=1, description="Micro train batch size per GPU")
    micro_forward_batch_size_per_gpu: int = Field(default=1, description="Micro forward batch size per GPU")
    update_ref_every_epoch: bool = Field(default=False, description="Update reference model every epoch")
    use_sample_packing: bool = Field(default=True, description="Use sample packing")
    eval_batch_size: int = Field(default=1024, description="Evaluation batch size")
    eval_before_train: bool = Field(default=True, description="Evaluate before training")
    eval_interval: int = Field(default=5, description="Evaluation interval, -1 to disable")
    max_prompt_length: int = Field(default=512, description="Maximum prompt length")
    flash_attn: bool = Field(default=True, description="Use flash attention")
    disable_fast_tokenizer: bool = Field(default=False, description="Disable fast tokenizer")
    project_name: str = Field(default="skyrl", description="Project name for logging")
    run_name: str = Field(default="test_run", description="Run name for logging")
    logger: str = Field(default="wandb", description="Logger backend")
    dump_data_batch: bool = Field(default=False, description="Dump data batches for debugging")
    dump_eval_results: bool = Field(default=True, description="Dump evaluation results")
    rope_scaling: Optional[Dict[str, Any]] = Field(default=None, description="RoPE scaling config")
    rope_theta: Optional[float] = Field(default=None, description="RoPE theta parameter")


# ============================================================================
# Generator Configuration
# ============================================================================

class ChatTemplateConfig(BaseModel):
    """Chat template configuration."""
    source: Literal["name", "file"] = Field(default="name", description="Template source type")
    name_or_path: Optional[str] = Field(default=None, description="Template name or file path")


class SamplingParamsConfig(BaseModel):
    """Sampling parameters for generation."""
    max_generate_length: int = Field(default=1024, description="Maximum generation length")
    repetition_penalty: float = Field(default=1.0, description="Repetition penalty")
    temperature: float = Field(default=1.0, description="Sampling temperature")
    top_p: float = Field(default=1.0, description="Top-p sampling parameter")
    min_p: float = Field(default=0.0, description="Min-p sampling parameter")
    top_k: int = Field(default=-1, description="Top-k sampling parameter")
    logprobs: int = Field(default=0, description="Number of log probabilities to return")
    stop: Optional[Union[str, List[str]]] = Field(default=None, description="Stop sequences")


class GeneratorConfig(BaseModel):
    """Generator (inference engine) configuration."""
    model_name: str = Field(description="Model name for generation")
    model_dtype: str = Field(default="bfloat16", description="Model dtype")
    run_engines_locally: bool = Field(default=True, description="Run inference engines locally")
    num_inference_engines: int = Field(default=1, description="Number of inference engines")
    backend: Literal["vllm", "sglang"] = Field(default="vllm", description="Inference backend")
    weight_sync_backend: str = Field(default="nccl", description="Weight sync backend")
    weight_transfer_threshold_cuda_ipc_GB: float = Field(
        default=1.0, description="CUDA IPC transfer threshold in GB"
    )
    inference_engine_tensor_parallel_size: int = Field(default=4, description="Tensor parallel size")
    inference_engine_pipeline_parallel_size: int = Field(default=1, description="Pipeline parallel size")
    inference_engine_expert_parallel_size: int = Field(default=1, description="Expert parallel size")
    inference_engine_data_parallel_size: int = Field(default=1, description="Data parallel size")
    n_samples_per_prompt: int = Field(default=5, description="Samples per prompt")
    async_engine: bool = Field(default=True, description="Use async inference engine")
    batched: bool = Field(default=False, description="Use batched generation")
    max_input_length: int = Field(description="Maximum input length")
    vllm_v1_disable_multiproc: bool = Field(default=True, description="Disable vLLM v1 multiprocessing")
    enable_prefix_caching: bool = Field(default=True, description="Enable prefix caching")
    enable_chunked_prefill: bool = Field(default=True, description="Enable chunked prefill")
    max_num_batched_tokens: int = Field(default=8192, description="Max batched tokens")
    enforce_eager: bool = Field(default=True, description="Enforce eager execution")
    fully_sharded_loras: bool = Field(default=False, description="Fully shard LoRAs")
    gpu_memory_utilization: float = Field(default=0.8, description="GPU memory utilization")
    max_num_seqs: int = Field(default=1024, description="Max number of sequences")
    remote_inference_engine_urls: List[str] = Field(
        default=["127.0.0.1:8001"], description="Remote inference engine URLs"
    )
    enable_http_endpoint: bool = Field(default=False, description="Enable HTTP endpoint")
    http_endpoint_host: str = Field(default="127.0.0.1", description="HTTP endpoint host")
    http_endpoint_port: int = Field(default=8000, description="HTTP endpoint port")
    max_turns: int = Field(default=1, description="Maximum conversation turns")
    chat_template: ChatTemplateConfig = Field(
        default_factory=ChatTemplateConfig, description="Chat template config"
    )
    chat_template_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Chat template kwargs")
    engine_init_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Engine initialization kwargs")
    override_existing_update_group: Literal["auto", "enable", "disable"] = Field(
        default="auto", description="Override existing update group"
    )
    sampling_params: SamplingParamsConfig = Field(description="Sampling parameters")
    use_conversation_multi_turn: bool = Field(default=True, description="Use conversation format for multi-turn")
    append_eos_token_after_stop_str_in_multi_turn: bool = Field(
        default=True, description="Append EOS token after stop string in multi-turn"
    )
    eval_sampling_params: SamplingParamsConfig = Field(description="Evaluation sampling parameters")
    eval_n_samples_per_prompt: int = Field(default=1, description="Eval samples per prompt")
    zero_reward_on_non_stop: bool = Field(default=False, description="Zero reward on non-stop generation")
    apply_overlong_filtering: bool = Field(default=False, description="Apply overlong filtering")
    rope_scaling: Optional[Dict[str, Any]] = Field(default=None, description="RoPE scaling config")
    rope_theta: Optional[float] = Field(default=None, description="RoPE theta parameter")
    step_wise_trajectories: bool = Field(default=False, description="Use step-wise trajectories")


# ============================================================================
# Environment Configuration
# ============================================================================

class Text2SQLConfig(BaseModel):
    """Text-to-SQL environment configuration."""
    db_path: str = Field(default="/home/ray/default/sql_data", description="Database path")


class LLMAsAJudgeConfig(BaseModel):
    """LLM-as-a-Judge environment configuration."""
    model: str = Field(default="gpt-4o-mini", description="Judge model name")
    base_url: Optional[str] = Field(default=None, description="API base URL")


class SearchConfig(BaseModel):
    """Search environment configuration."""
    log_requests: bool = Field(default=False, description="Log search requests")
    search_url: str = Field(default="http://127.0.0.1:8000/retrieve", description="Search service URL")
    topk: int = Field(default=3, description="Top-k results to retrieve")
    timeout: int = Field(default=30, description="Request timeout in seconds")


class SkyRLGymConfig(BaseModel):
    """SkyRL Gym environment configuration."""
    max_env_workers: int = Field(default=32, description="Max background workers for env step calls")
    text2sql: Text2SQLConfig = Field(default_factory=Text2SQLConfig, description="Text2SQL config")
    llm_as_a_judge: LLMAsAJudgeConfig = Field(
        default_factory=LLMAsAJudgeConfig, description="LLM-as-a-Judge config"
    )
    search: SearchConfig = Field(default_factory=SearchConfig, description="Search config")


class EnvironmentConfig(BaseModel):
    """Environment configuration."""
    env_class: str = Field(default="gsm8k", description="Environment class name")
    skyrl_gym: SkyRLGymConfig = Field(default_factory=SkyRLGymConfig, description="SkyRL Gym config")


# ============================================================================
# Megatron Configuration
# ============================================================================

class MegatronDDPConfig(BaseModel):
    """Megatron DDP configuration."""
    grad_reduce_in_fp32: bool = Field(default=True, description="Reduce gradients in FP32")
    overlap_grad_reduce: bool = Field(default=False, description="Overlap gradient reduction")
    overlap_param_gather: bool = Field(default=False, description="Overlap parameter gather")
    average_in_collective: bool = Field(default=True, description="Average in collective")


class MegatronLoRAConfig(BaseModel):
    """Megatron LoRA configuration."""
    lora_type: Literal["lora", "canonical_lora"] = Field(default="lora", description="LoRA type")


class MegatronTorchProfilerConfig(BaseModel):
    """Megatron torch profiler configuration."""
    enable: bool = Field(default=False, description="Enable profiler")
    ranks: List[int] = Field(default_factory=list, description="Ranks to profile")
    save_path: Optional[str] = Field(default=None, description="Profiler save path")


class MegatronOptimizerConfigKwargs(BaseModel):
    """Megatron optimizer config kwargs."""
    overlap_cpu_optimizer_d2h_h2d: bool = Field(default=False, description="Overlap CPU optimizer D2H/H2D")
    use_precision_aware_optimizer: bool = Field(default=False, description="Use precision-aware optimizer")
    optimizer_cpu_offload: bool = Field(default=False, description="Offload optimizer to CPU")
    optimizer_offload_fraction: float = Field(default=0.0, description="Optimizer offload fraction")


class MegatronTransformerConfigKwargs(BaseModel):
    """Megatron transformer config kwargs."""
    recompute_granularity: str = Field(default="full", description="Recompute granularity")
    recompute_modules: List[str] = Field(default=["core_attn"], description="Modules to recompute")
    recompute_method: str = Field(default="uniform", description="Recompute method")
    recompute_num_layers: int = Field(default=1, description="Number of layers to recompute")


class MegatronConfig(BaseModel):
    """Megatron-specific configuration."""
    tensor_model_parallel_size: int = Field(default=1, description="Tensor model parallel size")
    pipeline_model_parallel_size: int = Field(default=1, description="Pipeline model parallel size")
    context_parallel_size: int = Field(default=1, description="Context parallel size")
    expert_model_parallel_size: int = Field(default=1, description="Expert model parallel size")
    expert_tensor_parallel_size: Optional[int] = Field(default=None, description="Expert tensor parallel size")
    ddp_config: MegatronDDPConfig = Field(default_factory=MegatronDDPConfig, description="DDP config")
    model_config_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Model config kwargs")
    torch_profiler_config: MegatronTorchProfilerConfig = Field(
        default_factory=MegatronTorchProfilerConfig, description="Torch profiler config"
    )
    lora_config: MegatronLoRAConfig = Field(default_factory=MegatronLoRAConfig, description="LoRA config")
    optimizer_config_kwargs: MegatronOptimizerConfigKwargs = Field(
        default_factory=MegatronOptimizerConfigKwargs, description="Optimizer config kwargs"
    )
    transformer_config_kwargs: MegatronTransformerConfigKwargs = Field(
        default_factory=MegatronTransformerConfigKwargs, description="Transformer config kwargs"
    )
    empty_cuda_cache: bool = Field(default=True, description="Empty CUDA cache between passes")


# ============================================================================
# Data Configuration
# ============================================================================

class DataConfig(BaseModel):
    """Data configuration."""
    train_data: List[str] = Field(description="Training data paths")
    val_data: List[str] = Field(description="Validation data paths")


# ============================================================================
# Main Configuration
# ============================================================================

class SkyRLConfig(BaseModel):
    """
    Main SkyRL training configuration.

    This is the root configuration object that contains all training parameters.
    It mirrors the structure of ppo_base_config.yaml exactly for backward compatibility.

    TODO: Migrate validation logic from validate_cfg() to Pydantic validators.
    """
    data: DataConfig = Field(description="Data configuration")
    trainer: TrainerConfig = Field(description="Trainer configuration")
    generator: GeneratorConfig = Field(description="Generator configuration")
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig, description="Environment config")

    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        validate_assignment = True


# ============================================================================
# Helper Functions
# ============================================================================

def _validate_path(path: str) -> List[str]:
    """
    Validate and split a dot-notation path, rejecting unsafe patterns.

    Args:
        path: Dot-separated path string

    Returns:
        List of path components

    Raises:
        ValueError: If path is invalid or contains unsafe patterns
    """
    if not path or not isinstance(path, str):
        raise ValueError(f"Invalid path: {path}")
    parts = path.split(".")
    for part in parts:
        if not part:
            raise ValueError(f"Invalid path with empty component: {path}")
        if part.startswith("_"):
            raise ValueError(f"Access to private/dunder attributes not allowed: {path}")
    return parts


def set_nested_attr(obj: Any, path: str, value: Any) -> None:
    """
    Set a nested attribute on a Pydantic model using dot notation.

    Args:
        obj: The root Pydantic model
        path: Dot-separated path (e.g., "trainer.policy.model.path")
        value: Value to set

    Example:
        >>> cfg = SkyRLConfig(...)
        >>> set_nested_attr(cfg, "trainer.policy.model.path", "Qwen/Qwen2.5-1.5B")
    """
    parts = _validate_path(path)
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def get_nested_attr(obj: Any, path: str) -> Any:
    """
    Get a nested attribute from a Pydantic model using dot notation.

    Args:
        obj: The root Pydantic model
        path: Dot-separated path (e.g., "trainer.policy.model.path")

    Returns:
        The value at the specified path

    Example:
        >>> cfg = SkyRLConfig(...)
        >>> model_path = get_nested_attr(cfg, "trainer.policy.model.path")
    """
    parts = _validate_path(path)
    for part in parts:
        obj = getattr(obj, part)
    return obj


def dictconfig_to_pydantic(dict_cfg: "DictConfig") -> SkyRLConfig:
    """
    Convert an OmegaConf DictConfig to a Pydantic SkyRLConfig.

    This function enables loading from YAML via Hydra and then converting
    to a type-safe Pydantic model.

    Args:
        dict_cfg: OmegaConf DictConfig from Hydra

    Returns:
        SkyRLConfig: Type-safe Pydantic configuration

    Example:
        >>> from omegaconf import DictConfig
        >>> dict_cfg = ... # from Hydra
        >>> pydantic_cfg = dictconfig_to_pydantic(dict_cfg)
    """
    from omegaconf import OmegaConf

    # Convert OmegaConf to plain dict, resolving all interpolations
    plain_dict = OmegaConf.to_container(dict_cfg, resolve=True)

    # Parse into Pydantic model
    return SkyRLConfig(**plain_dict)


def pydantic_to_dictconfig(pydantic_cfg: SkyRLConfig) -> "DictConfig":
    """
    Convert a Pydantic SkyRLConfig to an OmegaConf DictConfig.

    This function enables using Pydantic configs with existing code
    that expects DictConfig.

    Args:
        pydantic_cfg: Pydantic configuration

    Returns:
        DictConfig: OmegaConf configuration compatible with existing code

    Example:
        >>> cfg = SkyRLConfig(...)
        >>> dict_cfg = pydantic_to_dictconfig(cfg)
    """
    from omegaconf import OmegaConf

    # Convert Pydantic model to dict
    config_dict = pydantic_cfg.model_dump()

    # Convert to OmegaConf DictConfig
    return OmegaConf.create(config_dict)


def load_config_from_yaml(config_path: str, overrides: Optional[List[str]] = None) -> SkyRLConfig:
    """
    Load configuration from a YAML file with optional overrides.

    This is a convenience function for loading configs without using Hydra's
    decorator system.

    Args:
        config_path: Path to the YAML config file
        overrides: Optional list of override strings (e.g., ["trainer.lr=1e-5"])

    Returns:
        SkyRLConfig: Type-safe Pydantic configuration

    Example:
        >>> cfg = load_config_from_yaml(
        ...     "config/ppo_base_config.yaml",
        ...     overrides=["trainer.policy.model.path=Qwen/Qwen2.5-1.5B"]
        ... )
    """
    from omegaconf import OmegaConf
    from hydra import compose, initialize_config_dir
    from pathlib import Path
    import os

    config_dir = os.path.dirname(os.path.abspath(config_path))
    config_name = os.path.splitext(os.path.basename(config_path))[0]

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        dict_cfg = compose(config_name=config_name, overrides=overrides or [])

    return dictconfig_to_pydantic(dict_cfg)


def create_default_config() -> SkyRLConfig:
    """
    Create a default SkyRLConfig with minimal required fields.

    This is useful as a starting point for building configs programmatically.

    Returns:
        SkyRLConfig: Default configuration

    Example:
        >>> cfg = create_default_config()
        >>> cfg.trainer.policy.model.path = "Qwen/Qwen2.5-1.5B"
        >>> cfg.data.train_data = ["~/data/gsm8k/train.parquet"]
    """
    import os

    home = os.path.expanduser("~")

    # Create minimal required fields
    return SkyRLConfig(
        data=DataConfig(
            train_data=[f"{home}/data/gsm8k/train.parquet"],
            val_data=[f"{home}/data/gsm8k/validation.parquet"],
        ),
        trainer=TrainerConfig(
            placement=PlacementConfig(),
            policy=PolicyConfig(
                model=ModelConfig(path="Qwen/Qwen2.5-1.5B-Instruct"),
                optimizer_config=OptimizerConfig(lr=1.0e-6),
            ),
            ref=RefConfig(
                model=ModelConfig(path="Qwen/Qwen2.5-1.5B-Instruct"),
            ),
            critic=CriticConfig(
                model=CriticModelConfig(path="Qwen/Qwen2.5-1.5B-Instruct"),
                optimizer_config=OptimizerConfig(lr=5.0e-6),
            ),
            algorithm=AlgorithmConfig(),
            ckpt_path=f"{home}/ckpts/",
            export_path=f"{home}/exports/",
        ),
        generator=GeneratorConfig(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            max_input_length=512,
            sampling_params=SamplingParamsConfig(),
            eval_sampling_params=SamplingParamsConfig(temperature=0.0),
        ),
        environment=EnvironmentConfig(),
    )

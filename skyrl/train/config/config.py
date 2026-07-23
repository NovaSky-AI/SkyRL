"""
Typed configuration dataclasses for SkyRL.

These mirror the YAML configuration structure 1:1. The top-level SkyRLTrainConfig
can be constructed from a Hydra DictConfig via SkyRLTrainConfig.from_dict_config().
"""

import copy
import dataclasses
import os
import typing
from abc import ABC
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional, Type, TypeVar, Union

import yaml
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
class DataLoaderConfig(BaseConfig):
    num_workers: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Prompt DataLoader worker processes. Default of None auto-derives to 8. "
                "Set 0 for in-process loading that never respawns workers at epoch boundaries."
            )
        },
    )
    persistent_workers: bool = field(
        default=False,
        metadata={
            "help": (
                "Keep DataLoader workers alive across epochs instead of respawning them at "
                "every epoch boundary. Setting this requires `num_workers > 0`"
            )
        },
    )

    def __post_init__(self) -> None:
        if self.num_workers is not None and self.num_workers < 0:
            raise ValueError(f"data.dataloader.num_workers must be None or >= 0, got {self.num_workers}.")


@dataclass
class DataConfig(BaseConfig):
    train_data: List[str] = field(default_factory=lambda: [os.path.expanduser("~/data/gsm8k/train.parquet")])
    val_data: List[str] = field(default_factory=lambda: [os.path.expanduser("~/data/gsm8k/validation.parquet")])
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)


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

    max_loras: int = 1
    """Maximum number of LoRA adapters that can be active concurrently in a
    single GPU batch. Maps to vLLM's ``max_loras``. Increase past 1 to enable
    multi-tenant LoRA serving via ``RemoteInferenceClient.load_lora_adapter``."""

    max_cpu_loras: Optional[int] = None
    """Total LoRA adapter capacity in vLLM's CPU LRU cache. Maps to vLLM's
    ``max_cpu_loras``; when None, vLLM defaults it to ``max_loras``. Must be
    >= ``max_loras`` if explicitly set."""


@dataclass
class ExpertMxfp8Config(BaseConfig):
    """MXFP8 compute for routed MoE experts."""

    enabled: bool = False
    training: bool = True
    rollout: bool = True
    persistent: bool = False


@dataclass
class FakeInt4QatConfig(BaseConfig):
    """Fake-INT4 quantization-aware training for MoE experts (Megatron only).

    When the inference engine serves the MoE experts as real ``compressed-tensors``
    INT4 (e.g. ``casperhansen/Qwen3.6-35B-A3B-INT4-RTN``) but the trainer holds
    BF16 masters, enabling this fake-quantizes the frozen expert GEMMs onto the
    same INT4 grid in the forward pass (straight-through backward), removing the
    train/infer weight mismatch. See
    ``skyrl.backends.skyrl_train.workers.megatron.fake_int4_qat``.
    """

    enabled: bool = False
    group_size: int = 32
    """Group size along the input dim; must match the served checkpoint (32)."""
    symmetric: bool = True
    scale_divisor: float = 7.5
    """Symmetric-INT4 scale divisor ``scale = amax / scale_divisor``:
    ``7.5`` = llm-compressor / compressed-tensors RTN (``[-8, 7]``; matches
    ``casperhansen/Qwen3.6-35B-A3B-INT4-RTN``); ``7.0`` = Kimi K2-Thinking / K2.6 /
    Miles (``[-7, 7]``). Set ``q_min`` consistently."""
    q_min: float = -8.0
    """Lower clamp of the INT4 code range: ``-8`` for llm-compressor RTN
    (``scale_divisor=7.5``), ``-7`` for Kimi/Miles (``scale_divisor=7.0``, whose
    QAT never emits ``-8``)."""
    bf16_base_path: Optional[str] = None
    """Megatron-Bridge cannot load a compressed-tensors INT4 checkpoint, so when
    ``model.path`` points at the INT4 model the trainer loads its BF16 master
    weights from this path instead. The INT4 ``model.path`` remains what the
    inference engine serves and the logical name. When None, the trainer loads
    weights from ``model.path`` directly (only valid if that path is already a
    BF16 checkpoint)."""


@dataclass
class ModelConfig(BaseConfig):
    path: Optional[str] = None
    lora: SkyRLLoraConfig = field(default_factory=SkyRLLoraConfig)
    expert_mxfp8: ExpertMxfp8Config = field(default_factory=ExpertMxfp8Config)
    fake_int4_qat: FakeInt4QatConfig = field(default_factory=FakeInt4QatConfig)

    def __post_init__(self) -> None:
        if self.fake_int4_qat.enabled:
            assert self.lora.rank > 0, (
                "`trainer.policy.model.fake_int4_qat.enabled=True` currently requires LoRA "
                "(`trainer.policy.model.lora.rank > 0`) because full-weight sync exports "
                "dense expert weights."
            )


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
    fp8_param_gather: bool = False
    average_in_collective: bool = True


TORCH_PROFILER_ACTIVITIES = ("cpu", "cuda")
TORCH_PROFILER_EXPORT_TYPES = ("chrome_trace", "stacks")


@dataclass
class TorchProfilerConfig(BaseConfig):
    """``torch.profiler`` config for policy training steps."""

    enable: bool = False
    ranks: List[int] = field(default_factory=lambda: [0])
    save_path: Optional[str] = None
    """Trace output dir. Required when ``enable=True``; must be a local absolute path."""

    # torch.profiler.schedule
    skip_first: int = 10
    """Steps to skip before scheduling begins."""
    wait: int = 0
    warmup: int = 1
    active: int = 1
    """Number of steps recorded per cycle."""
    repeat: int = 1
    """Number of cycles. 0 means forever."""

    # torch.profiler.profile
    activities: List[str] = field(default_factory=lambda: ["cpu", "cuda"])
    record_shapes: bool = True
    profile_memory: bool = False
    with_stack: bool = True
    with_flops: bool = False
    with_modules: bool = False
    export_type: str = "chrome_trace"
    """``chrome_trace`` or ``stacks``; stacks require ``with_stack=True``."""

    def validate(
        self,
        strategy: Optional[str] = None,
        colocate_all: Optional[bool] = None,
        colocate_policy_ref: Optional[bool] = None,
        fsdp_cpu_offload: Optional[bool] = None,
    ) -> None:
        """Fail fast on invalid or known-incompatible profiler settings."""
        if not self.enable:
            return
        if not self.ranks:
            raise ValueError("`torch_profiler_config.ranks` must be non-empty when profiling is enabled.")
        # Avoid implicit relative paths in Ray runtime working dirs.
        if not self.save_path:
            raise ValueError(
                "`torch_profiler_config.save_path` must be set when profiling is enabled. "
                "Use an absolute local path -- Ray workers run from a /tmp/ray runtime "
                "working dir, so a relative path would write traces there."
            )
        from skyrl.backends.skyrl_train.utils.io.io import is_cloud_path

        if is_cloud_path(self.save_path):
            raise ValueError(
                f"`torch_profiler_config.save_path` must be a local path; got cloud URI "
                f"{self.save_path!r}. torch.profiler cannot write to cloud storage."
            )
        # Empty activities record nothing.
        if not self.activities:
            raise ValueError("`torch_profiler_config.activities` must be non-empty when profiling is enabled.")
        bad_activities = [a for a in self.activities if a.lower() not in TORCH_PROFILER_ACTIVITIES]
        if bad_activities:
            raise ValueError(
                f"invalid `torch_profiler_config.activities` entries {bad_activities}. "
                f"Each must be one of {list(TORCH_PROFILER_ACTIVITIES)}."
            )
        if self.export_type not in TORCH_PROFILER_EXPORT_TYPES:
            raise ValueError(
                f"invalid `torch_profiler_config.export_type`: {self.export_type!r}. "
                f"Must be one of {list(TORCH_PROFILER_EXPORT_TYPES)}."
            )
        if self.export_type == "stacks" and not self.with_stack:
            raise ValueError(
                "`torch_profiler_config.export_type='stacks'` requires `with_stack=true` "
                "(torch.profiler.export_stacks needs stack records)."
            )
        for name in ("skip_first", "wait", "warmup", "repeat"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"`torch_profiler_config.{name}` must be >= 0, got {value}.")
        if self.active < 1:
            raise ValueError(f"`torch_profiler_config.active` must be >= 1, got {self.active}.")

        # FSDP manual CPU offload uses swap_tensors, which conflicts with profiler-held
        # parameter refs during colocated runs.
        if strategy == "fsdp" and fsdp_cpu_offload is False and (colocate_all or colocate_policy_ref):
            raise ValueError(
                "`torch_profiler_config.enable=true` is incompatible with this FSDP configuration: "
                "with the manual CPU-offload path (`policy.fsdp_config.cpu_offload=false`, the default) "
                "under colocation "
                f"(`placement.colocate_all={colocate_all}`, `placement.colocate_policy_ref={colocate_policy_ref}`), "
                "the trainer offloads models to CPU via `torch.utils.swap_tensors` while the profiler holds "
                "references to their parameters, which crashes mid-run with "
                "`RuntimeError: _apply(): Couldn't swap <param>`. "
                "To profile: set `policy.fsdp_config.cpu_offload=true` (FSDP2-native offload, no swap), or "
                "disable colocation (`placement.colocate_all=false` and `placement.colocate_policy_ref=false`), "
                "or use the Megatron backend (`trainer.strategy=megatron`)."
            )


@dataclass
class MegatronHFExportConfig(BaseConfig):
    distributed_save: bool = False
    """Fan the Megatron->HF safetensors export across ranks instead of writing the
    whole checkpoint from rank 0. The on-disk result is the standard HF sharded
    format either way; this only parallelizes the write, which is decisive for
    multi-hundred-GB checkpoints whose serial rank-0 write stalls every rank."""
    save_every_n_ranks: int = 1
    """In distributed save, only ranks 0, N, 2N, ... write shards (e.g. 8 = one
    writer per 8-GPU node). Ignored when ``distributed_save`` is False."""

    def __post_init__(self) -> None:
        # save_every_n_ranks indexes ranks via modulo/floor-div in the bridge's
        # distributed save; < 1 raises ZeroDivisionError there. Fail fast instead.
        if self.save_every_n_ranks < 1:
            raise ValueError(f"save_every_n_ranks must be >= 1, got {self.save_every_n_ranks}")


@dataclass
class MegatronLoraConfig(BaseConfig):
    lora_type: str = "lora"
    merge_lora: bool = True


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
    "gradient_accumulation_fusion": False,
}


@dataclass
class MegatronConfig(BaseConfig):
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: Optional[int] = None
    # MoE runtime configuration flags
    moe_token_dispatcher_type: str = "alltoall"
    moe_router_load_balancing_type: str = "none"
    """Set to "aux_loss", "seq_aux_loss", or "global_aux_loss" to enable aux loss-based load balancing and logging."""
    moe_aux_loss_coeff: float = 0.0
    """Scaling coefficient for the moe load balancing loss if moe_router_load_balancing_type is not 'none'. Will disable aux loss in megatron-core if set to 0."""
    moe_grouped_gemm: bool = True
    moe_router_score_function: Optional[str] = None
    moe_router_enable_expert_bias: Optional[bool] = None
    moe_enable_routing_replay: bool = False
    moe_per_layer_logging: bool = False
    """Enable per-layer logging of MoE metrics (i.e. per layer aux losses)."""
    moe_router_dtype: str = "fp32"
    """Pass through to Megatron-Bridge - can be set to 'fp64' for additional numerical stability."""
    ddp_config: MegatronDDPConfig = field(default_factory=MegatronDDPConfig)
    hf_export_config: MegatronHFExportConfig = field(default_factory=MegatronHFExportConfig)
    lora_config: MegatronLoraConfig = field(default_factory=MegatronLoraConfig)
    optimizer_config_kwargs: Dict[str, Any] = field(
        default_factory=lambda: copy.deepcopy(DEFAULT_MEGATRON_OPTIMIZER_KWARGS)
    )
    transformer_config_kwargs: Dict[str, Any] = field(
        default_factory=lambda: copy.deepcopy(DEFAULT_TRANSFORMER_CONFIG_KWARGS)
    )
    empty_cuda_cache: Optional[bool] = True
    model_config_kwargs: dict = field(default_factory=dict)
    dist_ckpt_optim_fully_reshardable: bool = False
    freeze_moe_router: bool = False
    """If True, freeze MoE router parameters so they are not updated during training. No-op on
    non-MoE models."""
    mtp_num_layers: Optional[int] = None
    """Number of Multi-Token Prediction (MTP) heads to build. ``None`` honors the model's HF config
    (``num_nextn_predict_layers``); an int overrides it (``0`` force-disables MTP). Active heads are
    trained with the decoupled draft loss and synced to vLLM for speculative decoding."""
    mtp_loss_weight: float = 0.1
    """Weight ``w`` of the draft loss in ``policy_loss + w * draft_loss``. The draft loss is fully
    decoupled: trunk, re-embedding, output weight and teacher are all detached, so its gradient
    reaches only the ``.mtp.`` head params. Only used when MTP heads are active."""
    mtp_loss_chunk_size: Optional[int] = 1024
    """Sequence-chunk size for the draft loss, with gradient checkpointing, to bound peak memory at
    large vocab (e.g. Qwen3.5's 248K, where the full-sequence softmax OOMs). Numerically identical to
    no chunking. ``None`` disables it; ignored when ``mtp_loss_topk`` is set."""
    mtp_loss_topk: Optional[int] = None
    """If set, use a top-k approximation of the soft-CE draft loss: distill only the teacher's top-k
    tokens (renormalized), ``O(seq*k)`` memory instead of ``O(seq*vocab)`` -- fits at large vocab
    without fragmentation. Reconciled across the TP group, so it scales to any parallel size.
    ``None`` uses the exact full-vocab loss. Typical: 64-128."""

    def __post_init__(self):
        # Backfill defaults for any keys the user didn't override so an override dict
        # doesn't have to repeat every default just to set one value.
        if self.transformer_config_kwargs is None:
            self.transformer_config_kwargs = {}
        for k, v in DEFAULT_TRANSFORMER_CONFIG_KWARGS.items():
            self.transformer_config_kwargs.setdefault(k, copy.deepcopy(v))
        if self.optimizer_config_kwargs is None:
            self.optimizer_config_kwargs = {}
        for k, v in DEFAULT_MEGATRON_OPTIMIZER_KWARGS.items():
            self.optimizer_config_kwargs.setdefault(k, copy.deepcopy(v))


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------


@dataclass
class PlacementConfig(BaseConfig):
    colocate_all: bool = True
    """When True, training and inference share the same GPUs."""
    colocate_policy_ref: bool = True
    """When colocate_all is False, True (default) still colocates policy and ref
    on the same GPUs (one shared placement group). Set this item to False to place
    policy and ref on separate GPUs (their own placement groups); needed when
    a large model's policy and ref shards can't both fit on one GPU."""
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
    torch_profiler_config: TorchProfilerConfig = field(default_factory=TorchProfilerConfig)
    """``torch.profiler`` config for policy training steps."""
    megatron_config: MegatronConfig = field(default_factory=MegatronConfig)
    model_config_kwargs: dict = field(default_factory=dict)
    """Pass-through kwargs for the HuggingFace model config (FSDP backends).
    For Megatron, use ``policy.megatron_config.transformer_config_kwargs`` instead."""
    language_model_only: bool = False
    """When True, skip vision encoder initialization for multimodal models (e.g. Qwen3.5).
    Loads only the language model backbone using AutoModelForCausalLM."""
    inference_only_init: bool = False
    """When True, set up the policy worker for inference-only flows (forward + weight
    sync, no train_step), skipping the training-only state that would otherwise OOM
    memory-constrained nodes (e.g. large MoE on 4xH100). NOT valid for actual training.
    Backend-specific behavior:
    - FSDP: initialize weights in bf16 instead of fp32 (skipping the fp32 master weights
      that mixed-precision training requires) and skip optimizer/LR-scheduler construction.
    - Megatron: skip optimizer/LR-scheduler construction (DistributedOptimizer eagerly
      materializes fp32 master + AdamW state on GPU)."""


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
    language_model_only: bool = False
    """When True, skip vision encoder initialization for multimodal models (e.g. Qwen3.5).
    Loads only the language model backbone using AutoModelForCausalLM."""


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------


@dataclass
class KLCtrlConfig(BaseConfig):

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

    clip_ratio: float = 0.0002
    """Fraction of tokens to clip based on covariance."""
    clip_cov_lb: float = 1.0
    clip_cov_ub: float = 5.0


@dataclass
class KLCovConfig(BaseConfig):

    kl_cov_frac: float = 0.2
    """Fraction of tokens to apply KL regularization to."""
    ppo_kl_coef: float = 1.0


@dataclass
class CISPOConfig(BaseConfig):

    cispo_eps_clip_low: float = 1.0
    """Offset for lower bound of importance sampling ratio clipping (as opposed to PPO token update clipping).
    
    The lower bound used is ``1-cispo_eps_clip_low``. The default lower bound is 0 following the ScaleRL recipe: https://arxiv.org/abs/2510.13786
    """
    cispo_eps_clip_high: float = 4.0
    """Offset for upper bound of importance sampling ratio clipping (as opposed to PPO token update clipping).
    
    The upper bound used is ``1+cispo_eps_clip_high``. The default upper bound is 5 following the ScaleRL recipe: https://arxiv.org/abs/2510.13786
    """


# DPPO parameters (only used when policy_loss_type="dppo")
# See: https://arxiv.org/abs/2602.04879
@dataclass
class DPPOConfig(BaseConfig):
    dppo_type: str = "binary_tv"
    """DPPO divergence variant: ``"binary_tv"`` or ``"binary_kl"``. Used if ``policy_loss_type="dppo"``."""
    delta_low: float = 0.2
    """Divergence threshold for negative advantages (0.2 for TV, 0.05 for KL recommended)."""
    delta_high: float = 0.2
    """Divergence threshold for positive advantages (0.2 for TV, 0.05 for KL recommended)."""

    def __post_init__(self):
        if self.dppo_type not in ["binary_tv", "binary_kl"]:
            raise ValueError("Invalid DPPO type")


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
    """Set to mask sequences with any token IS ratio below this threshold. Suggested: 1e-4. ``None`` to disable."""
    outlier_token_is_threshold_high: Optional[float] = None
    """Set to mask sequences with any token IS ratio above this threshold. Suggested: 100. ``None`` to disable."""
    token_mask_is_threshold_low: Optional[float] = None
    """Set to mask per-token when IS ratio < `token_mask_is_threshold_low`. ``None`` to disable."""
    token_mask_is_threshold_high: Optional[float] = None
    """Set to mask per-token when IS ratio > `token_mask_is_threshold_high`. ``None`` to disable."""


@dataclass
class AlgorithmConfig(BaseConfig):
    advantage_estimator: str = "grpo"
    """``"grpo"``, ``"gae"``, ``"rloo"``, ``"reinforce++"``, or custom via ``AdvantageEstimatorRegistry``."""
    kl_ctrl: KLCtrlConfig = field(default_factory=KLCtrlConfig)
    """Only used when ``use_kl_in_reward=True`` (not applied when ``use_kl_loss=True``).
    Uses ``kl_loss_coef`` as the initial KL coefficient."""
    kl_estimator_type: str = "k3"
    """``"k1"``, ``"k2"``, ``"k3"``, ``"abs"``. See http://joschu.net/blog/kl-approx.html."""
    use_kl_in_reward: bool = False
    """Apply KL penalty to rewards. Mutually exclusive with ``use_kl_loss``."""
    use_kl_loss: bool = True
    """Apply KL loss in the policy model. Mutually exclusive with ``use_kl_in_reward``."""
    kl_loss_coef: float = 0.001
    use_entropy_loss: bool = False
    entropy_loss_coef: float = 0.01
    temperature: Optional[float] = None
    """Temperature for scaling logits in policy loss computation.
    If ``None``, will be set to the temperature provided by ``generator.sampling_params.temperature`` during config validation.
    
    NOTE: When using HTTP endpoints directly, make sure to set this value to the temperature used during generation
    """
    advantage_batch_normalize: bool = False
    value_head_prefix: str = "value_head"
    policy_loss_type: str = "regular"
    """``"regular"``, ``"dual_clip"``, ``"gspo"``, ``"clip_cov"``, ``"kl_cov"``, ``cispo``, ``sapo``, ``"rollout_is"``, ``"dppo"``, or custom via ``PolicyLossRegistry``."""
    loss_reduction: str = "token_mean"
    """``"token_mean"``, ``"sequence_mean"``, ``"prompt_mean"``, or ``"seq_mean_token_sum_norm"``. ``max_seq_len`` must be set explicitly for ``"seq_mean_token_sum_norm"``."""
    grpo_norm_by_std: bool = True
    zero_variance_filter: bool = False
    """Loss-mask prompts with zero-variance rewards. Only applicable when rewards are response-level."""
    zero_variance_filter_tol: float = 1e-6
    """Two rewards within this absolute tolerance count as equal when detecting zero-variance groups.
    Only used when ``zero_variance_filter=True``. Defaults to 1e-6 so float (LLM-judge) rewards that are
    effectively identical are still treated as zero-variance; this is a no-op for integer rewards (e.g.
    0/1) where the spread is either 0 or >= 1. Set to 0.0 for exact equality."""
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
    """Only used when ``policy_loss_type="clip_cov"``."""
    kl_cov: KLCovConfig = field(default_factory=KLCovConfig)
    """Only used when ``policy_loss_type="kl_cov"``."""
    cispo: CISPOConfig = field(default_factory=CISPOConfig)
    """Only used when ``policy_loss_type="cispo"``."""
    dppo: DPPOConfig = field(default_factory=DPPOConfig)
    """Only used when ``policy_loss_type="dppo"``."""
    max_seq_len: Optional[int] = None
    """Used for ``seq_mean_token_sum_norm`` loss reduction.
    Must be set explicitly for that reduction mode; otherwise can remain ``None``."""


# ---------------------------------------------------------------------------
# Fully Async
# ---------------------------------------------------------------------------


@dataclass
class FullyAsyncConfig(BaseConfig):
    """Knobs for fully async training.
    See https://docs.skyrl.ai/docs/tutorials/fully_async#step-2-config-knobs-to-tune-for-fully-async-training."""

    enabled: bool = False
    """Indicates whether fully async training is enabled"""
    max_staleness_steps: int = 4
    """Maximum off-policy steps allowed. If a trajectory group is scheduled at step *i* and trained at step *j*,
    then ``j - i <= max_staleness_steps``. Larger values increase throughput but also off-policy-ness."""
    num_parallel_generation_workers: int = 768
    """Number of generation workers to spawn. Should be >= ``policy_mini_batch_size`` and
    <= ``policy_mini_batch_size * (max_staleness_steps + 1)``."""
    sample_full_batch: bool = False
    """Requires ``zero_variance_filter=True``. Drop zero-variance groups and keep pulling until the
    mini-batch is full of non-zero-variance groups (async-native DAPO ``dynamic_sampling="filter"``).
    Dropped groups are marked consumed (not regenerated on resume), so the per-epoch step count becomes
    an upper bound: if the epoch's prompts run out mid mini-batch, the partial batch is discarded and
    the epoch ends."""
    clear_kv_cache_on_weight_sync: bool = False
    """Whether or not to clear the KV cache on weight sync. Defaults to False.
    If False, we reuse KV cache from stale policies during generation
    (avoids recomputation at the cost of using slightly stale KV cache).
    """

    # --- Trainer simulation (no real trainer components) ---
    simulate_training: bool = False
    """If True, run fully-async generation with a SIMULATED trainer (see
    ``FullyAsyncTrainerSim``): no policy/critic/ref models are instantiated and no weight
    broadcast happens. Each step consumes a mini-batch from the generation buffer, sleeps for
    ``simulate_training_step_seconds``, then issues pause/resume generation (as a real weight
    sync would) but skips ``broadcast_to_inference_engines``. Used to benchmark the
    generation/inference side (e.g. router load-balancing policies) on large models without
    paying for trainer GPUs — typically pointed at already-served endpoints via
    ``external_proxy_url`` / ``external_server_urls``. The generation-side dynamics (staleness
    control, rate limiting, pause/resume) remain faithful."""
    simulate_training_step_seconds: float = 30.0
    """Wall-clock seconds the simulated dummy training step sleeps (stands in for fwd/bwd/optim)."""
    simulate_weight_sync_seconds: float = 0.0
    """Wall-clock seconds generation stays paused to stand in for the (skipped) weight broadcast.
    0.0 = pause then immediately resume."""


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
    fp8_weight_sync_mode: Optional[str] = None
    """Optional rollout weight format. ``"serialized_blockwise"`` serializes
    supported linears with 128x128 scales; ``"serialized_mxfp8"`` serializes
    routed experts with 1x32 E8M0 scales."""
    run_engines_locally: bool = True
    num_engines: int = 1
    backend: str = "vllm"
    """``"vllm"``."""
    weight_sync_backend: str = "nccl"
    weight_transfer_threshold_cuda_ipc_GB: float = 1.0
    """When using ``cuda_ipc``, send weights in batches of this size (GB)."""
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    expert_parallel_size: int = 1
    data_parallel_size: int = 1
    vllm_v1_disable_multiproc: bool = True
    """Sets ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` for reproducibility."""
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    enable_return_routed_experts: bool = False
    max_num_batched_tokens: int = 8192
    enforce_eager: bool = False
    """Disable CUDA graphs for stability. Set to ``False`` for higher performance,
    but this may affect convergence for long-running or long-context training jobs."""
    fully_sharded_loras: bool = False
    enable_ray_prometheus_stats: bool = True
    """Enable Ray Prometheus stats logger for inference engine metrics (vLLM v1 only)."""
    gpu_memory_utilization: float = 0.8
    use_expandable_segments: bool = False
    """Set ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` on the inference-engine
    processes to reduce fragmentation. Independent of the trainer-side
    ``TrainerConfig.use_expandable_segments``. Default ``False``: it is a safe opt-in
    on vLLM >= 0.20.1, where the CuMemAllocator auto-disables expandable segments around
    its sleep/wake memory pool. On older vLLM, sleep mode + expandable segments is a hard
    error, so leave this off."""
    max_num_seqs: int = 1024
    served_model_name: Optional[str] = None
    """Model name for HTTP endpoint validation. If set, must be used in the ``model`` field of
    ``/chat/completions`` requests instead of the model path. If ``None``, the model path is used."""
    distributed_executor_backend: str = "ray"
    """Distributed executor backend for vLLM. Set to ``"ray"`` to use the Ray backend
    or ``"mp"`` to use the multiprocessing backend (single-node serving only). Per-engine 
    placement groups are created when ``"mp"`` is used."""
    language_model_only: bool = False
    """When True, pass ``language_model_only=True`` to the vLLM engine so that
    multimodal models (e.g. Qwen3.5) skip vision encoder initialization."""
    engine_init_kwargs: Dict[str, Any] = field(default_factory=dict)
    """Pass-through kwargs for the vLLM engine. Names must match the engine's args."""
    speculative_config: Optional[Dict[str, Any]] = None
    """Speculative-decoding config passed through to vLLM for MTP drafter decoding. 
    (needs ``policy.megatron_config.mtp_num_layers`` > 0 to train mtp). ``None`` disables it."""
    external_proxy_url: Optional[str] = None
    """Data-plane URL (load-balanced router) for the new inference layer."""
    external_server_urls: Optional[List[str]] = None
    """Control-plane URLs (direct backend access) for the new inference layer."""
    enable_pd: bool = False
    """Enable prefill-decode disaggregation. Requires ``num_prefill > 0`` and ``num_engines >= 2``."""
    num_prefill: int = 0
    """Number of prefill engines when ``enable_pd=True``. Decode engines = ``num_engines - num_prefill``

    NOTE: SkyRL counts data parallel workers separately, so the total number of prefill workers will be ``data_parallel_size * num_prefill``."""
    router_init_kwargs: Dict[str, Any] = field(default_factory=dict)
    """Pass-through kwargs applied to ``RouterArgs`` for the vllm-router.
    Names must match ``vllm_router.RouterArgs`` fields (e.g. ``policy``, ``request_timeout_secs``)."""


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
    use_cache_salt: bool = True
    """Salt vLLM's prefix cache with the policy version so cache blocks are only shared across trajectories that started
    with the same policy weight version. The salt is keyed on the engine's weight version, captured at the start of each
    ``generate`` call. Matters for fully-async RL; a no-op for synchronous training (which resets the
    cache each sync) and when prefix caching is off, so it is safe to leave on by default."""
    apply_overlong_filtering: bool = False
    """Apply DAPO Overlong Filtering: mask out all tokens in the loss mask for trajectories that
    exceed max length (truncated, no EOS token)."""
    step_wise_trajectories: bool = False
    vision_language_generator: bool = False
    """If True, use SkyRLVLMGymGenerator (multi-modal text+image rollouts)"""
    merge_stepwise_output: bool = False
    """When True (and step_wise_trajectories is True), apply prefix-aware merging
    to collapse multi-turn step-wise sequences into single sequences before training."""

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


@dataclass
class MTPConfig(BaseConfig):
    enabled: bool = False
    """Whether to train MTP draft heads and use them for speculative decoding."""
    num_speculative_tokens: int = 1
    """Draft depth vLLM speculates per step, independent of the trained head count. Single-head
    checkpoints (Qwen3.5/Qwen3-Next/DeepSeek-V3) reuse the one head autoregressively at depths > 1;
    expect acceptance to decay with depth (the head is trained at depth 1)."""
    loss_weight: float = 0.1
    """Weight ``w`` of the draft loss in ``policy_loss + w * draft_loss``."""


# ---------------------------------------------------------------------------
# Trainer (top-level)
# ---------------------------------------------------------------------------


@dataclass
class TrainerConfig(BaseConfig):
    placement: PlacementConfig = field(default_factory=PlacementConfig)
    use_expandable_segments: bool = True
    """Enable PyTorch's CUDA ``expandable_segments`` allocator on the training
    workers to reduce GPU memory fragmentation across the offload/backload and
    forward/backward cycles. See ``InferenceEngineConfig`` for the
    equivalent inference-engine knob."""
    sequence_parallel_backend: str = "ulysses"
    strategy: str = "fsdp"
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    ref: RefConfig = field(default_factory=RefConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    mtp: MTPConfig = field(default_factory=MTPConfig)
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
    max_training_steps: Optional[int] = None
    """If set, stop training after this many steps regardless of epochs or dataset size.
    Useful for CI smoke tests and quick validation runs."""
    update_epochs_per_batch: int = 1
    """Number of gradient update passes over each training batch."""
    train_batch_size: int = 1024
    """See ``utils/utils.py::validate_batch_sizes`` for train, mini, and micro batch size constraints."""
    policy_mini_batch_size: int = 256
    critic_mini_batch_size: int = 256
    micro_train_batch_size_per_gpu: int = 1
    micro_forward_batch_size_per_gpu: int = 1
    max_tokens_per_microbatch: int = -1
    """Maximum number of tokens per microbatch for both forward and training steps. When > 0, microbatches 
    are formed by bin-packing samples based on their token counts (from attention_mask) instead of using a 
    fixed sample count, and micro_train_batch_size_per_gpu / micro_forward_batch_size_per_gpu are ignored.
    -1 means disabled (use sample-based micro_train_batch_size_per_gpu / micro_forward_batch_size_per_gpu).
    Applies to both forward and training micro-batching.

    NOTE: this is a *soft* cap. Sequences are never split across microbatches, so a single sequence
    longer than ``max_tokens_per_microbatch`` is placed alone in its own microbatch that exceeds the
    cap (no error, no truncation). The true peak microbatch size is therefore
    ``max(max_tokens_per_microbatch, longest_sequence_in_batch)``."""
    recompute_old_logprobs_per_minibatch: bool = True
    """When True, recomputes policy/ref model logprobs (and critic values) per mini-batch using
    the same mini-batch + DP partition as the training step. When False, a single full-batch forward is run.
    This makes the microbatch packing — and therefore the resulting logprobs/values — identical to
    what forward_backward recomputes, so the PPO ratio (and critic value clipping) is exact at the
    first inner step."""
    update_ref_every_epoch: bool = False
    remove_microbatch_padding: bool = True
    """Pack samples into the THD layout and strip intra-microbatch padding (requires flash attention)."""
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
    enable_ray_gpu_monitor: bool = True
    """Enable background Ray GPU/RAM metrics collection and logging to wandb."""
    tags: Optional[List[str]] = None
    """Optional list of tags to apply to the W&B run. Has no effect on other backends."""
    dump_data_batch: bool = False
    dump_eval_results: bool = True
    print_example_interval: int = 1
    """Pretty-print an example prompt/response/reward to stdout every N
    training steps; ``0``/``-1`` disables. Renamed from ``log_example_interval``."""
    num_logger_eval_samples: int = -1
    """Number of evaluation trajectory (prompt, response, score) tuples to upload to a wandb
    table on each eval. ``-1`` (default) or ``0`` disables. When positive,
    up to this many samples are taken from the start of each eval pass and
    logged via :class:`TrajectoryLogger`. Column count is fixed
    by the first call, so keep the eval set size and this value stable."""
    num_logger_train_samples: int = -1
    """Number of training trajectory (prompt, response, score) tuples to upload to a wandb
    table on each training step. ``-1`` (default) or ``0`` disables. When positive,
    up to this many samples are taken from the start of each training step and
    logged via :class:`TrajectoryLogger`. Column count is fixed
    by the first call, so keep the training set size and this value stable."""
    log_example_interval: int = -1
    """Log an example prompt every N training steps, ``0``/``-1`` to disable"""
    logprobs_chunk_size: Optional[int] = 1024
    """Chunk size along the sequence dimension when computing log-probs from logits.
    This lowers peak GPU memory at the cost of ~2x wall-clock time.
    ``None`` disables chunking (Megatron backend only; FSDP requires a positive int).
    See https://github.com/NovaSky-AI/SkyRL/pull/1610 for more details."""
    vocab_entropy_chunk_size: Optional[int] = 0
    """Chunk size along the sequence dimension when computing Megatron vocab entropy.
    ``0`` auto-sizes from the local vocab shard size and ``vocab_entropy_chunk_memory_mb``.
    ``None`` disables chunking."""
    vocab_entropy_chunk_memory_mb: int = 512
    """Approximate per-chunk temporary memory budget for auto-sized Megatron vocab entropy chunks."""
    fused_lm_head_logprob: bool = False
    """Megatron only. Fuse the LM-head projection into log-prob / entropy
    computation so the full ``[B, S, vocab//TP]`` logits tensor is never
    materialized. Uses ``logprobs_chunk_size`` to bound peak memory."""
    fused_lm_head_logprob_backend: str = "torch"
    """Fused LM-head backend: ``"torch"`` (default) or ``"triton"``.
    The Triton backend requires CUDA + triton and falls back to ``"torch"``
    when unavailable. Ignored unless ``fused_lm_head_logprob`` is true."""

    def __post_init__(self):
        # ref model defaults to the policy model
        if self.ref.model.path is None:
            self.ref.model.path = self.policy.model.path

        if self.log_example_interval > 0:
            print(
                f"log_example_interval has been renamed, use print_example_interval instead. Setting print_example_interval to {self.log_example_interval}"
            )
            self.print_example_interval = self.log_example_interval

        if self.policy.model.fake_int4_qat.enabled:
            assert (
                self.strategy == "megatron"
            ), "`trainer.policy.model.fake_int4_qat.enabled=True` is only supported with `trainer.strategy=megatron`."
            assert not self.policy.megatron_config.lora_config.merge_lora, (
                "`trainer.policy.model.fake_int4_qat.enabled=True` currently requires "
                "`trainer.policy.megatron_config.lora_config.merge_lora=False` so weight "
                "sync preserves the inference engine's INT4 base weights."
            )

        expert_mxfp8 = self.policy.model.expert_mxfp8
        if expert_mxfp8.enabled:
            if self.strategy != "megatron":
                raise ValueError("Expert MXFP8 is only supported with trainer.strategy=megatron")
            if not expert_mxfp8.training and not expert_mxfp8.rollout:
                raise ValueError("Expert MXFP8 must enable training, rollout, or both")
            if expert_mxfp8.persistent:
                if not expert_mxfp8.training:
                    raise ValueError("Persistent expert MXFP8 requires expert_mxfp8.training=true")
                if not self.policy.megatron_config.ddp_config.fp8_param_gather:
                    raise ValueError(
                        "Persistent expert MXFP8 requires policy.megatron_config.ddp_config.fp8_param_gather=true"
                    )
            if self.policy.model.fake_int4_qat.enabled:
                raise ValueError("Expert MXFP8 and fake INT4 QAT are mutually exclusive")
            if expert_mxfp8.training:
                if not self.policy.megatron_config.moe_grouped_gemm:
                    raise ValueError("Expert MXFP8 training requires moe_grouped_gemm=true")
                if self.policy.megatron_config.moe_router_dtype != "fp32":
                    raise ValueError("Expert MXFP8 training requires moe_router_dtype=fp32")
                transformer_overrides = self.policy.megatron_config.transformer_config_kwargs
                if transformer_overrides.get("moe_grouped_gemm") is False:
                    raise ValueError("Expert MXFP8 training requires moe_grouped_gemm=true")
                if transformer_overrides.get("moe_router_dtype", "fp32") != "fp32":
                    raise ValueError("Expert MXFP8 training requires moe_router_dtype=fp32")
                conflicts = {"fp8", "quant_recipe"} & transformer_overrides.keys()
                conflicts.update(
                    option
                    for option in ("fp8_dot_product_attention", "fp8_multi_head_attention", "fp8_output_proj")
                    if transformer_overrides.get(option)
                )
                if conflicts:
                    raise ValueError(f"Expert MXFP8 conflicts with transformer_config_kwargs: {sorted(conflicts)}")
            if (
                expert_mxfp8.rollout
                and self.policy.model.lora.rank > 0
                and not self.policy.megatron_config.lora_config.merge_lora
            ):
                raise ValueError("Expert MXFP8 rollout does not support unmerged LoRA")

        if self.logprobs_chunk_size is not None and (
            not isinstance(self.logprobs_chunk_size, int) or self.logprobs_chunk_size <= 0
        ):
            raise ValueError(
                f"logprobs_chunk_size must be a positive integer or None, got {self.logprobs_chunk_size!r}."
            )
        if self.logprobs_chunk_size is None and self.strategy != "megatron":
            raise ValueError(
                "logprobs_chunk_size=None (no chunking) is only supported with the Megatron backend. "
                f"Set a positive integer for strategy={self.strategy!r}."
            )
        if self.fused_lm_head_logprob and self.strategy != "megatron":
            raise ValueError(
                "fused_lm_head_logprob=True is only supported with the Megatron backend, "
                f"got strategy={self.strategy!r}."
            )
        if self.fused_lm_head_logprob_backend not in ("torch", "triton"):
            raise ValueError(
                "fused_lm_head_logprob_backend must be 'torch' or 'triton', "
                f"got {self.fused_lm_head_logprob_backend!r}."
            )
        if self.vocab_entropy_chunk_size is not None and (
            isinstance(self.vocab_entropy_chunk_size, bool)
            or not isinstance(self.vocab_entropy_chunk_size, int)
            or self.vocab_entropy_chunk_size < 0
        ):
            raise ValueError(
                "vocab_entropy_chunk_size must be a non-negative integer or None, "
                f"got {self.vocab_entropy_chunk_size!r}."
            )
        if (
            isinstance(self.vocab_entropy_chunk_memory_mb, bool)
            or not isinstance(self.vocab_entropy_chunk_memory_mb, int)
            or self.vocab_entropy_chunk_memory_mb <= 0
        ):
            raise ValueError(
                "vocab_entropy_chunk_memory_mb must be a positive integer, "
                f"got {self.vocab_entropy_chunk_memory_mb!r}."
            )


def validate_dict_keys_against_dataclass(datacls: Type[Any], d: dict):
    """
    Validate the keys of a dict against fields of a dataclass.

    Args:
        datacls: The dataclass class to validate
    """
    valid_fields = {f.name for f in dataclasses.fields(datacls)}
    if invalid_keys := set(d.keys() - valid_fields):
        raise ValueError(f"Invalid fields {invalid_keys} for {datacls.__name__}. Valid fields are {valid_fields}.")


def _has_nested_key(cfg: Any, path: str) -> bool:
    node = cfg
    for key in path.split("."):
        if not isinstance(node, (dict, DictConfig)) or key not in node:
            return False
        node = node[key]
    return True


_MISSING = object()


def _get_nested_value(cfg: Any, path: str) -> Any:
    node = cfg
    for key in path.split("."):
        if not isinstance(node, (dict, DictConfig)) or key not in node:
            return _MISSING
        node = node[key]
    if isinstance(node, DictConfig):
        return OmegaConf.to_container(node, resolve=True)
    return node


def _delete_nested_key(cfg: Any, path: str) -> None:
    keys = path.split(".")
    node = cfg
    for key in keys[:-1]:
        if not isinstance(node, (dict, DictConfig)) or key not in node:
            return
        node = node[key]
    if isinstance(node, (dict, DictConfig)) and keys[-1] in node:
        del node[keys[-1]]


def _resolve_class_type(type_annotation: Any) -> Optional[Type]:
    """Extract the concrete non-plain class type from a type annotation.

    Handles plain types, Optional[T], Union[T, None], and Annotated[T, ...].
    Returns None if no dataclass or Enum type can be resolved.
    """
    origin = typing.get_origin(type_annotation)

    if origin is Union:
        # Optional[X] is Union[X, None]. Find the non-None dataclass arg.
        for arg in typing.get_args(type_annotation):
            if arg is type(None):
                continue
            resolved = _resolve_class_type(arg)
            if resolved is not None:
                return resolved
        return None

    if origin is Annotated:
        return _resolve_class_type(typing.get_args(type_annotation)[0])

    # Plain class check
    if isinstance(type_annotation, type) and (
        dataclasses.is_dataclass(type_annotation) or issubclass(type_annotation, Enum)
    ):
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
        nested_cls = _resolve_class_type(f.type)
        if nested_cls is not None:
            if isinstance(value, dict) and dataclasses.is_dataclass(nested_cls):
                kwargs[f.name] = build_nested_dataclass(nested_cls, value)
            elif issubclass(nested_cls, Enum):
                kwargs[f.name] = nested_cls(value)
            else:
                kwargs[f.name] = value
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

        # Copy temperature from generator sampling params to algorithm config
        # so workers can access it without needing the generator config
        if self.trainer.algorithm.temperature is None:
            self.trainer.algorithm.temperature = self.generator.sampling_params.temperature

        if self.data.dataloader.num_workers is None:
            self.data.dataloader.num_workers = 8
        if self.data.dataloader.persistent_workers and self.data.dataloader.num_workers == 0:
            raise ValueError(
                "data.dataloader.persistent_workers requires num_workers > 0, but it was set explicitly to 0."
            )

        # TODO(devpatel): Bandaid solution, replace this once we have a better
        # solution for LoRA performance degradation on the vLLM side
        from skyrl.backends.skyrl_train.inference_servers.utils import (
            _uses_lora_weight_sync,
        )

        ie_cfg = self.generator.inference_engine
        if _uses_lora_weight_sync(self) and ie_cfg.enforce_eager and ie_cfg.backend == "vllm":
            import warnings

            warnings.warn(
                "LoRA is enabled but inference_engine.enforce_eager=true. "
                "This combination causes significant performance degradation (2-3x slower generation). "
                "Automatically setting enforce_eager=false for better performance. "
            )
            ie_cfg.enforce_eager = False

    @classmethod
    def from_cli_overrides(cls, args: Union[List[str], dict]) -> "SkyRLTrainConfig":
        """Construct a SkyRLTrainConfig from CLI arguments or a dict of overrides.

        Parses CLI arguments and builds a typed config. Dataclass field defaults
        are used for any values not specified on the command line.

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
            # OmegaConf's CLI parser only treats "null" as None; Python's
            # None stringifies to "None" which is parsed as the literal
            # string. Map None -> "null" so JSON-style overrides survive
            # the round-trip through OmegaConf.from_cli below.
            args = [f"{k}=null" if v is None else f"{k}={v}" for k, v in args.items()]

        # Check for unsupported '+' prefix
        for arg in args:
            if arg.startswith("+"):
                raise ValueError(
                    f"The '+' prefix for adding new config fields is not supported: '{arg}'. "
                    "To add custom config fields, subclass the relevant config dataclass."
                )
        overrides = OmegaConf.from_cli(args)
        unsupported_rope_paths = (
            "trainer.rope_scaling",
            "trainer.rope_theta",
            "trainer.rope_parameters",
            "generator.rope_scaling",
            "generator.rope_theta",
            "generator.rope_parameters",
            "generator.inference_engine.rope_scaling",
            "generator.inference_engine.rope_theta",
            "generator.inference_engine.rope_parameters",
            "generator.inference_engine.engine_init_kwargs.rope_scaling",
            "generator.inference_engine.engine_init_kwargs.rope_theta",
            "generator.inference_engine.engine_init_kwargs.rope_parameters",
            "generator.inference_engine.engine_init_kwargs.hf_overrides.rope_scaling",
            "generator.inference_engine.engine_init_kwargs.hf_overrides.rope_theta",
        )
        if any(_has_nested_key(overrides, path) for path in unsupported_rope_paths):
            raise ValueError(
                "`rope_scaling`, `rope_theta`, and `rope_parameters` are no longer supported as native "
                "config overrides, use `generator.inference_engine.engine_init_kwargs.hf_overrides.rope_parameters` "
                "and `trainer.policy.model_config_kwargs.rope_parameters` or "
                "`trainer.policy.megatron_config.transformer_config_kwargs.rope_parameters` instead"
            )
        inference_rope_parameters = _get_nested_value(
            overrides, "generator.inference_engine.engine_init_kwargs.hf_overrides.rope_parameters"
        )
        if inference_rope_parameters is not _MISSING:
            trainer_strategy = _get_nested_value(overrides, "trainer.strategy")
            trainer_strategy = "fsdp" if trainer_strategy is _MISSING else trainer_strategy
            trainer_rope_parameters_path = (
                "trainer.policy.megatron_config.transformer_config_kwargs.rope_parameters"
                if trainer_strategy == "megatron"
                else "trainer.policy.model_config_kwargs.rope_parameters"
            )
            trainer_rope_parameters = _get_nested_value(overrides, trainer_rope_parameters_path)
            if inference_rope_parameters != trainer_rope_parameters:
                raise ValueError(
                    "`generator.inference_engine.engine_init_kwargs.hf_overrides.rope_parameters` must match "
                    f"the trainer-side override at `{trainer_rope_parameters_path}`"
                )
        async_engine_path = "generator.inference_engine.async_engine"
        async_engine = _get_nested_value(overrides, async_engine_path)
        if async_engine is not _MISSING:
            if async_engine is True or (isinstance(async_engine, str) and async_engine.lower() == "true"):
                _delete_nested_key(overrides, async_engine_path)
            elif async_engine is False or (isinstance(async_engine, str) and async_engine.lower() == "false"):
                raise ValueError(
                    "`async_engine=False` is no longer supported; SkyRL always uses the async "
                    "HTTP/vLLM inference path. Remove the override."
                )
            else:
                raise ValueError("`async_engine` is no longer supported as a config field. Remove the override.")
        removed_inference_engine_overrides = {
            "generator.inference_engine.enable_http_endpoint": (
                "`enable_http_endpoint` is no longer supported; SkyRL always uses the HTTP/vLLM inference path. "
                "Remove the override."
            ),
            "generator.inference_engine.override_existing_update_group": (
                "`override_existing_update_group` is no longer supported; update-group handling is managed "
                "automatically by the vLLM-native inference path. Remove the override."
            ),
        }
        for path, message in removed_inference_engine_overrides.items():
            if _has_nested_key(overrides, path):
                raise ValueError(message)
        if (
            "generator" in overrides
            and "inference_engine" in overrides.generator
            and "remote_urls" in overrides.generator.inference_engine
        ):
            raise ValueError(
                "`remote_urls` is no longer supported, external inference servers can be used with "
                "`external_proxy_url` and `external_server_urls` instead"
            )
        # Accept the deprecated ``trainer.use_sample_packing`` key as an alias
        # for ``trainer.remove_microbatch_padding``. Remap it before
        # construction so the strict key validation does not reject the old
        # name.
        if "trainer" in overrides and "use_sample_packing" in overrides.trainer:
            if "remove_microbatch_padding" in overrides.trainer:
                raise ValueError(
                    "Specify only one of trainer.use_sample_packing (deprecated) and "
                    "trainer.remove_microbatch_padding, not both."
                )
            import warnings

            warnings.warn(
                "trainer.use_sample_packing has been renamed to "
                "trainer.remove_microbatch_padding; use "
                "trainer.remove_microbatch_padding instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            overrides.trainer["remove_microbatch_padding"] = overrides.trainer["use_sample_packing"]
            del overrides.trainer["use_sample_packing"]
        return cls.from_dict_config(overrides)


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

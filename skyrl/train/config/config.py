"""
Typed configuration dataclasses for SkyRL.

These mirror the YAML configuration structure 1:1. The top-level SkyRLTrainConfig
can be constructed from a Hydra DictConfig via SkyRLTrainConfig.from_dict_config().
"""

import copy
import dataclasses
import json
import os
import typing
from abc import ABC
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Type, TypeVar, Union

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
    """Training and validation dataset configuration.

    NOTE: datasets are currently loaded entirely into memory, so the maximum usable dataset
    size is bounded by the CPU memory available on a worker node.
    """

    train_data: List[str] = field(default_factory=lambda: [os.path.expanduser("~/data/gsm8k/train.parquet")])
    """Files for the training dataset.
    Each entry is a path to a parquet or json file, or the name of a HuggingFace dataset."""
    val_data: List[str] = field(default_factory=lambda: [os.path.expanduser("~/data/gsm8k/validation.parquet")])
    """Files for the evaluation dataset, in the same formats accepted by ``train_data``.
    When more than one is given, evaluation runs over all of them: both per-dataset metrics
    (keyed by each sample's ``data_source``) and aggregated ``eval/all/*`` metrics are logged,
    and ``trainer.dump_eval_results`` dumps the per-dataset and aggregated results."""
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)


# ---------------------------------------------------------------------------
# Model / LoRA
# ---------------------------------------------------------------------------


# added prefix SkyRL to avoid conflict with peft.LoraConfig
@dataclass
class SkyRLLoraConfig(BaseConfig):
    """LoRA configuration for parameter-efficient fine-tuning.

    Trains a small number of additional low-rank matrices instead of the full model weights.
    """

    rank: int = 0
    """Rank of the low-rank decomposition.
    ``0`` disables LoRA. Higher values increase capacity but also memory usage; 8, 16, 32, and 64 are common choices."""
    alpha: int = 16
    """Scaling factor for LoRA updates."""
    dropout: float = 0.0
    """Dropout probability applied to LoRA layers, to help prevent overfitting."""
    lora_sync_path: str = "/tmp/skyrl_lora_sync"
    """Directory where LoRA adapter weights are saved and synchronized between the training and inference processes.
    Must be accessible to all workers in distributed setups."""
    target_modules: str = "all-linear"
    """Modules to apply LoRA to.
    ``"all-linear"`` targets every linear layer for FSDP/PEFT, and is remapped to a fixed module list
    on Megatron. A list of specific module names can be given instead."""
    exclude_modules: Optional[str] = None
    """Modules to exclude from LoRA."""
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
    """HuggingFace model path (or local directory) for this model."""
    lora: SkyRLLoraConfig = field(default_factory=SkyRLLoraConfig)
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
    """Optimizer configuration, shared by the policy and critic models."""

    lr: float = 1e-6
    """Learning rate."""
    adam_betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    """Betas for the AdamW optimizer."""
    weight_decay: float = 1e-2
    """L2 regularization strength for AdamW."""
    max_grad_norm: float = 1.0
    """Gradient clipping. The total L2 norm of the model gradients is scaled to this value."""
    offload_after_step: bool = True
    """Offload optimizer state to CPU after each full training step.
    Applies under colocation (``colocate_all``, or ``colocate_policy_ref`` for policy/ref), and is
    inert when ``fsdp_config.cpu_offload=True`` since FSDP2 then offloads natively. Without
    colocation it can be preferable to leave optimizer state on GPU, avoiding both the offload cost
    and the extra CPU memory usage."""
    num_warmup_steps: int = 0
    """Number of mini-batch steps to warmup the optimizer."""
    scheduler: str = "constant_with_warmup"
    """Learning rate scheduler. Intended to align with ``transformers.SchedulerType``:
    https://huggingface.co/docs/transformers/main/en/main_classes/optimizer_schedules#transformers.SchedulerType"""


@dataclass
class MixedPrecisionConfig(BaseConfig):
    param_dtype: str = "bf16"
    reduce_dtype: str = "fp32"
    buffer_dtype: str = "fp32"


@dataclass
class FSDPConfig(BaseConfig):
    cpu_offload: bool = False
    """Offload params and optimizer state to CPU during the forward pass.

    Corresponds to FSDP2's ``offload_policy``:
    https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html
    Enabling this replaces SkyRL's manual colocation offload (and makes
    ``optimizer_config.offload_after_step`` inert) rather than stacking with it; see
    https://docs.skyrl.ai/docs/tutorials/placement for the difference."""
    reshard_after_forward: Union[bool, int] = True
    """FSDP2 only.
    Accepts True, False, or an int between 1 and ``fsdp_size``. See
    https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html. Setting False retains the full model
    parameters on each worker (similar to DeepSpeed ZeRO stage 2)."""
    fsdp_size: int = -1
    """Group size within which worker state is sharded, for hybrid sharding in multi-node runs.
    ``-1`` shards across all workers in the group. Example: with 8 workers across 2 nodes (4 each)
    and ``fsdp_size=4``, training state is fully sharded across the 4 ranks within each node and
    replicated (data-parallel) across nodes."""
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


TORCH_PROFILER_ACTIVITIES = ("cpu", "cuda")
TORCH_PROFILER_EXPORT_TYPES = ("chrome_trace", "stacks")


@dataclass
class TorchProfilerConfig(BaseConfig):
    """``torch.profiler`` config for policy training steps.

    Applies to FSDP and Megatron in both RL and SFT. With the default ``export_type="chrome_trace"``,
    writes one Kineto/HTA-friendly ``*.pt.trace.json`` per active window and profiled rank
    (https://github.com/facebookresearch/HolisticTraceAnalysis); with ``export_type="stacks"`` it
    instead writes a per-rank ``rank{N}_stacks.txt`` that later windows overwrite.

    Scope: policy workers only. Active windows capture policy-worker operations between profiler
    steps — in RL that includes policy log-prob forwards and policy training; in SFT it includes
    training and may include eval forwards. Critic/ref workers, the controller, and the
    generation/inference engines are not profiled.

    FSDP restriction: profiling is rejected under colocation without CPU offloading (i.e.
    ``policy.fsdp_config.cpu_offload=False`` together with ``placement.colocate_all`` or
    ``placement.colocate_policy_ref``), because that path offloads via ``torch.utils.swap_tensors``
    while the profiler holds references to the parameters. See ``validate``.
    """

    enable: bool = False
    """Enable profiling."""
    ranks: List[int] = field(default_factory=lambda: [0])
    """Global ranks to profile, e.g. ``[0]``."""
    save_path: Optional[str] = None
    """Trace output dir.
    Required when ``enable=True``; must be a local absolute path. Relative paths would land under Ray's
    ``/tmp/ray/.../working_dir_files``, and cloud URIs are rejected."""

    # torch.profiler.schedule
    skip_first: int = 10
    """Steps to skip before scheduling begins. Passed to ``torch.profiler.schedule``:
    https://docs.pytorch.org/docs/stable/profiler.html#torch.profiler.schedule"""
    wait: int = 0
    """Steps to wait before warmup in each cycle. Passed to ``torch.profiler.schedule``."""
    warmup: int = 1
    """Warmup steps per cycle. Passed to ``torch.profiler.schedule``."""
    active: int = 1
    """Number of steps recorded per cycle."""
    repeat: int = 1
    """Number of cycles. 0 means forever."""

    # torch.profiler.profile
    activities: List[str] = field(default_factory=lambda: ["cpu", "cuda"])
    """Subset of ``["cpu", "cuda"]``."""
    record_shapes: bool = True
    """Passed to ``torch.profiler.profile``."""
    profile_memory: bool = False
    """Passed to ``torch.profiler.profile``."""
    with_stack: bool = True
    """Passed to ``torch.profiler.profile``. Required when ``export_type="stacks"``."""
    with_flops: bool = False
    """Passed to ``torch.profiler.profile``."""
    with_modules: bool = False
    """Passed to ``torch.profiler.profile``."""
    export_type: str = "chrome_trace"
    """Either ``chrome_trace`` or ``stacks``.
    ``chrome_trace`` writes ``*.pt.trace.json``; ``stacks`` writes self-CUDA-time stacks and
    requires ``with_stack=True``."""

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
    """Fan the Megatron->HF safetensors export across ranks instead of writing it all from rank 0.

    The on-disk result is the standard HF sharded format either way; this only parallelizes the
    write. Only affects HF exports from ``hf_save_interval`` or explicit ``save_hf_model`` calls.
    Sharded multi-node exports require ``trainer.export_path`` to be a shared filesystem path
    visible to every rank; see https://docs.skyrl.ai/docs/checkpointing-logging/checkpointing."""
    save_every_n_ranks: int = 1
    """In distributed save, only ranks 0, N, 2N, ... write shards (e.g. 8 = one writer per 8-GPU node).
    Must be at least 1. Ignored when ``distributed_save`` is False."""

    def __post_init__(self) -> None:
        # save_every_n_ranks indexes ranks via modulo/floor-div in the bridge's
        # distributed save; < 1 raises ZeroDivisionError there. Fail fast instead.
        if self.save_every_n_ranks < 1:
            raise ValueError(f"save_every_n_ranks must be >= 1, got {self.save_every_n_ranks}")


@dataclass
class MegatronLoraConfig(BaseConfig):
    lora_type: str = "lora"
    """``"lora"`` or ``"canonical_lora"``.
    See https://docs.nvidia.com/nemo/megatron-bridge/0.2.0/apidocs/bridge/bridge.peft.lora.html"""
    merge_lora: bool = True
    """Merge LoRA weights into the base weights during weight sync."""


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
    """Megatron-Core backend configuration, used when ``trainer.strategy="megatron"``.

    The parallelism sizes must satisfy:

    - ``model_size = pipeline_model_parallel_size * tensor_model_parallel_size * context_parallel_size``
    - ``dp_size = world_size / model_size``
    - ``world_size % (pipeline_model_parallel_size * expert_model_parallel_size * expert_tensor_parallel_size) == 0``

    The last rule means ``expert_model_parallel_size * expert_tensor_parallel_size`` can scale
    independently of ``tensor_model_parallel_size * context_parallel_size``, and can span data
    parallel ranks. See https://docs.skyrl.ai/docs/examples/megatron for sizing guidance.
    """

    tensor_model_parallel_size: int = 1
    """Tensor model parallel size, reducing memory for model parameters and activations.
    Megatron sequence parallelism (unrelated to Ulysses ``sequence_parallel_size``) is enabled automatically whenever
    this is greater than 1."""
    pipeline_model_parallel_size: int = 1
    """Pipeline model parallel size, sharding model layers across GPUs."""
    context_parallel_size: int = 1
    """Context parallel size, reducing activation memory along the sequence-length dimension."""
    expert_model_parallel_size: int = 1
    """Expert parallel size, sharding expert modules across GPUs."""
    expert_tensor_parallel_size: Optional[int] = None
    """Tensor parallel size for each expert module.
    ``None`` lets Megatron resolve it to ``tensor_model_parallel_size``. Setting this to ``1`` is recommended for best
    performance when ``expert_model_parallel_size > 1``."""
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
    """Enable Megatron router replay.
    Used together with ``generator.inference_engine.enable_return_routed_experts`` to enable R3."""
    moe_per_layer_logging: bool = False
    """Enable per-layer logging of MoE metrics (i.e. per layer aux losses)."""
    moe_router_dtype: str = "fp32"
    """Pass through to Megatron-Bridge - can be set to 'fp64' for additional numerical stability."""
    ddp_config: MegatronDDPConfig = field(default_factory=MegatronDDPConfig)
    """Pass-through config for Megatron's ``DistributedDataParallelConfig``:
    https://github.com/NVIDIA/Megatron-LM/blob/core_r0.13.0/megatron/core/distributed/distributed_data_parallel_config.py"""
    hf_export_config: MegatronHFExportConfig = field(default_factory=MegatronHFExportConfig)
    lora_config: MegatronLoraConfig = field(default_factory=MegatronLoraConfig)
    optimizer_config_kwargs: Dict[str, Any] = field(
        default_factory=lambda: copy.deepcopy(DEFAULT_MEGATRON_OPTIMIZER_KWARGS)
    )
    """Pass-through kwargs for Megatron's ``OptimizerConfig``.

    https://github.com/NVIDIA/Megatron-LM/blob/core_r0.13.0/megatron/core/optimizer/optimizer_config.py
    Any keys overlapping with what SkyRL resolves from ``optimizer_config`` are overridden by the
    values here. ``*_dtype`` keys accept case-insensitive dtype names, coerced by
    ``distributed.megatron.optimizer_dtype``; see
    https://docs.skyrl.ai/docs/examples/megatron for the accepted names and per-field checks.
    ``use_precision_aware_optimizer=True`` can cause checkpointing to fail
    (https://github.com/nvidia/megatron-lm/issues/1820); leaving it ``False`` is recommended."""
    transformer_config_kwargs: Dict[str, Any] = field(
        default_factory=lambda: copy.deepcopy(DEFAULT_TRANSFORMER_CONFIG_KWARGS)
    )
    """Pass-through kwargs for Megatron's ``TransformerConfig``:
    https://github.com/NVIDIA/Megatron-LM/blob/core_r0.13.0/megatron/core/transformer/transformer_config.py
    Also the place to put HuggingFace config overrides (e.g. ``rope_parameters``) for the Megatron
    backend, where FSDP would use ``model_config_kwargs``."""
    empty_cuda_cache: Optional[bool] = True
    """Manually empty torch's CUDA cache between the forward/backward pass and the optimizer step.
    This frees reserved-but-unallocated memory and can help avoid OOMs in the optimizer."""
    model_config_kwargs: dict = field(default_factory=dict)
    """HF-config overrides read from the nested ``model_config`` key only.
    Used for bridge and RoPE resolution. Not a general HF-config override -- use
    ``transformer_config_kwargs`` instead."""
    dist_ckpt_optim_fully_reshardable: bool = False
    """When True, use the "fully-reshardable" format for the distributed-optimizer checkpoint.
    When False (default), use the "dp-reshardable" format, which is more efficient but only
    supports resharding along the data-parallel dimension. See
    https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.0/megatron/core/optimizer/distrib_optimizer.py"""
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
    async_dist_ckpt_save: bool = False
    """Write the torch_dist checkpoint from a background process so training resumes
    immediately; the pending write is finalized at the next checkpoint and at shutdown.
    The on-disk format is identical to a synchronous save. Only the sharded
    model/optimizer state is async -- the rank-0 HF config/tokenizer write stays inline.
    Falls back to synchronous for cloud paths."""
    async_dist_ckpt_strategy: str = "mcore"
    """Backend for the async write. ``mcore`` needs no extra deps; megatron-core's own
    default ``nvrx`` requires nvidia-resiliency-ext. Only used when async saves are on."""
    async_save_prestage_to_cpu: bool = False
    """Copy shards to host memory on the training rank before handing them to the async
    checkpoint writer, instead of letting the writer pull them over CUDA IPC.

    Enable this when using async save on machines with restricted ptrace permissions. 
    megatron-core hands the writer the *GPU* tensors and copies them in the writer process;
    with ``expandable_segments:True`` those handles are file descriptors that the writer 
    imports via ``pidfd_getfd``, which needs ptrace-attach permission on the rank. 
    Where that is refused (e.g. ``kernel.yama.ptrace_scope=1``, as on some CI runners) 
    the writer dies with ``pidfd_getfd: Operation not permitted`` and every rank then 
    hangs on the preload barrier.

    Off by default: prestaging shrinks the rank's blocking
    window but delays the background write (per-tensor shared-memory handoff cost).
    See ``_stage_async_request_to_host``."""

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
    """GPU placement and colocation for the policy, critic, ref, and inference engines.

    See the model placement and colocation guide for an in-depth walkthrough of how the
    options here interact: https://docs.skyrl.ai/docs/tutorials/placement
    """

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
    """Optimizer configuration for the policy model."""
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)
    """FSDP configuration, applicable when ``trainer.strategy="fsdp"``."""
    sequence_parallel_size: int = 1
    """Ulysses sequence parallel size (https://arxiv.org/abs/2309.14509).
    Distinct from the Megatron sequence parallelism implied by ``megatron_config.tensor_model_parallel_size > 1``."""
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
    """Critic model configuration.

    Supports a subset of the policy options (model/LoRA, optimizer, FSDP, sequence parallelism).
    FSDP only -- a critic is rejected under ``trainer.strategy="megatron"``.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer_config: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(lr=5e-6))
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)
    """FSDP configuration, applicable when ``trainer.strategy="fsdp"``."""
    sequence_parallel_size: int = 1
    """Ulysses sequence parallel size (https://arxiv.org/abs/2309.14509)."""
    model_config_kwargs: dict = field(default_factory=dict)
    """Pass-through kwargs for the HuggingFace model config (e.g. overriding vocab size)."""


# TODO: Have global config init so that the default value for the ref model path is the policy model path
@dataclass
class RefConfig(BaseConfig):
    """Reference model configuration.

    NOTE: the reference model is only used when base-model log probabilities are needed, either as
    part of the training loss or as part of the reward. So ``trainer.algorithm.use_kl_in_reward`` or
    ``trainer.algorithm.use_kl_loss`` must be True for it to be used at all — if both are False the
    reference model is never instantiated.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    """Reference model.
    ``model.path`` defaults to ``trainer.policy.model.path``, but can be set separately — e.g. for distillation-style
    approaches where the reference differs from the policy."""
    sequence_parallel_size: int = 1
    """Ulysses sequence parallel size (https://arxiv.org/abs/2309.14509)."""
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)
    """FSDP configuration, applicable when ``trainer.strategy="fsdp"``."""
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
    """SAPO parameters (https://arxiv.org/pdf/2511.20347). Only used when ``policy_loss_type="sapo"``."""

    tau_pos: float = 1.0
    """Temperature for the gating function on tokens with positive advantages."""
    tau_neg: float = 1.05
    """Temperature for the gating function on tokens with negative (or zero) advantages.
    The default matches the value used in the paper with Qwen3-30B-A3B-Base."""


@dataclass
class DynamicSamplingConfig(BaseConfig):
    """Dynamic sampling configuration: resample or filter training batches based on reward signal."""

    type: Optional[str] = None
    """Dynamic sampling strategy: ``"filter"``, ``"replace"``, or ``None`` for no dynamic sampling.
    ``"filter"`` is DAPO (https://dapo-sia.github.io/); ``"replace"`` is POLARIS
    (https://hkunlp.github.io/blog/2025/Polaris/) / WebSailor (https://arxiv.org/abs/2507.02592)."""
    max_sample_batches: int = 30
    """Sample at most this many batches before stopping. ``-1`` to sample forever."""
    min_replace_ratio: float = 0.3
    """Minimum proportion of good samples to replace bad samples. Only used with ``"replace"`` strategy."""


@dataclass
class ClipCovConfig(BaseConfig):
    """Clip-Cov parameters. Only used when ``policy_loss_type="clip_cov"``."""

    clip_ratio: float = 0.0002
    """Fraction of tokens to clip based on covariance."""
    clip_cov_lb: float = 1.0
    """Lower bound for covariance clipping."""
    clip_cov_ub: float = 5.0
    """Upper bound for covariance clipping."""


@dataclass
class KLCovConfig(BaseConfig):
    """KL-Cov parameters. Only used when ``policy_loss_type="kl_cov"``."""

    kl_cov_frac: float = 0.2
    """Fraction of tokens to apply KL regularization to."""
    ppo_kl_coef: float = 1.0
    """Coefficient for the KL regularization term."""


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
    cispo_anchor: str = "old"
    """Behavior policy the IS ratio is anchored on: ``"old"`` (default) uses the recomputed
    old log-probs (``ratio = pi_theta / pi_old``), matching the original CISPO paper. ``"rollout"``
    uses the rollout/sampler log-probs (``ratio = pi_theta / pi_rollout``), which makes the clamped
    objective engage under fully-async training where the sampler lags the trainer (with ``"old"``
    the ratio is ~1 at a single gradient step and the clamp never bites). With ``"rollout"`` the
    ratio is the full off-policy correction, so ``off_policy_correction.tis_ratio_type`` must be
    ``None`` (else the off-policy gap is double-counted)."""

    def __post_init__(self):
        if self.cispo_anchor not in ("old", "rollout"):
            raise ValueError(f"cispo_anchor must be 'old' or 'rollout', got {self.cispo_anchor!r}")


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
    """Apply KL penalty to rewards, as ``rewards - kl * kl_loss_coef``.
    Mutually exclusive with ``use_kl_loss``."""
    use_kl_loss: bool = True
    """Apply KL loss in the policy model, as ``policy_loss + kl * kl_loss_coef``.
    Mutually exclusive with ``use_kl_in_reward``."""
    kl_loss_coef: float = 0.001
    """Coefficient for the KL divergence loss."""
    use_entropy_loss: bool = False
    """Add an entropy bonus to the policy loss.
    This encourages exploration by penalizing low-entropy (overly confident) policies."""
    entropy_loss_coef: float = 0.01
    """Coefficient for the entropy loss term. Only used when ``use_entropy_loss=True``."""
    temperature: Optional[float] = None
    """Temperature for scaling logits in policy loss computation.
    If ``None``, will be set to the temperature provided by ``generator.sampling_params.temperature`` during config validation.
    
    NOTE: When using HTTP endpoints directly, make sure to set this value to the temperature used during generation
    """
    advantage_batch_normalize: bool = False
    """Normalize advantages by the (global) training-batch mean and standard deviation."""
    value_head_prefix: str = "value_head"
    """Name used to identify the value head in the critic model."""
    policy_loss_type: str = "regular"
    """Type of policy loss to use, or custom via ``PolicyLossRegistry``:

    - ``"regular"``: vanilla PPO loss with token-level importance sampling.
    - ``"dual_clip"``: dual-clip PPO loss (https://arxiv.org/pdf/1912.09729).
    - ``"gspo"``: Group Sequence Policy Optimization (https://arxiv.org/abs/2507.18071) with
      sequence-level importance sampling for improved training stability. Implements the
      "GSPO-token" variant from the paper.
    - ``"clip_cov"``: combines standard PPO clipping with covariance-based correction masking for
      improved stability (https://arxiv.org/abs/2505.22617).
    - ``"kl_cov"``: applies KL regularization to tokens selected by covariance value
      (https://arxiv.org/abs/2505.22617).
    - ``"cispo"``: Clipped Importance Sampling Weight Policy Optimization, from MiniMax-M1
      (https://arxiv.org/abs/2506.13585).
    - ``"sapo"``: Soft Adaptive Policy Optimization (https://arxiv.org/html/2511.20347v1).
    - ``"rollout_is"``: the agentic loss from section 4.1.2 of the GLM-5 tech report
      (https://arxiv.org/pdf/2602.15763). Uses rollout logprobs and Icepop-style clipping with an
      additional stop gradient for masked tokens.
    - ``"cross_entropy"`` and ``"importance_sampling"``: also registered; see ``PolicyLossRegistry``.
    - ``"dppo"``: DPPO, from Rethinking the Trust Region in LLM Reinforcement Learning
      (https://arxiv.org/pdf/2602.04879). Uses rollout logprobs and absolute probability
      divergences rather than probability ratios, improving on PPO clipping behavior.
    """
    loss_reduction: str = "token_mean"
    """Type of loss reduction to use, applied per mini-batch by rescaling advantages:

    - ``"token_mean"``: average loss over all valid tokens in the batch, as in DAPO
      (https://dapo-sia.github.io/).
    - ``"sequence_mean"``: per-sequence average token loss, then averaged over the batch.
    - ``"seq_mean_token_sum_norm"``: sum of token losses per sequence, normalized by
      ``max_seq_len``, then averaged over the batch, as in Dr. GRPO
      (https://arxiv.org/abs/2503.20783). ``max_seq_len`` must be set explicitly for this mode,
      because multi-turn/token budgets are workload-dependent.
    - ``"prompt_mean"``: average token loss within each prompt group (all
      ``generator.n_samples_per_prompt`` responses sampled for a prompt), then averaged over
      prompts. Unlike ``"token_mean"``, every prompt contributes equally regardless of how many
      tokens its responses contain.
    - ``"token_mean_legacy"``: also accepted, retaining the previous ``"token_mean"`` behavior.
    """
    grpo_norm_by_std: bool = True
    """Normalize advantages by the standard deviation in GRPO.
    Set to False for Dr. GRPO (https://arxiv.org/abs/2503.20783)."""
    zero_variance_filter: bool = False
    """Loss-mask prompts with zero-variance rewards. Only applicable when rewards are response-level."""
    zero_variance_filter_tol: float = 1e-6
    """Two rewards within this absolute tolerance count as equal when detecting zero-variance groups.
    Only used when ``zero_variance_filter=True``. Defaults to 1e-6 so float (LLM-judge) rewards that are
    effectively identical are still treated as zero-variance; this is a no-op for integer rewards (e.g.
    0/1) where the spread is either 0 or >= 1. Set to 0.0 for exact equality."""
    lambd: float = 1.0
    """Lambda parameter for GAE."""
    gamma: float = 1.0
    """Gamma (discount) parameter for GAE."""
    eps_clip_low: float = 0.2
    """Lower bound for PPO clipping."""
    eps_clip_high: float = 0.2
    """Upper bound for PPO clipping."""
    clip_ratio_c: float = 3.0
    """Dual-clip parameter."""
    tis_imp_ratio_cap: float = -1.0
    """Deprecated: use ``off_policy_correction.tis_ratio_type="token"`` and ``token_tis_ratio_clip_high`` instead."""
    use_tis: bool = False
    """Deprecated: use ``off_policy_correction`` instead.
    Enabled Truncated Importance Sampling (TIS) as proposed in https://fengyao.notion.site/off-policy-rl."""
    off_policy_correction: OffPolicyCorrectionConfig = field(default_factory=OffPolicyCorrectionConfig)
    """See https://docs.skyrl.ai/docs/algorithms/off_policy_correction for a full guide."""
    enable_sample_support_replay: bool = False
    """Renormalize policy logprobs over the sampler's recorded bounded support. Requires
    ``generator.inference_engine.enable_return_sample_support_set`` to capture it."""
    sapo: SAPOConfig = field(default_factory=SAPOConfig)
    """Only used when ``policy_loss_type="sapo"``."""
    value_clip: float = 0.2
    """Clip value for the value loss."""
    dynamic_sampling: DynamicSamplingConfig = field(default_factory=DynamicSamplingConfig)
    """Dynamic sampling configuration."""
    clip_cov: ClipCovConfig = field(default_factory=ClipCovConfig)
    """Only used when ``policy_loss_type="clip_cov"``."""
    kl_cov: KLCovConfig = field(default_factory=KLCovConfig)
    """Only used when ``policy_loss_type="kl_cov"``."""
    cispo: CISPOConfig = field(default_factory=CISPOConfig)
    """Only used when ``policy_loss_type="cispo"``."""
    dppo: DPPOConfig = field(default_factory=DPPOConfig)
    """Only used when ``policy_loss_type="dppo"``."""
    max_seq_len: Optional[int] = None
    """Sequence-length normalization constant used for ``seq_mean_token_sum_norm`` loss reduction.
    Must be set explicitly for that reduction mode; otherwise can remain ``None``. This often
    matches the model context window / vLLM ``max_model_len`` when that is the intended budget."""


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
    """If True, run fully-async generation with a SIMULATED trainer (see ``FullyAsyncTrainerSim``).

    No policy/critic/ref models are instantiated and no weight broadcast happens. Each step consumes
    a mini-batch from the generation buffer, sleeps for ``simulate_training_step_seconds``, then
    issues pause/resume generation (as a real weight sync would) but skips
    ``broadcast_to_inference_engines``. The generation-side dynamics (staleness control, rate
    limiting, pause/resume) remain faithful.

    Because no models are built, this requires ``trainer.eval_interval``, ``trainer.ckpt_interval``,
    and ``trainer.hf_save_interval`` to all be ``<= 0``, ``trainer.update_ref_every_epoch=False``,
    and resumption to be disabled. See
    https://docs.skyrl.ai/docs/tutorials/fully_async for usage."""
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
    """Sampling parameters passed to the inference engine during generation."""

    max_generate_length: int = 1024
    """Maximum length of the generated response."""
    repetition_penalty: float = 1.0
    """Repetition penalty. ``1.0`` applies no penalty.
    Not forwarded by the typed sampling-params path -- pass it via ``additional_kwargs``."""
    temperature: float = 1.0
    """Sampling temperature.
    Automatically propagated to ``trainer.algorithm.temperature`` during config initialization."""
    top_p: float = 1.0
    """Top-p (nucleus) sampling parameter."""
    min_p: float = 0.0
    """Min-p sampling parameter, as proposed in https://arxiv.org/pdf/2407.01082."""
    top_k: int = -1
    """Top-k sampling parameter. ``-1`` disables it."""
    logprobs: Optional[int] = 1
    """Number of logprobs to return from the inference engine.
    Must be ``None``, ``0``, or ``1``; both ``0`` and ``1`` return only the chosen token's
    logprob, and larger values are rejected."""
    stop: Optional[List[str]] = None
    """Optional list of stop strings for generation."""
    additional_kwargs: Optional[Dict[str, Any]] = None
    """Extra sampling kwargs passed through to the inference engine."""


@dataclass
class ChatTemplateConfig(BaseConfig):
    """Custom chat template configuration."""

    source: str = "name"
    """``"name"`` to select a built-in template, or ``"file"`` to load one from disk."""
    name_or_path: Optional[str] = None
    """Selects the template, interpreted according to ``source``.
    When ``source="name"``, one of the supported templates in ``skyrl/train/generators/utils.py``
    (e.g. ``"qwen3_with_thinking"``). When ``source="file"``, the path to a Jinja2 template file."""


# ---------------------------------------------------------------------------
# Inference Engine
# ---------------------------------------------------------------------------


@dataclass
class DeltaWeightSyncConfig(BaseConfig):
    """Disk/cloud checkpoint-delta weight sync configuration."""

    sync_dir: str
    """Shared directory/URI where the trainer publishes per-version delta payloads.
    Supports local paths, ``gs://`` URIs, and ``s3://`` URIs."""

    local_checkpoint_dir: Optional[str] = None
    """Receiver-side directory used to cache patched checkpoint versions.
    If unset, resolved in ``__post_init__`` to a ``sync_dir``-derived path under
    ``/tmp/skyrl_delta_checkpoints``."""

    publish_staging_dir: Optional[str] = None
    """Trainer-side local directory used to stage cloud payload files before upload.
    If unset, resolved in ``__post_init__`` to a ``sync_dir``-derived path under
    ``/tmp/skyrl_delta_publish_staging``."""

    max_file_size_in_gb: float = 1.0
    """Maximum compressed payload file size before starting a new safetensors file."""

    cloud_download_workers: int = 4
    """Maximum number of payload files to download concurrently for ``gs://`` and ``s3://`` sync dirs."""

    publish_num_workers: Optional[int] = None
    """Number of trainer-side worker threads used to compute and compress delta payloads.
    If unset, the publisher uses ``min(8, os.cpu_count())``."""

    checkpoint_load_format: Literal["vllm_multi_thread_safetensors", "vllm_fastsafetensors"] = (
        "vllm_multi_thread_safetensors"
    )
    """Receiver reload iterator for the prepared local checkpoint.
    
    `vllm_multi_thread_safetensors` loads safetensor files from disk to CPU storage with N parallel workers using vLLM's native safetensors iterator. Tensors are then loaded onto GPU memory iterately.

    `vllm_fastsafetensors` loads tensors from safetensor files on disk directly into GPU memory in a highly parallelized way.
    This setting is currently not recommended for large models because of large memory requirements.
    See: https://github.com/vllm-project/vllm/issues/48644 for more details
    """

    multi_thread_safetensors_max_workers: int = 8
    """Number of worker threads for ``vllm_multi_thread_safetensors``."""

    def __post_init__(self) -> None:
        from skyrl.backends.skyrl_train.weight_sync.delta_checkpoint import (
            _default_local_checkpoint_dir,
            _default_publish_staging_dir,
        )

        if self.local_checkpoint_dir is None:
            self.local_checkpoint_dir = str(_default_local_checkpoint_dir(self.sync_dir))
        if self.publish_staging_dir is None:
            self.publish_staging_dir = str(_default_publish_staging_dir(self.sync_dir))


@dataclass
class InferenceEngineConfig(BaseConfig):
    """Configuration for inference engine instantiation and management."""

    model_dtype: str = "bfloat16"
    """Should match the dtype used by the inference engine.
    Also used during full-weight sync, where policy weights are cast to this dtype before being sent
    to the inference engine. The LoRA-adapter sync path exports fp32 instead."""
    run_engines_locally: bool = True
    """Launch inference servers during the training run in the current Ray cluster.
    When ``False``, point SkyRL at an external HTTP/vLLM deployment via ``external_proxy_url`` and/or
    ``external_server_urls``. See https://docs.skyrl.ai/docs/tutorials/placement"""
    num_engines: int = 1
    """Number of inference engines to launch when ``run_engines_locally=True``."""
    backend: str = "vllm"
    """``"vllm"``."""
    weight_sync_backend: str = "nccl"
    """Backend used for weight synchronization.
    Use ``"nccl"`` (colocated ``nccl`` uses CUDA IPC internally), or ``"delta"`` for checkpoint-delta sync through
    shared storage in non-colocated vLLM runs. See https://docs.skyrl.ai/docs/examples/delta_weight_sync"""
    weight_transfer_threshold_cuda_ipc_GB: float = 1.0
    """When using ``cuda_ipc``, send weights in batches of this size (GB)."""
    delta_weight_sync: Optional[DeltaWeightSyncConfig] = None
    """Required when ``weight_sync_backend="delta"``."""
    tensor_parallel_size: int = 1
    """Tensor parallel size for the inference engine."""
    pipeline_parallel_size: int = 1
    """Pipeline parallel size for the inference engine. Currently only supported for vLLM."""
    expert_parallel_size: int = 1
    """Expert parallel size for the inference engine.
    Currently only supported for vLLM. When set > 1, must equal
    ``data_parallel_size * tensor_parallel_size``."""
    data_parallel_size: int = 1
    """Data parallel size for the inference engine. Currently only supported for vLLM."""
    vllm_v1_disable_multiproc: bool = True
    """Currently inert.
    ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` is set unconditionally whenever ``VLLM_USE_V1`` is absent
    from the environment, which makes vLLM scheduling deterministic.
    Useful for reproducibility."""
    enable_prefix_caching: bool = True
    """Enable vLLM prefix caching.
    Can be left at the default in most cases. With remote inference servers, this must match the setting the remote
    servers were initialized with."""
    enable_chunked_prefill: bool = True
    """Enable vLLM chunked prefill.
    Currently not plumbed through to the engine; set ``engine_init_kwargs.enable_chunked_prefill``
    to change it."""
    enable_return_routed_experts: bool = False
    """Return per-layer expert routing indices, for rollout router replay (R3) when training an MoE model.
    Used together with ``trainer.policy.megatron_config.moe_enable_routing_replay``."""
    enable_return_sample_support_set: bool = False
    """Return the bounded sampler support used to renormalize rollout logprobs."""
    max_num_batched_tokens: int = 8192
    """vLLM continuous-batching parameter: maximum number of tokens to pack into a batch."""
    enforce_eager: bool = False
    """Disable CUDA graphs.
    Enabling this trades performance for stability; leaving it off (the default) is faster but may affect convergence
    for long-running or long-context training jobs."""
    fully_sharded_loras: bool = False
    enable_ray_prometheus_stats: bool = True
    """Enable Ray Prometheus stats logger for inference engine metrics (vLLM v1 only)."""
    gpu_memory_utilization: float = 0.8
    """GPU memory utilization for the inference engine.
    Only applicable when ``run_engines_locally=True``."""
    offload_kv_for_weight_sync: bool = False
    """Non-colocated only. Sleep the engine (freeing the KV cache from GPU) during weight
    sync so ``gpu_memory_utilization`` can be pushed higher without OOMing on the weight-
    transfer buffers. On the fully-async trainer, in-flight requests are frozen (KEEP
    pause) and, unless ``trainer.fully_async.clear_kv_cache_on_weight_sync`` is set, their
    KV cache is offloaded to CPU and restored so they resume with no abort or prefill
    recompute (at the cost of a GPU<->CPU copy of the KV pool each sync). Requires
    non-colocated and non-LoRA weight sync."""
    use_expandable_segments: bool = False
    """Set ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` on the inference-engine
    processes to reduce fragmentation. Independent of the trainer-side
    ``TrainerConfig.use_expandable_segments``. Default ``False``: it is a safe opt-in
    on vLLM >= 0.20.1, where the CuMemAllocator auto-disables expandable segments around
    its sleep/wake memory pool. On older vLLM, sleep mode + expandable segments is a hard
    error, so leave this off."""
    max_num_seqs: int = 1024
    """vLLM continuous-batching parameter: maximum number of sequences to pack into a batch."""
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
    """Pass-through kwargs for the vLLM engine.
    Names must match the engine's args. Applied last, so they silently override config-derived
    engine args (e.g. ``tensor_parallel_size``).

    For HuggingFace config overrides such as RoPE scaling, use
    ``engine_init_kwargs.hf_overrides.rope_parameters`` and set the matching trainer-side override
    with ``trainer.policy.model_config_kwargs.rope_parameters`` (FSDP) or
    ``trainer.policy.megatron_config.transformer_config_kwargs.rope_parameters`` (Megatron). The two
    must agree, and are validated against each other."""
    speculative_config: Optional[Dict[str, Any]] = None
    """Speculative-decoding config passed through to vLLM for MTP drafter decoding. 
    (needs ``policy.megatron_config.mtp_num_layers`` > 0 to train mtp). ``None`` disables it."""
    external_proxy_url: Optional[str] = None
    """Data-plane URL (load-balanced router) for the new inference layer.
    Generation requests are sent here."""
    external_server_urls: Optional[List[str]] = None
    """Control-plane URLs (direct backend access) for the new inference layer.
    Used to fan out pause/resume, sleep/wake, and weight sync. If ``external_proxy_url`` is omitted,
    SkyRL starts an internal router over these servers."""
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
    """Number of samples to generate per prompt.
    The total size of the training batch is ``trainer.train_batch_size * n_samples_per_prompt``."""
    batched: bool = False
    """Use batched inference. Only applicable for single-turn generation."""
    max_turns: int = 1
    """Maximum number of turns for multi-turn RL generation."""
    max_input_length: Optional[int] = None
    """Max generator input length.
    For single-turn generation this can equal ``trainer.max_prompt_length`` (the initial prompt length); for multi-turn
    it is the maximum input length used for the conversation at each turn. Defaults to ``trainer.max_prompt_length``."""
    chat_template: ChatTemplateConfig = field(default_factory=ChatTemplateConfig)
    """Custom chat template configuration, if needed."""
    chat_template_kwargs: Dict[str, Any] = field(default_factory=dict)
    """Kwargs passed to ``tokenizer.apply_chat_template``.
    Requires non-batched generation: a non-empty value with ``batched=True`` raises."""
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    """Sampling parameters used during the trajectory generation phase."""
    use_conversation_multi_turn: bool = True
    """If ``True``, each multi-turn model response and env observation is stored in a separate assistant/user message.
    If ``False``, they are appended to the original assistant response as-is in token space and generation continues
    (after removing any EOS token in the response). Models can be sensitive to chat-history format (as observed in
    SkyRL-SQL), so ``False`` gives full control over the exact tokens added after environment interaction."""
    append_eos_token_after_stop_str_in_multi_turn: bool = True
    """When ``use_conversation_multi_turn=True`` and ``sampling_params.stop`` is set, append
    ``eos_token_id`` to generations that end with a matched stop string."""
    eval_sampling_params: Optional[SamplingParams] = None
    """Separate sampling params for evaluation. If ``None``, then it defaults to ``SamplingParams(temperature=0.0, max_generate_length=generator.sampling_params.max_generate_length)``."""
    eval_n_samples_per_prompt: int = 1
    """Number of samples to generate per prompt during evaluation."""
    zero_reward_on_non_stop: bool = False
    """Set reward to 0 when ``stop_reason`` is not ``"stop"`` (i.e., generation was truncated or aborted).
    Useful with format rewards, where an unfinished response should not earn format credit.
    Applies to all environments."""
    use_cache_salt: bool = True
    """Salt vLLM's prefix cache with the policy version so cache blocks are only shared across trajectories that started
    with the same policy weight version. The salt is keyed on the engine's weight version, captured at the start of each
    ``generate`` call. Matters for fully-async RL; a no-op for synchronous training (which resets the
    cache each sync) and when prefix caching is off, so it is safe to leave on by default."""
    apply_overlong_filtering: bool = False
    """Apply DAPO Overlong Filtering: mask out all tokens in the loss mask for trajectories that
    exceed max length (truncated, no EOS token)."""
    step_wise_trajectories: bool = False
    """Return outputs step-wise.
    When ``True``, multi-turn generations are returned with each turn's (prompt, response) pair as a separate
    trajectory. Advantages are computed from the last step of each trajectory and propagated to the previous steps. See
    https://docs.skyrl.ai/docs/tutorials/step-wise-training"""
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
    """Enable PyTorch's CUDA ``expandable_segments`` allocator on the training workers.

    Reduces GPU memory fragmentation across the offload/backload and forward/backward cycles.
    Automatically turned off around CUDA-IPC weight sync, since IPC handles are incompatible with the
    VMM addresses expandable segments uses; under ``colocate_all=False`` weight sync uses NCCL
    broadcast instead, so it stays on continuously.
    ``InferenceEngineConfig.use_expandable_segments`` is the independent inference-engine knob (the
    trainer and the engine are separate processes with separate allocators). See
    https://docs.skyrl.ai/docs/troubleshooting/troubleshooting for the fragmentation symptoms this
    addresses."""
    sequence_parallel_backend: str = "ulysses"
    strategy: str = "fsdp"
    """Training backend: either ``"fsdp"`` or ``"megatron"``.
    ``"fsdp"`` uses PyTorch's composable ``fully_shard`` API (formerly known as FSDP2)."""
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    ref: RefConfig = field(default_factory=RefConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    mtp: MTPConfig = field(default_factory=MTPConfig)
    fully_async: FullyAsyncConfig = field(default_factory=FullyAsyncConfig)
    gradient_checkpointing: bool = True
    """Use gradient checkpointing (activation recomputation) to trade compute for memory."""
    gradient_checkpointing_use_reentrant: bool = False
    seed: int = 42
    """Random seed for training."""
    resume_mode: Optional[str] = "latest"
    """``None``/``"none"``, ``"latest"``, or ``"from_path"``.
    See https://docs.skyrl.ai/docs/checkpointing-logging/checkpointing"""
    resume_path: Optional[str] = None
    """Checkpoint directory to resume from. Only used when ``resume_mode="from_path"``."""
    log_path: str = "/tmp/skyrl-logs"
    """Path for infrastructure log files.
    vLLM engine startup, model loading, and worker initialization logs are written to
    ``{log_path}/infra-YYMMDD_HHMMSS.log``. For multi-node training, use a shared filesystem path to
    consolidate logs into a single file. See https://docs.skyrl.ai/docs/checkpointing-logging/logging"""
    ckpt_path: str = field(default_factory=lambda: os.path.expanduser("~/ckpts/"))
    """Directory for resumable training checkpoints (model state, optimizer state, etc.).
    Accepts a local directory path or a cloud storage path (S3, GCS)."""
    max_ckpts_to_keep: int = -1
    """``-1`` to keep all checkpoints, ``N`` to keep only the last N."""
    ckpt_interval: int = 10
    """Save a full training checkpoint every N steps."""
    hf_save_interval: int = -1
    """Save HuggingFace-format model every N steps. ``-1`` to disable."""
    export_path: str = field(default_factory=lambda: os.path.expanduser("~/exports/"))
    """Path for exported artifacts (HF models, debug dumps, etc.).
    For sharded multi-node HF exports with ``policy.megatron_config.hf_export_config.distributed_save=True``, this must
    be a shared filesystem path visible to all Megatron ranks."""
    bf16: bool = True
    epochs: int = 1
    """Number of epochs (passes over the full dataset)."""
    max_training_steps: Optional[int] = None
    """If set, stop training after this many steps regardless of epochs or dataset size.
    Useful for CI smoke tests and quick validation runs."""
    update_epochs_per_batch: int = 1
    """Number of gradient update passes over each training batch.
    Equivalent to the concept of "PPO epochs", where the same experience is iterated over multiple times."""
    train_batch_size: int = 1024
    """Batch size of prompts used for each dataloader step.

    See ``utils/utils.py::validate_batch_sizes`` for the constraints relating this to
    ``policy_mini_batch_size``, ``micro_train_batch_size_per_gpu``, and
    ``micro_forward_batch_size_per_gpu``."""
    policy_mini_batch_size: int = 256
    """Mini batch size for the RL training step; each mini batch is one optimizer step.
    For example, with ``train_batch_size=4`` and ``policy_mini_batch_size=2`` there are 2 optimizer steps (model
    updates) per training batch. This is the *global* mini batch size, counted in prompts — the
    per-worker mini batch is
    ``policy_mini_batch_size * generator.n_samples_per_prompt / number of DP ranks``."""
    critic_mini_batch_size: int = 256
    """Like ``policy_mini_batch_size``, but for the critic model.
    The critic generally tolerates off-policy updates better than the policy, so setting this lower than
    ``policy_mini_batch_size`` (i.e. more critic updates) is usually preferable."""
    micro_train_batch_size_per_gpu: int = 1
    """Micro batch size during the training step, common to both policy and critic.
    Each mini batch is split into micro batches of this size, and gradients are accumulated over them."""
    micro_forward_batch_size_per_gpu: int = 1
    """Micro batch size during the forward pass, i.e. log probability or value computation.
    Common to both policy and critic. Each mini batch is split into micro batches of this size."""
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
    """Re-sync the reference model from the policy model at every epoch boundary."""
    remove_microbatch_padding: bool = True
    """Pack samples into the THD layout and strip intra-microbatch padding (requires flash attention).
    Common to all models."""
    eval_batch_size: int = 1024
    """Batch size for evaluation."""
    eval_before_train: bool = True
    """Evaluate the model once before training starts."""
    eval_interval: int = 5
    """Evaluate against the validation dataset every N steps. ``-1`` to disable evaluation."""
    max_prompt_length: int = 512
    """Maximum prompt length during training.
    Prompts longer than this are filtered out of the train/eval datasets at load time, not
    truncated."""
    flash_attn: bool = True
    disable_fast_tokenizer: bool = False
    project_name: str = "skyrl"
    """Project name in WandB and MLflow."""
    run_name: str = "test_run"
    """Run name in WandB and MLflow."""
    logger: str = "wandb"
    """Logger to use: ``"wandb"``, ``"mlflow"``, ``"swanlab"``, ``"tensorboard"``, or ``"console"``.
    See https://docs.skyrl.ai/docs/checkpointing-logging/logging"""
    enable_ray_gpu_monitor: bool = True
    """Enable background Ray GPU/RAM metrics collection and logging to wandb."""
    tags: Optional[List[str]] = None
    """Optional list of tags to apply to the W&B run. Has no effect on other backends."""
    dump_data_batch: bool = False
    """Dump each training data batch to a file for debugging.
    The batch at global step N is written to
    ``{export_path}/dumped_data/global_step_{N}_training_input.pkl``."""
    dump_eval_results: bool = True
    """Dump full evaluation results to a file.
    Results at global step N are written to ``{export_path}/dumped_evals/global_step_{N}_evals``, with both per-dataset
    and aggregated results when multiple validation datasets are configured."""
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


def overrides_dict_to_dotlist(args: Dict[str, Any]) -> List[str]:
    """Serialize a dict of config overrides into an OmegaConf dotlist.

    ``OmegaConf.from_cli`` re-parses the right-hand side of each ``key=value``
    entry using YAML scalar rules, so values are serialized as JSON: JSON is a
    subset of YAML, so ``None`` becomes ``null``, bools become ``true``/``false``,
    and strings are quoted, which suppresses YAML scalar interpretation of values
    like ``"null"``, ``"true"``, ``"1e5"``, ``"[a]"``, ``"a: b"`` and ``""``.

    ``ensure_ascii=False`` keeps non-ASCII characters literal. With the default
    ``\\uXXXX`` escaping, characters outside the basic multilingual plane are
    emitted as a surrogate pair, and OmegaConf decodes each half into a separate
    lone surrogate that later fails to UTF-8 encode.

    Values JSON cannot represent are serialized with ``str()``.

    Args:
        args: Mapping of dot-notation config keys to override values.

    Returns:
        A list of ``key=value`` strings suitable for ``OmegaConf.from_cli``.
    """
    dotlist = []
    for key, value in args.items():
        try:
            serialized = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            # TypeError: unsupported type. ValueError: circular reference.
            serialized = str(value)
        dotlist.append(f"{key}={serialized}")
    return dotlist


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
    """Root configuration object for SkyRL training with the ``fsdp`` and ``megatron`` backends.

    Every field is overridable from the CLI in ``key.path=value`` form (see
    ``from_cli_overrides``): ``trainer.policy.model.path=...`` sets
    ``SkyRLTrainConfig.trainer.policy.model.path``.
    """

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

        if self.trainer.algorithm.enable_sample_support_replay:
            if not self.generator.inference_engine.enable_return_sample_support_set:
                raise ValueError(
                    "trainer.algorithm.enable_sample_support_replay requires "
                    "generator.inference_engine.enable_return_sample_support_set"
                )
            if self.trainer.strategy != "megatron":
                raise ValueError(
                    "sample-support replay requires trainer.strategy=megatron, got " f"{self.trainer.strategy}"
                )

        # Eval requests opt out of capture and do not use these constraints.
        if self.generator.inference_engine.enable_return_sample_support_set:
            sampling_params = self.generator.sampling_params
            if sampling_params.temperature <= 0:
                raise ValueError("sample-support capture requires generator.sampling_params.temperature > 0")
            if sampling_params.top_k <= 1:
                raise ValueError("sample-support capture requires generator.sampling_params.top_k > 1")
            if sampling_params.repetition_penalty != 1.0:
                raise ValueError("sample-support capture requires repetition_penalty=1.0")
            if sampling_params.additional_kwargs:
                raise ValueError("sample-support capture does not support sampling_params.additional_kwargs")
            if self.generator.vision_language_generator:
                raise ValueError("sample-support capture does not support vision_language_generator")

        # The VLM generator does not populate routed-expert indices.
        if self.generator.inference_engine.enable_return_routed_experts and self.generator.vision_language_generator:
            raise ValueError("rollout router replay (r3) does not support vision_language_generator")

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
                  Dict values are serialized as JSON, so ``None``, bools, strings,
                  lists and nested dicts keep their types.

        Returns:
            A fully constructed SkyRLTrainConfig with CLI overrides applied.

        Raises:
            ValueError: If an argument uses the unsupported '+' prefix.
        """
        if isinstance(args, dict):
            args = overrides_dict_to_dotlist(args)

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

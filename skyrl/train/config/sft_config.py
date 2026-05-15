"""
SFT (Supervised Fine-Tuning) configuration.

Defines ``SFTConfig`` -- the user-facing config for SFT training -- and the
bridge function ``build_skyrl_config_for_sft`` that maps it to the internal
``SkyRLTrainConfig`` used by the SkyRL backend.
"""

import os
from dataclasses import dataclass, field
from enum import StrEnum
from typing import List, Optional, Union

from omegaconf import OmegaConf

from skyrl.train.config import (
    BaseConfig,
    FSDPConfig,
    MegatronConfig,
    ModelConfig,
    OptimizerConfig,
    SkyRLTrainConfig,
)

# ---------------------------------------------------------------------------
# TrainOnWhat enum
# ---------------------------------------------------------------------------


class TrainOnWhat(StrEnum):
    """Enum controlling which parts of the sequence to compute loss on.

    Members:
        LAST_ASSISTANT_MESSAGE: Train only on the final assistant message.
        ALL_ASSISTANT_MESSAGES: Train on every assistant message in the conversation.
    """

    LAST_ASSISTANT_MESSAGE = "last_assistant_message"
    ALL_ASSISTANT_MESSAGES = "all_assistant_messages"


# ---------------------------------------------------------------------------
# SFT-specific config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SFTPlacementConfig(BaseConfig):
    """Placement configuration for SFT training"""

    num_nodes: int = 1
    num_gpus_per_node: int = 4


@dataclass
class SFTConfig(BaseConfig):
    """Configuration for SFT training.

    Usage::

        cfg = SFTConfig(
            strategy="megatron",
            placement=SFTPlacementConfig(num_gpus_per_node=4),
            megatron_config=MegatronConfig(tensor_model_parallel_size=2,
                                    pipeline_model_parallel_size=2),
        )

    Or from CLI::

        cfg = SFTConfig.from_cli_overrides(sys.argv[1:])
    """

    @classmethod
    def from_cli_overrides(cls, args: Union[List[str], dict]) -> "SFTConfig":
        """Construct an SFTConfig from CLI arguments or a dict of overrides.

        Parses CLI dotlist arguments via OmegaConf and builds a typed config.
        Dataclass field defaults are used for any values not specified.

        Args:
            args: Either a list of CLI arguments in 'key.path=value' format, or a dict
                  mapping dot-notation keys to values.
                  Example list: ['strategy=megatron', 'model.path=Qwen/Qwen3-0.6B']
                  Example dict: {'strategy': 'megatron', 'model.path': 'Qwen/Qwen3-0.6B'}

        Returns:
            A fully constructed SFTConfig with CLI overrides applied.

        Raises:
            ValueError: If both ``num_epochs`` and ``num_steps`` are explicitly provided.
        """
        if isinstance(args, dict):
            args = [f"{k}={v}" for k, v in args.items()]

        overrides = OmegaConf.from_cli(args)
        # Check for mutual exclusion before constructing the full config
        if "num_epochs" in overrides and "num_steps" in overrides:
            raise ValueError("Cannot specify both num_epochs and num_steps")
        return cls.from_dict_config(overrides)

    # ---- Reused SkyRL config objects ----
    model: ModelConfig = field(default_factory=lambda: ModelConfig(path="Qwen/Qwen3-0.6B"))
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    placement: SFTPlacementConfig = field(default_factory=SFTPlacementConfig)
    megatron_config: MegatronConfig = field(
        default_factory=lambda: MegatronConfig(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
        )
    )
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)

    # Ulysses sequence parallelism
    sequence_parallel_size: int = 1
    """Ulysses sequence parallelism size"""

    model_config_kwargs: dict = field(default_factory=dict)
    """Pass-through kwargs for the HuggingFace model config (FSDP backends).
    For Megatron, use ``megatron_config.transformer_config_kwargs`` instead."""
    use_torch_compile: bool = False
    """Apply torch.compile to logits calculation."""
    record_memory: bool = False
    """Save memory snapshots to ``{ckpt_path}/memory_snapshots/``.
    Visualize by dragging pickle files to https://docs.pytorch.org/memory_viz."""

    # ---- SFT-specific flat fields ----
    strategy: str = "megatron"  # "megatron" or "fsdp"
    dataset_name: str = "yahma/alpaca-cleaned"
    dataset_split: str = "train[:100]"
    messages_key: str = "messages"  # column name for chat-format datasets
    tools_key: str = "tools"
    """Column name holding per-row tool/function schemas for tool-calling datasets
    (e.g. APIGen-MT, xLAM, ToolACE). May be a list[dict] or a JSON-encoded string.
    Ignored if the column is absent from the dataset."""
    system_key: str = "system"
    """Column name holding a per-row system prompt to prepend when ``messages``
    does not already start with a system turn. Ignored if absent."""

    # ---- Evaluation dataset ----
    eval_dataset_name: Optional[str] = None
    """HuggingFace dataset name (or path) used to compute eval loss during training.
    When ``None`` (default), eval is disabled."""
    eval_dataset_split: str = "validation"
    """Split of the eval dataset to load (e.g. ``"validation"``, ``"test[:500]"``)."""
    eval_interval: int = 0
    """Run eval every N training steps. Eval also runs once at the end of training
    when an eval dataset is configured. ``0`` disables periodic eval."""
    eval_before_train: bool = False
    """If True, run a baseline eval pass before training begins (logged at step 0)."""
    max_length: Optional[int] = None
    """Maximum length of tokenized sequences. If specified, all sequences will be truncated to this value
    By default, no truncation is performed"""
    num_steps: Optional[int] = None
    """Number of training steps. If None, num_epochs is used to derive the step count."""
    num_epochs: Optional[int] = 1
    """Number of training epochs. Used when num_steps is None. Default: 1 epoch."""
    batch_size: int = 4
    micro_train_batch_size_per_gpu: int = 2
    logger: str = "console"  # "console" or "wandb"
    project_name: str = "skyrl_sft"
    run_name: str = "skyrl_sft_run"
    ckpt_path: str = ""  # empty string = no checkpointing
    ckpt_interval: int = 0
    max_ckpts_to_keep: int = -1
    """-1 to keep all checkpoints, N to keep only the last N."""
    resume_from: str = ""  # "" = no resume, "latest" = latest checkpoint, or path to global_step_N dir

    # ---- HF export ----
    hf_save_interval: int = 0
    """Save HuggingFace-format weights every N steps. 0 = disabled."""
    export_path: str = ""
    """Directory for HF-format exports. Defaults to ckpt_path/hf_exports if empty."""

    seed: int = 42

    # ---- Training target ----
    train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE
    """Which tokens to compute loss on. See :class:`TrainOnWhat` for options."""

    # ---- Packing ----
    use_sample_packing: bool = True  # Pack multiple sequences per batch (requires flash_attn)

    # ---- Dummy run / benchmarking ----
    dummy_run_full_ctx: bool = False  # Skip real data; fabricate full-context sequences
    dummy_run_max_steps: int = 5  # Number of steps to run in dummy mode


# ---------------------------------------------------------------------------
# Bridge: SFTConfig -> SkyRLTrainConfig
# ---------------------------------------------------------------------------


_VALID_STRATEGIES = ("megatron", "fsdp")


def validate_sft_cfg(cfg: SFTConfig) -> None:
    """Validate SFT-specific configuration.

    Only checks fields that are relevant to SFT training, unlike
    ``validate_cfg`` which includes RL-specific validations.
    """
    if cfg.strategy == "fsdp2":
        import warnings

        warnings.warn(
            "strategy='fsdp2' has been renamed to 'fsdp'; use 'fsdp' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        cfg.strategy = "fsdp"
    if cfg.strategy not in _VALID_STRATEGIES:
        raise ValueError(f"Unknown strategy '{cfg.strategy}'. Must be one of {_VALID_STRATEGIES}.")
    if cfg.micro_train_batch_size_per_gpu <= 0:
        raise ValueError(f"micro_train_batch_size_per_gpu must be > 0, got {cfg.micro_train_batch_size_per_gpu}")
    if cfg.batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {cfg.batch_size}")
    if cfg.num_steps is not None and cfg.num_steps <= 0:
        raise ValueError(f"num_steps must be > 0, got {cfg.num_steps}")
    if cfg.num_steps is None:
        if cfg.num_epochs is None:
            raise ValueError("One of num_steps or num_epochs must be set")
        if cfg.num_epochs <= 0:
            raise ValueError(f"num_epochs must be > 0, got {cfg.num_epochs}")
    if not cfg.model.path:
        raise ValueError("model.path must be set")
    if cfg.dummy_run_full_ctx and cfg.dummy_run_max_steps <= 0:
        raise ValueError(f"dummy_run_max_steps must be > 0, got {cfg.dummy_run_max_steps}")

    # Eval config
    if cfg.eval_interval < 0:
        raise ValueError(f"eval_interval must be >= 0, got {cfg.eval_interval}")
    if cfg.eval_interval > 0 and not cfg.eval_dataset_name:
        raise ValueError("eval_interval > 0 requires eval_dataset_name to be set")
    if cfg.eval_before_train and cfg.eval_dataset_name is None:
        raise ValueError("eval_before_train=True requires eval_dataset_name to be set")

    #  checks for megatron
    if cfg.strategy == "megatron":
        tp = cfg.megatron_config.tensor_model_parallel_size
        pp = cfg.megatron_config.pipeline_model_parallel_size
        cp = cfg.megatron_config.context_parallel_size
        total_world_size = cfg.placement.num_nodes * cfg.placement.num_gpus_per_node
        if total_world_size % (tp * pp * cp) != 0:
            raise ValueError(
                f"For megatron strategy, total_world_size must be divisible by TP * PP * CP. "
                f"Got TP={tp}, PP={pp}, CP={cp}, (TP*PP*CP={tp * pp * cp}), "
                f"total_world_size={total_world_size} "
                f"(num_nodes={cfg.placement.num_nodes} * num_gpus_per_node={cfg.placement.num_gpus_per_node})."
            )
        # context parallel are not yet supported for megatron
        if cfg.megatron_config.context_parallel_size > 1:
            assert cfg.use_sample_packing, "context parallel is only supported with sample packing"
        # check that sequence parallel is not configured outside of megatron
        assert cfg.sequence_parallel_size == 1, (
            f"found sequence_parallel_size={cfg.sequence_parallel_size}, ulysses style sequence "
            f"parallel is not supported for megatron"
        )


# NOTE (sumanthrh): Ideally this is not needed, but our internal abstractions for workers and worker groups depend
# on the RL configuration dataclass so we add this translation layer.
def build_skyrl_config_for_sft(sft_cfg: SFTConfig) -> SkyRLTrainConfig:
    """Map user-facing SFTConfig to the internal SkyRL backend config."""
    validate_sft_cfg(sft_cfg)

    cfg = SkyRLTrainConfig()

    # Strategy
    cfg.trainer.strategy = sft_cfg.strategy

    # Model -- direct assignment (same type: ModelConfig)
    cfg.trainer.policy.model = sft_cfg.model

    # Optimizer -- direct assignment (same type: OptimizerConfig)
    cfg.trainer.policy.optimizer_config = sft_cfg.optimizer_config

    # Placement -- map SFTPlacementConfig fields to PlacementConfig
    cfg.trainer.placement.policy_num_nodes = sft_cfg.placement.num_nodes
    cfg.trainer.placement.policy_num_gpus_per_node = sft_cfg.placement.num_gpus_per_node
    # SFT overrides: no inference engine or ref model
    cfg.trainer.placement.colocate_all = False

    # Parallelism configs -- direct assignment (same types)
    if sft_cfg.strategy == "megatron":
        cfg.trainer.policy.megatron_config = sft_cfg.megatron_config
    if sft_cfg.strategy == "fsdp":
        cfg.trainer.policy.fsdp_config = sft_cfg.fsdp_config

    cfg.trainer.policy.sequence_parallel_size = sft_cfg.sequence_parallel_size
    cfg.trainer.policy.model_config_kwargs = sft_cfg.model_config_kwargs
    cfg.trainer.policy.use_torch_compile = sft_cfg.use_torch_compile
    cfg.trainer.policy.record_memory = sft_cfg.record_memory

    # SFT doesn't use KL/ref model
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.use_kl_in_reward = False

    # Training params
    cfg.trainer.micro_train_batch_size_per_gpu = sft_cfg.micro_train_batch_size_per_gpu
    cfg.trainer.use_sample_packing = sft_cfg.use_sample_packing

    # Logging & checkpointing
    cfg.trainer.logger = sft_cfg.logger
    cfg.trainer.project_name = sft_cfg.project_name
    cfg.trainer.run_name = sft_cfg.run_name
    if sft_cfg.ckpt_path:
        cfg.trainer.ckpt_path = sft_cfg.ckpt_path
        cfg.trainer.ckpt_interval = sft_cfg.ckpt_interval

    # HF export
    if sft_cfg.hf_save_interval > 0:
        cfg.trainer.hf_save_interval = sft_cfg.hf_save_interval
        if sft_cfg.export_path:
            cfg.trainer.export_path = sft_cfg.export_path
        elif sft_cfg.ckpt_path:
            cfg.trainer.export_path = os.path.join(sft_cfg.ckpt_path, "hf_exports")
        # else: leave cfg.trainer.export_path at its default (~/exports/)

    return cfg

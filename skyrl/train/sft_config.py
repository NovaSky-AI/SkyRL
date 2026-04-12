"""
SFT (Supervised Fine-Tuning) configuration.

Defines ``SFTConfig`` -- the user-facing config for SFT training -- and the
bridge function ``build_skyrl_sft_config`` that maps it to the internal
``SkyRLTrainConfig`` used by the SkyRL backend.
"""

from dataclasses import dataclass, field
from typing import List, Union

from omegaconf import OmegaConf

from skyrl.train.config import (
    BaseConfig,
    FSDPConfig,
    MegatronConfig,
    ModelConfig,
    OptimizerConfig,
    SkyRLTrainConfig,
)
from skyrl.train.utils.utils import validate_cfg

# ---------------------------------------------------------------------------
# SFT-specific config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SFTPlacementConfig(BaseConfig):
    """Placement configuration for SFT training.

    Simple alternative to the RL PlacementConfig -- SFT only has a single
    model (no critic, no ref), so we only need node/GPU counts.
    """

    num_nodes: int = 1
    num_gpus_per_node: int = 4


@dataclass
class SFTConfig(BaseConfig):
    """Configuration for SFT training.

    Reuses SkyRL config objects for model, optimizer, and parallelism.
    SFT-specific fields (dataset, training loop, logging) are flat.

    Usage::

        cfg = SFTConfig(
            strategy="megatron",
            placement=SFTPlacementConfig(num_gpus_per_node=4),
            megatron=MegatronConfig(tensor_model_parallel_size=2,
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
        """
        if isinstance(args, dict):
            args = [f"{k}={v}" for k, v in args.items()]

        overrides = OmegaConf.from_cli(args)
        return cls.from_dict_config(overrides)

    # ---- Reused SkyRL config objects ----
    model: ModelConfig = field(default_factory=lambda: ModelConfig(path="Qwen/Qwen3-0.6B"))
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    placement: SFTPlacementConfig = field(default_factory=SFTPlacementConfig)
    megatron: MegatronConfig = field(
        default_factory=lambda: MegatronConfig(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
        )
    )
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)

    # ---- SFT-specific flat fields ----
    strategy: str = "megatron"  # "megatron" or "fsdp2"
    dataset_name: str = "yahma/alpaca-cleaned"
    dataset_split: str = "train[:100]"
    messages_key: str = "messages"  # column name for chat-format datasets
    max_length: int = 512
    num_steps: int = 10
    batch_size: int = 4
    micro_train_batch_size_per_gpu: int = 2
    logger: str = "console"  # "console" or "wandb"
    project_name: str = "megatron-sft"
    run_name: str = ""
    ckpt_path: str = ""  # empty string = no checkpointing
    ckpt_interval: int = 0


# ---------------------------------------------------------------------------
# Bridge: SFTConfig -> SkyRLTrainConfig
# ---------------------------------------------------------------------------


_VALID_STRATEGIES = ("megatron", "fsdp2")


def build_skyrl_sft_config(sft_cfg: SFTConfig) -> SkyRLTrainConfig:
    """Map user-facing SFTConfig to the internal SkyRL backend config."""
    if sft_cfg.strategy not in _VALID_STRATEGIES:
        raise ValueError(f"Unknown strategy '{sft_cfg.strategy}'. Must be one of {_VALID_STRATEGIES}.")

    cfg = SkyRLTrainConfig()

    # Strategy
    cfg.trainer.strategy = sft_cfg.strategy

    # Model -- direct assignment (same type: ModelConfig)
    cfg.trainer.policy.model = sft_cfg.model

    # Optimizer -- direct assignment (same type: OptimizerConfig)
    cfg.trainer.policy.optimizer_config = sft_cfg.optimizer

    # Placement -- map SFTPlacementConfig fields to PlacementConfig
    cfg.trainer.placement.policy_num_nodes = sft_cfg.placement.num_nodes
    cfg.trainer.placement.policy_num_gpus_per_node = sft_cfg.placement.num_gpus_per_node
    # SFT overrides: no inference engine or ref model
    cfg.trainer.placement.colocate_all = False

    # Parallelism configs -- direct assignment (same types)
    if sft_cfg.strategy == "megatron":
        cfg.trainer.policy.megatron_config = sft_cfg.megatron
    if sft_cfg.strategy == "fsdp2":
        cfg.trainer.policy.fsdp_config = sft_cfg.fsdp

    # SFT doesn't use KL/ref model
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.use_kl_in_reward = False

    # Training params
    cfg.trainer.micro_train_batch_size_per_gpu = sft_cfg.micro_train_batch_size_per_gpu

    # Logging & checkpointing
    cfg.trainer.logger = sft_cfg.logger
    cfg.trainer.project_name = sft_cfg.project_name
    cfg.trainer.run_name = sft_cfg.run_name
    if sft_cfg.ckpt_path:
        cfg.trainer.ckpt_path = sft_cfg.ckpt_path
        cfg.trainer.ckpt_interval = sft_cfg.ckpt_interval

    validate_cfg(cfg)
    return cfg

# Configuration API

The configuration system uses Python dataclasses. The top-level
[`SkyRLTrainConfig`][skyrl_train.config.config.SkyRLTrainConfig] can be
constructed from CLI arguments via `SkyRLTrainConfig.from_cli_overrides()` or
from a Hydra `DictConfig` via `SkyRLTrainConfig.from_dict_config()`.

## Top-level Config

::: skyrl_train.config.config.SkyRLTrainConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl_train.config.config.make_config
    options:
      show_root_heading: true

## Data

::: skyrl_train.config.config.DataConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

## Trainer

::: skyrl_train.config.config.TrainerConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

## Policy / Critic / Ref

::: skyrl_train.config.config.PolicyConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl_train.config.config.CriticConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl_train.config.config.RefConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

## Model / LoRA

::: skyrl_train.config.config.ModelConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl_train.config.config.SkyRLLoraConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

## Optimizer / FSDP

::: skyrl_train.config.config.OptimizerConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl_train.config.config.FSDPConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl_train.config.config.MixedPrecisionConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

## Placement

::: skyrl_train.config.config.PlacementConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

## Algorithm

::: skyrl_train.config.config.AlgorithmConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl_train.config.config.KLCtrlConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl_train.config.config.DynamicSamplingConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl_train.config.config.OffPolicyCorrectionConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl_train.config.config.SAPOConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl_train.config.config.ClipCovConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl_train.config.config.KLCovConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl_train.config.config.CISPOConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

## Generator

::: skyrl_train.config.config.GeneratorConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl_train.config.config.SamplingParams
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl_train.config.config.ChatTemplateConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

## Inference Engine

::: skyrl_train.config.config.InferenceEngineConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

## Environment

::: skyrl_train.config.config.EnvironmentConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl_train.config.config.SkyRLGymConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

## Fully Async

::: skyrl_train.config.config.FullyAsyncConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

## Megatron

::: skyrl_train.config.config.MegatronConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl_train.config.config.MegatronDDPConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

## Utilities

::: skyrl_train.config.config.BaseConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl_train.config.config.build_nested_dataclass
    options:
      show_root_heading: true

::: skyrl_train.config.config.get_config_as_dict
    options:
      show_root_heading: true

::: skyrl_train.config.config.get_config_as_yaml_str
    options:
      show_root_heading: true

# Configuration Reference

SkyRL-Train uses Python dataclasses for configuration. The top-level
`SkyRLTrainConfig` mirrors the YAML configuration structure and can be
constructed from YAML files, CLI overrides, or plain dicts.

## Top-Level Config

::: skyrl.train.config.SkyRLTrainConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

## Trainer

::: skyrl.train.config.TrainerConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

### Placement

::: skyrl.train.config.PlacementConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

### Policy / Critic / Ref

::: skyrl.train.config.PolicyConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl.train.config.CriticConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl.train.config.RefConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

### Model & LoRA

::: skyrl.train.config.ModelConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl.train.config.SkyRLLoraConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

### Optimizer & Mixed Precision

::: skyrl.train.config.OptimizerConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl.train.config.MixedPrecisionConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

### FSDP

::: skyrl.train.config.FSDPConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

### Megatron

::: skyrl.train.config.MegatronConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl.train.config.MegatronDDPConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl.train.config.MegatronLoraConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl.train.config.MegatronTorchProfilerConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

### Algorithm

::: skyrl.train.config.AlgorithmConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl.train.config.SAPOConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl.train.config.CISPOConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl.train.config.ClipCovConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl.train.config.KLCovConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl.train.config.KLCtrlConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl.train.config.DynamicSamplingConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl.train.config.OffPolicyCorrectionConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

### Fully Async

::: skyrl.train.config.FullyAsyncConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

## Generator

::: skyrl.train.config.GeneratorConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

### Inference Engine

::: skyrl.train.config.InferenceEngineConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

### Sampling

::: skyrl.train.config.SamplingParams
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl.train.config.ChatTemplateConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

## Environment

::: skyrl.train.config.EnvironmentConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl.train.config.SkyRLGymConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

## Data

::: skyrl.train.config.DataConfig
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

## Utilities

::: skyrl.train.config.config.make_config
    options:
      show_root_heading: true

# Algorithm Registry API

The registry system in SkyRL Train provides a way to register and manage custom algorithm functions (like advantage estimators and policy loss functions) across distributed Ray environments. This system allows users to extend the framework with custom implementations without modifying the core codebase.

## Base Registry Classes

::: skyrl_train.utils.ppo_utils.BaseFunctionRegistry
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl_train.utils.ppo_utils.RegistryActor
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl_train.utils.ppo_utils.sync_registries
    options:
      show_root_heading: true

## Advantage Estimator Registry

The advantage estimator registry manages functions that compute advantages and returns for reinforcement learning algorithms.

::: skyrl_train.utils.ppo_utils.AdvantageEstimatorRegistry
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl_train.utils.ppo_utils.AdvantageEstimator
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl_train.utils.ppo_utils.register_advantage_estimator
    options:
      show_root_heading: true

## Policy Loss Registry

The policy loss registry manages functions that compute policy losses for PPO and related algorithms.

::: skyrl_train.utils.ppo_utils.PolicyLossRegistry
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl_train.utils.ppo_utils.PolicyLossType
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl_train.utils.ppo_utils.register_policy_loss
    options:
      show_root_heading: true

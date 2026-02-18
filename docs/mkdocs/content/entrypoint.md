# Entrypoint API

## Training Entrypoint

The main entrypoint is the `BasePPOExp` class which runs the main training loop.

::: skyrl_train.entrypoints.main_base.BasePPOExp
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

## Evaluation Entrypoint

The evaluation-only entrypoint is the `EvalOnlyEntrypoint` class which runs evaluation without training.

::: skyrl_train.entrypoints.main_generate.EvalOnlyEntrypoint
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

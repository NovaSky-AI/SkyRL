# Trainer API

The Trainer drives the training loop.

## Trainer Class

::: skyrl_train.trainer.RayPPOTrainer
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

## Dispatch APIs

::: skyrl_train.distributed.dispatch.Dispatch
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl_train.distributed.dispatch.MeshDispatch
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl_train.distributed.dispatch.PassThroughDispatch
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

## Actor APIs

The base worker abstraction in SkyRL:

::: skyrl_train.workers.worker.Worker
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl_train.workers.worker.PPORayActorGroup
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

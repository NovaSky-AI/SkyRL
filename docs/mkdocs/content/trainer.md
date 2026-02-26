# Trainer API

The Trainer drives the training loop.

## Trainer Class

::: skyrl.train.trainer.RayPPOTrainer
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

## Dispatch APIs

::: skyrl.backends.skyrl_train.distributed.dispatch.Dispatch
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl.backends.skyrl_train.distributed.dispatch.MeshDispatch
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl.backends.skyrl_train.distributed.dispatch.PassThroughDispatch
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

## Actor APIs

The base worker abstraction in SkyRL:

::: skyrl.backends.skyrl_train.workers.worker.Worker
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

::: skyrl.backends.skyrl_train.workers.worker.PPORayActorGroup
    options:
      show_root_heading: true
      members_order: source
      show_bases: true

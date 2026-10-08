# FSDP Backend

## Overview

Default backend (`trainer.strategy=fsdp`). Uses PyTorch FSDP2 for distributed training.

- **FSDPConfig** in `skyrl/train/config.py`.
- **FSDPStrategy** in `skyrl/backends/skyrl_train/distributed/fsdp_strategy.py`.
- **FsdpWeightSource** (`skyrl/backends/skyrl_train/weight_sync/sources.py`) presents the sharded model to weight sync as vLLM's `WeightSource`; the worker builds it in `FSDPPolicyWorkerBase._build_weight_source`.

## CPU Offload

- `trainer.fsdp_config.cpu_offload=true` offloads optimizer states to CPU.
- Also available for reference model: `ref.fsdp_config.cpu_offload=true`.
- Useful when GPU memory is low but adds overhead.
- NOT to be confused with `offload_after_step`: This is for colocated training where training state is offloaded to CPU after a training step is complete, so that the inference workers can be loaded on the same GPUs.

## Sharding

- `FULL_SHARD` (default): Shards parameters, gradients, and optimizer states.
- `NO_SHARD`: Falls back when world_size=1.
- `fsdp_size`: Controls sharding group size. `-1` = auto (full world). For Hybrid Sharded Data Parallelism (HSDP), use `fsdp_size=<num_gpus_per_node>`

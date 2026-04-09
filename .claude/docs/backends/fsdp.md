# FSDP Backend

## Overview

Default backend (`trainer.strategy=fsdp2`). Uses PyTorch FSDP2 for distributed training. Legacy FSDP1 is available via `trainer.strategy=fsdp`.

- **FSDPStrategy** in `skyrl/backends/skyrl_train/distributed/fsdp_strategy.py`.
- **FSDPWeightExtractor** for extracting weights from sharded parameters (in `skyrl/backends/skyrl_train/workers/fsdp/fsdp_worker.py`).

## Key Config

```yaml
trainer:
  strategy: fsdp2  # or "fsdp" for legacy FSDP1
  fsdp_config:
    reshard_after_forward: true  # FSDP2 only
    fsdp_size: -1  # auto
    cpu_offload: false
```

## CPU Offload

- `trainer.fsdp_config.cpu_offload=true` offloads optimizer states to CPU.
- Also available for reference model: `ref.fsdp_config.cpu_offload=true`.
- Useful when GPU memory is tight but adds overhead.

## Sharding

- `FULL_SHARD` (default): Shards parameters, gradients, and optimizer states.
- `NO_SHARD`: Falls back when world_size=1.
- `fsdp_size`: Controls sharding group size. `-1` = auto (full world).

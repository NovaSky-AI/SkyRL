# SkyRL-v2 (fleet-ai/SkyRL-v2)

Fork of SkyRL with Fleet-specific optimizations for multi-node FSDP2 training at scale.

## Fleet Integration

Fleet-specific changes, fixes, and context are documented in:
- **[integrations/fleet/CHANGELOG.md](integrations/fleet/CHANGELOG.md)** — detailed changelog with root causes and fixes

Always consult the changelog before modifying Fleet training paths (`fsdp_worker.py`, `worker.py`, `dispatch.py`, `fleet-*.sh`).

## Key Differences from Upstream SkyRL

1. **Multi-node FSDP2 stability**: Synchronous ref model offload/backload with `torch.distributed.barrier()` in `fsdp_worker.py`. Required because cross-node colocated training has no shared CUDA context.

2. **CUDA memory management for 35B**: `torch.cuda.empty_cache()` before backward pass in `worker.py` (policy + critic). Prevents OOM from fragmentation on large models with tight GPU memory margins.

3. **`stage_chunks` pre-staging**: `dispatch.py` has a `stage_chunks` optimization (not in upstream) that pre-stages mini-batch chunks in Ray object store. Includes dynamic `mini_batch_size` adjustment for hint augmentation's variable batch sizes.

4. **No `expandable_segments` with vLLM 0.18.0**: `fleet-35b-run.sh` passes `--no-pytorch-alloc-conf` because vLLM 0.18.0's `CuMemAllocator` uses `cuMemCreate`/`cuMemMap` and conflicts with PyTorch's `expandable_segments:True` (also cuMem-based). Anti-fragmentation is handled by `empty_cache()` before backward (fix #2). Old SkyRL (vLLM 0.17.0, `cudaMalloc`) doesn't have this conflict.

## Training Scripts

- `scripts/fleet-common-run.sh` — shared infra (Ray, NCCL, gIB detection, deps). Used by all runs.
- `scripts/fleet-35b-run.sh` — Qwen3.5-35B config. Calls `fleet-common-run.sh`.
- `scripts/fleet-9b-run.sh` — Qwen3.5-9B config. Calls `fleet-common-run.sh`.

All training flags live in these scripts. Never duplicate flags in SkyPilot YAMLs or fleet-research scripts.

## Branch

Primary development branch: `fleet/all`

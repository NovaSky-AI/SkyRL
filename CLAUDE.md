# SkyRL-v2 (fleet-ai/SkyRL-v2)

Fork of SkyRL with Fleet-specific optimizations for multi-node FSDP2 training at scale.

## Fleet Integration

Fleet-specific changes, fixes, and context are documented in:
- **[integrations/fleet/CHANGELOG.md](integrations/fleet/CHANGELOG.md)** — detailed changelog with root causes and fixes

Always consult the changelog before modifying Fleet training paths (`fsdp_worker.py`, `worker.py`, `model_wrapper.py`, `dispatch.py`, `fleet-*.sh`).

## Key Differences from Upstream SkyRL

1. **Multi-node FSDP2 stability**: Synchronous ref model offload/backload with `torch.distributed.barrier()` in `fsdp_worker.py`. Required because cross-node colocated training has no shared CUDA context.

2. **Chunked lm_head forward**: `model_wrapper.py` has `loss_chunk_size` support ported from the old fork. Avoids materializing full `(B, S, vocab_size)` logits — critical for 35B with 131K vocab at 97K sequence length. Without it, OOM/Xid 31 during training forward.

3. **CUDA memory management for 35B**: `torch.cuda.empty_cache()` before backward pass in `worker.py` (policy + critic). Prevents OOM from fragmentation. Especially important because `expandable_segments` can't be used (see #5).

4. **`flash_attn=false` for GatedDeltaNet**: `fleet-35b-run.sh` uses SDPA, not flash_attention_2. Qwen3.5-35B's GatedDeltaNet linear attention layers crash with flash_attn in multi-node FSDP2. Memory savings come from chunked lm_head (#2), not flash attention.

5. **No `expandable_segments` with vLLM 0.18.0**: `fleet-35b-run.sh` passes `--no-pytorch-alloc-conf` because vLLM 0.18.0's `CuMemAllocator` (`cuMemCreate`/`cuMemMap`) conflicts with PyTorch's `expandable_segments:True`. Old SkyRL uses vLLM 0.17.0 (`cudaMalloc`) which has no conflict.

6. **`stage_chunks` pre-staging**: `dispatch.py` has a `stage_chunks` optimization (not in upstream) that pre-stages mini-batch chunks in Ray object store. Includes dynamic `mini_batch_size` adjustment for hint augmentation's variable batch sizes.

## Training Scripts

- `scripts/fleet-common-run.sh` — shared infra (Ray, NCCL, gIB detection, deps). Used by all runs.
- `scripts/fleet-35b-run.sh` — Qwen3.5-35B config. Calls `fleet-common-run.sh`.
- `scripts/fleet-9b-run.sh` — Qwen3.5-9B config. Calls `fleet-common-run.sh`.

All training flags live in these scripts. Never duplicate flags in SkyPilot YAMLs or fleet-research scripts.

## Branch

Primary development branch: `fleet/all`

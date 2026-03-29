# Fleet Integration Changelog

## 2026-03-29: Multi-node FSDP stability + hint batch size fix

Ported from fleet-ai/SkyRL PR #328 and PR #333, plus a new fix for hint augmentation batch sizing.

### Problem

2-node (16 GPU) Qwen3.5-35B training on GCP H200 crashed with:
1. `cudaErrorIllegalAddress` segfaults during FSDP ref model offload/backload
2. OOM during backward pass from CUDA memory fragmentation
3. `AssertionError: data batch size must be divisible by mini_batch_size, got 160 and 128` when hints are enabled

### Root causes and fixes

#### 1. Synchronous ref offload + barrier (`fsdp_worker.py`)

**Where:** `FSDPRefWorkerBase.offload_to_cpu()` and `backload_to_gpu()`

**Problem:** With colocated models, the trainer cycles: ref on GPU -> ref offload to CPU -> policy on GPU. With `non_blocking=True`, the CPU<-GPU transfer is *queued* but returns immediately. On a single node, CUDA stream ordering serializes this naturally. Across nodes, there's no shared CUDA context -- node 0's policy worker can start touching GPU memory while node 1's ref worker is still mid-transfer. Result: `cudaErrorIllegalAddress`.

**Fix:** `non_blocking=False` (wait for transfer) + `torch.distributed.barrier()` (all ranks synchronize). Guarantees every GPU finishes offloading before any policy worker starts backloading.

**Why upstream SkyRL doesn't need this:** Designed for single-node where all workers share the same CUDA context and stream ordering prevents races.

#### 2. empty_cache before backward (`worker.py`)

**Where:** `PolicyWorkerBase._forward_backward_micro()` (both SFT and RL paths) and `CriticWorkerBase._forward_backward_micro()`

**Problem:** After the forward pass, freed intermediate tensors stay in PyTorch's CUDA cache as scattered blocks. The backward pass needs large contiguous allocations for gradients. On the 35B model with tight GPU memory margins, the fragmented cache can't satisfy these allocations -> OOM, even though total free memory is sufficient.

**Fix:** `torch.cuda.empty_cache()` before `strategy.backward()`. Returns cached blocks to CUDA which coalesces them into contiguous allocations.

**Why upstream SkyRL doesn't need this:** Targets smaller models (8B) with enough GPU headroom that fragmentation doesn't matter.

#### 3. Enable expandable_segments for 35B (`fleet-35b-run.sh`)

**Where:** Removed `--no-pytorch-alloc-conf` flag from `fleet-35b-run.sh`

**Problem:** Without `expandable_segments:True`, PyTorch allocates fixed-size CUDA segments. Triton autotuning (FlashAttention, MoE kernels) allocates many trial buffers then frees them, leaving a fragmented segment map. Subsequent large allocations fail even though total free memory is sufficient.

**Fix:** Remove `--no-pytorch-alloc-conf` so `fleet-common-run.sh` sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. GCP has writable `ptrace_scope=0` so vLLM's CuMemAllocator and expandable_segments coexist fine.

**Note:** 9B task-gen runs on RunPod still use `--no-pytorch-alloc-conf` because RunPod containers lack `CAP_SYS_PTRACE` and expandable_segments uses cuMem APIs that need `pidfd_getfd`.

#### 4. Dynamic mini_batch_size for hint augmentation (`dispatch.py`)

**Where:** `MeshDispatch.stage_chunks()`

**Problem:** `mini_batch_size` is computed as `policy_mini_batch_size * n_samples_per_prompt` (e.g., 16 * 8 = 128). But hint augmentation appends extra samples: 16 prompts * 2 hints = 32 additional, total batch = 160. The `stage_chunks` method asserted `160 % 128 == 0` -> crash.

The old fork's manual loop (`num_mini_batches = len(data) // mini_batch_size`) silently dropped the 32 hint samples -- no crash, but hint training was wasted.

**Fix:** When batch size isn't divisible by mini_batch_size, step down mini_batch_size (by `dp_size` increments to stay DP-divisible) until it divides evenly. For 160 samples with dp_size=16: adjusts from 128 -> 80, giving 2 mini-batches of 80. All 160 samples (including hints) are trained on.

**Why upstream SkyRL doesn't have this:** Upstream uses a simple `for` loop with `//` division (no `stage_chunks` optimization). The `stage_chunks` pre-staging is a SkyRL-v2 optimization that added a strict assert the old code path never had.

### Files changed

| File | Change |
|------|--------|
| `skyrl/backends/skyrl_train/workers/fsdp/fsdp_worker.py` | Synchronous ref offload + barrier |
| `skyrl/backends/skyrl_train/workers/worker.py` | empty_cache before backward (3 sites) |
| `scripts/fleet-35b-run.sh` | Remove `--no-pytorch-alloc-conf` |
| `skyrl/backends/skyrl_train/distributed/dispatch.py` | Dynamic mini_batch_size adjustment |

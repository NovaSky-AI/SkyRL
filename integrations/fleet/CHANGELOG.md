# Fleet Integration Changelog

## 2026-03-29: Multi-node 35B training parity with old SkyRL fork

Fixes for 2-node (16 GPU) Qwen3.5-35B GRPO training on GCP H200. Ported from fleet-ai/SkyRL PR #328 and PR #333, plus new fixes for SkyRL-v2-specific issues.

### Problems

2-node training crashed with:
1. `cudaErrorIllegalAddress` during FSDP ref model offload/backload (multi-node race)
2. OOM / Xid 31 FAULT_PDE during policy training forward+backward (missing chunked lm_head)
3. OOM / Xid 31 at 97K sequence length — SDPA too memory-hungry, flash_attn triggers GatedDeltaNet crash
4. `AssertionError: data batch size must be divisible by mini_batch_size, got 160 and 128` (hint augmentation)

### Root causes and fixes

#### 1. Synchronous ref offload + barrier (`fsdp_worker.py`)

**Where:** `FSDPRefWorkerBase.offload_to_cpu()` and `backload_to_gpu()`

**Problem:** With colocated models, the trainer cycles: ref on GPU → ref offload to CPU → policy on GPU. With `non_blocking=True`, the CPU←GPU transfer is *queued* but returns immediately. On a single node, CUDA stream ordering serializes this naturally. Across nodes, there's no shared CUDA context — node 0's policy worker can start touching GPU memory while node 1's ref worker is still mid-transfer. Result: `cudaErrorIllegalAddress`.

**Fix:** `non_blocking=False` (wait for transfer) + `torch.distributed.barrier()` (all ranks synchronize). Guarantees every GPU finishes offloading before any policy worker starts backloading.

**Why the old fork doesn't need this:** Designed for single-node where all workers share the same CUDA context and stream ordering prevents races.

#### 2. Port chunked lm_head forward (`model_wrapper.py`, `fsdp_worker.py`)

**Where:** `HFModelWrapper.forward()` and `HFModelWrapper._chunked_lm_head_forward()`

**Problem:** SkyRL-v2's `HFModelWrapper` was missing `loss_chunk_size` support entirely — the parameter existed in config but was never passed through `fsdp_worker.py` to the model wrapper. Without it, the model materializes the full `(B, S, 131072)` logits tensor during forward pass (~10 GB for 97K-length sequences on Qwen3.5-35B with vocab_size=131072). This consumed so much GPU memory that the subsequent training forward pass (with gradients enabled) hit OOM or Xid 31 FAULT_PDE when FSDP tried to unshard parameters.

**Fix:** Ported the chunked lm_head implementation from the old fork:
- Added `loss_chunk_size` parameter to `HFModelWrapper.__init__`
- Pass `loss_chunk_size` from `fsdp_worker.py` for both policy and ref model init
- During forward, replace `lm_head` with an identity module so the model returns hidden states `(B, S, 8192)` instead of logits `(B, S, 131072)` — 16x smaller
- Compute logits in chunks of 4096 tokens with gradient checkpointing, never materializing full logits

**Why the old fork doesn't have this problem:** It already has `loss_chunk_size` support and passes it correctly.

#### 3. `empty_cache` before backward (`worker.py`)

**Where:** `PolicyWorkerBase._forward_backward_micro()` (both SFT and RL paths) and `CriticWorkerBase._forward_backward_micro()`

**Problem:** After the forward pass, freed intermediate tensors stay in PyTorch's CUDA cache as scattered blocks. The backward pass needs large contiguous allocations for gradients. On the 35B model with tight GPU memory margins, the fragmented cache can't satisfy these allocations → OOM, even though total free memory is sufficient.

**Fix:** `torch.cuda.empty_cache()` before `strategy.backward()`. Returns cached blocks to CUDA which coalesces them into contiguous allocations. This is especially important because `expandable_segments:True` cannot be used (see fix #4).

**Why the old fork doesn't need this:** Targets smaller models (8B) with enough GPU headroom that fragmentation doesn't matter.

#### 4. Reduce sequence length to 72K and disable `expandable_segments` (`fleet-35b-run.sh`)

**Where:** `fleet-35b-run.sh` — `MAX_INPUT_LENGTH` and `--no-pytorch-alloc-conf` flag.

**Problem:** At 97K sequences (96000 input + 4096 generate), memory was too tight even with chunked lm_head and `empty_cache`:
- `flash_attn=false` (SDPA): OOM requesting 5.95 GiB during backward — SDPA's O(n²) attention memory is too large at 97K.
- `flash_attn=true`: Xid 31 FAULT_PDE in GatedDeltaNet layers during ref model forward — reproduced at both 97K and 72K. Not a memory issue; vLLM 0.18.0's CuMemAllocator corrupts CUDA memory mappings that FSDP2 DTensor operations later touch.
- `expandable_segments:True` would help with fragmentation but conflicts with vLLM 0.18.0's `CuMemAllocator` (`cuMemCreate`/`cuMemMap`).

**Fix:** Reduce `MAX_INPUT_LENGTH` from 96000 to 72000 (total seq ~76K) and use `flash_attn=false` (SDPA). At 72K, SDPA's O(n²) memory is ~55% of what it was at 97K — enough to fit with chunked lm_head + `empty_cache`. The `--no-pytorch-alloc-conf` flag passed to `fleet-common-run.sh` skips the default `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, avoiding the vLLM 0.18.0 CuMemAllocator conflict. The 9B VL script (`fleet-vl-run.sh`) also passes this flag for the same reason.

**Verified working:** 10 steps completed on GCP spot 2×H200:8 (asia-south1-b) with zero GPU errors over 12 hours. Step timing: generation ~7 min, ref forward ~8 min, policy backward ~44 min, total step ~70 min avg. Checkpoint saved to S3 at step 10. SDPA is slower than flash_attn but stable. WandB: `fleet_qwen35_35b_tool_use_2c0e13b7` (run ID `f6kw15p2`).

#### 5. Dynamic mini_batch_size for hint augmentation (`dispatch.py`)

**Where:** `MeshDispatch.stage_chunks()`

**Problem:** `mini_batch_size` is computed as `policy_mini_batch_size * n_samples_per_prompt` (e.g., 16 × 8 = 128). But hint augmentation appends extra samples: 16 prompts × 2 hints = 32 additional, total batch = 160. The `stage_chunks` method asserted `160 % 128 == 0` → crash.

The old fork's manual loop (`num_mini_batches = len(data) // mini_batch_size`) silently dropped the 32 hint samples — no crash, but hint training was wasted.

**Fix:** When batch size isn't divisible by mini_batch_size, step down mini_batch_size (by `dp_size` increments to stay DP-divisible) until it divides evenly. For 160 samples with dp_size=16: adjusts from 128 → 80, giving 2 mini-batches of 80. All 160 samples (including hints) are trained on.

**Why upstream SkyRL doesn't have this:** Upstream uses a simple `for` loop with `//` division (no `stage_chunks` optimization). The `stage_chunks` pre-staging is a SkyRL-v2 optimization that added a strict assert the old code path never had.

### Files changed

| File | Change |
|------|--------|
| `skyrl/backends/skyrl_train/workers/model_wrapper.py` | Port chunked lm_head forward (loss_chunk_size) |
| `skyrl/backends/skyrl_train/workers/fsdp/fsdp_worker.py` | Pass loss_chunk_size to HFModelWrapper; synchronous ref offload + barrier |
| `skyrl/backends/skyrl_train/workers/worker.py` | empty_cache before backward (3 sites) |
| `scripts/fleet-35b-run.sh` | Reduce seq length to 72K, flash_attn=false, --no-pytorch-alloc-conf, wandb project rename |
| `skyrl/backends/skyrl_train/distributed/dispatch.py` | Dynamic mini_batch_size adjustment |

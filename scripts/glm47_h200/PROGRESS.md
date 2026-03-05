# GLM-4.7-Flash H200 De-risking Progress

**Date**: 2026-03-05
**Environment**: Anyscale cluster, head node (10.0.12.152, no GPU), 4x worker nodes (10.0.20.62, 10.0.61.202, 10.0.19.2, 10.0.39.114), each with 8x H200-141GB GPUs

---

## Architecture Notes

- Head node has workspace at `/home/ray/default/SkyRL` (not visible on workers)
- Workers have SkyRL at `/home/ray/SkyRL` (cloned from PR #1280 branch, with main merged)
- Shared storage: `/efs` (s3fs mounted, accessible from all nodes) — s3fs has cache delay for cross-node writes (~minutes)
- Ray version: 2.51.1, already running; connect via `RAY_ADDRESS=auto`
- SKYRL_PYTHONPATH_EXPORT mechanism propagates PYTHONPATH to Ray workers
- **CRITICAL**: PYTHONPATH must be set BEFORE the inline command, not as an inline var: use `PYTHONPATH="..." RAY_ADDRESS=auto` not `WORKER_PP="..." PYTHONPATH="$WORKER_PP" ...`

## How to Run Training

```bash
PYTHONPATH="/home/ray/SkyRL:/home/ray/SkyRL/skyrl-gym:/home/ray/SkyRL/.venv/lib/python3.12/site-packages" \
RAY_ADDRESS=auto SKYRL_PYTHONPATH_EXPORT=true \
HF_HUB_ENABLE_HF_TRANSFER=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1 CUDA_DEVICE_MAX_CONNECTIONS=1 NVTE_FUSED_ATTN=0 \
RAY_memory_usage_threshold=1.0 \
.venv/bin/python -m skyrl.train.entrypoints.main_base [ARGS] 2>&1 | tee /tmp/phase_N.log &
```

## Fixes Applied

### Workers (all 4 nodes patched via Ray tasks)

1. **`qwen3_bridge.py`** — transformers 5.x rope_theta fix (only needed for Qwen3 model)
2. **Full skyrl package sync** — workers synced with tarball from head node
   - File: `/home/ray/SkyRL/skyrl/` on all workers
   - Includes `skyrl/env_vars.py` (from PR #1276)
3. **`ray_wrapped_inference_engine.py`** — vLLM bundle assignment fix (Phase 6 v4):
   - Use `get_reordered_bundle_indices(shared_pg)` for vLLM engine bundle assignment
   - Ray PACK assigns `{GPU:1}` bundles round-robin across nodes; the reorder sorts by (node_id, gpu_id) so TP groups stay within a node
   - Without this fix: cross-node TP groups cause `tcp://127.0.0.1:PORT` rendezvous to hang (PENDING_CREATION)

### Head node (`/home/ray/default/SkyRL/`)

4. **`skyrl/train/utils/utils.py`** — `peer_access_supported()`:
   - Returns `True` immediately on CPU-only head nodes
5. **`skyrl/backends/skyrl_train/distributed/megatron/megatron_strategy.py`**:
   - Merged from `main` to get PR #1268 (dp_reshardable checkpoint format)
   - Uses `_dist_ckpt_optim_metadata` property + `_ensure_optimizer_state_initialized()`
6. **`tests/backends/skyrl_train/gpu/gpu_ci/test_save_load_checkpoint.py`**:
   - File existence check uses `NodeAffinitySchedulingStrategy` on ALL GPU nodes (handles local worker storage)
   - Cleanup also runs on all GPU nodes
7. **Branch**: merged `main` into `pr-1280` to get PRs #1268, #1266, #1276

---

## Phase Status

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: Setup | DONE | Workers set up with SkyRL + deps on all 4 nodes |
| Phase 2: Megatron checkpoint test | DONE (PASSED) | test_save_load_checkpoint[megatron] passed in 66s |
| Phase 3: 8-GPU GLM smoke test | DONE (PASSED) | 2 training steps completed, all metrics healthy |
| Phase 4: 8-GPU 8K context | DONE (PASSED) | No OOM on H200 with 8K context |
| Phase 6: 32-GPU 4x8 full run | DONE (PASSED) | 2+ training steps complete, metrics healthy |
| Phase 7: vLLM Inference Benchmark | DONE | Full report in INFERENCE_BENCHMARK_REPORT.md |

---

## Phase 2 Results (Megatron Checkpoint Test)

- **Test**: `test_save_load_checkpoint[megatron]`
- **Result**: PASSED in 66.69s
- **Key fix**: NodeAffinitySchedulingStrategy to check files on ALL GPU nodes
  (local worker FS: checkpoint saved to `$HOME/ckpts/test/` by rank-0 worker)

---

## Phase 3 Results (8-GPU Smoke Test)

- **Config**: 8 GPUs (node 10.0.19.2), TP=1, EP=8, 2x vLLM TP=4, 1K context
- **Model**: zai-org/GLM-4.7-Flash (Glm4MoeLiteForCausalLM, 30B MoE)
- **Per-shard parameters**: 5.63B (with EP=8, each GPU gets 1 expert group)
- **Result**: PASSED after 3 steps

### Step Timing (8K GPU memory for context=1K)
| Phase | Time |
|-------|------|
| Model load + init | ~4 min |
| Generation (1K context) | ~5.5 min/step |
| Forward/scoring | ~21s/step |
| Training (backward + optim) | ~80s/step |
| Total per step | ~7.5 min |

### Metrics at Step 2
- `reward/avg_raw_reward`: 0.5879 (59% correct on GSM8K)
- `reward/avg_pass_at_8`: 0.8906 (89% pass@8)
- `grad_norm`: 0.180 (healthy)
- `policy_loss`: ~0 (as expected for first steps)
- Checkpoint saved successfully to `/home/ray/ckpts/phase3/`

---

## Phase 4 Results (8K Context Test)

- **Config**: 8 GPUs (node 10.0.39.114), max_generate_length=8192, max_model_len=8704
- **Result**: PASSED — no OOM on H200 with 8K context
- H200's 141GB VRAM provides 61GB more headroom than A100 (which OOMed on backward)

---

## Phase 6 Results (32-GPU 4x8 Full Run)

- **Config**: 32 GPUs (4 nodes × 8 H200), TP=4, EP=8, DP=4
- **vLLM**: 8 engines × TP=4 (TP=8 fails: 20 heads not divisible by 8)
- **Batch**: train_batch_size=1024, n_samples=16, context=8704 tokens
- **Run**: phase6_32gpu_4x8_v4
- **Result**: PASSED — 2+ steps completed, all metrics healthy

### Step Timing (32 GPUs, 8K context)
| Phase | Time |
|-------|------|
| Model load + init | ~5 min |
| Weight sync (Megatron→vLLM) | ~10s |
| Generation (8 engines × TP=4) | ~12 min/step |
| Forward/scoring | ~12 min/step |
| Training (backward + optim, CPU offload) | ~38 min/step |
| **Total per step** | **~63 min** |

### Step 1 Metrics
- `reward/avg_raw_reward`: 0.6978 (70% on GSM8K)
- `reward/avg_pass_at_16`: 0.9736 (97% pass@16)
- `grad_norm`: 0.1080 (healthy)
- `policy_loss`: 0.0017
- Step 2 generation reward jumped to 0.761 — model is learning

### Bugs Fixed for Phase 6

1. **vLLM TP=8 → TP=4**: GLM has 20 attention heads; TP must evenly divide 20
2. **vLLM Gloo barrier timeout**: Added `disable_custom_all_reduce=true` to skip TCP barrier in `CustomAllreduce.__init__`
3. **Cross-node TP groups (root cause)**: Ray PACK assigns `{GPU:1}` bundles round-robin across nodes. Megatron workers use `get_reordered_bundle_indices()` to sort by (node_id, gpu_id), but vLLM engines used raw indices → cross-node TP → `tcp://127.0.0.1:PORT` rendezvous failed (PENDING_CREATION). **Fix**: use `get_reordered_bundle_indices(shared_pg)` in `ray_wrapped_inference_engine.py`

### Performance Note
The 38-minute backward pass is due to:
- `micro_train_batch_size_per_gpu=1` with 1024 global batch / 8 DP ranks / 4 mini-batches = 32 gradient accumulation steps × 4 = 128 micro-batches
- CPU optimizer offload with D2H/H2D transfers
- Flash-attn v2 (v3 would be faster on H200 Hopper architecture)

---

## Key Learnings

1. **PYTHONPATH inline expansion bug**: In bash, `A="val" B="$A" command` does NOT expand `$A` in `B`'s value. Must export first or hardcode.
2. **s3fs for checkpoint storage**: Local worker FS (`$HOME/ckpts/`) is reliable; `/efs` has cross-node visibility delays.
3. **GLM-4.7-Flash**: Works perfectly with transformers 5.3.0, bridge at `skyrl/backends/skyrl_train/workers/megatron/megatron_worker.py`.
4. **Flash-Attn v2 warning**: TransformerEngine recommends v3 for H200 Hopper, but v2 works. Install v3 for performance.
5. **dp_reshardable checkpoint format**: Required for mcore 0.16.0 (flattened_range removed). PR #1268 properly handles this.
6. **Ray PACK + GPU bundles = round-robin**: With `{GPU:1}` bundles, Ray PACK assigns them round-robin across nodes (not per-node sequential). `get_reordered_bundle_indices()` corrects this by sorting by (node_id, gpu_id).
7. **vLLM disable_custom_all_reduce**: Required for multi-node colocate_all to skip the Gloo TCP barrier check in `CustomAllreduce.__init__`.
8. **GLM TP constraint**: attention_heads=20, so TP must divide 20 (valid: 1, 2, 4, 5, 10, 20). TP=8 fails.

---

---

## Phase 7 Results (vLLM Inference Benchmark)

**Full report**: `INFERENCE_BENCHMARK_REPORT.md`

### Baseline TP Sweep (TP=1,2,4 × context 16K-128K)
**TP=4 is optimal** at all context lengths (2.83–3.35× vs TP=1):

| TP | 16K tok/s | 32K tok/s | 64K tok/s | 128K tok/s |
|----|-----------|-----------|-----------|------------|
|  1 |    16,090 |    15,200 |    11,434 |      7,448 |
|  2 |    30,072 |    26,740 |    20,479 |     13,621 |
|  4 |    46,514 |    43,052 |    34,810 |     24,975 |

TP=5 fails with mp backend (NCCL issue with non-power-of-2 workers).

### Optimization Findings

**Incompatible with GLM MLA (vLLM 0.16.0)**:
- FP8 KV cache: no MLA backend supports it (head_size=576, qk_nope_head_dim=192)
- Prefill context parallelism: FlashAttnMLAImpl doesn't support PCP

**No benefit** (±0.2%): chunked prefill, block_size=32, block_size=64, FLASHMLA backend

**Works**: Decode context parallelism (DCP=2) using 8 GPUs:
- 32K: 49,385 tok/s (+14.7% vs 4-GPU baseline, but uses 2× GPUs)
- 64K: 38,151 tok/s (+9.6%)
- 128K: 25,452 tok/s (+3.3%)

### Recommendation
- **SkyRL training**: TP=4 per engine, default settings — already optimal
- **Max throughput on 8 GPUs**: 2 independent TP=4 engines (~86K tok/s estimated) vs 1 DCP=2 engine (49K)
- Default vLLM settings are near-optimal for GLM-4.7-Flash; no tuning knobs provide significant gains

---

## Issues Encountered

1. `ModuleNotFoundError: skyrl` on worker — PYTHONPATH empty due to inline bash expansion bug → fixed
2. `CheckpointingException: flattened_range` — MCore 0.16.0 removed it → dp_reshardable fix (PR #1268)
3. `File config.json not found` — checkpoint on local worker FS, not visible from head → NodeAffinitySchedulingStrategy on all nodes
4. `RuntimeError: File common.pt cannot be opened` — s3fs race condition → reverted to local FS
5. `ModuleNotFoundError: skyrl.env_vars` — PR #1276 moved module → full skyrl sync to workers
6. `ValidationError: Total number of attention heads (20) must be divisible by tensor parallel size (8)` — GLM TP constraint → TP=4 with 8 engines
7. `RuntimeError: [gloo/transport/tcp/...] Timed out waiting 1800000ms` — vLLM Gloo barrier on multi-node → disable_custom_all_reduce=true
8. `4 AsyncVLLMInferenceEngine stuck in PENDING_CREATION` — cross-node TP due to round-robin bundle assignment → get_reordered_bundle_indices fix

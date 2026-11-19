# Bounded-memory colocated weight sync: validation results

Companion to `colocated_weight_sync.md` (milestone names M0-M5 are its). Measured on the
`colocated-weight-sync-impl` branch, before the port to this one (where M1 became `weight_sync_from_offloaded_params`). Everything here
was run on 4 × 8×B200 (178 GiB) nodes, vLLM 0.33.0, with `Qwen/Qwen3-30B-A3B` (128 experts,
48 layers, 61 GB BF16) unless noted.

## How each feature was validated

**Correctness gate: byte parity.** `tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_colocated_weight_sync.py`
boots colocated vLLM from the checkpoint, records an integer checksum of *every* vLLM
parameter and buffer on every worker, sleeps the engines at level 2 (weights discarded), and
runs the production `WorkerDispatch.save_weights_for_sampler` path. After each sync every
tensor on every worker must be byte-identical to the fresh load (5,408 tensors on 8 workers;
11,584 on 16). A second test takes a real training step and checks the feature path writes the
same vLLM bytes as today's path on the same trained weights. This is much stricter than the
logprob-tolerance tests: a dropped, truncated, misrouted or stale tensor fails it.

Trainer memory/time come from `SKYRL_COLOCATED_SYNC_DEBUG=1` (per-rank peak allocation, a
`mem_get_info` poller for true shared-device peak, time inside the export generator,
round-trip count and latency); vLLM per-worker received bytes come from the receive engine.

## Results

Megatron TP2/EP8 → vLLM TP8 on one node, 1 GiB slots, steady-state sync (second sync):

| configuration | byte parity | round trips | bytes received / worker | transfer (s) | trainer peak alloc (GiB) |
|---|---|---|---|---|---|
| today's path | ✅ | 58 | 61.1 GB | 7.3 | 10.9 |
| today's path, 4 GiB slots | ✅ | 15 | 61.1 GB | 7.0 | 13.9 |
| M1 streaming | ✅ | 58 | 61.1 GB | 6.8 | **3.0** |
| M3 ring (2 slots) | ✅ | 58 | 61.1 GB | 7.0 | 11.9 |
| M4a selective sleep (3 syncs) | ✅ | 58 | 61.1 GB | 7.1 | 10.9 |
| M2 local shards + ring, vLLM EP | ✅ | **11** | **10.3 GB** | 2.6 | 11.9 |
| all of M1–M4a, vLLM EP (3 syncs) | ✅ | 11 | 10.3 GB | **2.0** | **4.0** |
| all, after a training step vs today's path | ✅ (identical bytes) | 11 vs 58 | 10.3 vs 61.1 GB | 2.1 vs 5.3 | 4.2 vs 11.2 |
| all, vLLM TP4 × 2 engines (EP4 each) | ✅ | 17 | 17.6 GB | 3.7 | 4.0 |
| all, **2 nodes**: Megatron TP2/EP16 → vLLM TP8 × 2 (EP) | ✅ (16 workers) | 11 | 10.3 GB | 2.4 | 4.0 |
| all + M4b wake-on-write, vLLM EP (3 syncs) | ✅ | 11 | 10.3 GB | 2.2 | 4.0 |
| M4a + M4b alone (3 syncs) | ✅ | 58 | 61.1 GB | 7.8 | 10.9 |

**OOM-shaped repro** (trainer capped at 6 GiB during the send with
`SKYRL_COLOCATED_SYNC_TRAINER_CAP_GIB`, mimicking "vLLM weights fill the GPU"): today's path
fails in `broadcast_to_inference_engines` with *"Tried to allocate 1024.00 MiB … 6.00 GiB
allowed; 8.20 GiB allocated"* (the reloaded BF16 shard); M1 completes the same sync with byte
parity at a 3.0 GiB peak.

**M5 LoRA** (rank 32, all-linear incl. experts, `sync_mode=memory`, after a training step):
parking the adapter export on pinned CPU stages byte-identical adapters on every worker
(37,248 tensors per worker). It costs ~1.9 s per sync here (dedupe/folds/3D→flat on the host).

**Serving check.** After each sync the harness also compares 4 greedy 48-token generations
with the fresh engine's. In eager mode they are identical through 3 syncs with every feature
(including M4b) on. With CUDA graphs on, exact equality is not a usable gate: the *fresh
engine repeated with no sync at all* already diverges at token 33 and 18 on two prompts,
today's path diverges the same way after a sync, and the feature path stays within that
noise. The harness therefore asserts only that no prompt diverges before token 8.

**M0 probes.** vLLM's `weights` pool for this model is 130 CuMem allocations (4–5 per layer,
26 shared across layer boundaries, 0.27 GiB in shared ones). `create_and_map` costs a flat
~2.3 ms per allocation from 64 to 256 MiB (≈100 GiB/s at 256 MiB); with the pinned H2D
restore, ~45 GiB/s. Staged-wake cost therefore scales with *allocation count*, not call count:
grouping wakes buys nothing at level 2, and per-layer is fine (~12 ms/layer here).

**Selective sleep findings.** Per worker (vLLM EP) 7.14 GiB of weights is covered by the reload
and discarded; 0.68 GiB is backed up. Of that, ~594 MiB is one allocation with live blocks that
no `named_parameters` / `named_buffers` reaches (likely MoE-backend scratch or tensors retained
from the initial load), plus the shared 20 MiB segments holding them and
`rotary_emb.cos_sin_cache`. Today's level 2 discards those and depends on the save-lists; the
selective rule backs them up without having to know what they are.

## Feasibility per milestone

| milestone | status | notes |
|---|---|---|
| M0 instrumentation | **done** | Debug env, worker RPCs, harness |
| M1 trainer streaming | **done, validated** | Hooks `_gather_gtp_weight` in `model_bridge` / `peft_bridge` / `quant_bridge`; one-ahead H2D prefetch. Rejects `overlap_param_gather` (needs params on GPU to publish optimizer shards). Not done: `weight_sync_gpu_budget_GB` (slot sizing from a budget); the gather transient is still one unsharded tensor (risk 5) |
| M2 local shards, Phase A | **done, validated** | Per-expert names only; fused 3D expert tensors (risk 4) still go whole. Falls back (logged) without vLLM EP |
| M3 slot ring | **done, validated** | No gain alone on this model: rank 0 waits ~3.8 s on sends; the receiver (clone + layerwise load), not the 0.8 s export, is the bottleneck. Matters when the EP gather is slow (motivating case) and composes with M2 |
| M4a selective sleep | **done, validated** | Coverage recorded empirically from the actual reload (no static rules), classified against the torch allocator's live blocks |
| M4b wake-on-write | **prototype, validated on this model** | Byte parity over 3 syncs; 124 of 130 `weights` allocations (7.2 GiB) mapped by the write hook as layers were written, 6 at finish. With CUDA graphs on, post-sync greedy output stays within the engine's own run-to-run noise (below), so replay at the re-mapped addresses works. Not done: the CUDA free-while-asleep guard (step 5, no frees mid-sync on this model), direct-reload rejection beyond the FP8 check, and a crashed sync leaves the pool unmapped |
| M5 LoRA | **step 1 done, validated** | Adapter parked on pinned CPU; per-receiver slicing of fused 3D expert adapters not done |
| M6 Phase B (TP slices) | **scoped, not built** | Needs the `FakeRDTTensor` bake from `sharded_rdt` run against the colocated worker's loader; only matters without vLLM EP |

## Operational notes for whoever picks this up

- vLLM 0.33 caches FlashInfer autotune results **per rank**. A run killed mid-autotune leaves
  some ranks cached and the next boot hangs in `profile_run` (cached ranks skip the
  synchronized autotune; the rest spin in a collective). The harness sets a fresh
  `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR` per run.
- Nested `ray.init` inside a Ray task can segfault at shutdown (seen with Ray's event export on) unless the task's
  `RAY_enable_ray_event` / `RAY_enable_core_worker_ray_event_to_aggregator` env is dropped.
- Ray's uv runtime-env hook mis-parses `uv run ... -- pytest`; drop the `--`. Use
  `UV_OFFLINE=1` so per-worker env builds don't hit GitHub (403s under load).

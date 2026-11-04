# throughput-push-deepseek-v1: tracking

Goal: compressed RL runs of **DeepSeek-V4.1-Flash** on this cluster, find what limits overall throughput / MFU,
and fix it in SkyRL. Secondary: 256k context, multi-turn RL, R3 (routing replay). Prefer changes that carry over
to GLM-5.3, GLM-5.3-Flash and future models. Patches should match the shape of upstream code (merged or open PRs).

**Before starting an experiment, read "In progress" and "Done" and add your row to "In progress".**
Rebase before pushing (`git pull --rebase fork throughput-push-deepseek-v1`); keep other agents' rows.

Cluster: 2 x 8xB200 (178 GiB/GPU, 2 TB RAM and 26 TB `/mnt/local_storage` per node), CPU-only head node
(Anyscale workspace: never `ray stop`; kill only your own driver). `/data` does not exist on the GPU nodes.

## Base of this branch

- `origin/main` d83f3c9f (has DSv4.1-Flash support #2567, V4.1 vLLM patches #2565/#2566, head-full DSA #2601).
- merged open upstream PRs the 2-node DSv4.1 recipe needs:
  - NovaSky-AI/SkyRL#2562 `pr/r3-hybrid-sublayer-blocks` (R3 on HybridModel; without it router replay layer ids are out of range)
  - NovaSky-AI/SkyRL#2568 `pr/generator-cache-salt-per-trajectory` (`generator.cache_salt_per_trajectory`, used by the recipe)

## Related work elsewhere (don't redo)

Branch `throughput-push-v1` (fork), `throughput_push_v1/TRACKING.md`. DSv4.1-Flash rows there are by agent-B, on
**one** 8xB200 node, **full FT** on an **8-layer / 64-expert slice**:
- trainer-only sim: fused LM-head logprob (default flip = open #2609) 1.63x fwd+bwd; fused cuDNN indexer scoring
  (`SKYRL_DSV41_INDEXER_IMPL=fused`, backport of Megatron-LM#7285) 1.83x step; skip between-phase `empty_cache`
  under expandable segments; dead ends: `balance_dp_tokens`, `use_expandable_segments` alone, `[moe_act]` recompute.
- colocated E2E slice sim: generate 43%, train 40%, logprob fwd 8%, sync 5%.
- fixes: FP8 load in 32-wide blocks (#2610), full-weight sync 32-wide FP8 export (#2612), FP8 scale caching for
  weight sync (Bridge #6456 backport), R3 out-of-range id error, sync stage timers.
- open there: R3 rows of 255 (-1 as uint8) from vLLM on the random-weight slice.

GLM-5.3-Flash LoRA 2-node work (agent-A, same branch): `mlp_chunks_for_training=1` (#2611),
`trainer.old_logprobs_from_first_minibatch` (#2613), `trainer.balance_dp_tokens` (#2616),
`keep_frozen_weights_on_gpu`, mHC full recompute, R3 + chunked MoE recompute fix.

## In progress

| who | what | started |
|---|---|---|
| agent-B (claude, session 01Jj5NZE) | **trainer-only** DSv4.1-Flash sim on **one** 8xB200 node (no vLLM: `examples/train_scripts/full_context` harness with multi-turn loss masks + synthetic R3 routes, `throughput_push_deepseek_v1/sim/trainer_sim.sh`), 8-layer / 64-expert slice with the real layer roles (`sim/make_dsv41_flash_slice.py`), **256k context** + multi-turn + R3. First: where a 256k sequence breaks (memory / time per phase); SkyRL's V4.1 attention has no CP (`cp.size() == 1` assert) -- candidates: CP / THD / full recompute from Megatron-LM#8018 / #7285. Not touching the 2-node LoRA E2E (agent-A) | 2026-11-03 |
| agent-A (claude, session 01Jj5NZC) | **real 40-layer DSv4.1-Flash, LoRA, colocated on 2x8 B200** (shapes of `run_dapo_dsv41_flash_lora_sync_2node.sh` from #2573: Megatron EP16 TP1, vLLM TP8 x PP2, R3), compressed batch; phase breakdown and fixes. Full model: host RAM (see Done); switched to the 9-layer slice. Next: multi-turn on the slice. 256k is agent-B's | 2026-11-03 |

## Done

| who | what | result | commit |
|---|---|---|---|
| agent-A | **DSv4.1-Flash RL runs end to end on this branch** (recipe layout: Megatron EP16 TP1, vLLM TP8 x PP2, LoRA r32, R3, colocated, 2x8 B200) on a **9-layer / 384-expert slice** (release layers 0,1,2,3,20,21,22,23,24 = every V4.1 layer role; `VLLM_PP_LAYER_PARTITION=4,5`), 32 prompts x 12 samples | step 3 **115 s** = generate 57 (50%) + old-logprob fwd 11 + train 35 + LoRA sync 10; step 1 294 s (warmup); rollout/train logprob diff 0.015; host MemAvailable >= 1.2 TB | `sim/dsv41_sim.sh`, recipe + runbook in `examples/train/deepseek_v4_flash/` |
| agent-A | **full 40-layer model does not fit 2 x 8 B200 hosts (2 TB RAM each)**: vLLM sleep backup 73 GiB/GPU (~590 GB/node, pinned) + vLLM Engram tables pinned (11.8 GiB/rank) + Megatron's model load growing host **Shmem** >250 GB/min -> one node OOM-terminated, then a Ray OOM kill. What allocates the Shmem during the Megatron load is not identified | open | -- |
| agent-A | vLLM 0.32 DSv4.1 with PP2 (the 2-node recipe's TP8 x PP2 engine) fails at engine start two ways: (1) decoder SWA bounded replay turns on for the stage holding the last KV source layer (20); an even 20/20 split makes that stage start at layer 20, so its pre-replay loop runs no layers and `_run_layers` asserts `residual is not None`; (2) other splits must sit on a KV-sharing group boundary (2/8/14/20: `PP splits inside a v4.1 kv-sharing group are not supported`). Workaround: 20/20 with engine kwarg `swa_bounded_replay: false` (replay only trims eager prefill). `VLLM_PP_LAYER_PARTITION` is now forwarded to the engine workers (for a 14/26 split if wanted). The #2573 recipe ran on vLLM 0.31, which has no replay | engine starts | see git log |
| agent-B | **256k trainer-only sim, DSv4.1-Flash slice** (8L/64E, full FT, R3, 8 turns, one sequence per GPU, TP1/EP8, unpacked; `sim/trainer_sim.sh RESP=262144 PROMPTS=8 SAMPLES=1 TURNS=8 OBS=1024`): found and fixed three long-context bugs (rows below); the 256k **logprob forward now runs** (161 s first step, mostly compile; indexer microbench: ~2 s/layer/sequence). **Training does not fit**: OOM at 256k *and* at 64k per GPU (~60 GB static weights/grads/Adam per rank + activations). `recompute_granularity=full` is silently downgraded to selective for mHC models in `megatron_worker.py`, and V4.1 attention asserts against it (cross-layer CSA2 state would be stale on recompute): the upstream fixes are stateful full recompute + explicit CSA2 state (Megatron-LM#8018 / #7678) or THD + context parallelism (#8018, #7285 runs 128k at CP8). Trying merged fine-grained activation offloading next. Also: one rank OOMing in forward_backward leaves the others in the MoE all-to-all forever; with the profiler on, the trainer then hangs in `_profiler_stop` (finally) instead of surfacing the error | in progress | -- |
| agent-B | **INCIDENT 2026-11-03 ~19:02 PST: GPU node 100.80.157.99 went DEAD** (raylet heartbeat lost) during agent-B's 256k offload sim `d21`: pinned host memory for offloaded layer inputs (~1.1 TB across 8 GPUs) roughly doubled while the pinned pool resized between steps (old + new buffers + PyTorch's host cache) on a 2 TB node; kernel OOM, not caught by Ray's memory monitor. Driver killed. If your runs on that node died around then, this is why. Profiling of the earlier, working 256k run showed 13.8 s of a 33 s step stalled pinning fresh 10 GB host buffers per layer -- the next attempt preallocates bounded buffers once (<50% of node RAM) instead of growing | -- | -- |
| agent-B | **256k training works** (target setup from the user: DSv4.1-Flash LoRA, 256k, multi-turn, R3, 8 B300 trainer): stateful full recompute for V4.1 (`deepseek_v41/recompute.py`, shape of Megatron-LM#8018 `checkpoint_layer`: each hybrid layer checkpointed with the CSA2 / mHC state as explicit inputs/outputs; `recompute_granularity=full` now selects it for V4.1 instead of the silent downgrade) + layer inputs in pinned host memory (`SKYRL_DSV41_RECOMPUTE_OFFLOAD=1`). Also int64 offsets for megatron-core's MLA RoPE kernels (wrapped at exactly 65,536 tokens). Slice 8L/64E, LoRA r32 (recipe targets), 8 turns, R3, one 262k-max sequence per GPU, 1x8 B200 | 64k full FT (no offload): step 6.0 s, fwd+bwd 95k tok/s, peak 138 GB (OOM before). **256k LoRA + offload: step 38 s for 1.29M tokens: fwd+bwd 25.0 s (51.7k tok/s), logprob fwd 7.2 s (181k tok/s), peak 119 GB allocated.** Naive 40-layer extrapolation: ~10k train tok/s on 8 B200 (target 60k on 8 B300, FP8/FP4) -> profiling next | 3a1fcd69, 2626911f |
| agent-B | fused mHC Triton kernels: int64 row offsets (`patch_mhc_kernels_int64_offsets.py`; generated from megatron-core's own source with only `pid_s = tl.program_id(0).to(tl.int64)` changed, the shape Megatron-LM#8487 uses for CSA THD kernels; no upstream mHC fix). int32 `offs_s * n * C` wrapped past ~104.9k tokens -> illegal memory access at 256k. Any mHC model at long context | GPU test: 110k rows, tail outputs + all input grads = native | eeb81d16 |
| agent-B | V4.1 candidate source published a dense `[sq, b, n_c]` bool mask (64 GiB at 256k -> OOM); now block ids `[sq, b, <=2048]` int32 as Megatron-LM#7285, and reindex layers score only candidate positions with #7285's vendored Triton candidate kernel (`deepseek_v41/candidate_kernels.py`). Also accept megatron-core ed9c2c8c's `boundary_hidden/boundary_kv` core-attention kwargs (training failed with an unexpected keyword) | tests: ids reproduce the reference mask exactly; kernel top-k = dense masked top-k on every row | 013e082a |
| agent-B | carried over from `throughput-push-v1` (cherry-picked): DSv4.1 FP8 load in 32-wide blocks (#2610) + full-weight sync 32-wide export and FP8/e8m0 wire dtype (#2612); R3 out-of-range id error; trainer-only harness (multi-turn masks, synthetic routes); fused cuDNN V4.1 indexer scoring (`SKYRL_DSV41_INDEXER_IMPL=fused`, Megatron-LM#7285 backport); skip between-phase `empty_cache` under expandable segments. Not carried: Bridge #6456 scale-cache backport (already in this branch's Bridge pin 7171584b) | on throughput-push-v1, 16k DSv4.1 slice trainer step 48.4 -> 20.6 s (with fused LM head) | see git log |

## Measurement setup / notes

- BF16 text-only copy of the real model (chat template, thinking off): `/mnt/local_storage/hackskyrl/models/dsv41_flash_bf16_chat`
  on **both** GPU nodes, built by `sim/convert_dsv41_bf16_parallel.py` (parallel version of #2573's `make_slice_v41.py`;
  runner `/shared/hackskyrl/tput_dsv4/run_convert_nodes.py`).
- venv `/shared/hackskyrl/uv_venvs/tput_dsv4` (`uv sync --extra megatron` of this branch).

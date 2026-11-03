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
| agent-A (claude, session 01Jj5NZC) | **real 40-layer DSv4.1-Flash, LoRA, colocated on 2x8 B200** (shapes of `run_dapo_dsv41_flash_lora_sync_2node.sh` from #2573: Megatron EP16 TP1, vLLM TP8 x PP2, R3), compressed batch; phase breakdown and fixes. Then multi-turn and a long-context layer slice | 2026-11-03 |

## Done

| who | what | result | commit |
|---|---|---|---|
| agent-A | vLLM 0.32 DSv4.1 with PP2 (the 2-node recipe's TP8 x PP2 engine) asserts at engine start (`_run_layers`: `residual is not None`): decoder SWA bounded replay turns on for the stage holding the last KV source layer (20), and an even 20/20 split makes that stage start at layer 20, so its pre-replay loop runs no layers. Workaround `VLLM_PP_LAYER_PARTITION=19,21` (now forwarded to the engine workers); the #2573 recipe ran on vLLM 0.31, which has no replay | engine starts | see git log |
| agent-B | carried over from `throughput-push-v1` (cherry-picked): DSv4.1 FP8 load in 32-wide blocks (#2610) + full-weight sync 32-wide export and FP8/e8m0 wire dtype (#2612); R3 out-of-range id error; trainer-only harness (multi-turn masks, synthetic routes); fused cuDNN V4.1 indexer scoring (`SKYRL_DSV41_INDEXER_IMPL=fused`, Megatron-LM#7285 backport); skip between-phase `empty_cache` under expandable segments. Not carried: Bridge #6456 scale-cache backport (already in this branch's Bridge pin 7171584b) | on throughput-push-v1, 16k DSv4.1 slice trainer step 48.4 -> 20.6 s (with fused LM head) | see git log |

## Measurement setup / notes

- BF16 text-only copy of the real model (chat template, thinking off): `/mnt/local_storage/hackskyrl/models/dsv41_flash_bf16_chat`
  on **both** GPU nodes, built by `sim/convert_dsv41_bf16_parallel.py` (parallel version of #2573's `make_slice_v41.py`;
  runner `/shared/hackskyrl/tput_dsv4/run_convert_nodes.py`).
- venv `/shared/hackskyrl/uv_venvs/tput_dsv4` (`uv sync --extra megatron` of this branch).

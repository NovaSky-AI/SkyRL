# GLM-5.3 (753B `glm_moe_dsa`) on 8×8 B200: FP8 rollouts, bf16 trainer

Handoff notes for this branch (2026-09-30). Everything here ran on the 8-node B200 cluster
(`b200-resv-ray-0..7`, Ray head 10.180.0.9:6380). W&B project names are given per run.

## Recipes

| Script | What it is | Status |
|---|---|---|
| `run_gsm8k_glm5p3_lora_fp8_rollout_8node.sh` | GSM8K GRPO, LoRA r64 (`share_expert_adapters=true`, `merge_lora=true`), TP8/EP64, 32×8 | 5-step smoke test passed: GSM8K eval 0.82 → 0.93 (run `eywye3sd`, project `glm5p3_gsm8k`) |
| `run_dapo_glm5p3_fullft_fp8_rollout_8node.sh` | DAPO full fine-tuning, 2k prompt / 8k response, 128×12, LR 1e-6, TP8/EP64, 20 steps | Stage 1 (no R3) ran 10 steps; stage 2 (R3) running, see below |
| `run_dapo_glm5p3_lora.sh` | Upstream BF16 DAPO LoRA reference recipe, 4×8 B300 (see [below](#upstream-bf16-lora-reference-recipe-b300)) | configuration-checked upstream, not rerun here |

Both serve GLM-5.3 from 8 vLLM engines at TP8 with online FP8 (`load_format=dummy`; the trainer
syncs bf16 weights and vLLM's layerwise reload re-quantizes them). bf16 would need ~176 GiB/GPU
and does not fit a B200; FP8 is ~89 GiB/GPU. No `fp8_weight_sync_mode`.

## Code changes on this branch

- `patches/vllm/patch_layerwise_reload_eager.py` — vLLM's layerwise reload buffered every
  incoming tensor at its **unsharded** size (the packed IPC consumer clones it) until the whole
  layer arrived: ~19 GiB per GLM-5.3 MoE layer on every TP rank. The patch loads each tensor into
  the materialized local shard as it arrives. Bitwise-identical to stock over repeated reloads
  (TP1 A/B); at TP8 it took vLLM's sync-time peak from ~122 to ~102 GiB/GPU.
- `patches/vllm/patch_per_block_fp8_param.py` — `quantization=fp8_per_block` crashed on the
  first sync (`InternalTorchDynamoError: RecursionError`): the torch.compile'd
  `per_block_cast_to_fp8` was handed a vLLM parameter subclass. The patch passes `.data`.
- `patches/vllm/patch_routed_experts_rebind.py` — backport of vllm-project/vllm#59455 (fixes
  #59449). R3's routed-experts capture callback is bound once at startup and was lost when a weight
  sync rebuilt the monolithic FlashInfer TRT-LLM FP8 MoE kernel, so capture returned the startup
  profile run's routing for every prefill token. The patch carries the callback over.
- All three are installed from `inference_servers/new_inference_worker_wrap.py`.
- `weight_sync/fp8/models/glm5.py` (+ test) — GLM-5.3 `ModelFp8Spec` for
  `fp8_weight_sync_mode=blockwise` (stage 3). Follows zai-org/GLM-5-FP8's split; the indexer's
  `wk`/`weights_proj` stay bf16 because vLLM fuses them into an unquantized linear.
  **Not exercised end to end yet.**

## Settings that matter (all in the recipes, with comments)

- `mlp_chunks_for_training=1`. The Flash recipes' 64 **deadlocks** `glm_moe_dsa` under EP:
  `torch.chunk` gives fewer chunks on ranks with fewer tokens, so EP all-to-all counts diverge.
  Symptom: every GPU at "100%" but ~236 W. (Flash is unaffected: its mHC layer never chunks.)
- `dsa_indexer_loss_coeff=0.0`. GLM5Bridge defaults to 0.001; its unfused backward OOMs at 8k.
  Same as the Flash bridge; the indexer stays at its pretrained weights.
- `dsa_kernel_backend=tilelang`. The default naive path materializes 32 heads × s² fp32 index
  scores (~10 GiB at 9k tokens). TileLang top-k is identical and SparseMLA fwd+bwd is within 0.3%
  on B200, ~10× less memory. Needs `CUDA_HOME` with nvcc (`/mnt/local_storage/cuda-13.0` here).
  (`cudnn` would need `flash_mla`, not installed.)
- `use_precision_aware_optimizer=true`. With CPU offload the default optimizer still keeps fp32
  master copies on the GPU (~50 GiB/GPU for full FT, ~6–11 GiB for the LoRA runs) while the CPU
  optimizer steps its own fp32 copies. Trainer peak went 157 → 117 GiB. Caveat: checkpoint saving
  (megatron-lm#1820); checkpoints are off.
- Trainer TP8 (not TP4) and LoRA r64 in the GSM8K recipe: at TP4 / r256 the post-step sync OOMed
  by 1–2 GiB (before the eager-reload patch and precision-aware optimizer existed; probably fits
  now, untested).
- R3 works on vLLM's default FP8 MoE backend (FlashInfer TRT-LLM) only with
  `patch_routed_experts_rebind`. Without it R3 made the logprob gap **worse** (0.063 → 0.164)
  because every prompt token was replayed with one stale expert set.
  `R3_MOE_BACKEND=deep_gemm` (router-side capture, not affected) is the fallback; it costs ~1.7×
  generation time.
- `FULL_FP8=true` (DAPO recipe): MXFP8 GEMMs on the trainer (`fp8=e4m3`, `fp8_recipe=auto`) plus
  `fp8_weight_sync_mode=blockwise`: the trainer casts its bf16 export to 128×128 FP8 with the GLM5
  `ModelFp8Spec` and SkyRL configures vLLM for checkpoint-format FP8 (the online
  `VLLM_QUANTIZATION` is then unused). Pre-training eval on this path matched the online-FP8 runs
  (55.8% vs 52–55%), so the spec and sync produce correct weights.

## Results so far (DAPO full FT, AIME-2024, 12 samples/problem)

The eval is truncation-bound at the 8k budget: accuracy among responses that finish is 98-100% at
every checkpoint, so the score is effectively the fraction of responses that finish. Report
`eval/all/mean_positive_reward` (fraction correct) alongside the ±1 `avg_score`.

| Fraction correct (eval step 0 / 5 / 10 / 15 / 20) | Run | Notes |
|---|---|---|
| 52.5% / 56.7% / – / – / – | no R3, bf16 trainer, `1o4x7v31` | died at the sync after step 10 (`unspecified launch failure`, one vLLM engine, no hardware Xid) |
| 54.4% / 62.2% / 51.7% / 48.3% / – | R3 + `deep_gemm`, bf16 trainer, `z9wad610` | stopped after 16 steps |
| 52.8% / 54.7% / 43.3% / – / – | R3 + TRT-LLM + rebind patch, bf16 trainer, `x7lm538i` | stopped after 10 steps for the R3 gradient check |
| **55.8% / 76.1% / 80.3% / 83.1% / 84.4%** | **full FP8 + R3** (`FULL_FP8=true`), `bz2t3wg9` | completed 20 steps; finished/truncated 200/160 → 309/51 |

Training-side, steps 1→10: no R3 shortened responses (4.46k → 3.81k tokens) while entropy
collapsed (0.107 → 0.043) and train reward rose (−0.09 → +0.21). Both R3 runs with the bf16
trainer lengthened them instead (4.5k → ~5.0k) and reward stayed mostly negative. Full FP8 + R3
shortened them fastest (4.2k → 3.1k by step 10, 2.75k by step 15), train reward −0.00 → +0.41
(step 10) → +0.55 (step 15), entropy 0.108 → 0.033.

| Step 1 | no R3, bf16 | R3, bf16 | full FP8 + R3 |
|---|---|---|---|
| rollout/old logprob gap (mean / max) | 0.063 / 17.6 | 0.037 / 6.9 | 0.028 / 9.6 |
| PPO clip ratio | 0.50% | 0.01% | 0.27% |
| generate / fwd_logprobs / policy_train | ~410 / ~320 / ~1,550 s | ~410 / ~320 / ~1,530 s | 406 / 658 / 2,485 s |
| step | ~40 min | ~40 min | ~60 min (43–60 as responses shortened) |
| trainer / sync peak per GPU | 117 / ~151 GiB | similar | 123 / 159 GiB |

Caveats: one run per arm. The full-FP8 run's rollouts already differed at step 1 (shorter, higher
reward from the same weights) because its FP8 weights come from the trainer's blockwise cast
(power-of-2 scales) instead of vLLM's online per-block quantization, so part of its gain may be
serving numerics rather than MXFP8 training. The two runs that learned (no R3; full FP8 + R3) both
clip ~0.3-0.5% of tokens per update, the two R3 + bf16-trainer runs that did not clip ~0.02%.
R3's gradients were checked directly (replay with vs without MoE recompute match to 7e-4 relative
on a 5-layer GLM-5.3 at TP8/EP8), so the divergence is not a replay bug.

## Open items

- Separate the full-FP8 gain into rollout numerics vs MXFP8 training: bf16 trainer with
  `fp8_weight_sync_mode=blockwise`.
- MXFP8 training is ~1.6× slower than bf16 at TP8/EP64 (fwd 658 vs ~320 s, train 2,485 vs ~1,550 s).
- `policy_train` is ~2× slower with TileLang DSA than with the naive kernels (~1,550 s vs ~850 s
  per step) even though the kernels are faster in isolation; not profiled.
- Why R3 + `deep_gemm` diverged from no-R3 after step 5 (`8l9ng63u` answers whether the MoE
  backend was the cause).
- Raise the response budget: at 8k both reward and eval mostly measure length.
- The step-10 `unspecified launch failure`; if it recurs, test the sync with and without the
  eager-reload patch.
- Eval at 8k response budget is truncation-bound: every finished AIME response was correct at
  step 0 and every wrong one hit the limit.

## Running on this cluster

Model at `/mnt/local_storage/models/glm5p3-bf16` on every node, DAPO data at `~/data/dapo`.
Launch with plain `uv run` (the recipes' `--isolated` makes every Ray worker build its own env on
this cluster) and these env vars:

```bash
export RAY_ADDRESS=10.180.0.9:6380
export NCCL_NET=IB NCCL_NVLS_ENABLE=0       # /etc/profile.d sets gIB, which fails under torch's NCCL
export CUDA_HOME=/mnt/local_storage/cuda-13.0
export MODEL_PATH=/mnt/local_storage/models/glm5p3-bf16 DATA_DIR=$HOME/data/dapo
export SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S=1800
ENABLE_ROUTING_REPLAY=true bash examples/train/glm5_3/run_dapo_glm5p3_fullft_fp8_rollout_8node.sh
```

## Upstream BF16 LoRA reference recipe (B300)

This BF16 recipe uses four 8×B300 nodes. Each node alternates between a TP8/EP8
trainer replica and a TP8/EP8 vLLM engine. It uses R3 routing replay, rank-32 LoRA
with shared expert adapters, and in-memory adapter synchronization.

Use SkyRL's Megatron installation and start a Ray cluster across the nodes.
The same checkout, model and data must be available on every node. `RUN_ROOT`
must be on shared storage and fresh for each run; the script refuses to reuse an
existing checkpoint directory.

```bash
bash examples/train/algorithms/dapo/prepare_dapo_data.sh
RUN_ROOT=/shared/glm53-dapo RAY_ADDRESS=auto \
  bash examples/train/glm5_3/run_dapo_glm5p3_lora.sh
```

`MODEL_PATH`, `DATA_DIR`, and `NUM_NODES` can be overridden in the environment.
Additional config overrides can be passed as arguments. Changing the node count
requires checking batch divisibility and available memory.

Each step collects 128 prompts × 16 responses (2,048 trajectories), then performs
four optimizer updates with 32-prompt minibatches. The 20-step run has 80 updates;
the learning rate warms up over the first 40 updates. AIME evaluation runs before
training and every five steps, with 32 responses per problem. Responses are capped
at 8,192 tokens, with DAPO's soft length penalty above 4,096 tokens and loss masking
for unfinished responses. Dynamic sampling is disabled.

This is a current-main adaptation of the full-model learning run, not its exact
historical configuration: it uses current main and in-memory synchronization,
and evaluates the base policy in the same run. The historical run used file-based
synchronization. This updated recipe has been configuration-checked but has not yet
been rerun end to end on GPUs; memory headroom must include adapter loading.

For GLM-5.3 Flash, use the existing [two-node DAPO LoRA recipe](../glm5_3_flash/run_dapo_glm5p3_flash_lora_sync_2node.sh).
It has separate topology, LoRA, and optimization settings; it is not the exact
Tinker API configuration used for the published Flash learning curve.

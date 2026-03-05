# GLM-4.7-Flash H200 Validation Plan

> **BLOCKING DEPENDENCY**: PRs **#1241** (transformers 5.x bump) and **#1280** (return_dict=False fixes) are NOT yet merged. You must either merge them first or apply them manually on each node. Without these, GLM-4.7-Flash will fail to load (`Glm4MoeLiteForCausalLM` only exists in transformers >=5.0.0). See "What's NOT Merged but Needed" section below for exact steps.


## Context

We are training GLM-4.7-Flash (30B MoE, DeepSeek-V3 architecture) with GRPO on SkyRL using Megatron backend + vLLM inference. All foundational work is done — 8 PRs merged, full pipeline proven on 8x A100-80GB. We now need to scale up to H200 clusters.

## What Has Been Proven (on 8x A100-80GB)

### Full GRPO Pipeline — WORKING
- Generate → forward → backward → optimizer step → weight sync → repeat
- Config: Megatron TP=1/EP=8, vLLM 2 engines × TP=4, colocated, optimizer CPU offload
- ~47.5s/step with short context (1024 gen tokens), stable across many steps
- Rewards on GSM8K: pass@8=0.96, raw_reward=0.68

### Checkpoint Save/Load — WORKING
- PR #1268 (merged) uses `dp_reshardable` format — preserves full Adam optimizer state
- GPU CI test `test_save_load_checkpoint[megatron]` passes on latest main
- Logits match exactly after save → load → retrain (bitwise reproducible)

### Scaling to 8K Context — OOM on A100, Expected to Work on H200
- With `max_generate_length=8192` and `gpu_memory_utilization=0.7`:
  - Generation works (909s, correct rewards)
  - Forward pass works (129s)
  - **Backward pass OOMs by ~5 GiB** (vLLM sleeping holds 17 GiB + Megatron needs 58 GiB = 75 GiB > 80 GiB)
- H200 has 141 GB per GPU — this gives ~61 GB extra headroom, so the OOM should be eliminated

### What's Merged on `origin/main`

| PR | What |
|----|------|
| #1212 | Megatron correctness fixes (grad_scale_func, PP seed, weight sync pause/resume) |
| #1213 | 5 MoE config fields on MegatronConfig |
| #1214 | GLM-4.7-Flash bridge registration + megatron-bridge bump |
| #1215 | GLM-4.7-Flash GRPO example config + launch script |
| #1240 | vLLM 0.16.0 + required dep updates |
| #1247 | megatron-core 0.15.0 → 0.16.0 (fixes numpy 2.x, enables dp_reshardable) |
| #1264 | Per-token hard masking for off-policy correction |
| #1266 | Refactor Megatron param/grad offload to use mcore builtins |
| #1268 | Fix checkpoint: dp_reshardable default, preserves full optimizer state |

### What's NOT Merged but Needed

| PR | What | Status |
|----|------|--------|
| #1241 | Bump transformers to >=5.0.0 | **REQUIRED** — GLM-4.7-Flash's `Glm4MoeLiteForCausalLM` only exists in transformers ≥5.0.0 |
| #1280 | return_dict=False fixes for transformers 5.x | **REQUIRED** — `apply_chat_template` returns BatchEncoding in 5.x, need return_dict=False at ~12 call sites |

**IMPORTANT**: After cloning SkyRL and `uv sync --extra megatron`, you MUST also run:
```bash
uv pip install --python .venv/bin/python "transformers>=5.0.0"
```
This overrides the `transformers<5` pin from vLLM/megatron-bridge. The override is safe — slime also uses transformers 5.x in production with GLM-4.7-Flash.

Then apply the return_dict=False fixes:
```bash
git fetch origin tgriggs/glm47-h200-scripts
git cherry-pick origin/tgriggs/glm47-h200-scripts  # just 1 commit: return_dict=False
```

## Available Clusters

| Cluster | GPUs | Memory/GPU |
|---------|------|-----------|
| 2x8xH200 | 16 | 141 GB |
| 4x8xH200 | 32 | 141 GB |

## Scripts

All scripts are at `/home/ubuntu/sky_workdir/scripts/` (on the A100 machine) and should be copied to the H200 clusters. They are also committed in PR #1280 branch but NOT in the scripts — you'll need to copy them manually or recreate them.

### `00_setup_and_sanity_check.sh` — Run on EVERY node
1. Verifies GPUs, CUDA, NVLink topology, network
2. Clones SkyRL, installs deps, installs transformers 5.x
3. Prepares GSM8K dataset
4. Runs Megatron checkpoint save/load GPU test (dp_reshardable + fully_reshardable)

### `01_train_2x8_h200.sh` — For the 2x8 cluster
```bash
bash 01_train_2x8_h200.sh phase3   # 8-GPU quick smoke test (1K context, 1 epoch, ~5 min)
bash 01_train_2x8_h200.sh phase4   # 8-GPU full slime match (8K context, batch=256, 20 epochs)
bash 01_train_2x8_h200.sh phase5   # 16-GPU multi-node (TP=4, EP=8, batch=512)
```

### `02_train_4x8_h200.sh` — For the 4x8 cluster
```bash
bash 02_train_4x8_h200.sh          # 32-GPU (TP=4, EP=8, DP=4, batch=1024, n_samples=16)
```

## Training Config (Matching Slime's GLM-4.7-Flash Settings)

### Parallelism (8-GPU / single node)
| Setting | Value | Source |
|---------|-------|--------|
| Megatron TP | 1 | Slime 8-GPU config |
| Megatron PP | 1 | Slime 8-GPU config |
| Megatron CP | 1 | Slime 8-GPU config |
| Megatron EP | 8 | Slime 8-GPU config |
| vLLM engines | 2 × TP=4 | SkyRL uses vLLM (slime uses SGLang with DP attention) |
| Colocated | true | Both use colocated |

### Parallelism (16-GPU / 2 nodes)
| Setting | Value | Source |
|---------|-------|--------|
| Megatron TP | 4 | Slime 16-GPU config |
| Megatron EP | 8 | Slime 16-GPU config |
| vLLM engines | 2 × TP=8 | Scaled up from 8-GPU |
| Nodes | 2 | Multi-node ray |

### Batch & Generation
| Setting | Value | Source |
|---------|-------|--------|
| train_batch_size | 256 (8-GPU) / 512 (16-GPU) / 1024 (32-GPU) | Slime uses 256 on 8-GPU |
| n_samples_per_prompt | 8 | Slime |
| max_prompt_length | 512 | GSM8K prompts are short |
| max_generate_length | 8192 | Slime |
| max_model_len (vLLM) | 8704 | 512 + 8192 |
| gpu_memory_utilization | 0.7 | Slime uses 0.7 |

### Optimizer
| Setting | Value | Source |
|---------|-------|--------|
| lr | 1e-6 | Slime |
| weight_decay | 0.1 | Slime |
| adam_beta1 | 0.9 | Slime |
| adam_beta2 | 0.98 | Slime (not default 0.999!) |
| optimizer_cpu_offload | true | Slime |
| optimizer_offload_fraction | 1.0 | Slime |
| overlap_cpu_optimizer_d2h_h2d | true | Slime |
| use_precision_aware_optimizer | true | Slime |
| num_warmup_steps | 0 | Constant LR (slime uses constant) |

### MoE Flags (all via transformer_config_kwargs)
| Setting | Value | Source |
|---------|-------|--------|
| moe_token_dispatcher_type | alltoall | Slime |
| moe_router_load_balancing_type | seq_aux_loss | Slime |
| moe_aux_loss_coeff | 0.0 | Slime (aux loss disabled) |
| moe_grouped_gemm | true | Slime |
| moe_permute_fusion | true | Slime |
| moe_router_score_function | sigmoid | Slime |
| moe_router_pre_softmax | true | Slime |
| moe_router_enable_expert_bias | true | Slime |
| moe_router_bias_update_rate | 0 | Slime (freeze bias for RL) |
| moe_router_topk_scaling_factor | 1.8 | Slime |
| moe_router_dtype | fp32 | Slime |

### Memory & Precision
| Setting | Value | Source |
|---------|-------|--------|
| recompute_granularity | full | Slime |
| recompute_method | uniform | Slime |
| recompute_num_layers | 1 | Slime |
| accumulate_allreduce_grads_in_fp32 | true | Slime |
| make_vocab_size_divisible_by | 64 | Slime |
| no_rope_fusion | true | Slime |
| flash_attn | true | GLM-4.7-Flash supports flash attention |
| empty_cuda_cache | true | Proven needed for memory on A100 |

### Algorithm (GRPO)
| Setting | Value | Source |
|---------|-------|--------|
| advantage_estimator | grpo | Slime |
| policy_loss_type | regular | Standard PPO clip |
| eps_clip_low | 0.2 | Slime |
| eps_clip_high | 0.28 | Slime |
| use_kl_loss | false | Slime sets KL coef=0.0 (effectively same) |

## Recommended Execution Order

### On 2x8xH200

1. **Setup both nodes**: `bash 00_setup_and_sanity_check.sh` on each node
2. **Phase 3** (quick smoke test): `bash 01_train_2x8_h200.sh phase3`
   - 8 GPUs, 1K context, batch=64, 1 epoch
   - Should complete in ~5 minutes
   - Validates: model loads, bridge works, generate+train+checkpoint all pass
3. **Phase 4** (the real test): `bash 01_train_2x8_h200.sh phase4`
   - 8 GPUs, 8K context, batch=256, 20 epochs
   - This is the config that OOMed on A100 — H200 should handle it
   - Validates: full slime-matching config works end-to-end
   - Expect: ~15 min/step for generation, ~2 min for training, ~18 min total/step
4. **Phase 5** (multi-node): `bash 01_train_2x8_h200.sh phase5`
   - Requires starting ray worker on node 2 first
   - 16 GPUs, TP=4, EP=8, batch=512
   - Validates: multi-node NCCL, cross-node weight sync

### On 4x8xH200

1. **Setup all 4 nodes**: `bash 00_setup_and_sanity_check.sh` on each
2. **Phase 6**: `bash 02_train_4x8_h200.sh`
   - Start ray workers on nodes 1-3 first
   - 32 GPUs, TP=4, EP=8, DP=4, batch=1024, n_samples=16
   - Validates: large-scale training

## Known Issues & Gotchas

### transformers 5.x Override
- vLLM 0.16.0 and megatron-bridge both declare `transformers<5`, but GLM-4.7-Flash requires ≥5.0.0
- Must install manually: `uv pip install "transformers>=5.0.0"`
- Must apply return_dict=False fixes (PR #1280 or cherry-pick)

### vLLM CUDA Context Overhead
- vLLM's sleeping process holds ~17 GiB per GPU regardless of gpu_memory_utilization
- On A100 (80GB) this caused OOM with 8K context. On H200 (141GB) this should not be an issue.
- If OOM occurs on H200 (unlikely), reduce gpu_memory_utilization to 0.5

### SkyRL Config System
- Does NOT support Hydra-style `+key=value` for new fields
- All kwargs go through `transformer_config_kwargs` (dict field) or `optimizer_config_kwargs` (dict field)
- These are set as `trainer.policy.megatron_config.transformer_config_kwargs.KEY=VALUE`

### Multi-Node Ray
- Head node: `ray start --head --node-ip-address=<HEAD_IP>`
- Worker nodes: `ray start --address=<HEAD_IP>:6379`
- Data must be on shared filesystem or replicated to all nodes
- GSM8K is small (~2MB) — just run the dataset prep script on each node

### Checkpoint Format
- Default is `dp_reshardable` (PR #1268) — preserves full optimizer state
- Only reshardable along DP dimension (cannot change TP/PP after save)
- For TP/PP resharding, set `dist_ckpt_optim_fully_reshardable=true` (slower, gathers to rank 0)

## Reference: Slime's Exact Config Sources
- 8-GPU script: https://github.com/THUDM/slime/blob/main/scripts/run-glm4.7-30B-A3B-8gpus.sh
- Model config: https://github.com/THUDM/slime/blob/main/scripts/models/glm4.7-30B-A3B.sh
- Docs: https://thudm.github.io/slime/examples/glm4.7-30B-A3B.html

## Reference: Key Files in SkyRL
- Megatron worker: `skyrl/backends/skyrl_train/workers/megatron/megatron_worker.py`
- Megatron strategy (checkpoint): `skyrl/backends/skyrl_train/distributed/megatron/megatron_strategy.py`
- Worker base (save/load checkpoint): `skyrl/backends/skyrl_train/workers/worker.py`
- Config: `skyrl/train/config/config.py`
- Trainer: `skyrl/train/trainer.py`
- vLLM engine: `skyrl/backends/skyrl_train/inference_engines/vllm_engine.py`
- GLM bridge registration: `skyrl/backends/skyrl_train/workers/megatron/megatron_worker.py` (lines 64-107)
- Example Qwen3-30B megatron script: `examples/train/megatron/run_megatron_dapo_qwen3_30b_a3b.sh`

## Reference: Progress Tracking
- Full project context: `/home/ubuntu/sky_workdir/progress.md`
- First-run log: `/home/ubuntu/sky_workdir/glm-47-flash-first-run.md`
- This plan: `/home/ubuntu/sky_workdir/h200_validation_plan.md`

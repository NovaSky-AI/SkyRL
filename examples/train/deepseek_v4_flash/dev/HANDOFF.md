# DeepSeek-V4-Flash (then V4.1-Flash) support in SkyRL — agent handoff

Status as of 2026-10-23. Branch `dsv4-flash` on `avigyabb/SkyRL`, cut from `NovaSky-AI/SkyRL` main at
`3a6e3b4e`. Follow `.agents/docs/new_model_support.md` (Eric Tang's staged plan: Stage 0 roundtrip
logprob tests -> Stage 1 GSM8K -> Stage 2 DAPO + R3 + checkpointing -> Stage 3 perf/MXFP8), and read the
routing docs in `CLAUDE.md` (megatron backend, weight sync, testing) before changing anything.

Plan agreed with the user: **do DeepSeek-V4-Flash first, then DeepSeek-V4.1-Flash-1017.**

## TL;DR

- DSv4-Flash is mostly integration work: every upstream piece exists at our pins.
- **Stage 1 done**: GSM8K LoRA (1 node) and full FT (4 nodes) both train end to end with stable logprob gaps.
- Stage 0 (slice): **both rows pass** -- full FT (rt16: Megatron vs vLLM 0.050 < 0.1; post-sync common-prefix
  0.015 < 0.05) and LoRA (lora5: zero adapter 0.047, stale sampler 0.184 > 0.1, updated adapter 0.048).
- Full model: forward matches (0.040); sync needs 2 nodes (OOM on one). Next: that, then Stage 1.
- Stage 2: checkpoint save/resume passes (LoRA + full FT); DAPO LoRA with R3 in progress (see Stage 2).

## What is on this branch

| File | What |
|---|---|
| `examples/train/deepseek_v4_flash/dev/make_slice.py` | Builds the BF16 4-layer slice from the local checkpoint (FP8 128x128 ue8m0 + MXFP4 experts dequantized, MTP dropped, `compress_ratios`/`num_hash_layers` rewritten, chat template embedded). |
| `examples/train/deepseek_v4_flash/chat_template.jinja` | DSv4 ships no Jinja template (only `encoding/encoding_dsv4.py`). Text-only system/user/assistant, `thinking` kwarg toggles thinking mode. Verified byte-identical to `encode_messages` for chat + thinking modes on 4 conversations. |
| `examples/train/deepseek_v4_flash/dev/run_roundtrip.sh` | Runs the roundtrip tests as a Ray job on a worker node (copied from the MiMo/Muse Spark pattern). |
| `tests/.../megatron/test_megatron_models.py` | New row `deepseek-v4-flash-4layer_h100_tp1_ep4` + per-model config: `remove_microbatch_padding=False`, `inference_only_init`, vLLM `max_model_len=4096`/`gmu=0.5`, 1800 s health timeout. Model path from `SKYRL_DEEPSEEK_V4_FLASH_4LAYER`. |
| `skyrl/backends/skyrl_train/workers/megatron/megatron_worker.py` | Drops `core_attn` from selective recompute for `experimental_attention_variant == "dsv4_hybrid"` (recompute off if nothing remains). |

## Machines (current, from 2026-10-23): 4 x GCP A4 VMs, 8 x B200 each

| Node | IP | Role |
|---|---|---|
| b200-resv-ray-0 | 10.180.0.9 | head; agent runs here; other users (etang, sumanthrh) also run jobs here |
| b200-resv-ray-1 | 10.180.0.10 | worker |
| b200-resv-ray-3 | 10.180.0.15 | worker |
| b200-resv-ray-8 | 10.180.0.14 | worker |

Each node: 8 x B200 (183 GB), 224 CPUs, 3.9 TB RAM, 8 x RoCE NICs (`rocep*`). ray-0 has passwordless ssh
(`~/.ssh/id_ed25519`) to the other three. No shared filesystem: every node has an identical copy of the
layout below (checkpoint, slice, GSM8K verified by file count + bytes; code at `~/avigyabb/SkyRL-dsv4`).
Inter-node TCP is open on arbitrary ports. No Ray cluster is running yet; single-node tests start their own
with `RAY_ADDRESS=local`. Check `nvidia-smi` before a GPU run and kill only your own processes.

**NCCL:** the interactive login env on these VMs sets Google's gIB net plugin (`NCCL_NET=gIB`,
`/usr/local/gib/lib64` on `LD_LIBRARY_PATH`, `NCCL_TUNER_CONFIG_PATH`), built for NCCL 2.27.5. The venv's
NCCL 2.29.7 can't load it ("Failed to initialize any NET plugin" -> `ncclInvalidUsage`). Unset those three;
NCCL's built-in IB transport then uses all 8 RoCE NICs: 3-node x 8-GPU all_reduce of 256 MiB at ~225-255
GB/s busbw. Non-interactive ssh shells don't get the gIB env, so `ray start` over ssh is already clean;
anything started from an interactive shell on ray-0 must unset it (run_roundtrip.sh does).

Storage: `/` (home) is a persistent disk with ~30 GB free -- keep caches off it. `/workspace` is a shared
persistent disk at 91%. Everything large lives on `/mnt/local_storage` (32 x local NVMe RAID0, 11 TB;
**lost if the VM is stopped/preempted** -- the checkpoint re-downloads and the slice rebuilds).

All under `DSV4_ROOT=/mnt/local_storage/avigyabb`:

| What | Path |
|---|---|
| Full checkpoint (deepseek-ai/DeepSeek-V4-Flash @ `60d8d707`, 46 shards, 149 GB, verified vs index) | `models/DeepSeek-V4-Flash` |
| 4-layer BF16 slice (52 GB) = `SKYRL_DEEPSEEK_V4_FLASH_4LAYER` | `models/DeepSeek-V4-Flash-4layer` |
| GSM8K parquet (`~/data/gsm8k` symlinks here) | `data/gsm8k` |
| uv cache (`UV_CACHE_DIR`; warm `--extra megatron` env) | `uv_cache` |
| HF cache (`HF_HOME`; downloads are unauthenticated) | `hf_cache` |
| vLLM cache (`VLLM_CACHE_ROOT`) / per-run FlashInfer autotune dirs | `vllm_cache/shared`, `vllm_cache/autotune/<run>` |
| Logs | `logs/` (`rt<N>-<ts>.log` per roundtrip run) |

Worktree: `/home/avigyabb/avigyabb/SkyRL-dsv4` (remote `fork` = avigyabb/SkyRL).

Not on the HF Hub yet (needs a token from the user; never publish with a token found in the env).

## Multi-node Ray cluster (Stage 1+)

Ray daemons run from a persistent venv pinned to the uv env's Ray (2.57.0, Python 3.12.14) -- the system
`/opt/ray-venv` is 2.58.0, and a `uv run --isolated` env is deleted when `ray start` returns. Workers still run in
`uv run --isolated` envs (Ray's uv hook ships the working dir and `py_executable`).

```bash
B=/mnt/local_storage/avigyabb   # same layout on every node
uv venv --python 3.12.14 $B/ray-venv && uv pip install --python $B/ray-venv/bin/python 'ray[default]==2.57.0'
export UV_CACHE_DIR=$B/uv_cache HF_HOME=$B/hf_cache VLLM_CACHE_ROOT=$B/vllm_cache/shared \
  CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH RAY_TMPDIR=$B/ray-tmp
unset NCCL_NET NCCL_TUNER_CONFIG_PATH   # gIB plugin vs NCCL 2.29 (see Machines)
$B/ray-venv/bin/ray start --head --port=6390 --node-ip-address=<head ip> --temp-dir=$B/ray-tmp --dashboard-host=127.0.0.1   # head
$B/ray-venv/bin/ray start --address=<head ip>:6390 --node-ip-address=<this ip>                                                # others
```

Current cluster (2026-10-25): head b200-resv-ray-1 (10.180.0.10) + ray-0, ray-3, ray-8 = 4 nodes, 32 GPUs. Drivers
run on the head with `RAY_ADDRESS=10.180.0.10:6390`. On ray-0 (an interactive login box) start the raylet from a clean
env (`env -i HOME=$HOME USER=$USER PATH=/usr/local/cuda/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin ...`), or
the gIB NCCL settings leak in -- and keep `/usr/local/bin` on PATH: Ray workers start via `uv run --isolated`, and a
raylet without `uv` fails every worker there ("worker startup repeatedly failed"; `uv: not found` in raylet.err). Full BF16 model (`models/DeepSeek-V4-Flash-bf16`, 138 files /
568,766,622,520 bytes) on every node.

## Stage 1 (GSM8K)

- LoRA, 1 node (`run_gsm8k_dsv4_flash_lora_1node.sh`, W&B `dsv4_flash_gsm8k/tm3osw8z`): ~15 steps, reward 0.80 ->
  0.86-0.93, rollout/train logprob gap 0.041 (R3 off), entropy ~0.19. Step ~13-16 min, of which training ~11 min:
  kernel-launch-bound (all ranks at 100% of one CPU core, GPUs mostly idle) -- the first Stage 3 perf target.
- Full FT, 4 nodes (`run_gsm8k_dsv4_flash_fullft_4node.sh`, W&B `dsv4_flash_gsm8k/4juk94b5`, 2026-10-25): **10/10
  steps, clean exit**. Reward 0.78 -> 0.96 (0.78, 0.86, 0.89, 0.91, 0.76, 0.95, 0.97, 0.98, 0.92, 0.96); rollout/train
  logprob gap 0.034-0.054 every step (so the post-update full-model sync is correct); entropy 0.16-0.22; grad norm
  0.1-0.65 (0 at step 7: every group's samples tied, so GRPO advantages were all zero). ~117-121 GiB/GPU during
  training. Step ~10.5 min (train ~9.5 min, full sync ~53 s, rollout ~8 s).
- Full FT: needs **4 nodes** (`run_gsm8k_dsv4_flash_fullft_4node.sh`). On 2 nodes (EP=16) step 1 OOM'd in the forward
  with 162.8 GiB allocated whatever the recompute: megatron-core's distributed optimizer keeps fp32 main params on GPU
  even with `optimizer_cpu_offload` (the CPU Adam works from them; SkyRL offloads them only while vLLM runs), and with
  EP == DP the expert main params are unsharded: ~50 bf16 + ~73 fp32 + ~50 grads = ~173 GiB/GPU. EP=32 halves the
  expert terms (~102 GiB). Recompute must be selective `[moe, mhc, layernorm]` (no core_attn for DSv4; mHC forbids
  full recompute and mlp). The first full-model weight sync (2 nodes, ~530 GiB) took 60 s with no errors.
## Stage 2

- Checkpoint save/resume (`test_trainer_full_checkpointing.py`, 4-layer slice): **LoRA and full-FT rows pass**
  (`megatron_deepseek_v4_flash_{lora,fullft}_h100`). Full FT needed three fixes:
  1. megatron-core `DistributedOptimizer` indexes model params by grad-buffer order (bf16 buffer, then fp32) but
     orders each optimizer group's main params `[fp32..., bf16...]`; in a group holding both dtypes every param
     resolved to the next one's main param/state (a 5-param cycle on the slice: `output_layer`, mHC
     `mapping_proj`, compressor/indexer `ape`). Only checkpoint save/load read that index -- training is fine.
     `patches/megatron/patch_distopt_mixed_dtype_group_index.py` rebuilds it. This is also the "distributed-optimizer
     shape mismatch" the GLM-5.3 full-FT row is skipped for (unverified for GLM: no GLM slice here).
  2. With `optimizer_cpu_offload`, megatron-core builds the `HybridDeviceOptimizer` with `init_state_fn=None`, so
     SkyRL's `_ensure_optimizer_state_initialized` skipped it and a save before the first step had no
     `exp_avg`/`exp_avg_sq` (load then `KeyError`s). It now creates the sub-optimizers' first-step state.
  3. The DSv4 full-FT test row uses optimizer CPU offload (as the recipes do): on 4 GPUs the slice's fp32 Adam
     state otherwise OOMs at load. Checkpoint tests write to `SKYRL_TEST_CKPT_DIR` (local NVMe; a full-FT
     slice checkpoint filled a node's 35 GB root disk under /tmp).
- DAPO LoRA (`run_dapo_dsv4_flash_lora_sync_4node.sh`, 4 nodes, R3 on, W&B project `dsv4_flash_dapo`, logs
  `/mnt/local_storage/avigyabb/logs/dapo-lora4-*.log` on node 1). Baseline AIME-2024 avg_score 0.35, pass@12 0.90,
  mean response ~1.6K tokens; step-1 rollout reward ~0.44. Fixes needed to get through step 1 (runs 1-10):
  1. R3 needs prefix caching on in vLLM (its AuxOutput connector); the script no longer disables it.
  2. `lora_sync_path` defaults to `/tmp/skyrl_lora_sync` (owned by another user on node 3, and ~20 GB adapters
     fill the 35 GB root disks): the script uses `LORA_SYNC_PATH` (set to `$B/lora_sync`).
  3. R3 layer indices: vLLM reports routes per DSv4 block (43), Megatron's HybridModel counts sub-layers (86).
     `replay_utils._hybrid_sublayers_to_block_indices` maps them back.
  4. R3 all-to-all split mismatch: the replay layout ignored the CSA `seq_align` padding that
     `remove_left_padding` applies; `build_token_metadata_layout(..., seq_align=...)` now matches it.
  5. `mlp_chunks_for_training>1` chunks hash-routed layers' replayed routes wrongly (`[1280, 6]` vs `[20, 256]`):
     dropped from the script.
  6. LoRA + `recompute_modules=[..., mhc, layernorm]`: `StopIteration` in backward, because megatron-core's
     `CheckpointWithoutOutput` saves nothing when no input requires grad (frozen embeddings / base weights)
     but still recomputes. `patches/megatron/patch_checkpoint_without_output_frozen_inputs.py` keeps the inputs
     (GPU test in `gpu_ci/patches/megatron/`). MoE-only recompute OOMs on 8K-token rows.
  7. `Failed to CUDA calloc 10485760 bytes` (NCCL "unhandled cuda error") in the post-backward metrics all-reduce:
     with the optimizer fully on CPU `empty_cuda_cache` defaults off and the caching allocator holds everything.
     The script sets `empty_cuda_cache=true`.
  8. Run 10 then OOMed in the step-1 policy forward (MoE grouped GEMM output; 146.7 GiB allocated + 13.5 GiB held
     by the sleeping vLLM engine). Run 9 had squeaked through the same forward, so 8K responses sit right at
     the limit. `MAX_RESPONSE_LENGTH` now defaults to 4096 (env-overridable); 8K needs a memory cut (Stage 3).
  Run 11 (`dapo-lora4-1792950053.log`, 4K responses, W&B `dsv4_flash_dapo`): **step 1 trains end to end** --
  step 922 s (generate 149 s, logprob fwd 119 s, policy_train 521 s, sync 118 s), rollout-vs-train logprob
  abs diff 0.040, peak 95 GiB allocated (so ~6K responses would likely fit), grad_norm 0.027.
  Steps 1-10 (W&B run `pktgjo3p`): ~12-15 min/step; rollout reward 0.40, 0.40, 0.46, 0.40, 0.50, 0.38, 0.30, 0.44,
  0.44, 0.49 (different prompts per batch, so noisy); logprob diff 0.040-0.049; grad_norm 0.027-0.031; clip ratio
  ~0.05%. AIME-2024 avg_score (+1/-1 reward, so acc = (score+1)/2) at 4K responses: **0.222 -> 0.278 at step 10**
  (61.1% -> 63.9%), pass@12 0.933 both (the 8K baseline was 0.35: 4K truncates long solutions). Step-10
  checkpoint saved in 42 s, ~9 GB/node on node-local NVMe: each node holds only its own ranks, so resume
  needs the same rank placement (or the dirs synced to every node).
  **Run 11 finished all 20 steps** (2026-10-25 22:10, ~12-15 min/step, no errors). AIME-2024 avg_score
  0.222 -> 0.278 (step 10) -> 0.306 (step 20), i.e. accuracy 61.1% -> 63.9% -> 65.3%; pass@12 0.93/0.93/0.90.
  Rollout reward per step (different prompts each batch): 0.40 0.40 0.46 0.40 0.50 0.38 0.30 0.44 0.44 0.49
  0.38 0.42 0.45 0.45 0.28 0.40 0.44 0.36 0.58 0.42; logprob diff 0.040-0.051 throughout; peak ~97 GiB/GPU.
  Same hyperparameters as the GLM-5.3 DAPO LoRA run except 4K vs 8K responses; DSv4 starts at a much higher
  reward (~0.4 vs ~0.17), so the climb is smaller. Resume test: step-10 dirs merged onto every node at
  `ckpts/dapo-lora4-resume` (74 files, 36 GB).
  **Resume passes** (`dapo-lora4-resume-1792966381.log`): loaded global_step_10 (policy dist-ckpt + LoRA adapters,
  trainer + dataloader state) in 12 s on all 4 nodes, then trained steps 11-12: logprob diff 0.050 / 0.044 and
  grad_norm 0.027 / 0.026, matching the original run's steps 11-12 (0.051 / 0.045, ~0.03); rewards 0.42 / 0.42.
- DAPO full FT (`run_dapo_dsv4_flash_fullft_sync_4node.sh`, R3): runs end to end (step ~8-10 min, logprob diff
  0.034, peak 147 GiB at 2K responses; 2K OOMed at step 4 -> default 1.5K). Step-10 checkpoint (fp32 main params +
  distributed optimizer, 4 nodes) saves in 582 s. At 1.5K responses AIME is mostly truncated: avg_score -0.033 ->
  -0.061, pass@12 0.80 -> 0.63 at step 10, so this budget shows the pipeline, not learning; a real full-FT DAPO
  curve needs more nodes (or the Stage 3 memory work) for >=4K responses. W&B `dsv4_flash_dapo/0wjvh0wo`.
  It then OOMed at step 13 even at 1.5K (a long-row batch; 4-node full FT sits at the memory edge). Not relaunched:
  V4.1 has priority, and the pipeline / checkpoint path is what Stage 2 needed from it. The 4 DSv4 nodes are free.

- Multi-node vLLM boot hang: the same warm FlashInfer autotune cache hang as roundtrip run 3. SkyRL forwards an
  allow-list of env vars to Ray workers; `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR` is now on it and the scripts set a
  fresh per-run dir.

## How to run

```bash
cd /home/avigyabb/avigyabb/SkyRL-dsv4
RUN_ID=rt<N>-$(date +%s) bash examples/train/deepseek_v4_flash/dev/run_roundtrip.sh \
  "tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_megatron_models.py::test_logprobs_matching_roundtrip[deepseek-v4-flash-4layer_h100_tp1_ep4]" \
  > /mnt/local_storage/avigyabb/logs/$RUN_ID.log 2>&1
```

The `h100` id is historical; the row runs on 4 of the B200s. Rebuild the slice with
`DSV4_LOCAL_DIR=$DSV4_ROOT/models/DeepSeek-V4-Flash uv run --no-project --with torch --with safetensors --with numpy python make_slice.py $DSV4_ROOT/models/DeepSeek-V4-Flash-4layer`
(~5 min, CPU). GSM8K: `examples/train/gsm8k/gsm8k_dataset.py --output_dir $DSV4_ROOT/data/gsm8k`.

Previous machine (runs 1-3): Anyscale workspace, 2 x 8xB200 workers, artifacts on `/shared/...` (gone).

## Upstream status at our pins (all verified by reading the installed code)

- **vLLM 0.31.0**: `vllm/models/deepseek_v4/nvidia/model.py` `DeepseekV4ForCausalLM` with `SupportsLoRA`
  (packed: `gate_up_proj=[w1,w3]`, `fused_wqa_wkv=[wq_a,wkv]`, `fused_wkv_wgate=[wkv,wgate]`), loads
  DSv4-native names via `hf_to_vllm_mapper`. Serves the BF16 slice at TP=4 fine (14 GiB/rank, boots,
  generates). `process_weights_after_loading` finalizes mega-MoE and the mHC broadcast weights —
  **watch that weight sync re-runs these** (cf. the Muse Spark reload bugs, where vLLM's layerwise reload
  skipped custom loaders; test reloads with CUDA graphs on, not just eager).
- **transformers 5.16.1**: has `deepseek_v4`; its config strips legacy `compress_ratios`/`num_hash_layers`
  into `layer_types`/`mlp_layer_types` (the bridge understands both).
- **Megatron-Bridge `c4067f50`**: `megatron/bridge/models/deepseek/deepseek_v4_bridge.py` `DeepSeekV4Bridge`
  (DeepseekV4ForCausalLM -> `HybridModel`), each DSv4 block = attention hybrid layer (W/C/H) + MoE
  layer (E), mHC wrappers, hash routing (`tid2eid`), MTP, FP8/MXFP4 dequant on import, export back to
  DSv4-native names (no requant when `weight_dtype` is set). It sets `moe_token_dispatcher_type="flex"`/hybridep;
  SkyRL overrides that with its own `alltoall` default.
- **megatron-core `07147d69`**: `transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py`, `csa.py`.
  Limits at this pin: **TP must be 1** (attention is replicated; scale with EP/DP), **no packed THD**
  (`packed_seq_params` must be None), **no `checkpoint_core_attention`**. Megatron-LM main
  (`~/hackskyrl/Megatron-LM`) already has THD + CP support for DSv4 (`test_dsv4_packed_cp.py`) — a
  backport/bump candidate for Stage 3.

## Run history / fixes

1. `dsv4-rt-1792780514` (TP=2): `AssertionError: DSv4 Hybrid Attention only supports TP size 1.` -> row switched to TP=1, EP=4.
2. `dsv4-rt-1792783189` (TP=1): `Checkpoint core attention is not supported in DSv4 Hybrid Attention.`
   SkyRL defaults to full recompute; for mHC models the worker downgrades it to selective `["core_attn"]`.
   Fixed in `megatron_worker.py` (drop `core_attn` for `dsv4_hybrid`). This leaves recompute off;
   find a DSv4-compatible selective set (e.g. `moe_act`, `layernorm`) before long-context training.
3. `dsv4-rt-1792785532`: **hung in vLLM boot**, see below. Stopped after 30 min.

New machine (logs in `$DSV4_ROOT/logs/rt<N>-*.log`):

4. rt4: vLLM init `NCCL error: invalid usage` -- gIB plugin vs NCCL 2.29.7 (see Machines). Fixed in run_roundtrip.sh.
5. rt5: vLLM booted (590 s incl. MoE autotune, all ranks tuned together with a fresh per-run
   `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR`). Router died: `/tmp/skyrl-logs` owned by another user. Added
   `SKYRL_TEST_LOG_PATH` override in `tests/backends/skyrl_train/gpu/utils.py`.
6. rt6: reached Megatron `init_model`; `NVTE_FLASH_ATTN=0` (which I had added, per the megatron doc) conflicts
   with SkyRL's `attention_backend="flash"` default. Removed from the script.
7. rt7: Megatron forward: `CompressedSparseAttention supports only an implicit causal mask`. SkyRL passes the
   padding mask on the unpacked path. `MegatronModelWrapper` now passes `attention_mask=None` into the model
   for `experimental_attention_variant == "dsv4_hybrid"` (rows are left-aligned by `remove_left_padding`, so
   causal is exact); the mask is still used for logprob extraction.
8. rt8: `seqlen_q (211) must be <= seqlen_k * ratio (208)` in cuDNN's lightning indexer: the unpacked length must
   be a multiple of every compress ratio. `remove_left_padding(seq_align=...)`; the wrapper passes
   `max(csa_compress_ratios)` (128) for dsv4_hybrid. Trailing pad only completes a compressed block that pad
   positions alone can attend to.
9. rt9: **Megatron vs vLLM forward matches: mean |dlogprob| 0.049** (threshold 0.1; same in rt10-12). Weight sync
   then died in vLLM: `Cannot copy out of meta tensor` in `finalize_mhc_broadcast_weights`. vLLM's
   `DeepseekV4ForCausalLM.load_weights` runs its model-level finalize after every chunk, against layers the
   layerwise reload still holds on meta. `patches/vllm/patch_deepseek_v4_reload_finalize.py` defers it to the end
   of the sync (hooked in `weight_receivers.py`).
10. rt10: sync completes; **post-sync vLLM-vs-vLLM diff 0.87** (threshold 0.3). Real, not near-tie noise: rt11
   (common-prefix mode, since reverted) showed 80/80 sequences diverging at the first token.
11. Diagnosis (see "Post-sync mismatch" below): export is bit-exact; the bug reproduces in plain vLLM.
12. rt12: same 0.87 with fp32 kept on the wire -> confirms the export was not the cause.
13. rt13: attn_sink fix alone: 0.89 (the MoE clamp loss remained).
14. rt14: both vLLM fixes: positional 0.49 -- at the measured nondeterminism floor (0.39), not a weight problem.
15. rt15: common-prefix check, threshold 0.05: **PASSED** (Megatron 0.048; post-sync prefix 0.016 over 3535 tokens).
16. rt16 (with the FlashMLA preload): **PASSED** (0.050; prefix 0.015 over 4032 tokens).

Other rows:
- LoRA `deepseek-v4-flash-4layer_h100_tp1_ep4_adapter` (node 1): **PASSED** -- zero adapter 0.050, stale sampler
  0.194 (> 0.1), updated adapter 0.048 (< 0.1). First attempt died in vLLM boot because TileLang picked the
  venv's pip CUDA-13 `nvcc` (no `/usr/local/cuda/bin` on PATH in a non-interactive shell): "CUDA compiler and
  CUDA toolkit headers are incompatible". run_roundtrip.sh now sets `CUDA_HOME`/`PATH`.
- Full model, BF16 (`deepseek-v4-flash-full-bf16_h100_tp1_ep8`, env-gated, not for the PR; node 3,
  `$DSV4_ROOT/models/DeepSeek-V4-Flash-bf16`, 530 GB, built by make_slice.py with all 43 layers): **forward
  matches, 0.040**. Sync OOMs on one node (Megatron ~87 GB + woken vLLM weights ~90 GB per GPU); rerun on 2 nodes.

## Post-sync mismatch (fixed)

Symptom: after a weight sync vLLM served a different model (rt10-14). Root cause: two pieces of vLLM state that a
**level-2 sleep** zeroes and a layerwise reload never rebuilds (both reproduced and confirmed fixed in plain vLLM on
node 8 with a per-tensor checksum diff across sleep2 + reload, `vllm_reload_check.py` in the session scratchpad):

1. `attn_sink` (every DSv4 attention layer): `DeepseekV4Model.load_weights` writes it with a direct
   `params_dict[name][:n].copy_()`, bypassing `weight_loader`, so under layerwise reload the copy lands in the meta
   placeholder and is dropped ("DeepseekV4FlashMLAAttention: Failed to load weights"); the restored pre-reload storage
   is zeroed after a level-2 sleep. Fix: `patches/vllm/patch_deepseek_v4_reload.py` stashes sinks whose param is on
   meta and writes them back (per-rank head slice, -inf padding) after `finalize_layerwise_reload`. Same module defers
   the model-level finalize (mHC broadcast) to the end of the sync.
2. `gemm1_clamp_limit` (FlashInfer CUTLASS MoE, `FlashInferExperts.__init__`, `torch.full` per expert from
   `swiglu_limit`=10): a plain attribute of the kernel object, zeroed by the sleep -> every routed expert clamped to 0.
   Model-agnostic fix: `patches/vllm/patch_moe_kernel_constants.py` refills `gemm1_clamp_limit` / `gemm1_alpha` /
   `gemm1_beta` / `fake_input_scale` from the kernel's quant config after every full sync.

With both: 0/122 tensors differ across sleep2 + reload. Test metric: vLLM's DSv4 kernels are not bit-reproducible run
to run -- with **no sync at all** 52/80 greedy generations diverge and the positional post-sync diff is 0.386 (> the
0.3 threshold), common prefixes agree to 0.014; sleep2 + patched reload gives 0.376 / 0.014. So the DSv4 rows use
the test's common-prefix comparison (as Kimi / FP8 do) with `vllm_threshold` tightened 0.3 -> 0.05 (the broken sync
measured 0.233 on prefixes, which 0.3 would have passed).

Also found on the way (not the cause): the export is bit-exact (temporary `test_dsv4_export_check.py`, 3186/3186
tensors) once `MegatronWeightSource._wire_dtype` keeps fp32 (mHC, sinks, router bias) and integer (`tid2eid`) tensors
in their own dtype.

## LoRA: FlashMLA TOPK_LENGTH (fixed)

lora3/lora4 failed in the Megatron forward (no-grad CSA path -> `csa_sparse_attn(topk_length=...)` ->
`flash_mla_sparse_fwd`): "The chosen implementation does not support all required features ... TOPK_LENGTH".
Not data: the dumped failing inputs run fine in a fresh process. The failing process had **vLLM's bundled
`vllm/_flashmla_C.abi3.so` loaded before the pinned `flash_mla/cuda*.so`** (nv_dev @ b7643bd). Both statically link
FlashMLA with the same exported symbols, so the pinned forward dispatched through vLLM's older implementation table
(its SM100 head-64 kernel lacks TOPK_LENGTH; the pinned source's has it). Reproduced: same inputs fail iff
`vllm._flashmla_C` is imported first; pinned-then-vLLM works. The LoRA test module imports vLLM at import time, which
is why the LoRA row hit it (lora2 passed by load order luck). Fix: `patches/flash_mla/preload.py`, called from
`skyrl/backends/skyrl_train/__init__.py` (like the TE nvrtc pin): loads the pinned FlashMLA first in processes that
have not imported vLLM; no-op in vLLM's own workers. Proper fix: build the wheel with hidden symbol visibility.

## FlashInfer MoE autotune hang at vLLM boot (worked around)

`vllm/model_executor/warmup/kernel_warmup.py::flashinfer_autotune`: rank 0 reads the cache file and broadcasts
it, then all ranks tune collectively over the world CPU group. Runs 1-2 (cold cache) booted; run 3 (warm cache)
hung with TP0 done and TP1-3 at `Tuning trtllm::fused_moe::gemm1: 0/21`. Consistent with ranks disagreeing on
cache hits so only some enter the collective tuning. Workaround: fresh `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR` per
run (costs ~1-2 min of tuning per boot). Not yet reproduced deliberately -- candidate vLLM issue.

## Next steps (DSv4-Flash)

1. Get the full-FT roundtrip passing (Megatron vs vLLM logprob diff, then post-sync vLLM vs vLLM).
   Expected trouble spots, from the earlier GLM-5.3 / Muse Spark work: hash-routing `tid2eid` needing
   input ids under SkyRL's forward, mHC + `alltoall` dispatcher, `hc_head` export, vLLM weight reload
   of the fused `fused_wqa_wkv` / `fused_wkv_wgate` and `finalize_mhc_broadcast_weights`.
2. Also run the roundtrip once with the **full** 149 GB model (not added to CI), e.g. 8xB200 EP=8, to
   confirm coherent generations.
3. LoRA row in `test_megatron_lora_models.py`. Trainer target modules must be spelled out (MLA names:
   `linear_q_down_proj`, `linear_q_up_proj`, `linear_kv_proj`, `linear_proj`, `linear_fc1`, `linear_fc2`,
   compare the GLM-5.3 lists there); vLLM side uses the packed names above plus `experts`.
4. Stage 1: `examples/train/deepseek_v4_flash/run_gsm8k_*` (model the GLM-5.3-Flash scripts), LoRA + full FT,
   using `chat_template.jinja` (pass via `generator.chat_template` or bake into a tokenizer copy).
5. Stage 2: DAPO LoRA + full FT with R3, checkpoint save/resume (`test_trainer_full_checkpointing.py`).
6. Stage 3: packed THD/CP + TP>1 attention (Megatron-LM main has THD/CP), selective recompute,
   MXFP8 model spec under `skyrl/backends/skyrl_train/weight_sync/fp8/models`.
7. Keep a running list of upstream issues to file (vLLM / megatron-core / Megatron-Bridge). So far:
   - vLLM: `DeepseekV4Model.load_weights` writes `attn_sink` via direct copy_, bypassing weight_loader (dropped under
     layerwise reload); `DeepseekV4ForCausalLM.load_weights` runs the model-level finalize per call (meta tensors
     under layerwise reload); `FlashInferExperts` per-expert constants are plain attributes zeroed by level-2 sleep;
     FlashInfer autotune boot hang with a warm cache (unconfirmed); DSv4 serving is not run-to-run deterministic.
   - FlashMLA wheel (skyrl-wheels): exported symbols collide with vLLM's bundled FlashMLA; build with hidden visibility.
   - megatron-core: DSv4 attention TP=1 only, no packed THD, no core_attn recompute (Megatron-LM main has THD/CP).

## DeepSeek-V4.1-Flash-1017 (after V4) — findings so far

Checkpoint: `/mnt/local_storage/avigyabb/models/DeepSeek-V4.1-Flash-1017` on node 0 (downloaded from the public HF repo
`deepseek-ai/DeepSeek-V4.1-Flash-1017`; was `/shared/models/...` on the old machine). Tech report + chat template from
`deepseek-ai/DeepSeek-V4.1-Flash` in `/mnt/local_storage/avigyabb/models/DeepSeek-V4.1-Flash-docs`. (476 GB; reference code in `inference/model.py`,
`inference/engram.py` — use these as the parity reference). `/shared/dsv41_logs` is an unrelated earlier
vLLM serving benchmark (nightly 0.32.1rc1), not training work.

- vLLM 0.31 has `deepseek_v41` (VL wrapper `DeepseekV41ForCausalLM` is **not** `SupportsLoRA`; inner
  `DeepseekV41LLMForCausalLM` is). No torch.compile for it. transformers 5.16.1 has no `deepseek_v41`.
  Megatron-Bridge/megatron-core have **nothing** for 4.1.
- Architecture deltas vs V4 (all need megatron-core implementations under `patches/megatron/`):
  compress ratios 1 and 2 (ratio 1 = plain projection, no gate; >1 pooled in fp32); cross-layer
  compressed-KV sharing (`kv_source_layer_ids` 2,8,14,20; consumers read the latest source); indexer
  sharing (`index_source_layer_ids`; index keys come from the source compressor latent via `wk`);
  two-level candidate block pre-filter (`candidate_source_layer_id` 20, 2048 blocks of 8); Engram
  n-gram hash tables at layers 1 and 14 (each 384M x 256 FP8, ~98 GB) with `wkv`/`q_weight`/`k_weight`
  gating into the mHC stream; gate `bias_vl` for image tokens; no hash routing; DSpark MTP x3; vision
  tower. FP8 block 32x32 ue8m0, MXFP4 experts, `rms_norm_eps=1e-20`, 384 experts, hidden 5120, 40 layers.
- Size: ~551B trainable backbone params + ~197B engram params.
- Decisions **confirmed by the user 2026-10-25** ("not strict requirements -- do whatever gets 4.1 supported
  fastest"): freeze engram tables (FP8 buffers, row-sharded,
  dequantized on lookup, excluded from weight sync — vLLM keeps its own copy); text-only (drop vision +
  MTP); 6-layer slice 0,1,2,3,20,24 (SWA, engram, KV/index source, consumer, candidate source, candidate
  consumer). PP stage boundaries must not split a KV-sharing group.
- Rough compute (B300 288 GB): LoRA ~2 nodes; full FT ~4 nodes with optimizer CPU offload, ~8 without.

### V4.1 work (from 2026-10-25)

Extra nodes (same user, same `/mnt/local_storage/avigyabb` layout, W&B creds in `~/.config/skyrl/wandb.env`,
repo at `~/avigyabb/SkyRL-dsv4`; not in the DSv4 Ray cluster):

| Node | IP | Notes |
|---|---|---|
| b200-resv-ray-4 | 10.180.15.246 | GPU 0 held by another user's idle vLLM engine (do not touch); root disk 97% full; another user's tmux |
| b200-resv-ray-5 | 10.180.15.203 | V4.1 slice test node; `models/DeepSeek-V4.1-Flash-6layer` |
| b200-resv-ray-6 | 10.180.15.248 | idle |
| b200-resv-ray-7 | 10.180.15.202 | idle |

- Slice: `make_slice_v41.py` -> `models/DeepSeek-V4.1-Flash-6layer` (251 GB; on nodes 0, 4, 5). Text only, layers
  0,1,2,3,20,24 -> compress_ratios [0,0,2,2,1,1], kv_source [2,4], index_source [2,4,5], candidate_source 4,
  engram [1] (keeps its id: hash seeds use it). Engram table + scale kept FP8; everything else BF16.
- Megatron: `skyrl/backends/skyrl_train/patches/megatron/deepseek_v41/` on megatron-core's dsv4_hybrid HybridModel:
  - attention: no per-head q RMS norm; compressed RoPE (compress_rope_theta + YaRN) on ratio 1 too; compressor
    without APE/overlap; KV / index-key / top-k / candidate sharing through a per-forward state; indexer under
    no_grad (chunked torch scoring); sparse attention via mcore's csa kernels. Layer roles in `config.dsv41_spec`
    (mcore forces C=4 per symbol and only allows ratios 0/4/128 in `csa_compress_ratios`).
  - mHC wiring differs from V4 (same formulas): a sublayer collapses with the *previous* sublayer's pre-mix (first
    = one-hot stream 0) and the final readout uses the last pre-mix; there is no hc_head (removed from the stack).
    mcore's mHC rms eps is hard-coded 1e-6 (reference uses rms_norm_eps=1e-20; negligible).
  - Engram: frozen table shard per rank over a node-local group of 8 (~12 GB/GPU per table), loaded lazily from the
    HF safetensors, all-gather ids -> local dequant -> reduce-scatter; trainable wkv + q/k gate weights.
  - bridge: registers vLLM's `DeepseekV41Config` with transformers and `DeepseekV41ForCausalLM` -> HybridModel.
- vLLM (survey of 0.31): only the VL wrapper is registered; `patch_deepseek_v41.py` (a `vllm.general_plugins` entry)
  registers the text-only `DeepseekV41LLMForCausalLM` under the same arch name. BF16 needs
  `attention_backend=FLASHMLA_SPARSE_DSV41` (SM100 default mega attention is FP8-only and permutes wq_b/wo_a once
  at load). The V4 reload patch (attn_sink copy_, per-call finalize) now covers the V4.1 classes. Engram tables are
  never synced (vLLM keeps them in pinned host memory). LoRA: keep compressors out of the targets (ratio-1
  `fused_wkv_wgate` has one slice).
- Stage 0 row: `test_megatron_models.py::test_logprobs_matching_roundtrip[deepseek-v4.1-flash-6layer_tp1_ep8]`
  (`SKYRL_DEEPSEEK_V41_FLASH_6LAYER`); first run on node 5: `logs/v41rt1-*.log`.
- Stage 0 status (2026-10-25): runs end to end. Megatron vs vLLM logprob diff 0.235-0.287 on the 6-layer slice
  (sub-slices: window+engram [0,1] 0.074, ratio-2 [0,2,3] 0.124-0.129, ratio-1 [0,4,5] 0.116-0.117). Against
  DeepSeek's reference (`dev/ref_logprobs_v41.py` on the test's dumped batch, `SKYRL_TEST_DUMP_LOGPROBS`):
  | slice | mean logprob ref(bf16) / megatron / vllm | ref-megatron | ref-vllm |
  |---|---|---|---|
  | ratio-1 | -1.7355 / -1.7358 / -1.6938 | 0.068 | 0.113 |
  | ratio-2 | -2.1410 / -2.1396 / -2.1062 | 0.060 | 0.125 |
  | full 6-layer | -1.5509 / -1.5515 / -1.4588 | 0.132 | 0.239 |
  Megatron tracks the reference; vLLM's BF16 V4.1 path is the outlier (more confident). With the reference's own
  activation fake-quant on, it is ~0.29-0.41 from both. KV fake-quant in Megatron (`SKYRL_DSV41_FAKE_QUANT=vllm`
  for fp8_ds_mla, `=reference` for the reference's FP8/FP4) does not close the gap. The gap is largest on the
  first response tokens (prefill logits). The row's Megatron threshold is 0.3 so the post-sync check runs; the
  vLLM-side gap is an open item (upstream issue candidate).
- Weight-sync buffer: SkyRL floors the packed IPC/NCCL buffer at the checkpoint's largest tensor; the Engram table
  asked for 183 GiB (fixed: `checkpoint_shapes._NOT_SYNCED`).
- **Stage 0 passes on the slice (2026-10-26)**:
  - full-FT row (`v41sync5-full-*.log`, 6-layer slice): Megatron vs vLLM 0.248 pre-sync, 0.258 post-sync (< 0.3).
  - LoRA row (`v41lora2-r1-*.log`, ratio-1 sub-slice [0,4,5]): zero adapter 0.122, stale sampler 0.462 (> 0.3 with
    LoRA-B x32), updated adapter 0.118.
  - Sync bug fixed on the way: vLLM's V4.1 text model writes `attn_sink` with a direct `copy_` (as V4 does); the
    V4 reload patch stashed sinks only when their layer was still on meta, which depends on the sync's chunk order,
    so some syncs served zeroed sinks. It now stashes every sink during a SkyRL receive.
  - vLLM's V4.1 serving is not reproducible run to run: regenerating the same prompts with no sync gives 0.121 on
    common prefixes (80/80 diverge), so the pre/post-sync vLLM comparison is noise for this model. The V4.1 row
    gates the sync on post-sync vLLM vs Megatron instead (`resync_vs_megatron`). Negative control
    (`SKYRL_DEBUG_SKIP_SINK_STASH=1`, zeroed sinks): post-sync 0.360 -> fails the 0.3 gate (with the fix: 0.258).
- Stage 1 (GSM8K LoRA, full model) setup:
  - Full text-only BF16 copy `models/DeepSeek-V4.1-Flash-bf16` (1.2 TB, 263 shards; `make_slice_v41.py <out>
    $(seq -s, 0 39)`, ~1 h) on nodes 5, 6, 7.
  - V4.1 Ray cluster: head b200-resv-ray-6 (10.180.15.248:6390) + b200-resv-ray-7, 16 GPUs, ray-venv 2.57.0 on
    each (started from a clean env as for the DSv4 cluster). Recipe: `run_gsm8k_dsv41_flash_lora_2node.sh`
    (Megatron EP=16; one vLLM engine TP8 x PP2, BF16 FlashMLA). Logs: `logs/v41-gsm8k-lora2-*.log` on node 6.
  - vLLM PP bug: a KV cache group with no layers on a PP stage keeps its full UniformTypeKVCacheSpecs, so stage 1
    allocates tensors for stage-0 layers -> StopIteration in allocate_kv_cache. Patched
    (`patches/vllm/patch_pp_kv_cache_projection.py`).
- **Stage 1 runs end to end on the full model (2026-10-26, `v41-gsm8k-lora2-1792975380.log`, W&B `dsv41_flash_gsm8k`)**:
  step 1 = 592 s (generate 57 s, logprob fwd 163 s, policy_train 257 s, LoRA sync 106 s); rollout-vs-train logprob
  diff **0.025** on the full 40-layer model (the 0.12-0.25 on 3-6-layer slices is a truncated-model artifact);
  peak 117 GiB/GPU; grad_norm 0.030; first-step GSM8K reward 0.586 (avg response 237 tokens).
  All 10 steps clean (~6 min/step after step 1; logprob diff 0.025-0.029, grad_norm 0.03-0.04, entropy 0.33-0.37);
  rollout reward 0.59 0.62 0.63 0.58 0.57 0.57 0.69 0.48 0.60 0.55; final GSM8K test pass@1 0.603 (W&B `yxaqr3nq`).
  The official V4.1 template defaults to **thinking mode** ("Reasoning Effort: 75" system line, `<think>`); wrong
  answers average 291 tokens (not truncation at 1024), so the GSM8K `####` extractor is likely missing thinking-mode
  answers. `chat_template_v41_chat.jinja` defaults to chat mode (used by the DAPO recipe).
- Stage 2 DAPO LoRA (`run_dapo_dsv41_flash_lora_sync_2node.sh`, R3 on, 2K responses, `v41-dapo-lora2-1792982729.log`,
  W&B `dsv41_flash_dapo/ycblqga1`): R3 rejects `generator.chat_template` (it retokenizes), so chat mode comes from a
  symlinked model dir whose tokenizer_config carries the chat template (`DeepSeek-V4.1-Flash-bf16-chat`, on all 7
  nodes). Baseline AIME-2024 (chat mode, 2K): avg_score 0.067, pass@12 0.77, 1.6K tokens. **Step 1 end to end with
  R3 over vLLM PP2**: rollout-vs-train logprob diff **0.009**, peak 132 GiB, grad_norm 0.014; step 1665 s (generate
  452, convert 122, fwd 198, train 768, sync 115) -- one engine for 1536 x 2K samples is the bottleneck.
- Full V4.1 BF16 copy now on nodes 0, 1, 3, 8, 5, 6, 7. Full-FT recipe `run_gsm8k_dsv41_flash_fullft_6node.sh` (EP=48,
  3 engines TP8 x PP2, Engram tables on host via `SKYRL_DSV41_ENGRAM_TABLE_DEVICE=cpu`).
- Full-FT attempt 1 (`v41-gsm8k-fullft6-1792987443.log`, 6 nodes = DSv4 cluster nodes 0,1,3,8 + 5,7, head node 1):
  boot + initial sync 79 s, generate 18 s, logprob fwd 316 s, reward 0.83 (chat mode), then **hung in policy_train**
  for 30 min until the NCCL watchdog fired: node 3's ranks sat in an EP-group `_ALLGATHER_BASE` of 384 elements
  (the per-expert token counts) that the others never joined. Cause: `chunked_mlp` (mHC layer's
  `mlp_chunks_for_training`) used `Tensor.chunk(64)`, which yields fewer than 64 pieces unless 64 divides the
  length (100 -> 50). Unpacked rows are padded per rank, so ranks made different numbers of MoE calls. DSv4 got away
  with it because it pads to a multiple of 128. Fixed with `tensor_split` (exactly n pieces; unit test adds
  (100, 64)); also froze the V4.1 indexer (never gets a gradient; selection is under no_grad). Commit 3a5e0d1f.
  Relaunched as `v41-gsm8k-fullft6-1792991642.log`.
- Attempt 2 (`...-1792991642`): first rollout after the initial sync returned NaN logprobs (HTTP 400 "Out of
  range float values are not JSON compliant: nan"). Reverted the indexer freeze (6627eb6a). The same freeze passes
  the 6-layer slice round trip on node 6 (post-sync gap 0.241), so the link to the NaN is unconfirmed.
- Attempt 3 (`...-1792994224`, no freeze): rollout fine (gen 13 s, logprob fwd 502 s), then **hung again in
  policy_train**: all 48 GPUs at 100% util and ~236 W (NCCL spin, not compute), memory frozen. Real cause: the
  V4.1 hybrid layers are megatron-core's `HyperConnectionHybridLayer`, whose `TransformerLayer._forward_mlp`
  also chunks with `Tensor.chunk(n)`, and only in training (so the eval logprob forward never hit it). SkyRL's
  `chunked_mlp` fix does not reach that path. Fix (ea6bb0f8): pad unpacked rows to a multiple of
  `mlp_chunks_for_training` (lcm with the compress-ratio alignment), so every rank gets exactly n chunks.
  The packed (THD) path is not covered. Also added opt-in `SKYRL_DSV41_FAULTHANDLER=1` (on in the 6-node script):
  `kill -USR1 <worker pid>` dumps Python stacks to the worker log, since py-spy needs ptrace.
- Checked: megatron-core's mHC recompute does not call `compute_mappings` again (only `aggregate` is
  checkpointed, with the delayed `h_pre` as input), so the thread-local delayed pre-mix state is safe under
  recompute.
- Remote note: in this worktree `origin` is upstream NovaSky-AI/SkyRL and `fork` is avigyabb/SkyRL -- push to `fork`.
- **V4.1 checkpoint save/resume passes** (2026-10-26, node 6, `logs/ckpt-v41.log`): `test_trainer_full_checkpointing.py
  -k deepseek_v41` -> `megatron_deepseek_v41_flash_lora_h100` and `..._fullft_h100` both PASSED (34 min) on the 2-layer
  sub-slice `DeepSeek-V4.1-Flash-sub-win-engram` (sliding-window layer + Engram layer; EP=4, TP2 vLLM, BF16 FlashMLA).
  Run: `bash examples/train/deepseek_v4_flash/dev/run_roundtrip.sh tests/backends/skyrl_train/gpu/gpu_ci/test_trainer_full_checkpointing.py -k deepseek_v41`.
- Attempt 4 (`...-1792997597`, padding fix): **policy_train completes** (674 s; step 1 902 s: gen 14 s, logprob fwd
  213 s), then optim_step OOMed at 170.6 GiB in `_copy_model_grads_to_main_grads`: without the precision-aware
  optimizer megatron-core copies every grad to fp32 on GPU before the CPU optimizer runs (~45 GiB of unsharded expert
  grads at EP=48). The 6-node script now sets `use_precision_aware_optimizer=true` (`PRECISION_AWARE_OPTIMIZER`). The
  GLM recipes keep it off over checkpoint-save worries; the V4.1 full-FT checkpoint save/resume row **passes with it
  on** (node 6, `logs/ckpt-v41-precaware.log`, 23 min). Attempt 5: `...-1793000285`.
- Attempt 5 (`...-1793000285`, no indexer freeze) also died at the first rollout with the NaN HTTP 400, so the NaN is
  intermittent (2/5 boots) and **not** the indexer freeze. It is vLLM's known occasional non-finite sampled logprob
  (see `build_logprobs_content`): without R3 the client used the native `/inference/v1/generate`, which fails the
  whole request at JSON encoding. Fix 5e915cd0: requests asking for logprobs go to `/skyrl/v1/generate`, which floors
  them and logs `clamped N/M ... non-finite sampled logprobs`. Indexer stays unfrozen for now (could be re-frozen:
  it never gets a gradient).
- Attempts 6-7 (`...-1793002127`, `...-1793004074`, precision-aware on): step 1 rollout + fwd (64-72 s) +
  policy_train (455-472 s) all fine, then **CUDA illegal memory access in optim_step** (one rank, no ECC errors).
  Reproduced on the 6-layer slice on one node (node 6: the 6-node script with `trainer.placement.policy_num_nodes=1
  ...expert_model_parallel_size=8 generator.inference_engine.num_engines=1 ...pipeline_parallel_size=1
  trainer.train_batch_size=16 trainer.policy_mini_batch_size=16 MODEL_PATH=<slice> RAY_ADDRESS=local`, ~15 min).
  Off -> passes; `CUDA_LAUNCH_BLOCKING=1` -> TE `multi_tensor_apply` in `clip_grads.get_grad_norm_fp32`. Cause: with
  the precision-aware optimizer the grad norm/clip lists are the `decoupled_grad` views, which keep each param's
  dtype, and mHC trains some params in fp32 -> one mixed bf16/fp32 list -> the kernel dispatches on the first
  dtype and reads past bf16 tensors. Fix 5ea94d72 (`patches/megatron/patch_clip_grads_mixed_dtype.py`): split those
  multi-tensor calls by dtype. Slice step passes with it (logs "splitting a mixed-dtype grad list").
- DAPO LoRA checkpoint test on the full model (`v41-dapo-lora-ckpt-*.log`, 2 steps, ckpt at 2, `LORA_SYNC_PATH` on
  local storage: the default `/tmp` filled node 3's root disk -- my old `~/.cache/uv` was 32 GB there and on node 8,
  now cleaned).
- **Attempt 8 (`v41-gsm8k-fullft6-1793011946`, W&B `dsv41_flash_gsm8k`, all fixes): V4.1 full FT trains on 6 nodes.**
  Step 1 724 s (sync 75, gen 13, fwd 80, train 558 incl. optimizer, sync 71), steps 2-3 ~530-560 s (gen ~15 s, fwd ~15 s,
  train ~430-450 s, sync ~70 s). Peak 93-97 GiB/GPU. Rollout reward 0.77 -> 0.85; grad_norm 1.69, 0.27.
  Rollout-vs-train logprob diff: step 2 **0.018** (max 1.74). Step 1's 8.68 mean is a handful of tokens at vLLM's
  -9999 floor (sampled token with -inf logprob) on the **first rollout after the initial full sync** -- the same spot
  as the NaN logprobs; post-training syncs are clean. Open: what is off in vLLM right after the first full-weight
  sync (LoRA runs never full-sync). Harmless here (TIS off); matters for DAPO full FT with TIS.
- DAPO LoRA checkpoint test (`v41-dapo-lora-ckpt-1793007559`): 2 steps (1525 s, 1478 s; peak 132-135 GiB), step-2
  checkpoint saved in 43 s. Pieces were split (trainer/data state on node 1, policy shards on nodes 3 and 5): merged
  into `$B/ckpts/v41-dapo-lora-ckpt-merged` on all six nodes for the resume test.
- **V4.1 Stage 1 full FT done (2026-10-26, `v41-gsm8k-fullft6-1793011946`, W&B `dsv41_flash_gsm8k`)**: 10/10 steps on
  6 nodes (EP=48, 3 vLLM engines TP8 x PP2, chat mode), ~530 s/step, peak 93-97 GiB/GPU. Rollout reward 0.77 0.85 0.81
  0.87 0.93 0.92 0.96 0.96 0.97 0.95; grad_norm 1.69 -> 0.034; rollout-vs-train logprob diff 0.016-0.019 (steps 1 and 8:
  8.68 / 0.31, a few vLLM -9999-floor tokens). **GSM8K test pass@1 after training 0.958** (chat mode).
- **V4.1 DAPO LoRA checkpoint resume passes on the full model** (`v41-dapo-lora-resume-1793019483`, W&B
  `dsv41_flash_dapo`): `resume_mode=latest` from the merged `v41-dapo-lora-ckpt-merged/global_step_2` ("Successfully
  loaded complete checkpoint state", 22 s), then step 3 in 1497 s with rollout-vs-train logprob diff 0.0093 (0.009
  before the save), grad_norm 0.015, reward -0.23. Remaining Stage 2: DAPO full FT with R3
  (`run_dapo_dsv41_flash_fullft_sync_6node.sh`).
- **V4.1 Stage 2 DAPO full FT with R3 done (2026-10-26, `v41-dapo-fullft6-1793022818`, W&B `dsv41_flash_dapo`,
  `run_dapo_dsv41_flash_fullft_sync_6node.sh`)**: 10/10 steps on 6 nodes, 2K responses, ~11.5 min/step (gen ~100 s
  with 3 engines vs 452 s for the 1-engine LoRA recipe). Peak 123-140 GiB/GPU. Reward -0.21 0.00 -0.06 0.00 0.24
  0.18 0.16 0.36 0.42 0.41; grad_norm 0.12-0.16; rollout-vs-train logprob diff 0.009-0.036, except steps 4/6/9 at
  1.7-2.2 (vLLM -9999-floor tokens, under investigation on node 6). **AIME-2024 avg_score 0.044 -> 0.272, pass@12
  0.73 -> 0.90.** Step-10 full-FT checkpoint saved in 632 s (`$B/ckpts/v41-dapo-fullft6-1793022818`).
  Stages 0-2 for V4.1 are complete.
- **Floored-logprob investigation (2026-10-26), resolved as one rare vLLM flake, not a full-FT/sync problem.**
  - `dev/probe_vllm_logprobs_v41.py`, fresh vLLM load (TP8, BF16, FLASHMLA_SPARSE_DSV41), no weight sync: full model
    4 bad / 2.23M sampled tokens; 6-layer slice 6 / 3.1M. Every one is the **first response token** with a fully NaN
    logits row (sampled id 1023 and top-1 id 0 both `nan`); the rest of the sequence decodes normally. So some
    requests' prefill last-position logits come out NaN; per request (different samples of a prompt), not per prompt.
  - Filling the KV cache with 0xFF (NaN) bytes first (`--poison-kv`) does not change the rate (4 vs 6 / 3.1M): not
    reads of unwritten KV slots.
  - Training-side counter (`generate/clamped_logprob_*`, a475c1b0) on a 5-step DAPO full-FT run
    (`v41-dapo-fullft6-floor-1793036246`, W&B `kq96cd4h`): step 1 0 floored tokens, step 2 **1** (first token of a
    14-token sequence), which alone moved `minibatch_rollout_logprobs_abs_diff_mean` from 0.01 to 0.47 (max 9979):
    that metric averages per-micro-batch means, so one floored token in a short sequence dominates it. The earlier
    0.3-2.2 "spikes" were a handful of such tokens, i.e. the same ~1e-6/token flake; my earlier "~100x in full FT"
    estimate was wrong (it assumed a plain token mean).
  - Full 5-step counts: floored tokens per step 0, 1, 1, 11, 0 (13 / 7680 requests = 0.17%), all at the first response
    token; the no-sync fresh-load probe gives 0.26% (4 / 1536). Weight sync does not raise the rate overall; it varies
    by step (step 4: 11), so batch composition or the current weights may modulate it.
  - Impact: the first response token of ~1 sequence per step is sampled from NaN logits. Harmless to the loss with TIS
    off; with TIS on, mask such sequences. Root cause (which layer yields NaN in prefill) still open: replay a captured
    failing prefill batch and bisect Engram / indexer top-k / sparse attention + sinks / mHC.
- **vLLM NaN first-token root cause (2026-10-26/27, `dev/probe_vllm_logprobs_v41.py`, 6-layer slice, max_tokens=1).**
  BF16 V4.1 prefill (`FLASHMLA_SPARSE_DSV41`, `vllm/models/deepseek_v41/nvidia/flashmla.py::_forward_prefill`) gathers
  each chunk's compressed + SWA KV into a reused, uninitialized workspace (`current_workspace_manager().
  get_simultaneous`), one `chunk_M`-row slab per request sized for the longest request in the chunk; shorter requests
  leave slab rows unwritten. `flash_mla_sparse_fwd` loads those rows and weights them 0, so finite garbage is harmless
  but NaN/Inf garbage turns the output row NaN (0 * NaN). Evidence (24,576 requests each unless noted):
  - workspace untouched 207 bad (0.84%); **zero-filled 0**; NaN-filled 11,479 (46.7%) -> ~half of batched requests
    touch unwritten rows. Zero- vs 1000-filled greedy first tokens (8,192): same token 99.9%, |dlogprob| 0.0006 vs
    0.0004 zero-vs-zero -> unwritten rows are masked, not attended (no silent corruption; does not explain the
    separate 0.12-0.24 vLLM-vs-reference gap).
  - Batching required: max_num_seqs 1/2/4 -> 0, 16 -> 5/16k, 64 -> 140/16k. Nondeterministic (same batch twice:
    disjoint failing requests). Ruled out: weight sync, KV-cache poisoning, CUDA graphs, Engram host offload,
    compressor latent rows, aux streams, attn_out buffer (all A/B'd; no change).
  - Localization: module hooks -> `layers.N.attn` (clean inputs, NaN out); function hooks ->
    `DeepseekV4Attention._sparse_indexer_and_attn` writes NaN in place.
  - Fix: zero the gathered-KV workspace before the gather (no visible prefill slowdown in the probe). Upstream:
    zero slab tails or select instead of multiply in the kernel.
  - Confirmed on the full BF16 model (node 6, TP8, CUDA graphs on, 32,768 distinct-prompt requests): as shipped
    193 bad (0.59%), with `patch_deepseek_v41_prefill_workspace` (installed on all 8 ranks) **0**. The patch
    (b6e8542b) loads via `new_inference_worker_wrap.py`; on PR #2547 as 8849c5c6. The planned 5-step DAPO
    confirmation was dropped (floored tokens too rare per step to add evidence beyond the probe).
- **DAPO full FT + R3 with every fix (2026-10-27, 88f62b39, 6 nodes, W&B `dsv41_flash_dapo/ds1css8s`).** Same recipe
  as the earlier 10-step run, now with the prefill workspace patch (engine log: "Installed DeepSeek-V4.1 FlashMLA
  prefill workspace zero-fill patch"), the HDO checkpoint fixes and the fp32 wire dtype.
  - `minibatch_rollout_logprobs_abs_diff_mean` 0.0090 / 0.0104 / 0.0104 / 0.0106 / 0.0117 / 0.0112 / 0.0120 /
    0.0122 / 0.0131 for steps 1-9: no spikes. The earlier run had 1.70 / 2.23 / 2.20 on three of ten steps (the
    floored NaN tokens) and 0.009-0.036 otherwise.
  - `avg_final_rewards` -0.196 / -0.037 / -0.104 / -0.047 / 0.185 / 0.142 / 0.067 / 0.325 / 0.339 / 0.393 (step
    10 rollouts); rollout pass@12 0.79 -> 0.91-0.95. ~700 s/step.
  - **Step-10 checkpoint save OOMed the host on node 7 (10.180.15.202)**: kernel OOM killer took one
    MegatronPolicyWorker (anon-rss 122 GB, shmem-rss 314 GB) during `save_checkpoints`; the run died with
    ActorUnavailableError before logging step 10 / the final eval. The earlier run's identical step-10 save
    succeeded, so this recipe's save sits at the edge of the 3.9 TB hosts (precision-aware CPU optimizer state +
    Engram pinned tables + dist-ckpt CPU staging). Not fixed; options: async/streamed save, fewer workers per
    node holding Engram, or lower `optimizer_offload_fraction`.
  - Rerun for a complete 10-step curve with `trainer.ckpt_interval=100` (only the end-of-training safety-net
    save, after step-10 metrics and eval are logged): log `v41-dapo-fullft6-curve-1793076638.log`.
  - **Complete 10-step curve (W&B `dsv41_flash_dapo/1sjty1ud`, log `v41-dapo-fullft6-curve-1793076638.log`).**
    `avg_final_rewards` -0.217 / -0.037 / -0.097 / -0.015 / 0.186 / 0.156 / 0.132 / 0.380 / 0.369 / 0.443;
    `minibatch_rollout_logprobs_abs_diff_mean` 0.0091 / 0.0106 / 0.0105 / 0.0109 / 0.0117 / 0.0119 / 0.0132 /
    0.0130 / 0.0140 / 0.0133 (no spikes); rollout `reward/avg_pass_at_12` 0.84 -> 0.91-0.95. AIME-2024 eval
    (`~/exports/dumped_evals/global_step_{0,10}_evals/aggregated_results.jsonl` on the head): avg_score 0.028 ->
    0.233, pass@12 0.767 -> 0.867. ~700 s/step, 2h04m for 10 steps. The end-of-training checkpoint save (640 s)
    succeeded this time, so the node-7 host OOM above is marginal rather than deterministic.

### Overnight perf / correctness pass (2026-10-27)

- **Step-time profile** (DAPO full FT + R3, 6 nodes, ~700 s/step): policy_train 480 s (66%), generate 95 s,
  fwd_logprobs 63-81 s, sync_weights 70 s. py-spy on one policy rank during step-2 policy_train (420 s window):
  forward_backward 80% (GPU bound: `custom_backward` + CUDA waits), optim_step 18%, of which torch's fused CPU AdamW
  ~53 s = ~13 s per optimizer step x 4 mini-batches.
- **CPU optimizer ran on one thread**: Ray starts the policy actor with num_cpus=1, so OMP_NUM_THREADS=1 and
  torch.get_num_threads()==1 (checked in a Ray actor). V4.1 has 544B routed-expert params = 11.3B per rank at EP=48;
  fused CPU AdamW is 1.36 s/B on 1 thread vs 0.14 s/B on 28. a0bf46f3 sets torch threads to the rank's share of the
  node's CPUs when optimizer_cpu_offload is on. A/B run `v41-dapo-perf-threads-*` in progress.
- **optimizer_offload_fraction=0.5** (half the optimizer on GPU, possible since the #2542 fix): GPU OOM in the first
  colocated weight sync (trainer 92 GiB + vLLM 86 GiB on one GPU); the GPU-resident optimizer state is not offloaded
  for the sync. Not usable colocated without offloading it.
- **Host memory**: each policy rank holds ~110 GB anon + ~240 GB pinned (fp32 master/grad/moments of 11.3B params on
  the host); 8 ranks/node is ~2.8 TB of the 3.9 TB before a checkpoint save stages anything, which is why the step-10
  save OOMed once. b4676690 empties PyTorch's pinned-host cache after building the optimizer and before saving; at
  init and at a step-3 save it released < 1 GiB (the pinned memory is live optimizer state); reverted. During that
  save node 7 stayed at ~1.0 TB used / ~970 GB available (peak), so the save fits with margin when nothing else
  grows; the earlier OOM remains unexplained beyond the ~2.8 TB baseline.
- **CPU-thread fix measured** (`v41-dapo-perf-threads-*` vs `v41-dapo-perf-base-*`, same recipe): policy_train
  488 -> 434 s (step 2) and 466 -> 420 s (step 3); fwd_logprobs 62-63 -> 54-55 s; whole step 727 -> 662 s (-9%).
- **Rollout-vs-trainer logprob outliers come from vLLM decode with prefix caching.** `dev/analyze_logprob_outliers.py`
  on `trainer.dump_data_batch` dumps: mean |diff| 0.014, only 2-6e-5 of tokens > 1, vLLM the more confident side
  (20/23, 55/70), no position / long-context trend, but clustered by prompt group (the 12 samples of one prompt) and
  sometimes at the first response token. `dev/rescore_vllm_v41.py`: re-scoring those tokens in a vLLM prefill lands on
  the trainer's value (step 1, same weights: 21/23 tokens; chunked prefill reading the KV cache: 22/23).
  `dev/decode_vs_prefill_v41.py` (vLLM alone, 69 outlier-group prompts, decode-time vs prefill logprob of the same
  sampled tokens): prefix caching off, 2 samples/prompt: 9e-6 of tokens > 1 (max 1.07); off, 12 samples/prompt:
  2.2e-5 (max 1.9, unclustered); **on, 12 samples/prompt (as SkyRL's rollout engines run): 6.9e-5 (max 2.8),
  clustered by prompt group**. A prefix-cache hit seems to restore V4.1's compressed/indexer state for the shared
  prompt inexactly. Mitigation to test: `generator.inference_engine.enable_prefix_caching=false`.
- **Checkpoint disk**: a V4.1 full-FT checkpoint (bf16 model + fp32 optimizer, dist-ckpt, written node-locally) is
  ~1.2 TB on each of the 6 nodes. Five kept full-FT checkpoints plus a run with `ckpt_interval=3` (two saves) filled
  node 0's 12 TB `/mnt/local_storage`; the final safety-net save then failed (`inline_container.cc ... unexpected
  pos`, i.e. ENOSPC) and the next launch failed in Ray's working_dir unpack (`No space left on device`). Deleted the
  two throwaway ones (`v41-dapo-perf-threads-*`, the partial `v41-dapo-fullft6-final-*`); still kept:
  `v41-dapo-fullft6-{1793022818,floor-1793036246,curve-1793076638}`, `dapo-fullft4-1792972963`, LoRA ckpts.
  Perf/debug runs should pass `trainer.ckpt_interval=0` (no saves at all, not even the end-of-training one).
- **Mitigation for the prefix-cache logprob drift: `generator.cache_salt_per_trajectory=true`** (17421d17; now on in
  all four V4.1 recipes). `enable_prefix_caching=false` is not an option with R3: vLLM's AuxOutput connector (routed
  expert capture) raises "AuxOutput Connector requires prefix caching". A per-trajectory salt keeps every sample off
  the others' cached prompt KV. DAPO full FT + R3, same recipe (`v41-dapo-perf-trajsalt-*` vs `v41-dapo-perf-base-*`
  dumps, ~1.1-1.2M response tokens per step):

  | step | tokens > 0.5 | tokens > 1 | max | generate |
  |---|---|---|---|---|
  | 1 (same base weights) | 104 -> 13 | 23 -> 0 | 10.17 -> 0.99 | 99 -> 100 s |
  | 2 | 85 -> 12 | 14 -> 0 | 2.00 -> 1.00 | 97 -> 98 s |
  | 3 | 349 -> 19 | 70 -> 0 | 4.37 -> 1.00 | 98 -> 99 s |

  The baseline tail is one-sided (16 of the top 17 have rollout > trainer: vLLM overconfident, i.e. systematic); with
  per-trajectory salts the top gaps have mixed signs and decay smoothly to ~1 (bf16 noise).

  Mean |diff| is unchanged (0.014): the bulk is bf16 noise; only the prefix-cache tail goes away.
- **10-step DAPO full FT + R3 with all fixes** (ffb2ac80: NaN prefill patch, CPU-optimizer threads,
  per-trajectory cache salt; W&B `dsv41_flash_dapo/ftk13x1o`, log `v41-dapo-fullft6-allfixes-1793105695.log`, no
  checkpoints). `policy/rollout_train_logprobs_abs_diff_max` per step vs the previous curve run `1sjty1ud`:
  3.70->0.97, 4.28->0.87, 5.99->1.01, 4.68->1.03, 5.11->1.00, **12.15->8.67**, 8.94->1.60, 5.27->1.16, 1.90->0.95,
  3.34->0.87 (mean 0.014 throughout). Reward -0.238 -> 0.415; step 612-675 s after warmup (was ~720-730 s). AIME-2024
  avg_score 0.078 -> 0.167, pass@12 0.800 -> 0.867. Step 6 is the outlier in both runs (same data order), so one
  step-6 prompt still trips something the salt does not cover; `v41-dapo-dump6-*` re-runs 6 steps with
  `dump_data_batch` to find it.
- **Step-6 outlier is not prompt-specific.** `v41-dapo-dump6-1793114210` (same recipe and data order, dumps on):
  step 6 max |diff| 1.07, 1 token > 1 of 1.23M, mixed signs. Responses are resampled each run, so the 8.67 (and
  12.1) at step 6 were one-off tokens: with per-trajectory salts a > ~1.5 gap is ~1 in 12M tokens. No further action.
- **No batch-size effect in vLLM's residual decode-vs-prefill gap** (`dev/decode_vs_prefill_v41.py`, node 6, prefix
  caching off, 12 samples/prompt, ~674k tokens each): `max_num_seqs` 32 / 128 / 512 / default -> 3.4e-5 / 2.8e-5 /
  1.5e-5 / 2.2e-5 of tokens > 1 (max 2.4 / 1.6 / 1.8 / 1.9). A flat ~2e-5 floor; the n=2 run's 9e-6 was within noise.
- **4K responses**: at the 2048-token cap, 35% (step 1) and 18% (step 6) of DAPO responses are truncated and
  overlong-filtered out of the loss. `v41-dapo-fullft6-resp4k-*` (MAX_RESPONSE_LENGTH=4096, 20 steps, evals at 0/10/20,
  no checkpoints) tests memory, truncation and reward at 4K.
  - **4096: base AIME-2024 eval avg_score 0.472, pass@12 0.933** (avg 2160 generated tokens), vs 0.03-0.08 at the
    2048 cap: the 2K recipe mostly measures truncation, and its earlier "0.04 -> 0.27" gains are largely learning to
    finish under 2K. Training at 4096 **OOMs in step-1 policy_train** (MoE expert forward under recompute; 162 GiB
    allocated + 7 GiB of the slept vLLM engine on a 178 GiB B200; rows up to ~6.1K tokens). Every memory knob the recipe
    can use is already on (Engram on host, grad_reduce_in_fp32=false, expandable segments, micro batch 1, selective
    recompute [moe, mhc, layernorm]; no trainer PP). Retrying at 3072 (`v41-dapo-fullft6-resp3k-*`).
  - **3072 fits** (`v41-dapo-fullft6-resp3k-1793123157.log`): base AIME-2024 avg_score 0.378, pass@12 0.867 (avg 1910
    tokens); step-1 reward 0.282, step 2 0.357 (2K: ~-0.22, -0.04). Step 765 s after warmup (generate 130 s, policy_train
    496 s) vs ~650 s at 2K. Running 20 steps, evals at 10 and 20.
  - **3072 OOMs at step 4** (TE linear forward in policy_train; 154.6 GiB allocated + 7.3 GiB slept vLLM; rows up to
    ~4.2K tokens): steps 1-3 fit, rewards 0.282 / 0.357 / 0.388 / 0.411 (step-4 rollout). 3K is at the edge of the
    6-node layout; longer responses need memory work (e.g. a memory snapshot of one rank to see what beyond the ~45 GB
    of bf16 params + grads holds the rest). Next: 2560 (`v41-dapo-fullft6-resp2560-*`, 20 steps).
- **Upstream status of the V4.1 NaN (2026-10-27):** root cause is stale slots in the reused sparse-index workspace
  (`combine_topk_swa_indices` writes only each row's valid prefix; FlashMLA loads the whole last 64-slot block), which
  can point at non-finite rows of the KV gather buffer our patch zeroes. Fixed on vLLM main by vllm-project/vllm#60613
  (2026-10-23, seeds `-1`; not in 0.32.0); DeepSeek-V4 got the same fix in #55299 (in 0.31.0+), so V4 needs no patch.
  #2538 (non-finite logprob routing + counter) was dropped from #2546/#2547 accordingly.
  - **2560 fits for 20 steps** (W&B `dsv41_flash_dapo/fqoqd8ux`, `v41-dapo-fullft6-resp2560-1793127190.log`, no
    checkpoints): rewards 0.071 / 0.183 / 0.206 / 0.220 / 0.403 / 0.311 / 0.247 / 0.460 / 0.553 / 0.547 / 0.492 /
    0.530 / 0.569 / 0.658 / 0.439 / 0.543 / 0.529 / 0.550 / 0.736 / 0.565 (2K: -0.22 -> 0.44 over 10 steps). AIME-2024
    avg_score / pass@12: step 0 0.211 / 0.833, step 10 0.333 / 0.967, step 20 0.344 / 0.833 (30 problems, pass@12 is
    noisy). Steps ~600-700 s after warmup. Largest response cap that is stable on 6 nodes; 3072 OOMs by step 4.

### Activation memory for longer responses (2026-10-28)

- **Where the memory goes** (`v41-dapo-memsnap3k-1793167092`, 3072 responses, `trainer.policy.record_memory=true`,
  rank 10's last forward_backward; peak 135.7 GiB): ~73.5 GiB params / grads / other state from before the
  recorded window, 62.2 GiB activations. Largest activation entries (80 = every attention + MoE sub-layer):
  - 17.8 GiB `fused_mhc_kernels._torch_proj_rms_compute_h` (80): the fp32 upcast of the n-stream hidden states
    that `HyperConnectionModule.compute_mappings` feeds the mapping projection. megatron-core's `mhc` recompute
    (`_forward_with_checkpoint`) checkpoints only `aggregate` and calls `compute_mappings` directly, so autograd
    keeps 2x the bf16 hidden states per sub-layer.
  - 8.8 GiB mHC `h_post_bda` outputs (the residual stream; inherent), 7.2 GiB FlashMLA sparse out/lse,
    7.1 GiB q up-projection output (`qkv_up_proj_and_rope_apply`), 7.1 GiB inverse-RoPE'd attention output (the
    output projection's input), then < 2.3 GiB each.
- **Fixes** (6da34b0e):
  - `patches/megatron/patch_mhc_recompute_mappings.py`: runs `compute_mappings` under non-reentrant
    `torch.utils.checkpoint` inside the `mhc` recompute path (saves only the bf16 input, a view of the hidden states
    kept anyway). `dev/check_mhc_recompute.py` (node 6, one HyperConnectionModule, 4096 tokens): outputs and all
    grads bit-identical, memory held after forward 321 -> 32 MiB per sub-layer. Applies to every mHC model
    (DeepSeek-V4, V4.1, GLM-5.3); worth upstreaming to megatron-core.
  - V4.1 now supports `mla_up_proj` recompute: the up-projection's recompute runs under the same
    `_without_q_head_norm` context as the forward (the old assert existed because a plain recompute would
    re-insert V4's q head norm).
- **Measured** (`v41-dapo-memsnap3k-fix-1793169962`, same recipe + `recompute_modules=[moe,mhc,layernorm,mla_up_proj]`):
  the mHC entry and the q up-projection entry are gone; the remaining per-sub-layer activations are the residual
  stream, FlashMLA out/lse and the inverse-RoPE'd output. ~25 GiB less at the baseline's row length (~62 -> ~37 GiB
  activations). Step-1 metrics match the baseline run: grad_norm 0.1114 / 0.1126, minibatch logprob diff
  0.0125 / 0.0121, entropy 0.50 / 0.49; policy_train 645 -> 598 s (both with memory recording on).
- 4096 responses with both fixes, 20 steps, evals at 0/10/20, no checkpoints:
  `v41-dapo-fullft6-resp4k-memfix-1793172399.log` (W&B `dsv41_flash_dapo`, run `dsv41_flash_dapo_fullft_r3_resp4k_memfix`).
- **Trainer PP is not available for DeepSeek-V4/V4.1** at the pinned megatron-core: `v41-slice-pp2-r3-*` (node 6,
  6-layer slice, PP=2) fails at model build with "enable_mhc_connections does not support
  pipeline_model_parallel_size > 1 yet" (p2p buffers are sized from hidden_size, inter-stage activations are
  n-stream). So the PP=2 silent-misrouting case of the R3 sub-layer bug (#2562) cannot happen for DSv4/V4.1 today;
  only the crash at PP=1 did. Upstream mHC PP support: NVIDIA/Megatron-LM#7553 (open). Static memory per rank is
  36.6 GiB bf16 params + 36.6 GiB bf16 grads (~19.7B params at EP=48), so PP would have been the next lever.
- Upstream NVIDIA/Megatron-LM#7521 (merged to main 2026-10-28) checkpoints the mHC projection only on the unfused
  path; `patch_mhc_recompute_mappings.py` covers the fused path the recipes use (TODO in the patch).
- 4096 run status: steps 1-4 fit (no OOM; before the fixes 4096 OOMed in step 1). Step-0 AIME-2024 avg_score
  0.533, pass@12 0.933 (avg 2140 tokens; 13.3% of eval responses still hit the 4096 cap, all scored 0, vs 88% of
  finished ones correct). Rollout reward 0.50 / 0.57 / 0.60 / 0.55 for steps 1-4; ~760 s/step.
- **4096 responses, 20 steps, complete** (`v41-dapo-fullft6-resp4k-memfix-1793172399.log`, W&B
  `dsv41_flash_dapo/dsv41_flash_dapo_fullft_r3_resp4k_memfix`, no checkpoints): no OOM.
  - Rollout reward 0.499 0.568 0.599 0.547 0.674 0.525 0.442 0.609 0.657 0.678 0.623 0.653 0.678 0.745 0.576 0.651
    0.632 0.654 0.787 0.651 (2560 run: 0.071 -> 0.565; 2048: -0.22 -> 0.44).
  - Step time 913 (step 1) then 763 -> 668 s (responses get shorter).
  - AIME-2024 (30 problems x 12) avg_score / pass@12 / truncated at 4096 / accuracy of finished answers:
    step 0 0.533 / 0.933 / 13.3% / 0.88; step 10 0.572 / 0.967 / 8.1% / 0.85; step 20 0.506 / 0.933 / 5.3% / 0.79.
    The policy learns to finish within the cap (eval avg tokens 2140 -> 1857), but AIME accuracy is flat within
    30-problem noise; the training reward gain is mostly on DAPO-17k.
  - Recipe defaults now: `MAX_RESPONSE_LENGTH=4096`, `recompute_modules=[moe,mhc,layernorm,mla_up_proj]` in both
    V4.1 full-FT recipes (LoRA recipes unchanged: mla_up_proj not tested under LoRA).
- **5120 responses fit** (`v41-dapo-fullft6-resp5k-1793189007.log`, 3 steps, recipe defaults otherwise): no OOM;
  rollout reward 0.606 / 0.605 / 0.692, grad_norm 0.086-0.094, minibatch logprob diff 0.0146-0.0149, step 950 / 808
  / 797 s (generate 174 s, policy_train 475-563 s). Not run longer: with 4096 the eval truncation is already 5-13%.

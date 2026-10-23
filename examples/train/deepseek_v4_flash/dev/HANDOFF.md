# DeepSeek-V4-Flash (then V4.1-Flash) support in SkyRL — agent handoff

Status as of 2026-10-23. Branch `dsv4-flash` on `avigyabb/SkyRL`, cut from `NovaSky-AI/SkyRL` main at
`3a6e3b4e`. Follow `.agents/docs/new_model_support.md` (Eric Tang's staged plan: Stage 0 roundtrip
logprob tests -> Stage 1 GSM8K -> Stage 2 DAPO + R3 + checkpointing -> Stage 3 perf/MXFP8), and read the
routing docs in `CLAUDE.md` (megatron backend, weight sync, testing) before changing anything.

Plan agreed with the user: **do DeepSeek-V4-Flash first, then DeepSeek-V4.1-Flash-1017.**

## TL;DR

- DSv4-Flash is mostly integration work: every upstream piece exists at our pins.
- Stage 0: Megatron forward matches vLLM (slice 0.049, full model 0.040). LoRA row passes. Full-FT row fails
  only on the post-sync check: vLLM's level-2 sleep loses DSv4 state that a reload doesn't restore (reproduced
  without SkyRL). See "Post-sync mismatch".
- Next action: find the lost tensor (checksum diff on node 8), patch it under patches/vllm, rerun the row,
  then the full model on 2 nodes.

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

Other rows:
- LoRA `deepseek-v4-flash-4layer_h100_tp1_ep4_adapter` (node 1): **PASSED** -- zero adapter 0.050, stale sampler
  0.194 (> 0.1), updated adapter 0.048 (< 0.1). First attempt died in vLLM boot because TileLang picked the
  venv's pip CUDA-13 `nvcc` (no `/usr/local/cuda/bin` on PATH in a non-interactive shell): "CUDA compiler and
  CUDA toolkit headers are incompatible". run_roundtrip.sh now sets `CUDA_HOME`/`PATH`.
- Full model, BF16 (`deepseek-v4-flash-full-bf16_h100_tp1_ep8`, env-gated, not for the PR; node 3,
  `$DSV4_ROOT/models/DeepSeek-V4-Flash-bf16`, 530 GB, built by make_slice.py with all 43 layers): **forward
  matches, 0.040**. Sync OOMs on one node (Megatron ~87 GB + woken vLLM weights ~90 GB per GPU); rerun on 2 nodes.

## Post-sync mismatch (open)

- Export check (temporary `test_dsv4_export_check.py`, not committed): `MegatronWeightSource` vs the slice
  safetensors -- 3186/3186 tensors bit-exact once fp32 params stay fp32 on the wire (before: 38 fp32 tensors --
  mHC `hc_*`, `attn_sink`, compressor `ape`, router `gate.bias` -- were rounded to bf16; `tid2eid` int64 went out as
  bf16). `MegatronWeightSource._wire_dtype` now keeps fp32 and integer tensors in their own dtype.
- vLLM-only (node 8, `vllm_reload_check.py`; 8 GSM8K prompts, greedy 64 tokens, common-prefix tokens / mean diff):
  no reload 365 / 0.015 (vLLM DSv4 is not bit-reproducible run to run); `reload_weights` from disk 189 / 0.043;
  **level-2 sleep + wake + reload from disk: 8 / 0.033 -- every sequence diverges at token 2.** So the post-sync
  failure is vLLM's level-2 sleep path for DSv4: some state outside the checkpoint is not restored. Registered
  buffers are saved/restored by the worker; plain tensor attributes are not. Bisecting with a per-tensor checksum
  diff across sleep2+reload.

Padding note: with `remove_microbatch_padding=False`, `remove_left_padding` left-aligns every sequence,
so the causal SWA/CSA/HCA attention never sees padding — correct for logprobs, just less efficient.

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
7. Keep a running list of upstream issues to file (vLLM / megatron-core / Megatron-Bridge).

## DeepSeek-V4.1-Flash-1017 (after V4) — findings so far

Checkpoint: `/shared/models/DeepSeek-V4.1-Flash-1017` (476 GB; reference code in `inference/model.py`,
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
- Decisions proposed to the user (not yet confirmed): freeze engram tables (FP8 buffers, row-sharded,
  dequantized on lookup, excluded from weight sync — vLLM keeps its own copy); text-only (drop vision +
  MTP); 6-layer slice 0,1,2,3,20,24 (SWA, engram, KV/index source, consumer, candidate source, candidate
  consumer). PP stage boundaries must not split a KV-sharing group.
- Rough compute (B300 288 GB): LoRA ~2 nodes; full FT ~4 nodes with optimizer CPU offload, ~8 without.

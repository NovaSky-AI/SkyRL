# DeepSeek-V4-Flash (then V4.1-Flash) support in SkyRL — agent handoff

Status as of 2026-10-23. Branch `dsv4-flash` on `avigyabb/SkyRL`, cut from `NovaSky-AI/SkyRL` main at
`3a6e3b4e`. Follow `.agents/docs/new_model_support.md` (Eric Tang's staged plan: Stage 0 roundtrip
logprob tests -> Stage 1 GSM8K -> Stage 2 DAPO + R3 + checkpointing -> Stage 3 perf/MXFP8), and read the
routing docs in `CLAUDE.md` (megatron backend, weight sync, testing) before changing anything.

Plan agreed with the user: **do DeepSeek-V4-Flash first, then DeepSeek-V4.1-Flash-1017.**

## TL;DR

- DSv4-Flash is mostly integration work: every upstream piece exists at our pins.
- Stage 0 full-FT roundtrip has not passed yet. Three runs so far; two real blockers were fixed (below).
  The third run hung in vLLM boot (FlashInfer MoE autotune, see "Current blocker") and was stopped.
  The Megatron forward has **not been reached yet**, so the next run may surface more issues there.
- Next action: rerun the roundtrip (command below), fix whatever the Megatron init/forward/weight sync
  hits, then add the LoRA row.

## What is on this branch

| File | What |
|---|---|
| `examples/train/deepseek_v4_flash/dev/make_slice.py` | Builds the BF16 4-layer slice from the local checkpoint (FP8 128x128 ue8m0 + MXFP4 experts dequantized, MTP dropped, `compress_ratios`/`num_hash_layers` rewritten, chat template embedded). |
| `examples/train/deepseek_v4_flash/chat_template.jinja` | DSv4 ships no Jinja template (only `encoding/encoding_dsv4.py`). Text-only system/user/assistant, `thinking` kwarg toggles thinking mode. Verified byte-identical to `encode_messages` for chat + thinking modes on 4 conversations. |
| `examples/train/deepseek_v4_flash/dev/run_roundtrip.sh` | Runs the roundtrip tests as a Ray job on a worker node (copied from the MiMo/Muse Spark pattern). |
| `tests/.../megatron/test_megatron_models.py` | New row `deepseek-v4-flash-4layer_h100_tp1_ep4` + per-model config: `remove_microbatch_padding=False`, `inference_only_init`, vLLM `max_model_len=4096`/`gmu=0.5`, 1800 s health timeout. Model path from `SKYRL_DEEPSEEK_V4_FLASH_4LAYER`. |
| `skyrl/backends/skyrl_train/workers/megatron/megatron_worker.py` | Drops `core_attn` from selective recompute for `experimental_attention_variant == "dsv4_hybrid"` (recompute off if nothing remains). |

## Machine: b200-resv-ray-0 (current, from 2026-10-23)

Single node, 8 x B200 (183 GB), 224 CPUs, 3.9 TB RAM. Shared login box (other users run jobs here): check
`nvidia-smi` before a GPU run, and kill only your own processes. No Ray cluster / Slurm / k8s; tests start
their own Ray with `RAY_ADDRESS=local`. Sibling nodes (`b200-resv-ray-1`, `-3`) are not ssh-reachable and
no filesystem is shared across nodes, so everything is single-node for now.

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

Padding note: with `remove_microbatch_padding=False`, `remove_left_padding` left-aligns every sequence,
so the causal SWA/CSA/HCA attention never sees padding — correct for logprobs, just less efficient.

## Current blocker: FlashInfer MoE autotune hang at vLLM boot

In run 3, TP0 loaded 42 configs from
`/home/ray/.cache/vllm/flashinfer_autotune_cache/0.7.0.post1/100a/...` and finished, while TP1-3 sat at
`[AutoTuner]: Tuning trtllm::fused_moe::gemm1: 0/21` for 30 min (EngineCore only printing
`shm_broadcast ... No available shared memory broadcast block`). Runs 1 and 2 booted vLLM fine on the
same nodes, so this looks like a cache race between ranks (one rank's cache file written mid-run) rather
than a DSv4 problem. Things to try, cheapest first: just rerun; delete the node-local
`~/.cache/vllm/flashinfer_autotune_cache` on the worker; point the cache at a fresh dir per job; disable
the boot autotune. If it recurs, py-spy the stuck `RayWorkerProc`s.

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

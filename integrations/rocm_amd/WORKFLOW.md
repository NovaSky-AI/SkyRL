# SkyRL Megatron GRPO on AMD ROCm

How SkyRL runs Megatron training and vLLM inference on AMD Instinct through `trainer.strategy=megatron`. Pins and reviewer commands live in [REPRODUCE.md](REPRODUCE.md).

## One GRPO step

```
┌─────────────────────────────────────────────────────────────────┐
│  Ray (skyrl_entrypoint on AMD GPU)                              │
│                                                                 │
│  1. ROLLOUT          vLLM HTTP (1 mp engine, TP across GPUs)    │
│     prompts → completions                                       │
│                                                                 │
│  2. FORWARD          Megatron policy + ref workers              │
│     logprobs, values, rewards → GRPO advantages                 │
│                                                                 │
│  3. WEIGHT SYNC      NCCL broadcast: policy → vLLM workers      │
│                                                                 │
│  4. POLICY UPDATE    Megatron GRPO loss + optimizer step        │
└─────────────────────────────────────────────────────────────────┘
```

With `colocate_all=true`, policy, ref, and vLLM share the same devices via fractional Ray GPU scheduling. The GSM8K recipe sets `enable_sleep_mode=false` so vLLM keeps its allocator instead of sleep/wake.

Typical smoke (Qwen2.5-0.5B, GSM8K subset, 2 GPUs): generate via local vLLM HTTP servers, sync weights with `weight_sync_backend=nccl`, train with `trainer.strategy=megatron`.

ROCm settings: `VLLM_USE_V1=0`, `VLLM_TARGET_DEVICE=rocm`, and aligned `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, and `CUDA_VISIBLE_DEVICES`.

## Megatron components

| Component | Role |
|-----------|------|
| megatron-core 0.19 | Distributed training (TP/PP, TE attention) |
| Megatron-Bridge | HF ↔ Megatron (`AutoBridge.from_hf`) |
| SkyRL `megatron_worker` | RL forward, GRPO loss, optimizer, weight export |
| Transformer Engine | Fused ops (`NVTE_USE_ROCM=1` in the base image) |

Install Megatron-Bridge with `--no-deps` plus `nvidia-modelopt`. Drop a stale `Megatron-LM` tree from `PYTHONPATH` if the image ships an older copy. See `install_megatron_bridge.sh`.

## Runtime

Supported GPUs: **MI300X**, **MI325X** (`gfx942`) and **MI355X** (`gfx950`). `rocm/primus:v26.4` covers both ISAs. Rebuild vLLM against the container torch if you change the image.

## End-to-end recipe

1. `run_in_container.sh` — mount the repo and expose `/dev/kfd` and `/dev/dri`.
2. `install_full_stack.sh` — megatron-core, Megatron-Bridge, SkyRL `[rocm-megatron]`, ROCm vLLM, Ray 2.57.
3. `run_gsm8k_megatron_rocm.sh` — Ray/vLLM preflight, tiny GSM8K parquet, then `python -m skyrl.train.entrypoints.main_base`.

Inside `main_base`: Ray init with explicit `num_gpus` → `ServerGroup` vLLM HTTP actors (smoke: one engine, TP = GPU count) → Megatron policy + ref on the same placement group → GRPO loop (generate → reward → forward → advantage → weight sync → train step). See the README for supported parallelism and Megatron pin compatibility.

## vLLM build and SkyRL integration

Never `pip install vllm` from PyPI. Build the pinned revision against the container torch and cache the wheel under `.vllm_rocm_cache/wheels/`. The cache is keyed by the vLLM revision, PyTorch and ROCm versions, Python ABI, and target GPU architectures so native wheels are never reused across incompatible environments. Run `verify_vllm_skyrl_compat.py` before GRPO.

| Area | Change |
|------|--------|
| `vllm_server_actor.py` | HIP/ROCR/CUDA mask alignment for mp backend |
| `utils.py` | HIP-gated `VLLM_USE_V1=0`, `get_ray_init_num_gpus`, and `VLLM_TARGET_DEVICE=rocm` |
| `inference_servers/` | Per-engine HIP pinning via Ray `runtime_env` (`server_group.py`, `engine_utils.py`) |
| GRPO config | `run_engines_locally=true`, `enable_sleep_mode=false` |
| `layerwise_reload.py` | `skyrl_start_weight_update` / `skyrl_finish_weight_update` (vLLM 0.20 Worker already defines `finish_weight_update`) |
| Ray imports | `PlacementGroupSchedulingStrategy` and `get_address_and_port` fallbacks for Ray 2.57 |

vLLM itself is the upstream ROCm build; SkyRL only adds integration and configuration.

## Weight sync (policy → vLLM)

The GSM8K recipe uses SkyRL's HTTP inference servers (`VLLMServerActor` via `ServerGroup`):

1. Policy Megatron worker exports weight chunks
2. NCCL broadcast (`weight_sync_backend=nccl`) to inference ranks — on ROCm this is RCCL
3. vLLM native weight-sync endpoints on each server (`/init_weight_transfer_engine`, `/update_weights`)
4. Weight chunks applied through the vLLM RLHF router when `VLLM_SERVER_DEV_MODE=1`

CUDA IPC weight sync is not the default on AMD.

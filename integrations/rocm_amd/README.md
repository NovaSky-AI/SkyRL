# SkyRL on AMD ROCm GPUs

Megatron training (`trainer.strategy=megatron`) and vLLM inference on AMD Instinct:

**MI300X**, **MI325X** (CDNA3, `gfx942`) · **MI355X** (CDNA4, `gfx950`)

| Doc | Purpose |
|-----|---------|
| [REPRODUCE.md](REPRODUCE.md) | Reviewer steps, pins, and pass criteria |
| [WORKFLOW.md](WORKFLOW.md) | GRPO loop, stack, and SkyRL/vLLM integration |

This README covers **how the colocated path works**, **which parallelisms are validated**, and **Megatron-Bridge / core compatibility** (arbitrary SHAs are not supported).

**Status:** End-to-end Megatron GRPO + vLLM rollout validated on MI355X (see `reports/grpo_amd_20260901T234529Z.log` — 4 training steps, exit 0). Upstream branch: `feat/rocm-amd-upstream`.

## Supported GPUs

| GPU | Architecture | LLVM target |
|-----|--------------|-------------|
| MI300X | CDNA3 | `gfx942` |
| MI325X | CDNA3 | `gfx942` |
| MI355X | CDNA4 | `gfx950` |

```bash
python3 integrations/rocm_amd/gpu_support.py
```

`build_vllm_rocm.sh` defaults to `PYTORCH_ROCM_ARCH=gfx942;gfx950`. The tested image is `rocm/primus:v26.4`. Another base image is fine if its PyTorch and Transformer Engine builds include the target ISA.

Override: `ROCM_IMAGE=your-image bash integrations/rocm_amd/run_in_container.sh`

## How it works

SkyRL on ROCm uses the same Megatron GRPO path as `main`, with HIP-only device and executor defaults.

With `colocate_all=true`, Ray places Megatron policy/ref workers and local vLLM HTTP servers on the same GPUs:

1. **Rollout.** One colocated vLLM engine generates completions over HTTP (`run_engines_locally=true`, `distributed_executor_backend=mp`).
2. **Train.** Megatron policy and reference workers compute logprobs, rewards, and the GRPO update (`trainer.strategy=megatron`).
3. **Weight sync.** Policy shards broadcast to vLLM with `weight_sync_backend=nccl`. On ROCm that collective is RCCL, not CUDA IPC.

The GSM8K recipe keeps vLLM awake (`enable_sleep_mode=false`) so sleep/wake does not fight the colocated allocator. Ray must not blank GPU masks on `num_gpus=0` inference actors (`RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0`). ROCr sees physical IDs; HIP/CUDA then see process-local `0..N-1`.

## Parallelism

Validated end-to-end on 2 Instinct GPUs (MI355X; same recipe is intended for MI300X/MI325X):

| Role | Layout | Notes |
|------|--------|--------|
| Megatron policy/ref | DP = number of GPUs, `TP=1`, `PP=1` | Recipe knobs: `MEGATRON_TP`, `MEGATRON_PP` (default `1`) |
| vLLM rollout | 1 engine, `TP = NUM_GPUS` | Recipe knobs: `NUM_ENGINES=1`, `VLLM_TP=${NUM_GPUS}` |

What is **wired but not AMD-validated**:

- Megatron `TP>1` or `PP>1` (SkyRL config accepts them; the smoke keeps both at 1).
- Megatron context / expert parallel (`CP`, `EP`). Single-GPU Bridge validation uses `CP=1`, `EP=1` only.
- Multi-node colocated vLLM. On ROCm, SkyRL rewrites `distributed_executor_backend=ray` to `mp` only when `TP*PP` fits on one node. Cross-node engines keep Ray; that path has not been run end-to-end here.
- FSDP / JAX trainers on AMD. This integration is Megatron + vLLM only.

Override the smoke layout with env vars, for example `NUM_GPUS=2 MEGATRON_TP=1 VLLM_TP=2`. If you change it, keep every vLLM engine's `TP*PP` on one node when using the mp backend.

## Compatibility (Megatron-Bridge / megatron-core)

The installer pins both and installs Bridge with `--no-deps` so CUDA-only extras (FlashInfer, `nvidia-resiliency-ext`, and similar) are not pulled onto ROCm:

| Component | Pin | Why it is pinned |
|-----------|-----|------------------|
| megatron-core | `71e418ea7d7b3a6c9a53238c543c3e0b43e11026` (0.19 line) | SkyRL Megatron workers and TE ROCm kernels |
| Megatron-Bridge | `91a15142a4b4442a8d46ab539d1b923bd08570d0` | `AutoBridge.from_hf_pretrained` / provider APIs used by `megatron_worker` |
| vLLM | `v0.20.2` source build vs container torch | SkyRL HTTP server + weight-sync RPCs |
| transformers | `>=5.6.1,<=5.8.0` | Matches this Bridge pin; later vLLM deps can otherwise upgrade it |

Also required, independent of those git SHAs:

- A ROCm PyTorch + Transformer Engine image that includes the GPU ISA (`NVTE_USE_ROCM=1`). The tested image is `rocm/primus:v26.4`.
- No stale `Megatron-LM` tree on `PYTHONPATH` (some ROCm images ship one).
- The vLLM wheel built against **this** image’s torch/HIP/ISA (cache key includes those).

`MCORE_REV` and `BRIDGE_REV` can be overridden for experiments. Treat that as untested: Bridge and core must stay API-compatible with each other and with SkyRL’s Megatron worker, and the pair must still run on the image’s ROCm TE.

vLLM is the same story: use the pinned ROCm source build, not PyPI `vllm` and not an untested vLLM SHA.

## Quick start

```bash
bash integrations/rocm_amd/run_in_container.sh
# inside the container:
bash integrations/rocm_amd/install_full_stack.sh
bash examples/train/gsm8k/run_gsm8k_megatron_rocm.sh
```

From the host:

```bash
ROCM_GPUS=2 bash integrations/rocm_amd/run_in_container.sh \
  bash examples/train/gsm8k/run_gsm8k_megatron_rocm.sh
```

Optional single-GPU Megatron check: `bash integrations/rocm_amd/validate_megatron_rocm.sh`

`build_image.sh` / `Dockerfile.rocm` are optional; `run_in_container.sh` is enough.

## Scripts

| Script | Purpose |
|--------|---------|
| `run_in_container.sh` | Docker/Podman launcher with ROCm devices |
| `run_on_cluster.sh` | Slurm wrapper (`verify` or `grpo` mode) |
| `gpu_cleanup.sh` | Free Ray/PyTorch VRAM between verify steps |
| `install_full_stack.sh` | Megatron-Bridge + vLLM ROCm wheel + verification |
| `install_megatron_bridge.sh` | megatron-core + Megatron-Bridge + SkyRL extra |
| `build_vllm_rocm.sh` | Build vLLM from source against container torch |
| `verify_vllm_skyrl_compat.py` | vLLM 0.20 APIs used by SkyRL |
| `validate_megatron_rocm.sh` | Single-GPU Megatron-Bridge check |
| `ray_preflight.sh` | Ray GPU visibility + Instinct allowlist |
| `gpu_support.py` | Detect MI300X/MI325X/MI355X and gfx ISA |
| `vllm_llm_smoke.sh` | Single-GPU vLLM `LLM` init via a Ray actor (same path as GRPO) |
| `run_smoke_test.sh` | Integration smoke (`python …/smoke_test.py --all`) |

Do **not** `pip install vllm` from PyPI (it pulls CUDA PyTorch). Build with `build_vllm_rocm.sh` and cache under `.vllm_rocm_cache/`.

The installer scripts bootstrap the active container environment intentionally: Megatron and
the source-built vLLM extensions must use the ROCm PyTorch shipped by the base image.
Running these bootstrap steps in an isolated `uv` environment would install a second PyTorch
and can produce ABI-incompatible native extensions. Use the repository's isolated `uv`
commands for development and tests outside this image-building workflow.

## Troubleshooting

| Symptom | Likely fix |
|---------|------------|
| `No module named vllm.entrypoints.openai.engine` | Rebuild vLLM; run `verify_vllm_skyrl_compat.py` |
| `register fat binary failed` | Base image missing the GPU ISA |
| Ray reports 0 GPUs | Set `NUM_GPUS`; run `ray_preflight.sh`; check `HIP_VISIBLE_DEVICES` |
| HIP agent crash / empty device | Keep HIP/ROCR/CUDA-compatible device masks aligned |
| vLLM engine core init failed | `VLLM_USE_V1=0`, `async_engine=false` |
| Worker `finish_weight_update` clash | Use SkyRL `skyrl_*` weight-sync RPC names (vLLM 0.20+) |

Weight sync uses `weight_sync_backend=nccl`; on ROCm, PyTorch routes collectives through RCCL.

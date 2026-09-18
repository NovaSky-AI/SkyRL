# SkyRL on AMD ROCm GPUs

Megatron training (`trainer.strategy=megatron`) and vLLM inference on AMD Instinct:

**MI300X**, **MI325X** (CDNA3, `gfx942`) · **MI355X** (CDNA4, `gfx950`)

| Doc | Purpose |
|-----|---------|
| [REPRODUCE.md](REPRODUCE.md) | Reviewer steps, pins, and pass criteria |
| [WORKFLOW.md](WORKFLOW.md) | GRPO loop, stack, and SkyRL/vLLM integration |

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

## Requirements

- Supported AMD Instinct GPU
- Docker or Podman with `/dev/kfd` and `/dev/dri`
- ROCm PyTorch + Transformer Engine matching the GPU ISA

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

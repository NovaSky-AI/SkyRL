# Reproduce Megatron GRPO

Same recipe on MI300X, MI325X, and MI355X. **2 GPUs**. Do **not** `pip install vllm` from PyPI.

## Pins

| Dependency | Version / pin | Link |
|------------|---------------|------|
| Base image | `rocm/primus:v26.4` | [Docker Hub `rocm/primus`](https://hub.docker.com/r/rocm/primus) |
| megatron-core | `71e418ea7d7b3a6c9a53238c543c3e0b43e11026` | [Megatron-LM](https://github.com/NVIDIA/Megatron-LM/commit/71e418ea7d7b3a6c9a53238c543c3e0b43e11026) |
| Megatron-Bridge | `91a15142a4b4442a8d46ab539d1b923bd08570d0` | [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge/commit/91a15142a4b4442a8d46ab539d1b923bd08570d0) |
| vLLM | [`v0.20.2`](https://github.com/vllm-project/vllm/releases/tag/v0.20.2) (`PYTORCH_ROCM_ARCH=gfx942;gfx950`) | [`build_vllm_rocm.sh`](build_vllm_rocm.sh) + [`patches/vllm_gptq_compat_rocm713.patch`](patches/vllm_gptq_compat_rocm713.patch) for ROCm ≥7.13 |
| Ray | 2.57.0 | [`install_full_stack.sh`](install_full_stack.sh) |
| Model | `Qwen/Qwen2.5-0.5B-Instruct` | [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) |
| Dataset | GSM8K (tiny subset) | [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) |

Need Docker or Podman, `/dev/kfd`, `/dev/dri`, and two supported GPUs.

## Commands

```bash
ROCM_IMAGE=rocm/primus:v26.4 ROCM_GPUS=2 \
  bash integrations/rocm_amd/run_in_container.sh

# inside the container
bash integrations/rocm_amd/install_full_stack.sh
python3 integrations/rocm_amd/gpu_support.py
NUM_GPUS=2 bash examples/train/gsm8k/run_gsm8k_megatron_rocm.sh
```

One-shot from the host:

```bash
ROCM_IMAGE=rocm/primus:v26.4 ROCM_GPUS=2 \
  bash integrations/rocm_amd/run_in_container.sh bash -lc '
    bash integrations/rocm_amd/install_full_stack.sh &&
    NUM_GPUS=2 bash examples/train/gsm8k/run_gsm8k_megatron_rocm.sh
  '
```

The GSM8K recipe uses `gpu_memory_utilization=0.6` and `max_model_len=512` so colocated vLLM has a positive KV cache on CDNA3 and CDNA4.

First `install_full_stack.sh` can take a long time if it must build vLLM; later runs reuse `integrations/rocm_amd/.vllm_rocm_cache/`.

## Pass criteria

- `gpu_support.py` reports `gfx942` or `gfx950`
- GRPO runs a short GSM8K loop (vLLM generate → weight sync → Megatron step)
- Process exits 0

# Nemotron Training with Megatron

## Nemotron-Mini-4B-Instruct

[Nemotron-Mini-4B-Instruct](https://huggingface.co/nvidia/Nemotron-Mini-4B-Instruct) is a standard transformer model that works out of the box with the Megatron backend — no dependency changes are needed. Use the existing megatron example scripts with `trainer.policy.model.path=nvidia/Nemotron-Mini-4B-Instruct`. See [`examples/train/megatron/`](../megatron/) for reference scripts.

## Nemotron-3-Nano-4B-BF16

This example trains [NVIDIA-Nemotron-3-Nano-4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16) on GSM8K using GRPO with the Megatron backend.

Nemotron-3-Nano is a hybrid Mamba+Attention+MoE architecture (52 layers, 128 experts, SSM state). It requires specific dependency versions that differ from the default SkyRL configuration.

### Required dependency changes

A patch is provided that makes the necessary `pyproject.toml` changes. It was tested against SkyRL commit [`c69f2488`](https://github.com/sky-org/SkyRL/commit/c69f2488).

```bash
git apply examples/train/nemotron/nemotron_support.patch
uv lock
```

The patch makes two changes:

1. **Enable Mamba dependencies** — Nemotron-3-Nano uses Mamba (SSM) layers. The default config disables `mamba-ssm` and `causal-conv1d` via `override-dependencies`; the patch enables them.

2. **Switch Megatron-Bridge to `nano-v3` branch** — This branch includes `Nemotron3NanoProvider` which handles the HF-to-Megatron model conversion for this hybrid architecture.

### Running

1. Prepare the GSM8K dataset:

```bash
uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
```

2. Run training (requires 8 GPUs):

```bash
bash examples/train/nemotron/run_nemotron_3_nano_4b_gsm8k.sh
```

### Expected results

On GSM8K with 8xH100/A100 GPUs, the model reaches ~96% pass@1 within 20 epochs. Training step time is approximately 60-80 seconds on 8xH100.

### Notes

- Numerical differences between HF and Megatron forward passes are higher for this hybrid architecture (~0.9 max, ~0.17 avg) compared to pure transformer models (~0.3 max, ~0.05 avg), likely due to implementation differences in HF's custom modeling code. The vLLM-vs-Megatron logprob difference is ~0.01 on average, similar to other models like Qwen 2.5, confirming that Megatron and vLLM agree closely. This does not affect training quality.
- The `nano-v3` branch of Megatron-Bridge also supports other Nemotron-H variants (Nemotron-H, Nemotron-Nano-V2-VL).

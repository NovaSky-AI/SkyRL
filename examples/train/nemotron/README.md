# Nemotron-3-Nano-4B Training with Megatron

This example trains [NVIDIA-Nemotron-3-Nano-4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16) on GSM8K using GRPO with the Megatron backend.

Nemotron-3-Nano is a hybrid Mamba+Attention+MoE architecture (52 layers, 128 experts, SSM state). It requires specific dependency versions that differ from the default SkyRL configuration.

## Required dependency changes

A patch is provided that makes the necessary `pyproject.toml` changes. It was tested against SkyRL commit [`48cf6035`](https://github.com/sky-org/SkyRL/commit/48cf6035f36d076426e100ea3c6164662028ab0d) (HEAD of `main` at time of writing).

```bash
git apply examples/train/nemotron/nemotron_support.patch
uv lock
```

The patch makes three changes:

1. **Enable Mamba dependencies** — Nemotron-3-Nano uses Mamba (SSM) layers. The default config disables `mamba-ssm` and `causal-conv1d` via `override-dependencies`; the patch enables them.

2. **Switch Megatron-Bridge to `nano-v3` branch** — This branch includes `Nemotron3NanoProvider` which handles the HF-to-Megatron model conversion for this hybrid architecture.

3. **Update megatron-core to 0.16.0** — The `nano-v3` branch requires `megatron-core>=0.15.0,<0.17.0`.

## Running

1. Prepare the GSM8K dataset:

```bash
uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
```

2. Run training (requires 8 GPUs):

```bash
bash examples/train/nemotron/run_nemotron_3_nano_4b_gsm8k.sh
```

## Configuration

| Parameter | Value | Notes |
|---|---|---|
| Training parallelism | TP=4, PP=1, CP=1 | Uses 4 GPUs for Megatron training |
| Inference engines | 8 engines, TP=1 | Colocated with training GPUs |
| Batch size | 128 train, 1024 eval | |
| Samples per prompt | 5 | For GRPO advantage estimation |
| Learning rate | 1e-6 | |
| Algorithm | GRPO with KL loss | |

## Expected results

On GSM8K with 8xH100/A100 GPUs, the model reaches ~96% pass@1 within 20 epochs. Training step time is approximately 60-80 seconds.

## Notes

- Numerical differences between HF and Megatron forward passes are higher for this hybrid architecture (~0.9 max, ~0.17 avg) compared to pure transformer models (~0.3 max, ~0.05 avg), due to recurrent state accumulation in Mamba layers. This does not seem to affect training quality - training has been verified against GSM8K task, with stable rewards for 1k+ steps.
- The `nano-v3` branch of Megatron-Bridge also supports other Nemotron-H variants (Nemotron-H, Nemotron-Nano-V2-VL).

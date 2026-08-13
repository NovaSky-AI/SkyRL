# Nemotron 3 Training with Megatron

## Nemotron 3.5 Lightning 30B-A3B-BF16

This recipe trains
[NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)
on GSM8K with four Megatron workers and a four-GPU vLLM engine:

```bash
bash examples/train/nemotron_3/run_nemotron_3p5_lightning_30b_gsm8k.sh
```

The one-node recipe uses TP=1 and EP=4 for LoRA training, TP=4 for inference, and disables the
checkpoint's MTP layer on the trainer side.

## Nemotron-3-Nano-4B-BF16

This example trains [NVIDIA-Nemotron-3-Nano-4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16) on GSM8K using GRPO with the Megatron backend.

Nemotron-3-Nano is a hybrid Mamba+Attention+MoE architecture (52 layers, 128 experts, SSM state).

### Running

1. Prepare the GSM8K dataset:

```bash
uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
```

2. Run training (requires 8 GPUs):

```bash
bash examples/train/nemotron_3/run_nemotron_3_nano_4b_gsm8k.sh
```

### Expected results

On GSM8K with 8xH100/A100 GPUs, the model reaches ~96% pass@1 within 20 epochs. Training step time is approximately 60-80 seconds on 8xH100.

### Notes

- Numerical differences between HF and Megatron forward passes are higher for this hybrid architecture (~0.9 max, ~0.17 avg) compared to pure transformer models (~0.3 max, ~0.05 avg), likely due to implementation differences. The vLLM-vs-Megatron logprob difference is ~0.01 on average, similar to other models like Qwen 2.5.

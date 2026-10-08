# PiSSA with Megatron

PiSSA initialization is an offline step. It writes a Hugging Face residual model and a standard SkyRL step-zero checkpoint containing the initialized adapter and optimizer state.

Current support is experimental. Dense and grouped-expert Megatron models are supported.

Prepare GSM8K, then produce the initialization artifacts:

```bash
uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir "$HOME/data/gsm8k"
PISSA_INIT_DIR="$HOME/pissa/qwen3_0.6b_r32" \
  bash examples/train/pissa/run_megatron_pissa_qwen3-0.6b_producer.sh
```

The producer creates:

```text
residual_base/
global_step_0/
```

Start training from those artifacts:

```bash
PISSA_INIT_DIR="$HOME/pissa/qwen3_0.6b_r32" \
  bash examples/train/pissa/run_megatron_pissa_qwen3-0.6b.sh
```

For Qwen3-30B-A3B, use the corresponding MoE producer and launcher:

```bash
PISSA_INIT_DIR="$HOME/pissa/qwen3_30b_a3b_r64" \
  bash examples/train/pissa/run_megatron_pissa_qwen3-30b-a3b_producer.sh
PISSA_INIT_DIR="$HOME/pissa/qwen3_30b_a3b_r64" \
  bash examples/train/pissa/run_megatron_pissa_qwen3-30b-a3b.sh
```

The policy loads the residual model and resumes the initialized adapter from `global_step_0`; the frozen KL reference loads the original source model. Additional arguments appended to the launcher override its defaults.

The base model, LoRA rank, target modules, and expert-adapter sharing setting must match between the producer and training launcher.

PiSSA uses `alpha=rank`, `target_modules=all-linear`, and no exclusions. MoE uses one adapter per local expert. Apart from loading the PiSSA artifacts, each training example matches its corresponding LoRA example under `examples/train/megatron/`.

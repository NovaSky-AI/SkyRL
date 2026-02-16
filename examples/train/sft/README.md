# SFT (Supervised Fine-Tuning) Example

This example demonstrates how to use SkyRL's training infrastructure for supervised fine-tuning (SFT).

## Usage

```bash
uv run --isolated --extra vllm python examples/sft/sft_trainer.py
```

## How It Works

1. **Load Dataset**: Uses a small subset of the Alpaca dataset
2. **Tokenize**: Converts instruction/output pairs into token sequences
3. **Create Batch**: Builds a `TrainingInputBatch` with:
   - `sequences`: Token IDs (left-padded)
   - `attention_mask`: 1 for real tokens, 0 for padding
   - `loss_mask`: 1 for response tokens to compute loss on
4. **Train**: Calls `forward_backward(loss_fn="cross_entropy")` for SFT

## Loss Functions

The `loss_fn` parameter supports:

| Loss Function | Use Case |
|--------------|----------|
| `cross_entropy` | Supervised fine-tuning |
| `regular` / `ppo` | PPO with clipping |
| `gspo` | Group Sequence Policy Optimization |
| ... | See `PolicyLossRegistry` for all options |

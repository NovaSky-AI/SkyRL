### Turn-level rewards (GSM8K multi-turn)

This example shows how to train with turn-level rewards in a multi-turn environment using GSM8K.

### How the GSM8K multi-turn environment works

We give the model up to `max_turns` attempts to answer the question. The episode ends early if a correct final numeric answer is produced.

We assign the following per-turn rewards:

- **Correct strict match**: 1.0 and the episode terminates.
- **Well-formatted but incorrect** (answer includes `#### ANSWER`): `0.2 / max_turns`.
- **Otherwise**: 0.0.

**Per-token credit assignment**: Turn-level rewards are converted to token level by assigning the entire turn reward to the final token of the assistant's response for that turn (all other tokens receive 0). This is how rewards are passed to the trainer.

**Custom advantage estimators**: You can plug in custom advantage estimators to operate on the per-turn rewards via `AdvantageEstimatorRegistry` (e.g., using `@register_advantage_estimator`). See `examples/algorithms/custom_advantage_estimator` for an example.

## How to run the example

### 1) Generate the dataset

This pulls `openai/gsm8k` from Hugging Face, adds a short instruction and the ground-truth answer:

```bash
# Adjust output_dir and max_turns as needed
uv run examples/turn_level_rewards/gsm8k_multi_turn_dataset.py \
  --output_dir "$HOME/data/gsm8k_multi_turn" \
  --max_turns 5
```

Outputs:
- `$HOME/data/gsm8k_multi_turn/train.parquet`
- `$HOME/data/gsm8k_multi_turn/validation.parquet`

If you change `--output_dir`, update the `DATA_DIR` variable in the run script below accordingly.

### 2) Launch training

Modify training config parameters in the training script as needed. Commonly modified parameters are: `NUM_GPUS`, `LOGGER`, and `INFERENCE_BACKEND`.

Then run the training script:

```bash
bash examples/turn_level_rewards/run_gsm8k_multi_turn.sh
```
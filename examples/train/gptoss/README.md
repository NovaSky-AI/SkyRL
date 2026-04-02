# GPT-OSS Examples

This folder contains SkyRL examples for training `unsloth/gpt-oss-20b-BF16`.

## Requirements and caveats

- GPT-OSS support in SkyRL currently depends on a Transformers version that exposes `GptOssConfig`.
  In practice, use `transformers>=4.56.2`.
- Flash attention must be disabled for GPT-OSS because attention sinks are not supported there yet.
- Sample packing is also disabled in these recipes.
- These examples use BF16.
- GPT-OSS chat templating benefits from passing `generator.chat_template_kwargs={reasoning_effort:'low'}`.

## Single-turn GSM8K

Generate the dataset:

```bash
uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir "$HOME/data/gsm8k"
```

Run training:

```bash
bash examples/train/gptoss/run_gsm8k_gptoss.sh
```

## Multi-turn GSM8K

This is the minimal GPT-OSS multi-turn example in the repo. It uses the built-in
`gsm8k_multi_turn` environment with turn-level rewards, so it exercises real multi-turn RL
without requiring extra infrastructure.

Generate the dataset:

```bash
uv run examples/train/turn_level_rewards/gsm8k_multi_turn_dataset.py \
  --output_dir "$HOME/data/gsm8k_multi_turn" \
  --max_turns 5
```

Run training:

```bash
bash examples/train/gptoss/run_gsm8k_multi_turn_gptoss.sh
```

If you generate the dataset with a different `--max_turns`, update `MAX_TURNS` in
`run_gsm8k_multi_turn_gptoss.sh` to match.

## When to use other multi-turn examples

If you want more operationally involved agentic tasks, see:

- `examples/train/search/` for SearchR1
- `examples/train/text_to_sql/` for SkyRL-SQL
- `examples/train/step_wise/` for step-wise multi-turn training

Those examples are useful follow-ups, but the GPT-OSS multi-turn GSM8K script here is the
smallest end-to-end recipe to start from.

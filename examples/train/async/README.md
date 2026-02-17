# Async Training Example

One-step off-policy GRPO for Qwen2.5-1.5B-Instruct on GSM8K.

## Usage

```bash 
# prepare the dataset
uv run -- python examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k

export WANDB_API_KEY=<your_key_here>

bash examples/async/async_run_gsm8k.sh
```

For more details, refer to the [documentation](https://docs.skyrl.ai/docs/tutorials/async)

For difference between `async` and `fully_async`, see the documentation: https://docs.skyrl.ai/docs/tutorials/fully_async

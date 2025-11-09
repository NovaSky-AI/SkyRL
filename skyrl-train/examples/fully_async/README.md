# Fully Async Training Example

Fully asynchronous (PipelineRL / AReal style) GRPO for Qwen2.5-1.5B-Instruct on GSM8K.

## Usage

```bash 
# prepare the dataset
uv run -- python examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k

export WANDB_API_KEY=<your_key_here>

bash examples/async/async_run_gsm8k.sh
```

For more details, refer to the documentation. (TO BE ADDED)

Currently, only support generators that use `/chat/completions`. Hence for demonstration we
implement a `skyrl_gym_http_generator.py` (which normally uses a `.generate()`).

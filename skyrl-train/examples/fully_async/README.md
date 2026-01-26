# Fully Async Training Example

Fully asynchronous (PipelineRL / AReal style) GRPO for Qwen2.5-1.5B-Instruct on GSM8K.

## Usage

```bash 
# prepare the dataset
uv run -- python examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k

export WANDB_API_KEY=<your_key_here>

bash examples/fully_async/fully_async_run_gsm8k.sh
```

For more details, refer to the documentation: https://docs.skyrl.ai/docs/tutorials/fully_async

Especially, refer to the section on what knobs to tune: https://docs.skyrl.ai/docs/tutorials/fully_async#step-2-config-knobs-to-tune-for-fully-async-training


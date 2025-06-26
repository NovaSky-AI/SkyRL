# Async Training Example

An example where training and generation can happen asynchronously with each other (off-by-one). 


## Usage

```bash 

uv run -- python examples/gsm8k/gsm8k_dataset.py --output_dir data/gsm8k

export WANDB_API_KEY=<your_key_here>

bash examples/async/async_run_gsm8k.sh
```

For more details, refer to the [documentation](https://skyrl.ai/en/latest/tutorials/async.html)

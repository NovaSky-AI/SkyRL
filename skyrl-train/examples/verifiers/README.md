## Verifiers examples

This directory shows a minimal two-step workflow to train verifiers:

1) Prepare a training dataset from a Prime Hub environment
2) Launch training on that dataset

### 1) Prepare the dataset
Run:
```bash
bash prepare_dataset.sh <ENV_ID>
```
For example:
```bash
bash prepare_dataset.sh primeintellect/reverse-text
```
This will:
- Resolve and install the environment specified by `ENV_ID`
- Generate Parquet files under `DATA_DIR` (default: `$HOME/data/$ENV_ID`):
  - `train.parquet`
  - `validation.parquet`

Notes:
- Internally, the script runs `verifiers_dataset.py`, which accepts optional limits:
  - `--num_train` and `--num_eval` (set to `-1` for no limit). You can run it directly if you need custom sizing.

### 2) Launch training
Run:
```bash
bash run_verifiers.sh
```

To change basic settings, edit the variables at the top of `run_verifiers.sh`:
- `ENV_ID` (to match the dataset you generated)
- `DATA_DIR` (path where the Parquet files live -- defaults to the same as `prepare_dataset.sh`)
- `NUM_GPUS` (number of GPUs to use)
- `LOGGER` (e.g., `console` or `wandb`)

You can also modify other config overrides when running the script and they will be forwarded to the trainer, such as the model choice (`trainer.policy.model.path="Qwen/Qwen2.5-1.5B-Instruct"`, the GRPO group size (`generator.n_samples_per_prompt`), or the training batch size (`trainer.train_batch_size`).
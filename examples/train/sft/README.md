# SFT (Supervised Fine-Tuning) Example

This example demonstrates supervised fine-tuning using SkyRL, with support for both FSDP and Megatron backends.

## Dataset

By default, the example uses the [Alpaca-Cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) dataset (`yahma/alpaca-cleaned`). No manual download is required -- the dataset is loaded automatically via HuggingFace `datasets`.

You can switch to a different dataset by overriding `dataset_name` and `dataset_split` on the command line.

## Quickstart

### FSDP (single GPU)

```bash
bash examples/train/sft/run_sft_fsdp.sh
```

Trains `Qwen/Qwen2.5-0.5B-Instruct` on 1 GPU with FSDP. Key defaults: max length 512, batch size 4, 10 training steps.

### Megatron (multi-GPU with TP/PP)

```bash
bash examples/train/sft/run_sft_megatron.sh
```

Trains `Qwen/Qwen3-0.6B` on 4 GPUs with Megatron (TP=2, PP=2). Key defaults: max length 512, batch size 4, 10 training steps.

### LoRA (FSDP, single GPU)

```bash
bash examples/train/sft/run_sft_lora.sh
```

Trains `Qwen/Qwen2.5-0.5B-Instruct` with LoRA adapters (rank 32, alpha 16) on 1 GPU using FSDP. Only adapter parameters are trainable, significantly reducing memory usage. Key defaults: max length 512, batch size 4, 10 training steps, sample packing enabled. Override LoRA settings with e.g. `model.lora.rank=64 model.lora.alpha=32`.

All scripts accept extra overrides as positional arguments:

```bash
bash examples/train/sft/run_sft_megatron.sh num_steps=20 batch_size=8
```

## Dummy/Benchmarking Mode

For profiling throughput or verifying the training pipeline without real data, use the dummy-run scripts. These fabricate full-context random sequences and skip dataset loading.

```bash
# FSDP dummy run
bash examples/train/sft/run_sft_dummy_fsdp.sh

# Megatron dummy run
bash examples/train/sft/run_sft_dummy_megatron.sh
```

Override the number of steps with:

```bash
bash examples/train/sft/run_sft_dummy_megatron.sh dummy_run_max_steps=10
```

## Configuration Reference

All SFT configuration is defined in [`skyrl/train/config/sft_config.py`](../../../skyrl/train/config/sft_config.py). Key knobs:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `strategy` | `megatron` | Backend: `megatron` or `fsdp2` |
| `model.path` | `Qwen/Qwen3-0.6B` | HuggingFace model ID or local path |
| `dataset_name` | `yahma/alpaca-cleaned` | HuggingFace dataset name |
| `dataset_split` | `train[:100]` | Dataset split/slice |
| `max_length` | `512` | Maximum sequence length |
| `num_steps` | `10` | Number of training steps |
| `batch_size` | `4` | Global batch size |
| `micro_train_batch_size_per_gpu` | `2` | Micro-batch size per GPU |
| `seed` | `42` | Random seed for data shuffling and reproducibility |
| `use_sample_packing` | `true` | Pack multiple sequences per batch (requires flash attention) |
| `ckpt_path` | `""` | Checkpoint directory (empty = no checkpointing) |
| `ckpt_interval` | `0` | Save a checkpoint every N steps (0 = only at end, if `ckpt_path` set) |
| `resume_from` | `""` | Resume training: `""` = fresh start, `"latest"` = latest checkpoint, or path to `global_step_N` dir |
| `megatron_config.tensor_model_parallel_size` | `2` | Tensor parallelism degree (Megatron only) |
| `megatron_config.pipeline_model_parallel_size` | `2` | Pipeline parallelism degree (Megatron only) |
| `megatron_config.context_parallel_size` | `1` | Context parallelism degree (Megatron only) |
| `logger` | `console` | `console` or `wandb` |
| `project_name` | `skyrl_sft` | W&B project name (when `logger=wandb`) |
| `dummy_run_full_ctx` | `false` | Enable dummy/benchmarking mode |
| `dummy_run_max_steps` | `5` | Steps to run in dummy mode |

## Entrypoint

The SFT trainer is invoked as a module:

```bash
python -m skyrl.train.main_sft [key=value overrides...]
```

See [`skyrl/train/main_sft.py`](../../../skyrl/train/main_sft.py) for the CLI entrypoint and
[`skyrl/train/sft_trainer.py`](../../../skyrl/train/sft_trainer.py) for the full implementation.

## Limitations

- **No evaluation support.** : Currently we do not support using an evaluation dataset.
- **Last assistant message only**: The current SFT trainer only supports training on the last assistant message.
- **Two data formats only.** Supports chat-template (`messages` column) and Alpaca (`instruction`/`output` columns). Raw pre-tokenized or plain-text continuation formats are not supported.
- **Single dataset.** No built-in multi-dataset mixing or weighting. Only one `dataset_name` + `dataset_split` pair can be specified.

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

### VLM SFT (Megatron, multi-GPU)

```bash
# One-time: build a small messages-format dataset from HuggingFaceM4/the_cauldron
uv run examples/train/sft/prepare_cauldron_vlm.py --output-dir $HOME/data/cauldron-vlm

bash examples/train/sft/run_sft_megatron_vlm.sh
```

Fine-tunes the vision-language model `Qwen/Qwen3-VL-2B-Instruct` on 4 GPUs with the Megatron backend (pure DP=4 by default). Scale to larger models by overriding `megatron_config.tensor_model_parallel_size` / `pipeline_model_parallel_size`.

VLM SFT mirrors the constraints of the FSDP VLM RL path (3D RoPE + image-token positions tie image tensors to specific sequence positions), so the following are required and enforced:

- `remove_microbatch_padding=false` — no microbatch padding / sequence packing.
- `megatron_config.sequence_parallel_size=1` and `megatron_config.context_parallel_size=1`.
- `train_on_what=last_assistant_message` — VLM tokenization only supports last-assistant training.

Mixed text+image batches are not supported: every sample in a VLM batch must carry image(s). The dataset must use the chat `messages` format with image content encoded as `{"type": "image", "image": <data-uri>}` (see `prepare_cauldron_vlm.py`).

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
| `strategy` | `megatron` | Backend: `megatron` or `fsdp` |
| `model.path` | `Qwen/Qwen3-0.6B` | HuggingFace model ID or local path |
| `dataset_name` | `yahma/alpaca-cleaned` | HuggingFace dataset name |
| `dataset_split` | `train[:100]` | Dataset split/slice |
| `max_length` | `None` | Maximum sequence length |
| `num_steps` | `None` | Number of training steps |
| `batch_size` | `4` | Global batch size |
| `micro_train_batch_size_per_gpu` | `2` | Micro-batch size per GPU |
| `seed` | `42` | Random seed for data shuffling and reproducibility |
| `sampler` | `random` | Training sampler: `random` (shuffle each epoch), `sequential` (in-order), or `custom` (load from `sampler_class_path`) |
| `sampler_class_path` | `None` | Import path (`module.path.ClassName`) to a custom stateful sampler; required when `sampler=custom` |
| `sampler_kwargs` | `{}` | Keyword args forwarded to the custom sampler constructor |
| `dataloader_num_workers` | `0` | Worker processes for the training/eval `StatefulDataLoader` (0 = main process) |
| `dataloader_persistent_workers` | `false` | Keep dataloader workers alive across epochs (only with `dataloader_num_workers > 0`) |
| `remove_microbatch_padding` | `true` | Pack multiple sequences per batch (requires flash attention) |
| `use_sequence_packing` | `false` | Enable controller-level bin-packing across the global mini-batch. Megatron-only. Requires `remove_microbatch_padding=true` and `max_length` set. |
| `max_tokens_per_microbatch` | `null` | Token budget per worker micro-batch when `use_sequence_packing=true`; must be a positive multiple of `max_length` (gives `max_tokens_per_microbatch / max_length` bin rows per micro-batch). `null` = `max_length` (one bin row per micro-batch). |
| `ckpt_path` | `""` | Checkpoint directory (empty = no checkpointing) |
| `ckpt_interval` | `0` | Save a checkpoint every N steps (0 = only at end, if `ckpt_path` set) |
| `resume_from` | `""` | Resume training: `""` = fresh start, `"latest"` = latest checkpoint, or path to `global_step_N` dir |
| `megatron_config.tensor_model_parallel_size` | `2` | Tensor parallelism degree (Megatron only) |
| `megatron_config.pipeline_model_parallel_size` | `2` | Pipeline parallelism degree (Megatron only) |
| `megatron_config.context_parallel_size` | `1` | Context parallelism degree (Megatron only) |
| `logger` | `console` | `console` or `wandb` |
| `project_name` | `skyrl_sft` | W&B project name (when `logger=wandb`) |
| `train_on_what` | `last_assistant_message` | Which tokens to train on. See `TrainOnWhat` enum: `last_assistant_message` (default, loss on final assistant reply only) or `all_assistant_messages` (loss on every assistant message). |
| `dummy_run_full_ctx` | `false` | Enable dummy/benchmarking mode |
| `dummy_run_max_steps` | `5` | Steps to run in dummy mode |

## Entrypoint

The SFT trainer is invoked as a module:

```bash
python -m skyrl.train.main_sft [key=value overrides...]
```

See [`skyrl/train/main_sft.py`](../../../skyrl/train/main_sft.py) for the CLI entrypoint and
[`skyrl/train/sft_trainer.py`](../../../skyrl/train/sft_trainer.py) for the full implementation.

## Minibatch packing (controller-level FFD, Megatron only)

When `use_sequence_packing=true`, `SFTTrainer` collates with
`PackedDataCollator` instead of `DefaultCollator`. Every training step:

1. The controller's collator runs FFD bin-packing over the global
   mini-batch using `max_length` as the bin capacity.
2. The bin count is forced to a multiple of `dp_size` via empty-bin
   padding (`min_bin_count`/`bin_count_multiple` knobs in
   [`bin_packing._adjust_bin_count`](../../../skyrl/train/dataset/bin_packing.py)).
3. Each bin becomes one row of the dispatched `TrainingInputBatch`. SkyRL additionally
   tracks some metadata for demarcating sequences.

Example overrides on top of `run_sft_megatron_tulu3_50k.sh`:

```bash
bash examples/train/sft/run_sft_megatron_tulu3_50k.sh \
    use_sequence_packing=true \
    max_tokens_per_microbatch=4096
```


## Samplers and the stateful dataloader

`SFTTrainer` feeds the training loop from a
[`torchdata.stateful_dataloader.StatefulDataLoader`](https://pytorch.org/data/main/stateful_dataloader.html).
The sampling position is written into each checkpoint (`data.pt`) so a
`resume_from` run continues from the exact next example of the in-progress
epoch.

Three sampler strategies are built in:

- `sampler=random` (default) — reshuffles every epoch using `seed`. The
  in-progress epoch resumes bit-exactly; later epochs are re-shuffled into a
  valid (but not byte-identical) order, matching the RL trainer's behavior.
- `sampler=sequential` — iterates the dataset in order
  ([`StatefulSequentialSampler`](../../../skyrl/train/dataset/samplers.py)).
- `sampler=custom` — loads your own stateful sampler from `sampler_class_path`,
  instantiated as `ClassName(tokenized, **sampler_kwargs)`. A custom sampler
  only needs `__iter__`/`__len__` plus `state_dict`/`load_state_dict` to be
  checkpointable. The import runs inside a Ray task, which does **not** inherit
  the driver's `PYTHONPATH` -- use a dotted path importable from the repo root
  (e.g. `examples.train.sft.curriculum_sampler.CurriculumLearningSampler`) and
  launch from the repo root. No `__init__.py` is needed (namespace packages).

### Curriculum learning example

[`curriculum_sampler.py`](curriculum_sampler.py) is a reference custom sampler
(`CurriculumLearningSampler`) that walks through difficulty-ordered subsets,
progressively unlocking harder data. Order the dataset easy→hard and give the
per-stage `lengths`. Set `num_samples=num_steps*batch_size` so the whole
schedule is covered in a single pass (this keeps the curriculum state intact
across epoch boundaries and makes resume bit-exact across the entire run).
Pass the sampler config as overrides on the base SFT example script, run from
the repo root so the dotted path resolves (`lengths` must sum to the dataset
size — 100 for the script's `train[:100]` split — and `num_samples` should be
`num_steps * batch_size` to cover the whole schedule):

```bash
bash examples/train/sft/run_sft_megatron.sh \
    sampler=custom \
    sampler_class_path=examples.train.sft.curriculum_sampler.CurriculumLearningSampler \
    'sampler_kwargs={lengths: [34, 33, 33], num_samples: 40, seed: 42}'
```

### Data mixing example

[`data_mixing_sampler.py`](data_mixing_sampler.py) is a reference custom sampler
(`DataMixingSampler`) that mixes a concatenation of sources with per-source
`weights`, using torch's native `WeightedRandomSampler` for the weighted draw.
Each source's weight is divided across its examples, so the source-level mixing
proportion matches `weights` regardless of how many examples each source has.
Order the dataset by source and give `lengths`/`weights` per source:

```bash
bash examples/train/sft/run_sft_megatron.sh \
    sampler=custom \
    sampler_class_path=examples.train.sft.data_mixing_sampler.DataMixingSampler \
    'sampler_kwargs={lengths: [80, 20], weights: [0.5, 0.5], num_samples: 40, seed: 42}'
```

## Limitations

- **Limited `train_on_what` options**: Supports training on all or the last assistant message.
- **Two data formats only.** Supports chat-template (`messages` column) and Alpaca (`instruction`/`output` columns). Raw pre-tokenized or plain-text continuation formats are not supported.
- **Single dataset.** No built-in multi-dataset mixing or weighting. Only one `dataset_name` + `dataset_split` pair can be specified.

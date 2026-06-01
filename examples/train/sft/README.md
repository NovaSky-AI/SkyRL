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
| `strategy` | `megatron` | Backend: `megatron` or `fsdp` |
| `model.path` | `Qwen/Qwen3-0.6B` | HuggingFace model ID or local path |
| `dataset_name` | `yahma/alpaca-cleaned` | HuggingFace dataset name |
| `dataset_split` | `train[:100]` | Dataset split/slice |
| `max_length` | `None` | Maximum sequence length |
| `num_steps` | `None` | Number of training steps |
| `batch_size` | `4` | Global batch size |
| `micro_train_batch_size_per_gpu` | `2` | Micro-batch size per GPU |
| `seed` | `42` | Random seed for data shuffling and reproducibility |
| `remove_microbatch_padding` | `true` | Pack multiple sequences per batch (requires flash attention) |
| `use_sequence_packing` | `false` | Enable controller-level FFD bin-packing across the global mini-batch. Megatron-only. Requires `remove_microbatch_padding=true` and `max_length` set. |
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
   [`bin_packing._adjust_bin_count`](../../../skyrl/backends/skyrl_train/distributed/megatron/bin_packing.py)).
3. Each bin becomes one row of the dispatched `TrainingInputBatch`. The
   per-row `sub_seq_lengths` metadata flows through `BatchIterator` into
   the worker's `forward_step`, where `preprocess_packed_seqs` emits
   `cu_seqlens` entries for every sub-sequence (not one per row).
4. The per-row loss mask zeros out positions at sub-seq boundaries so the
   `target.roll(shifts=-1)` inside `from_parallel_logits_to_logprobs` never
   contributes a cross-sub-seq prediction.

Constraints: Megatron backend only, `remove_microbatch_padding=true`, and
`max_length` set. See
[`docs/agent_packing_design.md`](../../../docs/agent_packing_design.md) and
[`docs/packing_architecture_comparison.md`](../../../docs/packing_architecture_comparison.md).

Example overrides on top of `run_sft_megatron_tulu3_50k.sh`:

```bash
bash examples/train/sft/run_sft_megatron_tulu3_50k.sh \
    use_sequence_packing=true \
    max_tokens_per_microbatch=4096
```

## Packing simulators (CPU only)

For research on packing strategies, two pure-Python simulators live alongside
the runnable training scripts. They tokenize a real chat dataset (Tulu3 SFT
mixture by default), iterate over mini-batches, run either SkyRL's
micro-batch-level packing or mini-batch-level FFD packing, and use
`time.sleep` to approximate fwd/bwd cost. They require no GPU and run with
just the `fsdp` extra.

```bash
# Run both simulators with defaults (matches run_sft_megatron_tulu3_50k.sh
# knobs: max_length=4096, batch_size=24, micro_batch_size=6, 10 mini-batches)
bash examples/train/sft/simulate_packing_compare.sh

# Or run them individually
uv run --isolated --extra fsdp -- \
    python examples/train/sft/simulate_packing_skyrl.py --num-batches 10
uv run --isolated --extra fsdp -- \
    python examples/train/sft/simulate_packing_nemo.py --num-batches 10

# Render a side-by-side comparison plot from the result JSONs
uv run --isolated --extra fsdp -- \
    python examples/train/sft/simulate_packing_plot.py \
    --input "SkyRL::results/<date>/skyrl.json" \
    --input "Minibatch-FFD::results/<date>/nemo.json" \
    --out results/<date>/comparison.png
```

`simulate_packing_common.py` carries the shared bin-packing implementation
(an inline FirstFitDecreasingPacker) and the synthetic
fwd/bwd cost model.

## Limitations

- **No evaluation support.** : Currently we do not support using an evaluation dataset.
- **Limited `train_on_what` options**: Supports training on all or the last assistant message.
- **Two data formats only.** Supports chat-template (`messages` column) and Alpaca (`instruction`/`output` columns). Raw pre-tokenized or plain-text continuation formats are not supported.
- **Single dataset.** No built-in multi-dataset mixing or weighting. Only one `dataset_name` + `dataset_split` pair can be specified.

# Megatron examples

GRPO/DAPO training with the Megatron backend. Every script here is self-contained
— the parallelism, batch sizes and memory arithmetic are set in the script's own
header, which is also where the reasoning for each choice lives.

Prepare GSM8K data first (the scripts default to `$HOME/data/gsm8k`):

```bash
uv run --isolated examples/train/gsm8k/gsm8k_dataset.py --output_dir "$HOME/data/gsm8k"
```

For anything above ~30B, put the HF cache on shared storage so every node reads
one copy of the weights, and set a wandb key if you want the run recorded:

```bash
export HF_HOME=/mnt/shared_storage/hf     # 235B is ~470 GB
export WANDB_API_KEY=<your_key_here>
```

Then, e.g.:

```bash
bash examples/train/megatron/run_megatron_qwen3-30b-a3b.sh
```

Most scripts accept `LOGGER=console` to print metrics to stdout instead of wandb,
and `NUM_STEPS=<n>` to cut a run short.

## Sharded-RDT weight sync (`run_megatron_rdt_*.sh`)

These use `weight_sync_backend=sharded_rdt`, where the vLLM workers **pull** the
slices they consume from the trainer over NIXL/RDMA rather than the trainer
broadcasting to every worker. On Qwen3-235B-A22B this moves a full merged-LoRA
sync (~470 GB) in ~3.5 s against ~65 s for the NCCL broadcast.

| script | model | nodes (8xGPU) | trainer | inference |
|---|---|---|---|---|
| `run_megatron_rdt_qwen3_30b_a3b_lora.sh` | Qwen3-30B-A3B | 2 trainer + 1 inference | tp2/ep8 | 1 engine, TP8 |
| `run_megatron_rdt_qwen3_235b_a22b_lora.sh` | Qwen3-235B-A22B | 2 trainer + 1 inference | tp4/pp2/ep8 | 1 engine, TP8 |
| `run_megatron_rdt_dpep_qwen3_235b_a22b_lora.sh` | Qwen3-235B-A22B | 2 trainer + 1 inference | tp4/pp2/ep8 | 1 engine, DP8/EP8 |
| **`run_megatron_rdt_dpep_qwen3_235b_a22b_lora_4node.sh`** | Qwen3-235B-A22B | **2 trainer + 2 inference** | tp4/pp2/ep8 | 1 engine, **DP16/EP16** |
| `run_megatron_rdt_dpep_glm45_air_lora.sh` | GLM-4.5-Air | 1 trainer + 1 inference | ep8 | 1 engine, DP8/EP8 |

### Requirements

- **Non-colocated placement.** `sharded_rdt` rejects `colocate_all=true`: the
  inference workers pull from separate trainer actors.
- **`aws-efa-installer >= 1.47` on every GPU node** for cross-node NIXL. Below
  that, NIXL silently falls off the LIBFABRIC provider and the transfer is neither
  fast nor representative — check for `Backend LIBFABRIC was instantiated` in the
  engine logs. EFA is commonly reset by node recycles, so re-check after one.
- **Ray >= 2.56.**
- `expert_tensor_parallel_size=1`. At ETP>1 no rank holds a whole expert, so the
  optimized weight source falls back to a slower whole-model export.
- Weights prefetched into a shared `HF_HOME`, and `DATA_DIR` on shared storage.

### The 4-node 235B recipe

`run_megatron_rdt_dpep_qwen3_235b_a22b_lora_4node.sh` is the reference
configuration for the published numbers. It spends the extra inference node on
**halving the per-rank weight footprint**, not on a second model replica: one
16-way data/expert-parallel engine spanning both inference nodes, so the per-rank
weights drop from 67.8 GiB (the 3-node DP8/EP8 shape) to ~41 GiB predicted, 43.6
GiB observed. That is what buys enough headroom for CUDA graphs and a usable KV
cache at the same time; the script's header carries the full per-GPU ledger.

```bash
export HF_HOME=/mnt/shared_storage/hf
export WANDB_API_KEY=<your_key_here>
# recommended with compilation on: share one torch.compile cache across both
# inference nodes instead of each paying the compile on first start
export VLLM_CACHE_ROOT=/mnt/shared_storage/vllm_cache
bash examples/train/megatron/run_megatron_rdt_dpep_qwen3_235b_a22b_lora_4node.sh
```

As written it trains LoRA r128 with `merge_lora=true`, so each sync still streams
the full merged weights — LoRA changes the trainer's memory, never the weight-sync
path or its timing. Full fine-tuning at this size does not fit 4 nodes; the
script's header has the per-GPU arithmetic.

Expect ~13-15 min to the first sync with a warm checkpoint page cache, and
substantially longer on the first run after a node recycle (the ~440 GB
safetensors load dominates).

### Checking a run is healthy

The metric that matters is `policy/minibatch_rollout_logprobs_abs_diff_mean`: how
far the rollout engine's logprobs are from the trainer's for the same tokens.

- **~0.05** — healthy; the engine and the trainer agree.
- **>1** — the weights in vLLM are not the weights in the trainer.

A bad sync does not announce itself: reward can look normal at step 0 and then sit
at exactly 0 forever, with `grad_norm` pinned at 0. Watch the logprob gap rather
than the loss curve. `timing/sync_weights` is the enclosing pause/resume bracket;
`timing/sync_weights_only_transfer` is the transfer on its own.

Which weight source a run actually chose is logged once at init, e.g.
`[rdt-config] source=MegatronStackedWeightSource lookahead_env=1 ...` — it prints
the class it selected, so it cannot disagree with what is running.

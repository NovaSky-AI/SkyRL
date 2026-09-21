# Megatron examples with sharded weight transfer using RDT

GRPO/DAPO training with the Megatron backend with sharded RDT weight transfer engine. Every script here is self-contained — the parallelism, batch sizes and memory arithmetic are set in the script's own
header, which is also where the reasoning for each choice lives.

## Pre-requisites

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
bash examples/train/megatron/sharded_rdt/run_megatron_rdt_qwen3_30b_a3b_lora.sh
```

## Running the scripts

All the scripts use `weight_sync_backend=sharded_rdt`, where the vLLM workers **pull** the
slices they consume from the trainer over NIXL/RDMA rather than the trainer
broadcasting to every worker. On Qwen3-235B-A22B this moves a full merged-LoRA
sync (~470 GB) in ~3.5 s against ~65 s for the NCCL broadcast.

Most scripts accept `LOGGER=console` to print metrics to stdout instead of wandb,
and `NUM_STEPS=<n>` to cut a run short.

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
- **`aws-efa-installer >= 1.47` on every GPU node** for cross-node NIXL on AWS. Below
  that, NIXL silently falls off the LIBFABRIC provider and the transfer is neither
  fast nor representative — check for `Backend LIBFABRIC was instantiated` in the
  engine logs. EFA is commonly reset by node recycles, so re-check after one.
- **Ray >= 2.56.**
- `expert_tensor_parallel_size=1`. At ETP>1 no rank holds a whole expert, so the
  optimized weight source falls back to a slower whole-model export.
- Weights prefetched into a shared `HF_HOME`, and `DATA_DIR` on shared storage.

### The 4-node 235B recipe

`run_megatron_rdt_dpep_qwen3_235b_a22b_lora_4node.sh` is the reference
configuration script for Qwen 235B benchmarks.

```bash
export HF_HOME=/mnt/shared_storage/hf
export WANDB_API_KEY=<your_key_here>
# recommended with compilation on: share one torch.compile cache across both
# inference nodes instead of each paying the compile on first start
export VLLM_CACHE_ROOT=/mnt/shared_storage/vllm_cache
bash examples/train/megatron/sharded_rdt/run_megatron_rdt_dpep_qwen3_235b_a22b_lora_4node.sh
```

As written it trains LoRA r128 with `merge_lora=true`, so each sync still streams
the full merged weights — LoRA changes the trainer's memory, never the weight-sync
path or its timing. Thus, it is still representative for a weight sync benchmark for a 235B model. Full fine-tuning at this size does not fit 4 nodes; the
script's header has the per-GPU arithmetic.
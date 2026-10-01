#!/bin/bash
# Context-length sweep of the GLM-5.3-Flash dummy SFT benchmark (random token ids, random-init
# weights), using the 1M-token configuration at every length: TP8 x CP2 (dp 4, batch 4), full
# recompute, pinned checkpoint offload, TP-sharded DSA indexer, KDA all-gather CP exchange and the
# node-deduplicated MoE all-to-all. FP32 gradients. One full-length sequence per DP rank per step.
#
# Needs: a Ray cluster with 8 nodes x 8 B200 (or set NUM_NODES / NUM_GPUS_PER_NODE), started from
# the SkyRL repo, and MODEL_PATH pointing at a directory, readable on every node, with
# zai-org/GLM-5.3-Flash's config.json (drop its "quantization_config") and tokenizer files
# (tokenizer.json, tokenizer_config.json, chat_template.jinja). No weights are loaded.
#
# Usage (from the repo root):
#   MODEL_PATH=/path/to/glm5p3_flash_cfg bash examples/train/sft/sweep_sft_dummy_glm5p3_flash_context.sh
#   MODEL_PATH=... LENGTHS="16384 65536" NUM_STEPS=2 bash examples/train/sft/sweep_sft_dummy_glm5p3_flash_context.sh
#
# Each length writes $LOG_DIR/L<length>.log; the sweep stops at the first failing length unless
# STOP_ON_FAILURE=0. At the end it prints (and writes $LOG_DIR/summary.md and .csv) one row per
# length, with timings averaged over the warm steps (step 0 includes kernel compilation and is
# excluded; with NUM_STEPS=2 there is one warm step). Measured on 64 B200 whose inter-node NCCL ran
# over TCP (~1 GB/s per GPU); 1M takes ~1,000 s per step there, the whole default sweep ~2.5 h.
#
# Host memory: pinned checkpoint offload holds ~1.4 MiB per token per GPU during fwd+bwd
# (~720 GB per node at 1M tokens with TP8 CP2). Set SKYRL_OFFLOAD_CHECKPOINT_INPUTS_PINNED=0 for
# pageable offload (same bytes, released automatically, ~10% slower).
set -euo pipefail

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to a GLM-5.3-Flash config + tokenizer directory}"
LENGTHS="${LENGTHS:-16384 32768 65536 131072 262144 524288 1048576}"
NUM_STEPS="${NUM_STEPS:-3}"
NUM_NODES="${NUM_NODES:-8}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-1}"
LOG_DIR="${LOG_DIR:-logs/glm5p3_ctx_sweep_$(date +%Y%m%d_%H%M%S)}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p "$LOG_DIR"

# Parallelism and memory knobs read by run_sft_dummy_glm5p3_flash_megatron.sh.
export MODEL_PATH NUM_STEPS NUM_NODES NUM_GPUS_PER_NODE
export MEGATRON_TP="${MEGATRON_TP:-8}"
export MEGATRON_CP="${MEGATRON_CP:-2}"
export MEGATRON_EP="${MEGATRON_EP:-32}"
export MEGATRON_ETP="${MEGATRON_ETP:-2}"
export RECOMPUTE_GRANULARITY=full RECOMPUTE_METHOD=uniform RECOMPUTE_NUM_LAYERS=1 RECOMPUTE_MODULES='[core_attn]'
# Opt-in patches (all forwarded to the Ray workers as SKYRL_* variables).
export SKYRL_OFFLOAD_CHECKPOINT_INPUTS="${SKYRL_OFFLOAD_CHECKPOINT_INPUTS:-1}"
export SKYRL_OFFLOAD_CHECKPOINT_INPUTS_PINNED="${SKYRL_OFFLOAD_CHECKPOINT_INPUTS_PINNED:-1}"
export SKYRL_DSA_INDEXER_TP_SHARD="${SKYRL_DSA_INDEXER_TP_SHARD:-1}"
export SKYRL_KDA_CP_EXCHANGE="${SKYRL_KDA_CP_EXCHANGE:-allgather}"
export SKYRL_MOE_NODE_DEDUP="${SKYRL_MOE_NODE_DEDUP:-1}"

echo "sweep: lengths=[$LENGTHS] steps=$NUM_STEPS TP=$MEGATRON_TP CP=$MEGATRON_CP logs=$LOG_DIR"
for L in $LENGTHS; do
  echo "=== $(date +%T) L=$L"
  rc=0
  MAX_LENGTH=$L bash examples/train/sft/run_sft_dummy_glm5p3_flash_megatron.sh run_name="ctx_sweep_L$L" \
    > "$LOG_DIR/L$L.log" 2>&1 || rc=$?
  echo "=== $(date +%T) L=$L exit=$rc"
  if [ "$rc" -ne 0 ] && [ "$STOP_ON_FAILURE" = "1" ]; then
    echo "stopping: L=$L failed (see $LOG_DIR/L$L.log)"
    break
  fi
done

python3 - "$LOG_DIR" $LENGTHS <<'EOF'
import csv, os, re, sys

log_dir, lengths = sys.argv[1], [int(x) for x in sys.argv[2:]]
cols = ["context", "status", "warm fwd+bwd (s)", "fwd+bwd tok/s/GPU", "warm step (s)", "step tok/s",
        "peak alloc (GiB)", "peak reserved (GiB)"]
rows = []
for L in lengths:
    path = os.path.join(log_dir, f"L{L}.log")
    if not os.path.exists(path):
        continue
    txt = open(path, errors="ignore").read()
    fb = [float(x) for x in re.findall(r"'timing/forward_backward': '([\d.]+)'", txt)]
    st = [float(x) for x in re.findall(r"'timing/step': '([\d.]+)'", txt)]
    alloc = [float(x) for x in re.findall(r"peak_mem_allocated_gb_max=([\d.]+)", txt)]
    resv = [float(x) for x in re.findall(r"peak_mem_reserved_gb_max=([\d.]+)", txt)]
    batch = int(re.search(r"batch_size=(\d+)", txt).group(1))
    nodes = int(re.search(r"placement\.num_nodes=(\d+)", txt).group(1))
    gpn = int(re.search(r"placement\.num_gpus_per_node=(\d+)", txt).group(1))
    if "out of memory" in txt.lower() or "OutOfMemoryError" in txt:
        status = "OOM"
    elif "Dummy SFT training complete" in txt:
        status = "ok"
    else:
        status = "failed"
    warm = lambda xs: (sum(xs[1:]) / len(xs[1:])) if len(xs) > 1 else (xs[0] if xs else None)
    f, s = warm(fb), warm(st)
    tokens = batch * L
    rows.append([L, status,
                 f"{f:.1f}" if f else "-", f"{tokens / f / (nodes * gpn):.1f}" if f else "-",
                 f"{s:.1f}" if s else "-", f"{tokens / s:,.0f}" if s else "-",
                 f"{max(alloc):.1f}" if alloc else "-", f"{max(resv):.1f}" if resv else "-"])

md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
md += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
print("\n".join(md))
open(os.path.join(log_dir, "summary.md"), "w").write("\n".join(md) + "\n")
with open(os.path.join(log_dir, "summary.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(cols)
    w.writerows(rows)
EOF

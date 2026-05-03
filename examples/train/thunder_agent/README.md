# ThunderAgent + SkyRL: R2EGym 32B Training Recipe

Train Qwen3-32B on R2EGym with ThunderAgent-accelerated rollout scheduling.
ThunderAgent pipelines Harbor agent trials through a capacity-aware token router
so vLLM GPUs stay busy instead of waiting for slow coding tasks.

![TA vs no-TA stitched 40-step Harbor timeline](./docs/speedup.png)

**3.01× wall-clock speedup** over the no-TA baseline (8.84 h vs 26.58 h for 40 training steps / 10 epochs).

---

## Hardware

| Role | Nodes | GPUs | Notes |
|---|---|---|---|
| Head / merged | 1 | 0 GPU (CPU-only SLURM step) | Ray head, ThunderAgent proxy, training driver |
| Rollout | 1 | 8 × H100 | 4 vLLM servers at TP=2 |
| Trainer | 4 | 8 × H100 each | FSDP2 policy + ref model |

Total: 5 SLURM nodes, 40 H100 GPUs.

---

## Quick Start (3 steps)

### Step 1 — Download the R2EGym dataset

Download the four base difficulty-bucket datasets from HuggingFace:

```bash
for BUCKET in trivial easy medium hard; do
  python examples/train_integrations/harbor/prepare_harbor_dataset.py \
    --dataset NovaSky-AI/r2egym-${BUCKET} \
    --output_dir ~/data/harbor/r2egym-${BUCKET}
done
```

Then generate the curated train/eval MANIFEST subsets used by this recipe:

```bash
python examples/train/thunder_agent/prepare_r2egym_subset.py \
  --data-root ~/data/harbor
```

This creates:
- `~/data/harbor/r2egym-train256-medium-hard-v1/MANIFEST.json` (256 tasks)
- `~/data/harbor/r2egym-eval64-medium-hard-v1/MANIFEST.json`  (64 tasks)

### Step 2 — Set up the cluster

The recipe uses `run_harbor_benchmark.sh` as the cluster orchestration layer.
It handles Ray cluster init, rollout server startup, Docker setup, and pre-flight checks.

```bash
export WRAPPER="$REPO_ROOT/examples/train_integrations/harbor/run_harbor_benchmark.sh"

# Allocate SLURM jobs (head+rollout on one node, 4 trainer nodes):
export MERGED_JOB_ID=<merged_job_id>
export MERGED_NODE=<merged_node_hostname>
export ROLLOUT_JOB_ID=<rollout_job_id>      # same as MERGED_JOB_ID if co-located
export ROLLOUT_NODE=<rollout_node_hostname>  # same as MERGED_NODE if co-located
export TRAINER_NODE_SPECS="<node1>:<jobid1>,<node2>:<jobid2>,<node3>:<jobid3>,<node4>:<jobid4>"

# Common env (adjust paths):
export PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
export MODEL_PATH="/path/to/Qwen3-32B"
export DATA_ROOT=~/data/harbor
export DOCKER_MODE=rootful
export ROLLOUT_SERVER_PORTS_CSV="18000,18001,18002,18003"
export SKYRL_INFERENCE_ROUTER_PORT=18080

# Run the setup stages (in order):
bash "$WRAPPER" prepare    # Harbor patches, validation, image checks
bash "$WRAPPER" head       # Docker runtime setup on head node
bash "$WRAPPER" ray        # Start Ray head + 4 trainer workers
bash "$WRAPPER" rollout    # Start 4 vLLM servers, wait for /health
bash "$WRAPPER" status     # Confirm all stages ready
```

### Step 3 — Launch training

```bash
export TRAIN_DATA="['$DATA_ROOT/r2egym-train256-medium-hard-v1']"
export EVAL_DATA="['$DATA_ROOT/r2egym-eval64-medium-hard-v1']"
export MAX_TRAIN_TASKS=256
export MAX_EVAL_TASKS=64
export FULL_EPOCHS=10
export EVAL_INTERVAL_STEPS=4
export RUN_NAME_OVERRIDE="r2egym-ta-mediumhard256-10epoch-$(date +%Y%m%d_%H%M%S)"
export LOG_DIR_OVERRIDE=~/tmp_logs/$RUN_NAME_OVERRIDE

# ROLLOUT_SERVER_URLS is built automatically from ROLLOUT_HOST_IP + ROLLOUT_SERVER_PORTS_CSV,
# or you can set it directly:
export ROLLOUT_HOST_IP=<rollout_node_ip>
# The ThunderAgent proxy URL is derived automatically from RAY_HEAD_IP:
export RAY_HEAD_IP=<head_node_ip>

bash examples/train/thunder_agent/run_harbor_thunder_agent_32b.sh full
```

To run the no-TA baseline for comparison, add `SKYRL_DISABLE_THUNDERAGENT=1`:

```bash
SKYRL_DISABLE_THUNDERAGENT=1 \
bash examples/train/thunder_agent/run_harbor_thunder_agent_32b.sh full
```

---

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `MAX_TRAIN_TASKS` | 256 | Tasks per epoch |
| `MAX_EVAL_TASKS` | 64 | Eval tasks |
| `FULL_EPOCHS` | 10 | Training epochs |
| `EVAL_INTERVAL_STEPS` | 4 | Steps between evals |
| `HARBOR_AGENT_MAX_TURNS` | 25 | Max agent turns per trial |
| `AGENT_TIMEOUT_SEC` | 9000 | Hard timeout per Harbor trial |
| `MINI_SWE_MODEL_TIMEOUT_SEC` | 1200 | Per-LLM-call timeout |
| `ROLLOUT_ENGINES` | 4 | Number of vLLM rollout servers |
| `ROLLOUT_TP_SIZE` | 2 | Tensor-parallel size per server |
| `THUNDER_AGENT_MODE` | `tr` | TA scheduler mode (`tr` = token-rate) |

All training hyperparameters can be overridden on the command line, e.g.:

```bash
bash run_harbor_thunder_agent_32b.sh full \
  trainer.policy.optimizer_config.lr=5e-7 \
  generator.sampling_params.temperature=0.6
```

---

## Environment Variables (URL auto-construction)

The run script accepts two input conventions for rollout server endpoints:

| Input | How to set |
|---|---|
| Direct URL list | `ROLLOUT_SERVER_URLS='["http://1.2.3.4:18000","http://1.2.3.4:18001",...]'` |
| Host + port CSV | `ROLLOUT_HOST_IP=1.2.3.4` + `ROLLOUT_SERVER_PORTS_CSV=18000,18001,18002,18003` |

`EXTERNAL_PROXY_URL` (TA proxy) is resolved in this order:
1. `EXTERNAL_PROXY_URL` if set explicitly
2. `THUNDERAGENT_URL` if set (matches `run_harbor_benchmark.sh` convention)
3. `http://$RAY_HEAD_IP:$SKYRL_INFERENCE_ROUTER_PORT` if `RAY_HEAD_IP` is set
4. Not set → an embedded `ThunderAgentRouter` is started inside the trainer process

---

## Dataset Subsets (reproducible selection)

`prepare_r2egym_subset.py` reproduces the exact train/eval splits using a deterministic
SHA-256 selection:

```
seed: r2egym-medium-hard-v1-20260325
train256: trivial=4  easy=16  medium=120  hard=116  (total 256)
eval64:   trivial=1  easy=4   medium=30   hard=29   (total 64)
```

Train tasks do not overlap with eval tasks. Eval preserves `r2egym-eval32-medium-major-v1`
as a prefix subset; train preserves `r2egym-train128-medium-major-v1` as a prefix.

---

## GSM8K Quickstart (single-node dev)

For a fast single-node sanity check with ThunderAgent on GSM8K:

```bash
uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
bash examples/train/thunder_agent/run_thunder_agent_gsm8k.sh
```

---

## Speedup Details

| | No-TA | With ThunderAgent |
|---|---|---|
| 40-step wall clock | 26.58 h | 8.84 h |
| Speedup | 1× | **3.01×** |
| Wait-for-generate | 22.91 h (86%) | 6.31 h (71%) |
| Training compute | 0.56 h | 0.64 h |

Hardware: 4 × 8 H100 training nodes, 1 × 8 H100 rollout node (TP=2, 4 engines).
Strategy: fully-async GRPO, `n_samples_per_prompt=4`, `train_batch_size=64`.

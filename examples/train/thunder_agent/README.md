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

The canonical entrypoint is:

```bash
examples/train/thunder_agent/run_r2egym_32b_recipe.sh
```

Use this wrapper for the 32B recipe unless you are already inside a running
Ray cluster and intentionally want to call the lower-level trainer script
directly. The wrapper owns the PR-core environment selection, SLURM cross-job
launch, rootful Docker setup, Harbor image pre-pull, Ray startup, rollout
startup, and driver launch.

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

Use the recipe-local wrapper. It encodes the
`thunderagent_medium_hard_256_10epoch_no_preflight` run shape from the
handoff runbook, but runs the PR-core ThunderAgent entrypoint and scripts.

```bash
export REPO_ROOT=/path/to/skyrl-ta-pr-core
cd "$REPO_ROOT"

# Build the recipe Python env once. Default stack:
# torch 2.11.0+cu129, vLLM 0.20.1+cu129, Ray 2.51.1.
export RECIPE_VENV="/data/zy/models/$USER/venvs/skyrl-ta-pr-core-thunder-agent-vllm0201-cu129"
bash examples/train/thunder_agent/setup_env.sh sync
source <(RECIPE_VENV="$RECIPE_VENV" bash examples/train/thunder_agent/setup_env.sh print)

export WRAPPER="$REPO_ROOT/examples/train/thunder_agent/run_r2egym_32b_recipe.sh"

# Allocate SLURM jobs (head+rollout on one node, 4 trainer nodes):
export MERGED_JOB_ID=<merged_job_id>
export MERGED_NODE=<merged_node_hostname>
export ROLLOUT_JOB_ID=<rollout_job_id>      # same as MERGED_JOB_ID if co-located
export ROLLOUT_NODE=<rollout_node_hostname>  # same as MERGED_NODE if co-located
export TRAINER_NODE_SPECS="<node1>:<jobid1>,<node2>:<jobid2>,<node3>:<jobid3>,<node4>:<jobid4>"

# Common env (adjust paths). The env must pass `bash "$WRAPPER" prepare`.
export MODEL_PATH="/data/zy/models/$USER/models/Qwen3-32B"
export DATA_ROOT="/home/$USER/zthunder_yagent/data/harbor"
export DOCKER_MODE=rootful
export PREPULL_R2EGYM_IMAGES=true
export ROLLOUT_SERVER_PORTS_CSV="18000,18001,18002,18003"
export ROLLOUT_ENFORCE_EAGER=true
export VLLM_SERVER_MODULE=skyrl.backends.skyrl_train.inference_engines.vllm.vllm_server
export SKYRL_INFERENCE_ROUTER_PORT=18080
export RUN_NAME_OVERRIDE="r2egym-ta-mediumhard256-10epoch-nopf-$(date +%Y%m%d_%H%M%S)"

# Run the strict stages in order. `driver` is the only stage that starts
# training; the earlier stages are setup and readiness checks.
bash "$WRAPPER" cleanup-stage all
bash "$WRAPPER" prepare    # Python/model/dataset validation
bash "$WRAPPER" head       # Docker/network/image pre-pull/shared mini-swe-agent setup
bash "$WRAPPER" ray        # Start Ray head + 4 trainer workers
bash "$WRAPPER" rollout    # Start 4 vLLM servers, wait for /health
bash "$WRAPPER" status     # Confirm all stages ready
```

### Step 3 — Launch training

```bash
bash "$WRAPPER" driver
```

The wrapper defaults match the benchmark variant:

- `TRAIN_DATA=['$DATA_ROOT/r2egym-train256-medium-hard-v1']`
- `EVAL_DATA=['$DATA_ROOT/r2egym-eval64-medium-hard-v1']`
- `MAX_TRAIN_TASKS=256`, `MAX_EVAL_TASKS=64`
- `FULL_EPOCHS=10`, `EVAL_INTERVAL_STEPS=4`, `CKPT_INTERVAL=4`
- `USE_KL_LOSS=false`, `KL_LOSS_COEF=0.0`
- `RUN_PREFLIGHT_CHECKS=false`, `AGENT_RUNTIME_PREFLIGHT=false`

The rollout launcher uses PR-core's native vLLM server module with
`skyrl.backends.skyrl_train.inference_servers.vllm_worker.WorkerWrap` directly.

For this cluster's CUDA 12.9 driver, `setup_env.sh` installs the official
vLLM release asset `vllm-0.20.1+cu129`; installing plain `vllm==0.20.1` from
PyPI can select the CUDA 13 wheel and fail on these nodes.

`prepare` imports the PR-core ThunderAgent entrypoint before any long-lived
SLURM stages start. If it fails on a missing dependency such as `omegaconf`,
fix the selected `PYTHON_BIN` environment first; do not continue to `head`,
`ray`, or `rollout`.

`head` uses system Docker by default (`DOCKER_MODE=rootful`). If Docker Hub
rate-limits anonymous pulls, run `docker login` on the head/rollout node before
`head`, or keep `PREPULL_R2EGYM_IMAGES=true` so the wrapper fails early while
pre-pulling the 320 curated train/eval task images instead of silently masking
many Harbor trials during training. To inspect the exact image set:

```bash
python examples/train/thunder_agent/prepull_harbor_images.py \
  --train-data "$TRAIN_DATA" \
  --eval-data "$EVAL_DATA" \
  --mode list
```

For a temporary escape hatch while debugging non-Docker code paths, set
`PREPULL_R2EGYM_IMAGES=false`; do not use that for a production recipe run.

To install test-only dependencies into the same recipe venv, run:

```bash
INSTALL_DEV=1 bash examples/train/thunder_agent/setup_env.sh sync
```

---

## Verification Before a Full Run

Run this checklist before spending a full 10-epoch allocation. It validates the
same code path without starting training.

```bash
export REPO_ROOT=/path/to/skyrl-ta-pr-core
cd "$REPO_ROOT"
export WRAPPER="$REPO_ROOT/examples/train/thunder_agent/run_r2egym_32b_recipe.sh"
export RECIPE_VENV="/data/zy/models/$USER/venvs/skyrl-ta-pr-core-thunder-agent-vllm0201-cu129"
source <(RECIPE_VENV="$RECIPE_VENV" bash examples/train/thunder_agent/setup_env.sh print)

# These checks do not use GPUs.
RECIPE_VENV="$RECIPE_VENV" bash examples/train/thunder_agent/setup_env.sh check
bash -n \
  examples/train/thunder_agent/run_r2egym_32b_recipe.sh \
  examples/train/thunder_agent/run_harbor_thunder_agent_32b.sh \
  examples/train/thunder_agent/setup_env.sh \
  examples/train/thunder_agent/start_harbor_rollout_servers.sh \
  examples/train/thunder_agent/cleanup_harbor_docker.sh
"$PYTHON_BIN" -m py_compile \
  examples/train/thunder_agent/prepull_harbor_images.py \
  examples/train/thunder_agent/harbor_runtime_setup.py \
  examples/train/thunder_agent/main_harbor_thunder_agent.py \
  examples/train/thunder_agent/skyrl_integration/harbor_generator.py

# Requires the SLURM env variables from Step 2, but still does not use GPUs.
bash "$WRAPPER" prepare

# Optional: list the exact Docker images that `head` will pre-pull.
"$PYTHON_BIN" examples/train/thunder_agent/prepull_harbor_images.py \
  --train-data "$TRAIN_DATA" \
  --eval-data "$EVAL_DATA" \
  --mode list
```

Startup validation uses the real distributed services but still stops before
training:

```bash
export PREPULL_R2EGYM_IMAGES=false  # startup dry-run only
bash "$WRAPPER" cleanup-stage all
bash "$WRAPPER" head
bash "$WRAPPER" ray
bash "$WRAPPER" rollout
bash "$WRAPPER" status
bash "$WRAPPER" cleanup-stage all
```

The startup validation occupies the rollout node's 8 GPUs while `rollout` is
running. Turn `PREPULL_R2EGYM_IMAGES` back to `true` for production; otherwise
Docker Hub rate limits can turn Harbor setup failures into masked trajectories
during training.

Validated recipe status on 2026-05-04:

- A vLLM 0.20.1+cu129 environment imports cleanly on the GPU node with
  torch 2.11.0+cu129, Ray 2.51.1, PR-core's native vLLM server module, and
  native vLLM weight-transfer support.
- Four Qwen3-32B rollout servers became healthy on ports 18000-18003 using
  vLLM 0.20.1+cu129, PR-core's native server module, and `ROLLOUT_ENFORCE_EAGER=true`.
- Without eager, vLLM 0.20.1+cu129 loaded weights but failed during
  torch.compile/KV-cache initialization with
  `AttributeError: 'AlwaysHitShapeEnv' object has no attribute 'var_to_hint_override'`.
- The PR-core recipe completed training through `global_step=4`, including eval
  JSONL export and all 32 FSDP checkpoint shard files.
- A full clean 10-epoch proof still requires Docker Hub auth or a pre-populated
  Harbor image cache so the 320 curated task images do not hit anonymous pull
  limits.

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
| `PREPULL_R2EGYM_IMAGES` | `true` | Pull all train/eval Harbor images during `head` |

All training hyperparameters can still be overridden through environment
variables before invoking the wrapper, or by running the driver script directly
when Ray and rollout servers are already up:

```bash
bash examples/train/thunder_agent/run_harbor_thunder_agent_32b.sh full \
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

# Fleet Training Scripts

Shared scripts for Fleet GRPO training. Work with both SkyPilot (cloud) and bare clusters (RunPod pods, on-prem).

## Architecture

```
                         SkyPilot YAML (or bare cluster)
                         ┌──────────────────────────────┐
                         │  envs: FLEET_API_KEY, ...     │
                         │  setup: bash <task-setup>.sh  │
                         │  run:   bash <task-run>.sh    │
                         └──────┬───────────┬────────────┘
                                │           │
                    ┌───────────┘           └──────────┐
                    ▼                                   ▼
            SETUP PHASE                          RUN PHASE
            ═══════════                          ═════════

    ┌─────────────────────────┐       ┌─────────────────────────┐
    │   fleet-<task>-setup.sh │       │   fleet-<task>-run.sh   │
    │  (task-gen, tool-use…)  │       │  (task-gen, tool-use…)  │
    │                         │       │                         │
    │  • custom dataset prep  │       │  • hydra overrides      │
    │  • extra pip packages   │       │  • entrypoint/env-class │
    └────────┬────────────────┘       └────────┬────────────────┘
             │ bash (subshell)                  │ bash (subshell)
             ▼                                  ▼
    ┌─────────────────────────┐       ┌─────────────────────────┐
    │  fleet-common-setup.sh  │       │  fleet-common-run.sh    │
    │                         │       │                         │
    │  1. Validate env vars   │       │  1. NCCL / gIB fix      │
    │  2. uv venv + uv sync   │       │  2. Fabric Manager      │
    │  3. pip deps (wandb,    │       │  3. Ray cluster start   │
    │     boto3, litellm)     │       │  4. wandb login         │
    │  4. Extra pip packages  │       │  5. Build CMD_ARGS:     │
    │  5. Model-specific hook │       │     common overrides    │
    │     ┌─────────────────┐ │       │     + task overrides    │
    │     │ fleet-qwen35-   │ │       │     (after --)          │
    │     │ extra-setup.sh  │ │       │  6. Launch training     │
    │     │                 │ │       │  7. Crash diagnostics   │
    │     │ • transformers  │ │       └─────────────────────────┘
    │     │ • flash-attn    │ │
    │     │ • CUDA toolkit  │ │
    │     │ • causal-conv1d │ │
    │     │ • .cuda_env     │ │
    │     └─────────────────┘ │
    │  6. OpenEnv install     │
    │  7. S3 dataset download │
    │  8. prepare_dataset     │
    │     (or --skip-prepare) │
    └─────────────────────────┘
```

### Script Inventory

| Script | Layer | Purpose |
|--------|-------|---------|
| `fleet-common-setup.sh` | Shared | Env validation, venv, pip deps, OpenEnv, S3 download, prepare_dataset |
| `fleet-common-run.sh` | Shared | Ray cluster, wandb login, NCCL/gIB, Fabric Manager, training launch |
| `fleet-qwen35-extra-setup.sh` | Model | transformers 5.3.0, flash-attn, CUDA toolkit, causal-conv1d, `.cuda_env` |
| `fleet-task-gen-setup.sh` | Task | Task-gen dataset prep (schema/tools cache, Supabase) |
| `fleet-task-gen-run.sh` | Task | Task-gen hydra overrides (TP=2, evaluator config, multi-turn) |

### How Overrides Flow

```
fleet-common-run.sh injects:          fleet-<task>-run.sh appends:
────────────────────────────           ─────────────────────────────
data.train_data=...                    trainer.policy.model.path=...
data.val_data=...                      generator.num_inference_engines=4   ← overrides common
environment.env_class=...              trainer.train_batch_size=4
trainer.placement.*                    ++environment.skyrl_gym.task_gen.*
trainer.strategy=fsdp2                 ...
generator.num_inference_engines=$GPUS
trainer.ckpt_path=...

                    ──── later overrides win ────►
```

## Usage

### Via SkyPilot (cloud)

Task-gen:
```bash
sky launch skyrl-train/tasks/task-gen-grpo-qwen3_5-9b.yaml --env FLEET_API_KEY=sk_xxx --env WANDB_API_KEY=xxx
```

Tool-use:
```bash
sky launch skyrl-train/tasks/openenv-fleet-grpo-qwen3_5-9b.yaml --env FLEET_API_KEY=sk_xxx --env WANDB_API_KEY=xxx
```

### Via GitHub Actions

Trigger `Fleet Task Training (SkyPilot)` workflow with `task_name = task-gen-grpo-qwen3_5-9b`.

### Bare Cluster (RunPod, on-prem)

Run the same scripts directly — no SkyPilot needed. Set the env vars that SkyPilot would inject:

```bash
# Clone repo
git clone https://github.com/fleet-ai/SkyRL.git && cd SkyRL
git checkout deniz/multi-turn-task-gen  # or main for tool-use

# Set env vars (SkyPilot normally injects these)
export SKYPILOT_NUM_GPUS_PER_NODE=8
export SKYPILOT_NODE_IPS="$(hostname -i)"
# SKYPILOT_NUM_NODES and SKYPILOT_NODE_RANK default to 1 and 0

# Set training env vars
export FLEET_API_KEY="sk_xxx"
export WANDB_API_KEY="xxx"
export AWS_ACCESS_KEY_ID="xxx"
export AWS_SECRET_ACCESS_KEY="xxx"
export MODALITY="tool_use"
export DATA_VERSION="v4"
export INFERENCE_BACKEND="vllm"
export S3_DATASET_BUCKET="fleet-internal-datasets"
export LOGGER="wandb"

# Task-gen specific
export SUPABASE_URL="https://ehefoavidbttssbleuyv.supabase.co"
export SUPABASE_KEY="xxx"
export JUDGE_MODEL="anthropic/claude-sonnet-4-6"
export K_ROLLOUTS=4
export ALPHA="0.5"
export MAX_EVAL_STEPS=30
export MAX_TURNS=10
export MAX_INPUT_LENGTH=8192
export MAX_GENERATE_LENGTH=8192
export NUM_EPOCHS=20

# Setup then run (same scripts SkyPilot uses)
bash skyrl-train/scripts/fleet-task-gen-setup.sh
bash skyrl-train/scripts/fleet-task-gen-run.sh
```

This eliminates the double source of truth — no separate `setup.sh`/`train.sh` per pod.

Data root is auto-detected: `/workspace` if writable (RunPod), otherwise `$HOME` (GCP, Lambda).

## Setup Script Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--openenv-branch BRANCH` | `deniz/fleet_client` | OpenEnv git ref to install |
| `--extra-setup SCRIPT` | *(none)* | Script to source after `uv sync` (model-specific deps) |
| `--extra-pip PACKAGES` | *(none)* | Additional pip packages to install before extra-setup |
| `--data-root DIR` | auto-detect | Root for dataset download (`DIR/data/fleet/`) |
| `--skip-uv-isolated` | `false` | Flag for configs that use `python` directly |
| `--skip-prepare` | `false` | Skip default `prepare_dataset` (for custom preparation) |

## Run Script Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--data-root DIR` | auto-detect | Root for dataset files |
| `--ckpt-root DIR` | same as data-root | Root for checkpoints, exports, and tmp dir |
| `--use-python-direct` | `false` | Use `python -m` instead of `uv run --isolated` |
| `--cuda-env FILE` | *(none)* | Source this file for CUDA_HOME (e.g., `$HOME/.cuda_env`) |
| `--set-ulimit` | `false` | Set `ulimit -n 65536` (needed for Ray+vLLM) |
| `--no-pytorch-alloc-conf` | `false` | Skip `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| `--nccl-heartbeat SEC` | *(none)* | Set `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` |
| `--entrypoint MODULE` | `main_fleet` | Python module for training entry |
| `--env-class CLASS` | `fleet_task` | Environment class name |
| `--data-dir-name NAME` | `$MODALITY` | Data subdirectory under `data/fleet/` |
| `--` | | Everything after this is passed as hydra overrides |

## Common Hydra Overrides (auto-injected)

Set by `fleet-common-run.sh` — don't repeat unless overriding:

- `data.train_data`, `data.val_data` (from `--data-root` + `--data-dir-name`)
- `environment.env_class` (from `--env-class`)
- `environment.skyrl_gym.fleet_task.tasks_file` (only when `--env-class fleet_task`)
- `trainer.placement.colocate_all=true`, `trainer.strategy=fsdp2`
- `trainer.placement.{policy,ref}_num_gpus_per_node`, `{policy,ref}_num_nodes`
- `generator.num_inference_engines=$TOTAL_GPUS`
- `trainer.ckpt_path`, `trainer.export_path`

Later overrides win, so task scripts can override any of these.

## Environment Variables

### Required (all tasks)

| Variable | Description |
|----------|-------------|
| `FLEET_API_KEY` | Fleet API key |
| `WANDB_API_KEY` | Weights & Biases key |
| `AWS_ACCESS_KEY_ID` | S3 dataset download |
| `AWS_SECRET_ACCESS_KEY` | S3 dataset download |
| `MODALITY` | `tool_use` or `computer_use` |
| `DATA_VERSION` | S3 dataset version (e.g., `v54`) |
| `S3_DATASET_BUCKET` | Dataset bucket name |
| `INFERENCE_BACKEND` | `vllm` or `sglang` |

### Required (bare cluster only)

| Variable | Default | Description |
|----------|---------|-------------|
| `SKYPILOT_NUM_GPUS_PER_NODE` | *(none)* | GPUs per node (e.g., `8`) |
| `SKYPILOT_NODE_IPS` | *(none)* | Space-separated node IPs (head first) |
| `SKYPILOT_NUM_NODES` | `1` | Number of nodes |
| `SKYPILOT_NODE_RANK` | `0` | This node's rank (0 = head) |

### Task-gen specific

| Variable | Default | Description |
|----------|---------|-------------|
| `JUDGE_MODEL` | *(required)* | LLM-as-judge for task validation |
| `K_ROLLOUTS` | *(required)* | Number of evaluator rollouts per task |
| `ALPHA` | *(required)* | Weight for variance vs hint gap |
| `MAX_EVAL_STEPS` | *(required)* | Max steps per evaluator rollout |
| `SUPABASE_URL` | *(required)* | Supabase URL for DB schema discovery |
| `SUPABASE_KEY` | *(required)* | Supabase service role key |
| `OPENROUTER_API_KEY` | *(optional)* | For LLM judge via OpenRouter |

## Adding a New Task Type

1. Create `fleet-<task>-setup.sh`: call `fleet-common-setup.sh` with flags, then custom dataset prep
2. Create `fleet-<task>-run.sh`: call `fleet-common-run.sh` with `--entrypoint`, `--env-class`, and hydra overrides
3. Create `tasks/<task>.yaml`: thin wrapper — just `envs:`, `resources:`, and 2-line `setup:`/`run:`
4. Add the task name to `.github/workflows/openenv-fleet-train.yaml`

## Adding a New Model

1. Create `fleet-<model>-extra-setup.sh` for model-specific deps
2. Reference it via `--extra-setup` in the task setup script
3. Update hydra overrides in the task run script (model path, flash_attn, batch sizes, etc.)

## GCP Notes

GCP requires the correct GPU image for H200/B200 (default SkyPilot image has driver 535, needs 550+):

```yaml
- accelerators: H200:8
  cloud: gcp
  image_id: projects/deeplearning-platform-release/global/images/common-cu128-ubuntu-2204-nvidia-570-v20260305
```

The run script auto-handles GCP's gIB/NCCL quirks (strips gIB when no RDMA devices present).

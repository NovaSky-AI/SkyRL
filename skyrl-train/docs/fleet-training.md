# SkyRL Fleet Training Guide

RL training on Fleet-hosted environments using SkyPilot for GPU provisioning.

## Quick Start

**Run training via GitHub Actions:**
1. Go to **Actions** → **"Fleet Task Training (SkyPilot)"** → **"Run workflow"**
2. Select `modality` (`tool_use` or `computer_use`), optionally filter by `env_key`, and set `task_name` to point to a YAML file in `skyrl-train/tasks/`
3. Training provisions H100, logs to WandB, uploads checkpoints to S3

**Run training via CLI:**
```bash
sky launch skyrl-train/tasks/openenv-fleet-grpo.yaml \
    --env FLEET_API_KEY=$FLEET_API_KEY \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env MODALITY=tool_use \
    -d -y
```

## Available Tasks (4,881 total)

| Environment | tool_use | computer_use | Total |
|-------------|----------|--------------|-------|
| github | 1,865 | 0 | 1,865 |
| booking | 1,089 | 0 | 1,089 |
| google-maps | 13 | 739 | 752 |
| reddit | 246 | 24 | 270 |
| ticketmaster | 222 | 25 | 247 |
| amazon | 0 | 145 | 145 |
| rops | 0 | 113 | 113 |
| zillow | 76 | 30 | 106 |
| walmart | 0 | 82 | 82 |
| dmv | 0 | 55 | 55 |
| fira | 54 | 0 | 54 |
| google-flights | 0 | 33 | 33 |
| instacart | 0 | 32 | 32 |
| outlook | 24 | 0 | 24 |
| hubspot | 12 | 0 | 12 |
| dropbox | 2 | 0 | 2 |

## Dataset Split Strategy

The `prepare_dataset.py` script creates train/eval splits with the following strategy:

**Stratified by Environment:** Each environment maintains the same train/eval ratio (default 98/2), ensuring balanced representation across all environments.

**Hash-based Deterministic Assignment:** Tasks are assigned to splits using MD5 hash of task key, ensuring the same task always goes to the same split across runs.

**Minimum Eval Threshold:** Environments with fewer than 5 expected eval samples have all tasks go to train (prevents statistically unreliable eval sets).

**Held-out Eval Environments:**
| Modality | Held-out Env |
|----------|--------------|
| tool_use | outlook |
| computer_use | instacart |

These environments go entirely to eval (not used in training).

## GitHub Secrets

| Secret | Required | Description |
|--------|----------|-------------|
| `FLEET_API_KEY` | Yes | Fleet API access |
| `WANDB_API_KEY_TOOL_USE` | Yes | WandB for tool_use |
| `WANDB_API_KEY_COMPUTER_USE` | Yes | WandB for computer_use |
| `LAMBDA_API_KEY` | Yes | Lambda Labs GPU |
| `RUNPOD_API_KEY` | Yes | RunPod GPU |
| `VAST_API_KEY` | Yes | Vast.ai GPU |
| `PRIME_INTELLECT_API_KEY` | Yes | Prime Intellect GPU |
| `AWS_ACCESS_KEY_ID` | Yes | S3 dataset download & checkpoint upload |
| `AWS_SECRET_ACCESS_KEY` | Yes | S3 dataset download & checkpoint upload |

## Key Configuration

Edit `skyrl-train/tasks/openenv-fleet-grpo.yaml`:

| What to change | Parameter |
|----------------|-----------|
| Base model | `trainer.policy.model.path` (default: `Qwen/Qwen2.5-1.5B-Instruct`) |
| Algorithm | `trainer.algorithm.advantage_estimator` (default: `grpo`) |
| Learning rate | `trainer.policy.optimizer_config.lr` (default: `1e-6`) |
| Batch size | `trainer.train_batch_size` (default: `4`) |
| Epochs | `trainer.epochs` (default: `20`) |
| Max turns | `generator.max_turns` (default: `50`) |
| Checkpoint interval | `trainer.ckpt_interval` (default: `10`) |

## Checkpoints

Uploaded to S3 (if AWS credentials set):
```
s3://skyrl-checkpoints/{project}/{model}/{run_name}/global_step_N/
```
Local checkpoints deleted after upload to save disk.

## Monitoring

- **WandB:** https://wandb.ai/thefleet/fleet-task-grpo
- **Slack:** `#fleet-training-runs`
- **Logs:** `sky logs --follow <cluster-name>`
- **Status:** `sky status`

## Files Reference

| File | Purpose |
|------|---------|
| `.github/workflows/openenv-fleet-train.yaml` | GitHub Actions workflow |
| `skyrl-train/tasks/openenv-fleet-grpo.yaml` | SkyPilot task config |
| `skyrl-train/integrations/fleet/env.py` | Fleet environment (BaseTextEnv) |
| `skyrl-train/integrations/fleet/entrypoints/main_fleet.py` | Training entrypoint |
| `skyrl-train/integrations/fleet/s3_checkpoints.py` | S3 upload module |
| `skyrl-train/integrations/fleet/prepare_dataset.py` | Dataset prep |
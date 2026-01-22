# Fleet Task Training Guide

This guide explains how to trigger RL training on Fleet-hosted environments using SkyPilot for GPU provisioning.

## Overview

The training pipeline:
1. **GitHub Actions** triggers a workflow
2. **SkyPilot** provisions a GPU instance (H100) on Lambda/RunPod/Vast
3. **SkyRL** runs training using Fleet environments (supports GRPO, PPO, etc.)
4. **WandB** logs metrics and training progress

## Prerequisites

### GitHub Secrets Required

The following secrets must be configured in the repository settings:

| Secret | Description |
|--------|-------------|
| `FLEET_API_KEY` | Fleet API key for environment access |
| `WANDB_API_KEY_TOOL_USE` | WandB API key for tool_use modality runs |
| `WANDB_API_KEY_COMPUTER_USE` | WandB API key for computer_use modality runs |
| `LAMBDA_API_KEY` | Lambda Labs API key |
| `RUNPOD_API_KEY` | RunPod API key |
| `VAST_API_KEY` | Vast.ai API key |
| `SLACK_BOT_TOKEN` | (Optional) Slack bot token for notifications |

### Local Testing

For local testing, set environment variables:
```bash
export FLEET_API_KEY="sk_..."
export WANDB_API_KEY="..."
```

## Triggering a Training Run

### Via GitHub Actions (Recommended)

1. Go to **Actions** tab in the SkyRL repository
2. Select **"Fleet Task Training (SkyPilot)"** workflow
3. Click **"Run workflow"**
4. Configure options:

| Option | Description | Default |
|--------|-------------|---------|
| `modality` | Task modality: `tool_use` or `computer_use` | `tool_use` |
| `env_key` | Filter by environment (e.g., `booking`, `github`) | empty (all) |
| `max_tasks` | Limit number of tasks for testing | empty (all) |
| `cloud` | Preferred cloud: `any`, `lambda`, `runpod`, `vast` | `any` |

5. Click **"Run workflow"**

### Via CLI (Manual)

```bash
# Install SkyPilot
pip install "skypilot[lambda,runpod,vast]"

# Configure credentials
mkdir -p ~/.lambda_cloud
echo "api_key = YOUR_LAMBDA_KEY" > ~/.lambda_cloud/lambda_keys

# Launch training
sky launch skyrl-train/tasks/openenv-fleet-grpo.yaml \
    --env FLEET_API_KEY=$FLEET_API_KEY \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env MODALITY=tool_use \
    -d -y
```

## Configuration

### SkyPilot Task YAML

The main configuration is in `skyrl-train/tasks/openenv-fleet-grpo.yaml`:

```yaml
resources:
  accelerators: H100:1      # GPU type and count
  disk_size: 40             # Disk size in GB

envs:
  MODALITY: "tool_use"      # or "computer_use"
  MAX_TURNS: 50             # Max agent turns per episode
  NUM_EPOCHS: 20            # Training epochs
```

### Training Hyperparameters

Key training parameters (in the YAML `run` section):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `trainer.algorithm.advantage_estimator` | `grpo` | Algorithm (grpo, ppo, etc.) |
| `trainer.policy.model.path` | `Qwen/Qwen2.5-1.5B-Instruct` | Base model |
| `trainer.train_batch_size` | 4 | Training batch size |
| `trainer.epochs` | 20 | Number of epochs |
| `generator.n_samples_per_prompt` | 4 | Samples per prompt |
| `generator.max_turns` | 50 | Max agent turns |
| `trainer.policy.optimizer_config.lr` | 1e-6 | Learning rate |

To change the algorithm, modify `trainer.algorithm.advantage_estimator` in the YAML.

### Sample Dataset

The training uses a pre-committed sample dataset:
- **File:** `skyrl-train/data/fleet_booking_sample.json`
- **Tasks:** 100 booking environment tasks
- **Split:** 90/10 train/validation

## Monitoring

### WandB Dashboard

Training metrics are logged to WandB:
- **Project:** `fleet-task-grpo`
- **URL:** https://wandb.ai/thefleet/fleet-task-grpo

### Slack Notifications

If configured, notifications are sent to `#fleet-training-runs`:
- Run started
- Run completed
- Run failed

### SkyPilot Status

Check cluster status:
```bash
sky status
sky logs <cluster-name>
```

## Troubleshooting

### Common Errors

**`FleetVersionNotFoundError`**
- The environment version doesn't exist in ECR
- Solution: Use versions available in the Fleet API

**`ModuleNotFoundError: No module named 'envs'`**
- OpenEnv not installed correctly
- Solution: Ensure `git+https://github.com/fleet-ai/OpenEnv.git@deniz/fleet_client` is in dependencies

**Cluster won't start**
- Check cloud credentials: `sky check lambda runpod vast`
- Check GPU availability on the selected cloud

### Logs

View training logs:
```bash
# Stream logs
sky logs --follow <cluster-name>

# Get cluster name from workflow output or:
sky status
```

### Cleanup

Terminate a running cluster:
```bash
sky down <cluster-name> -y
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     GitHub Actions                          │
│  .github/workflows/openenv-fleet-train.yaml                │
└─────────────────────┬───────────────────────────────────────┘
                      │ sky launch
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                      SkyPilot                               │
│  Provisions H100 on Lambda/RunPod/Vast                     │
└─────────────────────┬───────────────────────────────────────┘
                      │ runs
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              skyrl-train/tasks/openenv-fleet-grpo.yaml     │
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │   Setup     │───▶│  Prepare    │───▶│    Train    │    │
│  │  (uv sync)  │    │  Dataset    │    │   (GRPO)    │    │
│  └─────────────┘    └─────────────┘    └─────────────┘    │
│                                               │             │
└───────────────────────────────────────────────┼─────────────┘
                                                │
                      ┌─────────────────────────┼─────────────┐
                      │                         ▼             │
                      │  ┌─────────────────────────────────┐ │
                      │  │         Fleet API               │ │
                      │  │  (Environment provisioning)     │ │
                      │  └─────────────────────────────────┘ │
                      │                                       │
                      │  ┌─────────────────────────────────┐ │
                      │  │           WandB                 │ │
                      │  │    (Metrics & logging)         │ │
                      │  └─────────────────────────────────┘ │
                      └───────────────────────────────────────┘
```

## Files Reference

| File | Purpose |
|------|---------|
| `.github/workflows/openenv-fleet-train.yaml` | GitHub Actions workflow |
| `skyrl-train/tasks/openenv-fleet-grpo.yaml` | SkyPilot task definition |
| `skyrl-train/integrations/fleet/env.py` | SkyRL environment wrapper |
| `skyrl-train/integrations/fleet/prepare_dataset.py` | Dataset preparation |
| `skyrl-train/integrations/fleet/entrypoints/main_fleet.py` | Training entrypoint |
| `skyrl-train/data/fleet_booking_sample.json` | Sample task dataset |

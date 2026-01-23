# Fleet Task Training Guide

This guide explains how to trigger RL training on Fleet-hosted environments using SkyPilot for GPU provisioning.

## Overview

The training pipeline:
1. **GitHub Actions** triggers a workflow
2. **SkyPilot** provisions a GPU instance (H100) on Lambda/RunPod/Vast
3. **SkyRL** runs training using Fleet environments (supports GRPO, PPO, etc.)
4. **WandB** logs metrics and training progress
5. **S3** stores checkpoints (optional, prevents disk exhaustion)

## Available Tasks

### Task Breakdown by Modality

| Modality | Total Tasks |
|----------|-------------|
| tool_use | 3,603 |
| computer_use | 1,278 |
| **Total** | **4,881** |

### Task Breakdown by Environment

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
| **Total** | **3,603** | **1,278** | **4,881** |

**Notes:**
- `github` and `booking` are tool_use only
- `amazon`, `rops`, `walmart`, `dmv`, `google-flights`, `instacart` are computer_use only
- Currently training uses sample dataset (`fleet_booking_sample.json` with 100 booking tasks)

## Prerequisites

### GitHub Secrets Required

The following secrets must be configured in the repository settings:

| Secret | Description | Required |
|--------|-------------|----------|
| `FLEET_API_KEY` | Fleet API key for environment access | Yes |
| `WANDB_API_KEY_TOOL_USE` | WandB API key for tool_use modality runs | Yes |
| `WANDB_API_KEY_COMPUTER_USE` | WandB API key for computer_use modality runs | Yes |
| `LAMBDA_API_KEY` | Lambda Labs API key | Yes |
| `RUNPOD_API_KEY` | RunPod API key | Yes |
| `VAST_API_KEY` | Vast.ai API key | Yes |
| `AWS_ACCESS_KEY_ID` | AWS access key for S3 checkpoint upload | No |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key for S3 checkpoint upload | No |
| `SLACK_BOT_TOKEN` | Slack bot token for notifications | No |

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

### Via SkyPilot (Remote GPU)

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
  disk_size: 100            # Disk size in GB

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
| `trainer.ckpt_interval` | 10 | Save checkpoint every N steps |

To change the algorithm, modify `trainer.algorithm.advantage_estimator` in the YAML.

### Sample Dataset

The training uses a pre-committed sample dataset:
- **File:** `skyrl-train/data/fleet_booking_sample.json`
- **Tasks:** 100 booking environment tasks
- **Split:** 90/10 train/validation

## S3 Checkpoint Upload

To prevent disk exhaustion on cloud instances, checkpoints can be automatically uploaded to S3.

### Setup

1. Add AWS credentials to GitHub Secrets:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`

2. Checkpoints are uploaded to:
   ```
   s3://skyrl-checkpoints/{project_name}/{model_name}/{run_name}/global_step_N/
   ```

3. Local checkpoints are deleted after successful upload to save disk space.

### S3 Bucket Structure

```
s3://skyrl-checkpoints/
  └── fleet-task-grpo/                          # project_name
      ├── Qwen2.5-1.5B-Instruct/                # model_name
      │   └── fleet_tool_use_booking_pr0/       # run_name
      │       ├── global_step_10/
      │       │   ├── policy/
      │       │   ├── critic/
      │       │   ├── data.pt
      │       │   └── trainer_state.pt
      │       ├── global_step_20/
      │       └── ...
      └── Qwen2.5-7B-Instruct/                  # different model
          └── ...
```

### Without S3

If AWS credentials are not configured:
- Training continues normally
- Checkpoints are stored locally on 100GB disk
- Disk may fill up on long training runs

## Monitoring

### WandB Dashboard

Training metrics are logged to WandB:
- **Project:** `fleet-task-grpo`
- **URL:** https://wandb.ai/thefleet/fleet-task-grpo

### Slack Notifications

If configured, notifications are sent to `#fleet-training-runs-test`:
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

**`OSError: [Errno 28] No space left on device`**
- Disk filled up with checkpoints
- Solution: Add AWS credentials for S3 checkpoint upload, or increase `trainer.ckpt_interval`

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
                      │                                       │
                      │  ┌─────────────────────────────────┐ │
                      │  │      S3 (optional)              │ │
                      │  │  (Checkpoint storage)          │ │
                      │  └─────────────────────────────────┘ │
                      └───────────────────────────────────────┘
```

## Files Reference

| File | Purpose |
|------|---------|
| `.github/workflows/openenv-fleet-train.yaml` | GitHub Actions workflow |
| `skyrl-train/tasks/openenv-fleet-grpo.yaml` | SkyPilot task for GPU provisioning |
| `skyrl-train/integrations/fleet/env.py` | Fleet environment wrapper (BaseTextEnv) |
| `skyrl-train/integrations/fleet/entrypoints/main_fleet.py` | Training entrypoint |
| `skyrl-train/integrations/fleet/s3_checkpoints.py` | S3 checkpoint upload module |
| `skyrl-train/integrations/fleet/prepare_dataset.py` | Dataset preparation script |
| `skyrl-train/data/fleet_booking_sample.json` | Sample task dataset (100 booking tasks) |

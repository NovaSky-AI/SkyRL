# Fleet Training Runbook

Quick reference for launching training runs on Fleet tasks.

## Backends

| Backend | Infrastructure | Best For |
|---------|---------------|----------|
| **SkyRL (SkyPilot)** | Self-managed GPU (Lambda/RunPod/Vast) | Full control, longer runs |
| **Tinker** | Hosted GPU (Theseus) | Quick iteration, no GPU setup |

## Launching a Run

### Via GitHub Actions (Recommended)

1. Go to **Actions** → Select workflow:
   - `Fleet Task Training (SkyPilot)` - SkyRL backend
   - `Fleet Task Training (Tinker)` - Tinker backend

2. Click **Run workflow** and configure:

#### Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `modality` | `tool_use` or `computer_use` | `tool_use` |
| `env_key` | Filter by environment (empty=all) | `""` |
| `max_tasks` | Limit tasks for testing | `""` (all) |

#### SkyRL-Specific

| Parameter | Description | Default |
|-----------|-------------|---------|
| `cloud` | `any`, `lambda`, `runpod`, `vast` | `any` |
| `task_name` | SkyPilot YAML in `skyrl-train/tasks/` | `openenv-fleet-grpo` |

#### Tinker-Specific

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model_name` | Model to train | `Qwen/Qwen3-8B` |
| `max_steps` | Training steps | `50` |
| `batch_size` | Batch size | `8` |
| `n_samples_per_prompt` | GRPO samples per prompt | `4` |

## Monitoring

### Slack Notifications

Channel: `#fleet-training-runs`

- Started: Run name, backend, modality, config
- Completed: Final status, WandB link
- Failed: Error logs link
- Cancelled: Cleanup status

### WandB Dashboards

| Backend | Project |
|---------|---------|
| SkyRL | [fleet-task-grpo](https://wandb.ai/thefleet/fleet-task-grpo) |
| Tinker | [fleet-tinker-grpo](https://wandb.ai/thefleet/fleet-tinker-grpo) |

### Key Metrics

- `reward/avg_pass_at_1` - Primary success metric
- `reward/avg_raw_reward` - Average reward per episode
- `reward/{env_key}/pass_at_1` - Per-environment breakdown

## Testing a Small Run

Quick validation before full training:

```
modality: tool_use
env_key: github          # Single environment
max_tasks: 4             # Few tasks
max_steps: 10            # Tinker only
```

## Troubleshooting

### MCP Connection Timeouts

```
httpx.ConnectTimeout
ERROR mcp.client.streamable_http: Error in post_writer
```

**Cause**: Too many concurrent Fleet environment connections.

**Fix**: Tinker uses `max_concurrent=2` semaphore to limit parallel connections.

### "No tools found" Errors

```
RuntimeError: Task X: no tools found in observation
```

**Cause**: MCP client failed to fetch tools from Fleet.

**Fix**: OpenEnv retries with exponential backoff (3 attempts).

### SkyPilot Cluster Issues

```bash
# Check cluster status
sky status

# View logs
sky logs <cluster-name>

# Force cleanup
sky down <cluster-name> -y
```

## Required Secrets

| Secret | Used By |
|--------|---------|
| `FLEET_API_KEY` | Both |
| `WANDB_API_KEY_TOOL_USE` | Both (tool_use modality) |
| `WANDB_API_KEY_COMPUTER_USE` | Both (computer_use modality) |
| `TINKER_API_KEY` | Tinker |
| `LAMBDA_API_KEY` | SkyRL |
| `RUNPOD_API_KEY` | SkyRL |
| `VAST_API_KEY` | SkyRL |
| `AWS_ACCESS_KEY_ID` | SkyRL (S3 datasets) |
| `AWS_SECRET_ACCESS_KEY` | SkyRL (S3 datasets) |
| `SLACK_BOT_TOKEN` | Notifications |

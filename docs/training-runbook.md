# Fleet Training Runbook

Quick reference for launching training runs on Fleet tasks.

## Backends

| Backend | Infrastructure | Best For |
|---------|---------------|----------|
| **SkyRL (SkyPilot)** | Self-managed GPU (Lambda/RunPod/Vast) | Full control, longer runs |
| **Tinker** | Hosted GPU| Quick iteration, no GPU setup |

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
| `data_version` | Dataset version in S3 (e.g., `v0.3`) | `v0.3` |
| `enable_context_tools` | Enable context management tools (`check_context`, `manage_context`, etc.) | `false` |

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

## Context Management Tools

When `enable_context_tools=true`, the model gets 4 additional tools for managing long contexts:

| Tool | Description |
|------|-------------|
| `check_context` | Check current token count and context budget |
| `manage_context` | Drop old turns to free up context space |
| `search_tool_output` | Search within truncated tool outputs |
| `view_tool_output` | View specific lines of truncated tool outputs |

**Requirements for context tools:**
- `step_wise_trajectories=true` (set automatically in YAML)
- `inject_context_status=true` (adds context status to observations)
- YaRN rope scaling extends context to 65K tokens

**When to use:** Enable for environments with long tool schemas (e.g., GitHub with 100+ tools) where the model needs to proactively manage context before hitting the 48K limit.

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
| `AWS_ACCESS_KEY_ID` | Both (S3 datasets) |
| `AWS_SECRET_ACCESS_KEY` | Both (S3 datasets) |
| `SLACK_BOT_TOKEN` | Notifications |

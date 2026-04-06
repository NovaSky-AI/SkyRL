# OpenReward + SkyRL Integration

Train language models on [OpenReward](https://openreward.ai) environments using SkyRL's GRPO trainer.

## Prerequisites

- [Modal](https://modal.com) account and CLI (`pip install modal && modal setup`)
- OpenReward API key from [openreward.ai/keys](https://openreward.ai/keys)
- (Optional) WandB API key for logging

## Quick Start (Modal)

### 1. Generate dataset

```bash
MODAL_GPU=L4:1 modal run examples/train_integrations/modal/main.py \
  --command "pip install openreward pyarrow && \
    OPENREWARD_API_KEY=<your-key> python examples/train_integrations/openreward/prepare_tasks.py \
    --env GeneralReasoning/WhoDunit --split train --max-tasks 50 \
    --output /root/data/openreward/train.parquet"
```

### 2. Run training

```bash
MODAL_GPU=A100:4 modal run examples/train_integrations/modal/main.py \
  --command "pip install openreward && \
    OPENREWARD_API_KEY=<your-key> WANDB_API_KEY=<your-key> \
    OPENREWARD_UPLOAD_ROLLOUT=true \
    bash examples/train_integrations/openreward/run_openreward.sh"
```

### 3. (Optional) Override config

```bash
MODAL_GPU=A100:4 modal run examples/train_integrations/modal/main.py \
  --command "pip install openreward && \
    OPENREWARD_API_KEY=<your-key> \
    OPENREWARD_UPLOAD_ROLLOUT=true \
    LOGGER=console \
    bash examples/train_integrations/openreward/run_openreward.sh \
    trainer.epochs=2 generator.max_turns=8"
```

### Environment Variables

| Variable                    | Default            | Description                                                   |
| --------------------------- | ------------------ | ------------------------------------------------------------- |
| `OPENREWARD_API_KEY`        | (required)         | API key from [openreward.ai/keys](https://openreward.ai/keys) |
| `OPENREWARD_UPLOAD_ROLLOUT` | `true`             | Whether to upload rollouts to OpenReward for visualization    |
| `OPENREWARD_RUN_NAME`       | `skyrl-openreward` | Run name used for rollout uploads                             |

## Training Config Summary

| Parameter              | Default                  | Description                                          |
| ---------------------- | ------------------------ | ---------------------------------------------------- |
| `NUM_GPUS`             | 4                        | Number of GPUs (colocated: policy + ref + inference) |
| `MODEL`                | Qwen/Qwen2.5-3B-Instruct | Base model                                           |
| `train_batch_size`     | 32                       | Unique prompts per training step                     |
| `n_samples_per_prompt` | 4                        | Rollouts per prompt (GRPO group size)                |
| `max_turns`            | 10                       | Max agent-environment interaction turns              |
| `stop`                 | `["</tool_call>"]`       | Stop string for generation                           |

Total rollouts per step: 32 × 4 = 128.

## File Structure

```
openreward/
├── prepare_tasks.py          # Fetch tasks from OpenReward API → Parquet dataset
├── env.py                    # OpenRewardEnv(BaseTextEnv) adapter
├── entrypoints/
│   └── main_openreward.py    # Register env + launch training
└── run_openreward.sh         # Training launch script
```

## How It Works

1. **`prepare_tasks.py`** queries OpenReward for tasks, creates a temporary session per task to fetch the initial prompt and available tools, then writes a Parquet dataset.

2. **`OpenRewardEnv`** implements `BaseTextEnv`. On `init()`, it opens an OpenReward session. On each `step()`, it parses `<tool_call>` from the model output, calls `session.call_tool()`, and returns the result as `<tool_response>`.

3. SkyRL's `agent_loop` handles the multi-turn generation loop: generate → stop at `</tool_call>` → env.step() → append observation → repeat.

## Adding More Environments

The `env_class` is always `"openreward"` — the specific environment is determined by `env_name` in the dataset. To train on multiple environments:

```bash
python prepare_tasks.py \
  --env GeneralReasoning/WhoDunit \
  --env GeneralReasoning/CTF \
  --split train --max-tasks 50 \
  --output /root/data/openreward/train.parquet
```

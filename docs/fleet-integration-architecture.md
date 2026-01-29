# Fleet Integration Architecture

This document describes how Fleet environments integrate with SkyRL and Tinker training loops.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Training Loops                                  │
├───────────────────────────────────┬─────────────────────────────────────────┤
│      SkyRL Training Loop          │         Tinker Training Loop            │
│   (grpo_trainer.py +              │      (main_fleet_tinker.py)             │
│    skyrl_gym_generator.py)        │                                         │
│                                   │                                         │
│  • generate_batched() collects    │  • collect_batch_rollouts() collects    │
│    rollouts via vLLM (GPU)        │    rollouts via Tinker inference API    │
│  • Computes GRPO advantages       │  • Computes GRPO advantages             │
│  • Trainer runs forward/backward  │  • prepare_training_data() builds batch │
│    on local GPU                   │  • Tinker forward_backward() (hosted)   │
│                                   │                                         │
└───────────────────────────────────┴─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SkyRL FleetTaskEnv Wrapper                               │
│                  (integrations/fleet/env.py)                                │
│                                                                             │
│  FleetTaskEnv (extends BaseTextEnv)                                         │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  __init__(env_config, extras)                                               │
│    • Loads task config from JSON file (tasks_file)                          │
│    • Validates task_key exists in config                                    │
│    • Sets up API key and TTL for Fleet environments                         │
│                                                                             │
│  init() / init_async()                                                      │
│    • Creates OpenEnv FleetTaskEnv instance                                  │
│    • Calls openenv.reset_async() to create Fleet env and fetch tools        │
│    • Builds system prompt with tool definitions and date context            │
│    • Returns initial chat_history: [system_msg, user_task_prompt]           │
│                                                                             │
│  step() / step_async(action)                                                │
│    • Parses tool call from LLM response (<tool_call>...</tool_call>)        │
│    • Executes tool via openenv.step_async() → Fleet platform                │
│    • Tracks turns, tool_calls for metrics                                   │
│    • Returns BaseTextEnvStepOutput with observation, reward, done           │
│                                                                             │
│  close()                                                                    │
│    • Cleanup Fleet environment resources                                    │
│                                                                             │
│  State: chat_history, tools, turns, tool_calls, openenv_task_env            │
│  Note: sync methods wrap async methods with asyncio.run()                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OpenEnv FleetTaskEnv                                │
│                    (envs/fleet_env/task_env.py)                             │
│                                                                             │
│  reset_async()                                                              │
│    • Creates Fleet environment via fleet.make()                             │
│    • Fetches available tools via MCP list_tools                             │
│    • Returns observation with tools list                                    │
│                                                                             │
│  step_async(action)                                                         │
│    • Executes tool call on Fleet environment                                │
│    • Runs verifier if episode done (reward computation)                     │
│    • Returns (observation, reward, done, info)                              │
│                                                                             │
│  close()                                                                    │
│    • Destroys Fleet environment instance                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ HTTP/MCP
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Fleet Platform                                    │
│                                                                             │
│  Hosted task environments with MCP tool interfaces:                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Outlook    │  │    GitHub    │  │   Booking    │  │    Reddit    │    │
│  │  (email,     │  │  (repos,     │  │  (search,    │  │  (posts,     │    │
│  │  calendar)   │  │  issues, PR) │  │  reserve)    │  │  comments)   │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Abstractions

| Layer | Component | Purpose |
|-------|-----------|---------|
| Training | `main_fleet_tinker.py` | Tinker-based training: hosted inference + training |
| Training | `grpo_trainer.py` + `skyrl_gym_generator.py` | SkyRL training: vLLM inference (GPU) + local training |
| Wrapper | `FleetTaskEnv` (SkyRL) | Chat history management, tool call parsing, prompt construction |
| Env | `FleetTaskEnv` (OpenEnv) | Low-level Fleet API: env creation, tool execution, verifier |
| Platform | Fleet | Hosted task environments (Outlook, GitHub, etc.) |

## Dataset Configuration

Both training loops use the same dataset preparation process:

### Data Versioning

Tasks are stored in S3 at `s3://fleet-internal-datasets/{data_version}/openenv/`:
- `all_tool_use.json` - Tool use tasks
- `all_computer_use.json` - Computer use tasks

The `data_version` parameter (e.g., `v0.1`, `v0.2`) is configurable in both workflows and validated against S3 before training starts.

### Train/Test Split (`prepare_dataset.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `eval_ratio` | 10% | Fraction for evaluation |
| `MAX_EVAL_SAMPLES` | 30 | Cap per environment |
| `MIN_EVAL_SAMPLES` | 1 | Minimum to create eval split |

Split strategy:
- **Stratified by environment**: Each env maintains the train/eval ratio
- **Hash-based assignment**: Deterministic (same task → same split)
- **Held-out envs**: `instacart` (computer_use only) goes to eval only

## Communication Flow

### Tinker Training Loop
```python
# 1. Create environment
env = FleetTaskEnv(env_config, extras={"task_key": task_key, ...})

# 2. Initialize - creates Fleet env, fetches tools, builds system prompt
chat_history, metadata = await env.init_async([])

# 3. Rollout loop
while not done:
    input_ids = tokenize(env.chat_history)
    output = tinker_sample(input_ids)           # Tinker hosted inference
    step_output = await env.step_async(output)  # Parse tool call, execute on Fleet
    done = step_output.done

# 4. Cleanup
env.close()
```

### SkyRL Training Loop
```python
# 1. Create environment
env = skyrl_gym.make("fleet_task", env_config, extras)

# 2. Initialize (uses asyncio.run internally)
chat_history, metadata = env.init(prompt)

# 3. Rollout loop
while not done:
    output = inference_engine.generate(chat_history)  # Local vLLM inference
    step_output = env.step(output)                    # Uses asyncio.run internally
    done = step_output.done

# 4. Cleanup
env.close()
```

## Data Flow

```
┌────────────┐    prompt     ┌────────────┐    action     ┌────────────┐
│  Training  │ ───────────▶  │  FleetTask │ ───────────▶  │   Fleet    │
│    Loop    │               │    Env     │               │  Platform  │
│            │ ◀───────────  │  (SkyRL)   │ ◀───────────  │            │
└────────────┘  chat_history └────────────┘  observation  └────────────┘
                step_output
```

## Key Data Structures

**init() / init_async() returns:**
- `chat_history`: List of messages `[{role, content}, ...]`
  - System message with tools JSON and instructions
  - User message with task prompt
- `metadata`: `{task_key, env_key, tools, modality}`

**step() / step_async() returns:**
- `BaseTextEnvStepOutput`:
  - `observations`: New messages to append (tool result or error)
  - `reward`: Float (0.0 during rollout, verifier reward on done)
  - `done`: Boolean (agent said `<done>` or max_turns reached)
  - `metadata`: `{task_key, turn, tool_call, tool_result, error}`

## Training Data Preparation

Both training loops follow the same pattern for preparing training data:

1. **DAPO Overlong Filtering**: Zero out loss mask for responses not ending with EOS token
2. **Sequence Truncation**: Truncate to `max_sequence_length`, preserving prompt
3. **Advantage Computation**: GRPO advantages from rewards grouped by prompt
4. **Build Training Batch**: Combine into format expected by trainer

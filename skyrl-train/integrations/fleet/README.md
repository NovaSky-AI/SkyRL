# Fleet Integration for SkyRL

This module provides a SkyRL-compatible environment wrapper for Fleet-hosted tasks using OpenEnv as the abstraction layer.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ SkyRL FleetTaskEnv (integrations/fleet/env.py)                              │
│                                                                             │
│  __init__(env_config, extras)                                               │
│      └── Stores config, task_key, api_key                                   │
│      └── self.openenv_task_env = None (not created yet)                     │
│                                                                             │
│  init(prompt)  ←── Called by SkyRL trainer to start episode                 │
│      │                                                                      │
│      ├── OpenEnvFleetTaskEnv(task_config, ...)                              │
│      │       ├── fleet.make()      ←── Creates Fleet env (provisions VM)    │
│      │       └── list_tools()      ←── Fetches & caches tools               │
│      │                                                                      │
│      ├── reset_async()                                                      │
│      │       ├── _orch.reset()     ←── Resets episode state                 │
│      │       └── Returns cached tools                                       │
│      │                                                                      │
│      └── Builds system prompt with tools                                    │
│                                                                             │
│  step(action)                                                               │
│      ├── Parse tool call from LLM response                                  │
│      ├── Execute via OpenEnv step_async()                                   │
│      └── Return observation, reward, done                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ OpenEnv FleetTaskEnv (envs.fleet_env.task_env)                              │
│                                                                             │
│  __init__(task_config, api_key, ...)                                        │
│      ├── FleetEnvClient.from_fleet()  ←── HTTP: Creates Fleet env instance  │
│      │       └── fleet.make()         ←── Provisions cloud VM/container     │
│      └── list_tools()                 ←── MCP: Fetches available tools      │
│              └── Cached in _tools_cache                                     │
│                                                                             │
│  reset_async()                                                              │
│      ├── _orch.reset()  ←── HTTP: Resets episode (logs warning if fails)    │
│      └── Returns obs with cached _tools_cache                               │
│                                                                             │
│  step_async(action)                                                         │
│      └── _tools.call_tool()  ←── MCP: Executes tool                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Fleet SDK                                                                   │
│                                                                             │
│  fleet.make(env_key, ...)                                                   │
│      └── HTTP call to Fleet API                                             │
│      └── Provisions environment instance (VM/container)                     │
│      └── Returns env handle with URLs for HTTP + MCP endpoints              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

1. **Environment created in `__init__()`**: Fleet environment is provisioned and tools are fetched when `OpenEnvFleetTaskEnv` is instantiated, not during `reset()`. This ensures tools are available immediately.

2. **Tools cached**: Tools are fetched once during initialization and cached in `_tools_cache`. Every `reset_async()` and `step_async()` returns the cached tools.

3. **Reset failure handling**: If `_orch.reset()` fails (timeout), a warning is logged but the episode continues with empty observation. Tools remain available from cache.

4. **OpenEnv as abstraction layer**: SkyRL does not call Fleet SDK directly. All Fleet interactions go through OpenEnv's `FleetTaskEnv`.

## Tool Flow

```
init()
    └── Tools fetched via MCP list_tools()
    └── Tools injected into system prompt (JSON format)
    └── Tools cached for episode

step()
    └── LLM generates: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    └── SkyRL parses tool call
    └── OpenEnv executes via MCP call_tool()
    └── Result returned to LLM
```

## Configuration

### Environment Config
```yaml
environment:
  env_class: fleet_task
  skyrl_gym:
    fleet_task:
      tasks_file: /path/to/tasks.json  # Exported from Fleet
      api_key: ${FLEET_API_KEY}        # Or set via environment
      ttl_seconds: null                  # Auto: CUA=1800s, tool_use=600s (or override)
```

### Task JSON Format
```json
{
  "tasks": [
    {
      "key": "task-001",
      "prompt": "Search for flights from NYC to LA",
      "env_id": "booking-com",
      "version": "v1.2.3",
      "task_modality": "tool_use",
      "verifier_code": "async def verify(env): ..."
    }
  ]
}
```

## Error Handling

| Failure Point | Behavior |
|--------------|----------|
| `fleet.make()` fails | RuntimeError raised, episode fails |
| `list_tools()` fails | RuntimeError raised, episode fails |
| `_orch.reset()` fails | Warning logged, continues with empty observation |
| `call_tool()` fails | Error returned in observation, episode continues |

## Task Generation (GRPO)

RL-based task generation: trains Qwen3.5-9B to produce (prompt, verifier) pairs for Fleet environments using GRPO.

**Reward formula**: `R(task) = gate * (base_quality + alpha * var(raw_scores) + (p_hint - p_raw))`

- `gate`: LLM judge validity (0/1), currently disabled (gate=1.0)
- `base_quality`: 0.1 for tasks passing sandbox+judge gate (creates GRPO variance between valid/invalid)
- `var(raw_scores)`: Bernoulli variance from k raw evaluator rollouts
- `p_hint - p_raw`: Hint gap — solvable with hints but not without
- `alpha`: Weight for variance vs hint gap (default 0.5)

### Dataset Preparation

`prepare_task_gen_dataset.py` builds GRPO training data by:
1. Loading validated tasks from S3 (`all_tool_use.json`)
2. Discovering tool schemas from live Fleet environments via OpenEnv MCP
3. Fetching DB schemas from Supabase `seed_versions` -> S3 `schema.sql`
4. Storing env context (tools, schema, env_variables) in each parquet record

### Training Runs & Fixes

See [fleet-research/threads/task-rl/](https://github.com/fleet-ai/fleet-research/tree/main/threads/task-rl) for:
- [runs.md](https://github.com/fleet-ai/fleet-research/blob/main/threads/task-rl/runs.md) — detailed per-iteration analysis
- [changelog.md](https://github.com/fleet-ai/fleet-research/blob/main/threads/task-rl/changelog.md) — concise fix history

Fix log: [changelog.md](https://github.com/fleet-ai/fleet-research/blob/main/threads/task-rl/changelog.md)

## Dependencies

- **OpenEnv**: `pip install openenv[fleet]` or add to PYTHONPATH
- **Fleet SDK**: Installed as OpenEnv dependency
- **FLEET_API_KEY**: Must be set in environment or config

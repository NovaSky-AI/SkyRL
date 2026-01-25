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
      ttl_seconds: 600                  # Environment TTL
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

## Dependencies

- **OpenEnv**: `pip install openenv[fleet]` or add to PYTHONPATH
- **Fleet SDK**: Installed as OpenEnv dependency
- **FLEET_API_KEY**: Must be set in environment or config

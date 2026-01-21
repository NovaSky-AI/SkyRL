# Claude Code Instructions for SkyRL

## Critical Rules - DO NOT VIOLATE

1. **NEVER push to main** - Always create a branch and open a PR for review.

2. **DO NOT use Fleet SDK directly** - Use OpenEnv as the abstraction layer. The integration should go through OpenEnv, not call Fleet SDK APIs directly.
   - Fleet SDK repo: `/Users/deniz/repos/fleet-sdk`
   - OpenEnv repo: `/Users/deniz/repos/OpenEnv`
   - Use OpenEnv's `FleetTaskEnv` from `envs.fleet_env.task_env` for Fleet task integration

3. **Always look at API documentation** - Do not guess flags or API parameters. Read the actual documentation first.
   - Check the local fleet-sdk and OpenEnv repos for documentation
   - Look at existing examples in those repos

## Project Context

- This is SkyRL, a reinforcement learning training framework
- Fleet integration should use OpenEnv's `FleetTaskEnv` class (`envs/fleet_env/task_env.py`)
- OpenEnv provides: `FleetEnvClient`, `FleetMCPTools`, `FleetTaskEnv`, `make_fleet_task_env`
- OpenEnv repo: https://github.com/fleet-ai/OpenEnv

## Fleet Task Integration Pattern

The correct way to integrate Fleet tasks with SkyRL:

```python
# In SkyRL's Fleet integration env.py:
from envs.fleet_env import FleetTaskEnv, make_fleet_task_env

# Create task environment from config
task_config = {
    "task_key": "...",
    "prompt": "...",
    "env_key": "...",
    "env_version": "...",
    "verifier_code": "...",
    "task_modality": "tool_use",
}
env = FleetTaskEnv(task_config, api_key=...)

# Use Gymnasium-style interface
obs = env.reset()
obs, reward, done, info = env.step(action)
env.close()
```

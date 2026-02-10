# Claude Code Instructions for SkyRL

## Learnings

The `learnings/` folder contains documented knowledge about SkyRL internals:
- `training-mechanics.md` - step_wise_trajectories, retokenize_chat_history, chat templates, loss reduction

**Always check learnings/ before asking questions about SkyRL training mechanics.**

## Project Documentation

- `docs/experiments.md` - Main experiment log: dataset, method, changelog (v0.2→v0.2.4), analysis, future work

## Critical Rules - DO NOT VIOLATE

1. **NEVER push to main** - Always create a branch and open a PR for review. Do NOT merge PRs - the user will review and merge them.

2. **Always push to origin (fleet-ai/SkyRL)** - Never push to upstream (NovaSky-AI/SkyRL). When creating PRs, use `--repo fleet-ai/SkyRL`.

3. **Always run black before commits** - Format Python code before every commit:
   ```bash
   black skyrl-train/integrations/fleet/
   ```

   Or run full pre-commit before creating a PR:
   ```bash
   uv pip install pre-commit
   pre-commit run --all-files --config .pre-commit-config.yaml
   ```

   If pre-commit reformats files, stage them and commit again.

4. **DO NOT use Fleet SDK directly** - Use OpenEnv as the abstraction layer. The integration should go through OpenEnv, not call Fleet SDK APIs directly.
   - Fleet SDK repo: `/Users/deniz/repos/fleet-sdk`
   - OpenEnv repo: `/Users/deniz/repos/OpenEnv`
   - Use OpenEnv's `FleetTaskEnv` from `envs.fleet_env.task_env` for Fleet task integration

5. **Always look at API documentation** - Do not guess flags or API parameters. Read the actual documentation first.
   - Check the local fleet-sdk and OpenEnv repos for documentation
   - Look at existing examples in those repos

6. **Tinker integration must mimic SkyRL's skyrl_gym_generator** - Any development in Tinker code (`integrations/fleet/entrypoints/main_fleet_tinker.py`) must follow the same patterns as `skyrl_train/generators/skyrl_gym_generator.py`:
   - Use same parameter names: `max_input_length`, `max_generate_length`, `max_sequence_length`
   - Use same context length handling: check `if len(input_ids) > max_input_length` to end rollout
   - Use DAPO overlong filtering: truncate sequences > `max_sequence_length` and zero out loss mask
   - Match metrics naming: `pass_at_n`, per-environment metrics, etc.

7. **Always run tests before creating PRs** - Ensure tests pass before pushing code:
   ```bash
   # Run all CPU tests
   cd skyrl-train && uv run --isolated --extra dev pytest tests/cpu/ -v

   # Run specific test file
   uv run --isolated --extra dev pytest tests/cpu/test_trainer_utils.py -v
   ```

   Test requirements:
   - All existing tests must pass before creating or updating a PR
   - New functionality should include test coverage
   - Test edge cases and error conditions, not just happy paths
   - When fixing bugs, add regression tests that would have caught the bug
   - Use descriptive test names that explain what is being tested

## Eval Trajectory Analysis (Required for Each Training Run)

When documenting a training run, you MUST analyze eval trajectories from S3.

### 1. Download Eval Trajectories

```bash
aws s3 cp s3://skyrl-trajectories/evals/{run_name}/global_step_0/{env}.jsonl /tmp/{env}_step0.jsonl
```

### 2. Trajectory Format

- `input_prompt`: Full prompt string
- `output_response`: Model's response string
- `score`: List of per-token scores (check if any > 0 for success)
- `stop_reason`: Why generation stopped
- `env_extras`: Contains `task_key`, `data_source`

### 3. Compute These Metrics Per Environment

| Metric | How to compute |
|--------|----------------|
| Avg Turns | Count `<\|im_start\|>assistant` in `output_response` |
| Avg Tool Calls | Count `"name":` patterns in `output_response` |
| Success Rate | From `aggregated_results.jsonl` pass@3 |

### 4. Get Results from aggregated_results.jsonl

```bash
aws s3 cp s3://skyrl-trajectories/evals/{run_name}/global_step_X/aggregated_results.jsonl -
```

Extract `eval/{env}/pass_at_3` for each environment.

### 5. Document Error Patterns

Categorize failures by pattern:
- Tool argument errors (validation failures)
- Authorization errors (permission denied)
- Resource not found errors
- False completions (model confident but wrong)
- Context overflow (hit max_input_length)

### 6. Training Signal Analysis (variance_per_prompt)

Extract `reward/{env}/variance_per_prompt` from WandB to measure GRPO learning signal:

```python
import wandb
api = wandb.Api()
run = api.run("thefleet/fleet-task-grpo/{run_id}")
df = run.history(samples=1000)

variance_cols = [c for c in df.columns if 'variance_per_prompt' in c and c.startswith('reward/')]
for col in variance_cols:
    env = col.split('/')[1]
    vals = df[col].dropna().tolist()
    avg_var = sum(vals) / len(vals) if vals else 0
    pct_signal = sum(1 for v in vals if v > 0) / len(vals) * 100 if vals else 0
    print(f"{env}: avg={avg_var:.4f}, signal={pct_signal:.1f}%")
```

**Key metrics:**
- **Avg Variance**: Higher = more learning signal. 0.25 max (1 of 4 samples differs)
- **% Steps w/ Signal**: % of batches where variance > 0

**Interpretation:**
- Variance = 0: All 4 samples got same reward (no GRPO signal)
- Variance > 0: At least one sample differed (model can learn)
- < 25% signal = environment is too hard or too easy for current model

### 7. Update experiments.md

Document results in `fleet-research/threads/tool-use-training/experiments.md` following the template in Section 3.1.

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

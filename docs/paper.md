# SkyRL Fleet Training: Tool-Use Agent Development

## 1. Overview

Training an 8B parameter model (Qwen3-8B) on Fleet tool-use tasks using SkyRL's GRPO algorithm. Evaluation on held-out Fleet environments to measure cross-environment generalization.

## 2. Dataset

### v0.1 (Initial)

**Summary**: 3,603 tasks (3,522 train / 81 eval)
- 0% with env_variables
- Missing runtime context (dates, user info)
- Held-out environment: `outlook`

| Environment | Train | Eval | Total |
|-------------|-------|------|-------|
| booking | 1,070 | 19 | 1,089 |
| dropbox | 2 | 0 | 2 |
| fira | 54 | 0 | 54 |
| github | 1,827 | 38 | 1,865 |
| google-maps | 13 | 0 | 13 |
| hubspot | 12 | 0 | 12 |
| outlook | 0 | 24 | 24 |
| reddit | 246 | 0 | 246 |
| ticketmaster | 222 | 0 | 222 |
| zillow | 76 | 0 | 76 |
| **TOTAL** | **3,522** | **81** | **3,603** |

### v0.2 (Current)

**Summary**: 3,485 tasks (3,409 train / 76 eval)
- 95% with env_variables (`CURRENT_DATE`, `LOGGED_IN_USER`, `LOGGED_IN_NAME`)
- Held-out environment: `outlook`
- Fixes: YaRN rope_scaling (65K context), system prompt improvements (see [model_issues.md](model_issues.md))

| Environment | Train | Eval | Total |
|-------------|-------|------|-------|
| booking | 1,070 | 19 | 1,089 |
| dropbox | 1 | 0 | 1 |
| fira | 51 | 0 | 51 |
| github | 1,827 | 38 | 1,865 |
| google-maps | 7 | 0 | 7 |
| hubspot | 10 | 0 | 10 |
| outlook | 0 | 19 | 19 |
| reddit | 188 | 0 | 188 |
| ticketmaster | 222 | 0 | 222 |
| zillow | 33 | 0 | 33 |
| **TOTAL** | **3,409** | **76** | **3,485** |

**S3 path**: `s3://fleet-internal-datasets/v0.2/openenv/all_tool_use.json`

### Data Preparation Pipeline

**Task Validation Process**:
1. **Tool call replay**: Each task's tool calls are replayed against the Fleet environment to ensure they execute successfully
2. **Teacher model verification**: Tasks are validated by running a teacher model (e.g., Claude) to confirm solvability
3. **Filtering**: Tasks that fail replay or teacher verification are excluded from the training set

This ensures the training data contains only valid, solvable tasks with working tool chains.

## 3. Results

### Held-Out Environment

| Environment | Checkpoint 0 pass@k | Best pass@k | Best Step |
|-------------|---------------------|-------------|-----------|
| outlook | | | |

### Held-Out Tasks (from training environments)

| Environment | Checkpoint 0 pass@k | Best pass@k | Best Step |
|-------------|---------------------|-------------|-----------|
| github | | | |
| booking | | | |

### Training Environments

| Environment | Checkpoint 0 pass@k | Best pass@k | Best Step |
|-------------|---------------------|-------------|-----------|
| booking | | | |
| github | | | |
| reddit | | | |
| ticketmaster | | | |
| fira | | | |
| zillow | | | |
| hubspot | | | |
| google-maps | | | |
| dropbox | | | |

## 4. TODO

- [ ] **Scaling training samples**: Increase dataset size for better generalization
- [ ] **Trace analysis**: Compare trajectories before and after training to measure behavioral changes
- [ ] **Kimi K2.5 training**: Train on Kimi K2.5 model

## 5. Future Directions

- **Browser use (computer_use modality)**: Train on browser-based tasks with visual grounding
- **Combined training**: Joint training on `tool_use` + `computer_use` for generalist agents

## 6. References

- SkyRL repo: https://github.com/fleet-ai/SkyRL
- OpenEnv repo: https://github.com/fleet-ai/OpenEnv
- Model issues: [model_issues.md](model_issues.md)
- Training config: `skyrl-train/tasks/openenv-fleet-grpo-qwen3-8b.yaml`
- Trajectories: `s3://skyrl-trajectories/evals/`

# SkyRL Fleet Training: Tool-Use and Browser-Use Agent Development

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

### v0.3 (Current)

**Summary**: ~3,485 tasks (~3,304 train / ~181 eval)
- 10% eval ratio, capped at 30 samples per env
- No held-out environments (all envs split normally)
- ticketmaster now has ~22 eval tasks for trace analysis

| Environment | Train | Eval | Total |
|-------------|-------|------|-------|
| booking | ~1,059 | ~30 | 1,089 |
| dropbox | 1 | 0 | 1 |
| fira | ~46 | ~5 | 51 |
| github | ~1,835 | ~30 | 1,865 |
| google-maps | 7 | 0 | 7 |
| hubspot | ~9 | ~1 | 10 |
| outlook | ~17 | ~2 | 19 |
| reddit | ~169 | ~19 | 188 |
| ticketmaster | ~200 | ~22 | 222 |
| zillow | ~30 | ~3 | 33 |
| **TOTAL** | **~3,304** | **~181** | **3,485** |

**Changes from v0.2:**
- Increased eval_ratio: 2% → 10%
- Added MAX_EVAL_SAMPLES: 30 per env (caps large envs)
- Lowered MIN_EVAL_SAMPLES: 5 → 1
- Removed outlook from held-out (now split like other envs)

---

### v0.2

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

### Run: `mk6nr5ij` ([WandB](https://wandb.ai/thefleet/fleet-task-grpo/runs/mk6nr5ij))

**Status**: Step 19 (running)

#### Held-Out Environment

| Environment | Checkpoint 0 pass@3 | Best pass@3 | Best Step | Avg Turns |
|-------------|---------------------|-------------|-----------|-----------|
| outlook | 36.8% | 36.8% | 0 | - |

#### Held-Out Tasks (from training environments)

| Environment | Checkpoint 0 pass@3 | Best pass@3 | Best Step | Avg Turns |
|-------------|---------------------|-------------|-----------|-----------|
| github | 42.1% | 47.4% | 10 | 8.5 |
| booking | 52.6% | 63.2% | 10 | 3.0 |

#### Training Environments

| Environment | Best pass@4 | Best Step | Avg Turns |
|-------------|-------------|-----------|-----------|
| booking | 100% | 2 | 3.0 |
| github | 100% | 6 | 8.5 |
| reddit | 100% | 1 | 4.0 |
| ticketmaster | 0% | - | 10.0 |
| fira | - | - | - |
| zillow | 100% | 15 | 27.0 |
| hubspot | - | - | - |
| google-maps | - | - | - |
| dropbox | - | - | - |

---

## 4. TODO

- [ ] **Hillclimbing on tool-use eval set**: Iterate on model/training to improve eval performance
- [ ] **Scaling dataset**: Increase training samples to demonstrate performance improvement with data scale
- [ ] **Trace analysis**: Compare trajectories before and after training to measure behavioral changes (`s3://skyrl-trajectories/evals/`)
- [ ] **Kimi K2.5 training**: Train on Kimi K2.5 model
- [ ] **OpenEnv browseComp integration**: Integrate with OpenEnv browseComp for browser-use tasks
- [ ] **Browser use (computer_use modality)**: Train on browser-based tasks with visual grounding
- [ ] **Combined training**: Joint training on `tool_use` + `computer_use` for generalist agents

## 5. References

- SkyRL repo: https://github.com/fleet-ai/SkyRL
- OpenEnv repo: https://github.com/fleet-ai/OpenEnv
- Model issues: [model_issues.md](model_issues.md)
- Training config: `skyrl-train/tasks/openenv-fleet-grpo-qwen3-8b.yaml`
- Trajectories: `s3://skyrl-trajectories/evals/`

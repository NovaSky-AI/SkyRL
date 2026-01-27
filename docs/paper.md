# SkyRL Fleet Training: Tool-Use Agent Development

## 1. Overview

Training tool-use agents using SkyRL's GRPO algorithm on Fleet-hosted environments via OpenEnv's FleetTaskEnv abstraction.

- **Model**: Qwen3-8B (8B parameters)
- **Algorithm**: GRPO (Group Relative Policy Optimization)
- **Infrastructure**: SkyPilot on B200:2 GPUs
- **Target Environments**: GitHub, Booking, Outlook

## 2. Dataset Evolution

### v0.1 (Initial)
- 3,603 tasks
- 0% with env_variables
- Missing runtime context (dates, user info)

### v0.2 (Current)
- 3,485 tasks
- 95% with env_variables (`CURRENT_DATE`, `LOGGED_IN_USER`, `LOGGED_IN_NAME`)
- Modalities: `tool_use`, `computer_use`
- Train/validation split with stratified sampling by `env_key`
- S3 path: `s3://fleet-internal-datasets/v0.2/openenv/all_tool_use.json`

### Data Preparation Pipeline

**Task Validation Process**:
1. **Tool call replay**: Each task's tool calls are replayed against the Fleet environment to ensure they execute successfully
2. **Teacher model verification**: Tasks are validated by running a teacher model (e.g., Claude) to confirm solvability
3. **Filtering**: Tasks that fail replay or teacher verification are excluded from the training set

This ensures the training data contains only valid, solvable tasks with working tool chains.

## 3. Baseline Eval Results

**Overall**: 2.9% success rate (7/243 trajectories)

| Environment | Success Rate | Key Issue |
|-------------|--------------|-----------|
| GitHub | 0% | Tool calls go unanswered (infra bug) |
| Booking | ~5% | Tool repetition, can't signal done |
| Outlook | ~5% | Excessive thinking, max turns hit |

Trajectories analyzed from: `s3://skyrl-trajectories/evals/fleet_tool_use_92d02656/`

## 4. Critical Issues Identified

### Issue 0: Environment Not Returning Tool Results (CRITICAL)
- 88% of GitHub trajectories make a tool call but NEVER receive a response
- Trajectory ends at `</tool_call><|im_end|>` with no tool result
- Root cause: Unknown - likely Fleet GitHub environment bug

### Issue 1: Context Budget Exhaustion (CRITICAL)
- GitHub: 100 tools = ~26K tokens for tool definitions alone
- Exceeds context at turn 0 with 24K max input length
- Model spends 81% of token budget on `<think>` blocks

### Issue 2: Cannot Signal Task Completion (CRITICAL)
- Model completes task but can't give final response
- Gets error: "No tool call found. Use <tool_call>{...}</tool_call> format."
- 40% of Booking, 36% of Outlook trajectories hit this

### Issue 3: Tool Call Repetition (HIGH)
- Same tool call made up to 20x with identical parameters
- Booking: 56% of trajectories, Outlook: 54%
- Model doesn't track previous calls or process results properly

### Issue 4: Hitting Max Turns (HIGH)
- 90% of trajectories end by hitting max turns (50), not completing task
- GitHub: 100%, Booking: 84%, Outlook: 81%

### Issue 5: Thinking Pattern Repetition (MEDIUM)
- Model repeats same reasoning within trajectories
- Example: "If today is October 2023, then February 9th, 2024 is in the future..." repeated 14 times

### Issue 6: Hallucinated Parameters (MEDIUM)
- Model invents parameter values (e.g., `repo_id: "123"`)
- Doesn't fetch required data before making calls

## 5. Fixes Implemented

| Fix | Issue Addressed | Change |
|-----|-----------------|--------|
| YaRN rope_scaling | Context Budget | 32K → 65K context via `hf_overrides` for vLLM 0.13.0 |
| Done signal prompt | Task Completion | System prompt requires `<tool_call>` OR `<done>` every turn |
| Error handling prompt | Tool Repetition | "Do NOT repeat same call with identical arguments" |
| env_variables | Date/user errors | Dataset v0.2 with runtime variables (95% coverage) |
| Hydra config | Config errors | Use `++` prefix for new nested keys |
| vLLM hf_overrides | vLLM 0.13.0 API | Pass rope_scaling via `hf_overrides` instead of direct arg |

### Technical Details

**YaRN Configuration**:
```yaml
++trainer.rope_scaling.rope_type=yarn
++trainer.rope_scaling.factor=2.0
++trainer.rope_scaling.original_max_position_embeddings=32768
```

**System Prompt Additions**:
```
## Error Handling
If a tool call returns an error:
- Read the error message carefully
- Do NOT repeat the same call with identical arguments
- Change your approach: use different parameters, try a different tool, or break the task into smaller steps
```

## 6. GitHub Tools Analysis

100 tools causing context overflow (~26K tokens):

| Category | Count | Examples |
|----------|-------|----------|
| Legitimate | ~51 | getRepository, createIssue, listPullRequests |
| Fake endpoints | ~14 | NOT A REAL ENDPOINT errors |
| Redundant/convenience | ~35 | Wrappers that could be consolidated |

**Token Reduction Opportunities**:
- Reduce to 50 tools: Save ~12,500 tokens
- Reduce to 30 tools: Save ~17,500 tokens

## 7. Future Directions

### Short-term
- **Tool pruning**: Consolidate GitHub tools from 100 → 30-50
- **Reduce thinking overhead**: Shorter prompts or disable `<think>` blocks for tool-use
- **Debug Fleet GitHub env**: Investigate 88% no-response issue

### Medium-term
- **Reward shaping**: Penalize repeated identical calls, reward efficient tool chains
- **Multi-turn improvements**: Better context management across turns

### Long-term
- **Browser use (computer_use modality)**: Train on browser-based tasks with visual grounding
- **Combined training**: Joint training on `tool_use` + `computer_use` for generalist agents that can use both APIs and browser interfaces

## 8. References

- SkyRL repo: https://github.com/fleet-ai/SkyRL
- OpenEnv repo: https://github.com/fleet-ai/OpenEnv
- Model issues doc: `docs/model_issues.md`
- Training config: `skyrl-train/tasks/openenv-fleet-grpo-qwen3-8b.yaml`

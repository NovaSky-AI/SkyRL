# SkyRL Fleet Training - Model Issues Analysis

Analysis of eval trajectories from `s3://skyrl-trajectories/evals/fleet_tool_use_92d02656/`

**Summary Stats:**
- Total trajectories analyzed: 243 (57 booking, 114 github, 72 outlook)
- Overall success rate: 2.9% (7/243 non-zero scores)

---

## Changelog

### v0.2.3 (2026-01-29) - Run `eaqz4o21` Analysis

**Run:** [WandB](https://wandb.ai/thefleet/fleet-task-grpo/runs/eaqz4o21) | Step 69 (running)

#### Eval Performance (Held-out Tasks)

- **Ckpt 0**: Baseline pass@3 before any training (step 0)
- **Best**: Highest pass@3 achieved during training
- **pass@3**: % of tasks where at least 1 of 3 rollouts succeeded

| Env | Samples | Avg Turns | Ckpt 0 | Best | Best Step | Δ |
|-----|---------|-----------|--------|------|-----------|---|
| hubspot | 4 | 12.8 | 100.0% | 100.0% | 0 | +0.0% |
| outlook | 4 | 12.8 | 100.0% | 100.0% | 0 | +0.0% |
| reddit | 8 | 6.9 | 66.7% | 76.2% | 30 | +9.5% |
| github | 12 | 11.0 | 46.7% | 70.0% | 30 | **+23.3%** |
| booking | 4 | 8.8 | 53.8% | 61.5% | 60 | +7.7% |
| fira | 4 | 7.5 | 40.0% | 60.0% | 20 | +20.0% |
| zillow | 4 | 7.2 | 50.0% | 50.0% | 0 | +0.0% |
| ticketmaster | 4 | 8.5 | 3.6% | 7.1% | 10 | +3.6% |
| **OVERALL** | - | - | 43.1% | **49.1%** | 30 | +6.0% |

#### GRPO Signal Analysis (Training)

- **Steps**: Number of training steps where this env was sampled
- **Signal**: Steps with within-prompt variance (some rollouts pass, some fail → GRPO can learn)
- **No Signal**: Steps where all 4 rollouts had same result (all pass or all fail → no gradient)
- **% Signal**: Fraction of steps that provide learning signal
- **Avg Raw Reward**: Mean success rate per rollout (e.g., 33.6% = ~1.3/4 rollouts succeed on average)

| Env | Steps | Signal | No Signal | % Signal | Avg Raw Reward |
|-----|-------|--------|-----------|----------|----------------|
| **github** | 74 | 61 | 13 | **82.4%** | 33.6% |
| **fira** | 5 | 5 | 0 | **100.0%** | 65.0% |
| hubspot | 2 | 1 | 1 | 50.0% | 75.0% |
| booking | 59 | 29 | 30 | 49.2% | 28.2% |
| reddit | 16 | 7 | 9 | 43.8% | 62.5% |
| zillow | 3 | 1 | 2 | 33.3% | 16.7% |
| outlook | 5 | 1 | 4 | 20.0% | 5.0% |
| **ticketmaster** | 16 | 1 | 15 | **6.2%** | 4.7% |

- **Signal** = within-prompt variance exists (not all 0/4 or all 4/4 rollouts same result)
- **Avg Raw Reward** = mean success rate per rollout across all steps

#### Error Modes (Step 10 Eval Traces)

| Env | Fail Rate | Top Errors |
|-----|-----------|------------|
| ticketmaster | 98% | Resource not found (27%), Token limit (22%), No tool call (21%) |
| zillow | 100% | No tool call/stuck (17%) |
| booking | 65% | Token limit (24%), Date in past (10%) |
| github | 67% | Resource not found (50%), No tool call (10%) |
| fira | 67% | API error (70%), No tool call (30%) |
| reddit | 43% | Token limit (26%), API error (11%) |

#### Key Findings

1. **github** is the best learning source (82% signal, +23% eval improvement)
2. **ticketmaster** provides almost no signal (6%) - always fails, can't learn
3. **Token limits** are a major issue: booking (24%), ticketmaster (22%), reddit (26%)
4. **fira** has 100% signal but only 5 training steps (undersampled)

---

### v0.2.2 (2025-01-27) - Run `mk6nr5ij` Analysis

**Comparison: Baseline (2.9% success) → Current (43.4% pass@3)**

| Issue | Baseline | Current | Status |
|-------|----------|---------|--------|
| **Issue 0**: GitHub no env response | 88% no response, 1.1 tool calls | 14.3 avg tool calls | ✅ FIXED |
| **Issue 1**: Excessive thinking | 81% token budget on thinking | 6K tokens (success) vs 11K (failure) | ⚠️ IMPROVED |
| **Issue 2**: Cannot signal done | 40% booking stuck | max_turns=6 (booking), tasks completing | ✅ FIXED |
| **Issue 3**: Tool repetition | 56% booking, 20x repeats | - | ❓ NEEDS ANALYSIS |
| **Issue 4**: Hitting max turns | 90% hit 50 turns | booking max=6, reddit max=5 | ✅ IMPROVED |

**Per-Environment Improvement:**

| Environment | Baseline | Current pass@3 | Delta |
|-------------|----------|----------------|-------|
| GitHub | 0% | 42.1% | **+42.1%** |
| Booking | ~5% | 52.6% | **+47.6%** |
| Outlook | ~5% | 36.8% | **+31.8%** |
| Overall | 2.9% | 43.4% | **+40.5%** |

**Key Observations:**
- GitHub environment is now responding (14.3 avg tool calls vs 1.1)
- Tasks completing before max turns (booking max=6, not 50)
- Successful trajectories use fewer tokens (6K vs 11K for failures)

---

### v0.2.1 (2025-01-27)

**Infrastructure fix: rope_scaling for vLLM 0.13.0 and HuggingFace**

| Fix | Issue | Change |
|-----|-------|--------|
| vLLM rope_scaling | vLLM 0.13.0 removed direct `rope_scaling` param | Pass via `hf_overrides["rope_parameters"]` per [vLLM docs](https://docs.vllm.ai/en/latest/examples/offline_inference/context_extension/) |
| HuggingFace rope_scaling | HF models reject `rope_scaling` as kwarg | Set `model_config.rope_scaling` instead of passing to `from_pretrained()` |

**PR:** [NovaSky-AI/SkyRL#976](https://github.com/NovaSky-AI/SkyRL/pull/976)

---

### v0.2 (2026-01-26)

**Dataset:** `s3://fleet-internal-datasets/v0.2/openenv/all_tool_use.json`

| Fix | Issue Addressed | Change |
|-----|-----------------|--------|
| Context extension | Issue 1 (Context Budget) | YaRN rope_scaling (65K context), MAX_INPUT_LENGTH 24K→48K, MAX_GENERATE_LENGTH 4K→8K |
| Done signal prompt | Issue 2 (Task Completion) | System prompt requires `<tool_call>` OR `<done>` every turn |
| Error handling prompt | Issue 3 (Tool Repetition) | System prompt: "Do NOT repeat the same call with identical arguments" |
| env_variables fix | Date/user errors | Training data now includes `CURRENT_DATE`, `LOGGED_IN_USER`, `LOGGED_IN_NAME` (95% coverage) |

**Commits:**
- `de6b4b4` - Enable YaRN for 65K context, increase generation length
- `6a9b813` - Strengthen system prompt for task completion signaling
- `<pending>` - Add error handling to system prompt

**Dataset stats (v0.2 vs v0.1):**
- Total tasks: 3,485 (was 3,603)
- With env_variables: 95% (was 0%)

---

## Issue 0: Environment Not Returning Tool Results (GitHub)

**Severity: CRITICAL - INFRASTRUCTURE BUG**

88% of GitHub trajectories make a tool call but NEVER receive an environment response. The trajectory ends at `</tool_call><|im_end|>` with no tool result.

| Environment | No Env Response |
|-------------|-----------------|
| **GitHub**  | **88% (100/114)** |
| Booking     | 19% (11/57)     |
| Outlook     | 19% (14/72)     |

**Impact:** GitHub has 0% success rate because tool calls go unanswered.

**Root Cause:** Unknown - could be:
- Bug in GitHub Fleet environment
- Timeout issues
- Malformed tool calls crashing the env

**Recommendation:** Debug Fleet GitHub environment, check logs for errors.

---

## Issue 1: Excessive Thinking (81% of token budget on GitHub)

**Severity: CRITICAL**

The model spends excessive tokens on `<think>` blocks, leaving insufficient budget for actual tool calls and actions.

| Environment | Avg Response | Thinking % | Tool Calls Avg |
|-------------|--------------|------------|----------------|
| GitHub      | 7,616 chars  | **81.3%**  | 1.1            |
| Outlook     | 21,262 chars | 58.4%      | 8.1            |
| Booking     | 26,493 chars | 47.6%      | 14.3           |

**Impact:** GitHub has 0% success rate because model only completes ~1 tool call before hitting generation limit.

**Recommendation:**
- Reduce thinking verbosity via system prompt
- Consider removing `<think>` blocks for tool-use tasks
- Increase `max_generate_length` significantly

---

## Issue 2: Cannot Signal Task Completion

**Severity: CRITICAL**

When the model completes a task and tries to give a final response (without a tool call), the environment rejects it:
```
"No tool call found. Use <tool_call>{...}</tool_call> format."
```

**Stats:**
- Booking: 40% of trajectories have this error
- Outlook: 36% of trajectories have this error

**Example (Booking):**
Model says: "The hotel offers a price per night of $132, which is within your budget. If you need further assistance, feel free to ask!"
→ Environment rejects: "No tool call found"

**Impact:** Model cannot communicate task completion, forcing infinite tool call loops.

**Recommendation:**
- Add a `task_complete` tool for signaling completion
- Or accept `<done>` tag without requiring tool call
- Or modify system prompt to explain how to signal completion

---

## Issue 3: Tool Call Repetition (up to 20x same call)

**Severity: HIGH**

Model makes identical tool calls repeatedly, wasting turns.

| Environment | Trajectories with Repetition | Max Repetitions |
|-------------|------------------------------|-----------------|
| Booking     | 56%                          | 20x             |
| Outlook     | 54%                          | 9x              |
| GitHub      | 17%                          | 3x              |

**Example (Booking):**
Same `hotels_search` call made 20 times with identical parameters.

**Root Cause:** Model doesn't track which calls it already made, or doesn't properly process results.

**Recommendation:**
- Add tool call history to context
- Penalize repeated identical calls in reward
- Improve system prompt to track progress

---

## Issue 4: Hitting Max Turns (90% of trajectories)

**Severity: HIGH**

Most trajectories end by hitting max turns (50), not by completing the task.

| Environment | Hit Max Turns | Stopped Clean |
|-------------|---------------|---------------|
| GitHub      | 100%          | 0%            |
| Booking     | 84%           | 16%           |
| Outlook     | 81%           | 19%           |

**Root Cause:** Combination of Issues 1-3 (excessive thinking, can't signal done, repetition).

---

## Issue 5: Thinking Pattern Repetition

**Severity: MEDIUM**

Model repeats the same reasoning/thoughts within trajectories.

**Example (Booking):**
"If today is October 2023, then February 9th, 2024 is in the future..." repeated 14 times.

**Impact:** Wastes token budget on redundant reasoning.

---

## Issue 6: Hallucinated Parameters

**Severity: MEDIUM**

Model invents parameter values when they're not available.

**Example (GitHub):**
```json
{"name": "createPullRequest", "arguments": {"repo_id": "123", "user_id": "456"}}
```
These are made-up IDs that don't exist.

**Recommendation:**
- System prompt should emphasize fetching required data first
- Provide examples of proper tool chaining

---

## Issue 7: Done Signal Without Task Completion

**Severity: LOW**

18% of Outlook trajectories emit `<done>` but have 0 score.

**Root Cause:** Model says done prematurely without actually completing task.

---

## Priority Fixes

1. **[CRITICAL] Fix completion signaling** - Add `task_complete` tool or accept non-tool-call responses
2. **[CRITICAL] Reduce thinking overhead** - Shorter prompts, no think blocks, or increased budget
3. **[HIGH] Add tool call tracking** - Prevent identical repeated calls
4. **[HIGH] Increase max_generate_length** - Current limit too restrictive for multi-step tasks
5. **[MEDIUM] Improve tool chaining guidance** - System prompt examples for proper API sequences

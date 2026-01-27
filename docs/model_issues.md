# SkyRL Fleet Training - Model Issues Analysis

Analysis of eval trajectories from `s3://skyrl-trajectories/evals/fleet_tool_use_92d02656/`

**Summary Stats:**
- Total trajectories analyzed: 243 (57 booking, 114 github, 72 outlook)
- Overall success rate: 2.9% (7/243 non-zero scores)

---

## Changelog

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

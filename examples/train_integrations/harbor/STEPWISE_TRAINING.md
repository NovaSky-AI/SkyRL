# Step-Wise Training with TIS for Harbor in SkyRL

## Table of Contents
- [Motivation](#motivation)
- [How Step-Wise Training Works in SkyRL](#how-step-wise-training-works-in-skyrl)
- [Comparison with Other Frameworks](#comparison-with-other-frameworks)
- [Implementation](#implementation)
- [Caveats and Design Decisions](#caveats-and-design-decisions)
- [Configuration](#configuration)
- [Running](#running)
- [Files Changed](#files-changed)

---

## Motivation

### The Re-Tokenization Problem

Currently, Harbor's `HarborGenerator` re-tokenizes the final chat history (string) after the agent finishes. The flow is:

1. vLLM generates tokens → Harbor agent executes tool calls → environment returns observations
2. Harbor returns the full chat history as a list of message dicts (strings)
3. `HarborGenerator` re-tokenizes the entire chat history using `get_response_ids_and_loss_mask_from_messages()`

This re-tokenization can produce **different token IDs** than what the model actually generated — a phenomenon called **"retokenization drift"** (see [vLLM Agent Lightning blog post](https://blog.vllm.ai/2025/10/22/agent-lightning.html)). Causes include:
- Non-unique tokenization (e.g., `"HAVING"` → `H`+`AVING` vs `HAV`+`ING`)
- Tool-call serialization changes during parsing/re-rendering
- Chat template differences across frameworks

### Why This Breaks TIS

TIS (Truncated Importance Sampling) corrects for off-policy drift between the rollout policy and the current policy:

```
TIS ratio = π_current(token) / π_rollout(token)
          = exp(current_logprobs - rollout_logprobs)
```

If the training tokens differ from the generation tokens due to retokenization, then `rollout_logprobs` (recorded during generation) don't correspond to the actual tokens being trained on. The TIS ratios become meaningless.

### The Solution: Step-Wise Training

Instead of re-tokenizing, use the **exact per-turn token IDs and logprobs from vLLM** via Harbor's `collect_rollout_details` feature. Each agent turn becomes a separate (prompt, response) training sample, where:
- `prompt_ids` = the full context vLLM saw (from `rollout_details.prompt_token_ids[turn]`)
- `response_ids` = the exact tokens vLLM generated (from `rollout_details.completion_token_ids[turn]`)
- `logprobs` = the exact per-token logprobs from vLLM (from `rollout_details.logprobs[turn]`)

This eliminates retokenization drift entirely and enables correct TIS computation.

---

## How Step-Wise Training Works in SkyRL

### Per-Step Reward Assignment

Each multi-turn trajectory of N turns is decomposed into N separate training samples. Rewards are assigned as per-token lists:

```
Step 1: reward = [0.0, 0.0, ..., 0.0]           # all zeros (intermediate step)
Step 2: reward = [0.0, 0.0, ..., 0.0]           # all zeros (intermediate step)
...
Step N: reward = [0.0, 0.0, ..., final_reward]   # reward at last token only
```

Only the final step of each trajectory receives the actual reward (from Harbor's verifier), placed at the last token position. This follows the same pattern as `SkyRLGymGenerator` (see `skyrl_gym_generator.py:446-451`).

### Advantage Computation

Advantages are computed **only for last steps**, then broadcast to all steps in the same trajectory:

```python
# trainer.py:784-815
# 1. Filter to last steps only
last_step_advantages = compute_advantages(rewards[is_last_step], ...)

# 2. Build trajectory ID mapping
traj_ids = cumsum(shifted_is_last_step)  # maps each step to its trajectory

# 3. Broadcast: all steps in trajectory i get the same advantage
advantages = last_step_advantages[traj_ids]
```

**This is mathematically equivalent to normal (non-step-wise) training** from an advantage perspective — the advantage signal comes entirely from the final trajectory reward. The difference is purely operational: each step gets its own (prompt, response) pair with exact token IDs.

### TIS (Truncated Importance Sampling)

TIS is implemented in `off_policy_correction_utils.py`. Two modes:

- **Token-level TIS** (`tis_ratio_type="token"`): Clamp per-token `exp(old_logprobs - rollout_logprobs)` to `[0, token_tis_ratio_clip_high]`, multiply with loss. Recommended clip: 1.5-5.0.
- **Sequence-level TIS** (`tis_ratio_type="sequence"`): Product of all token ratios (sum in log space), clamped. Recommended clip: 2.0-10.0.

Additionally, **outlier token masking** rejects entire sequences where any token has an extreme importance ratio (configurable thresholds).

Step-wise training enables correct TIS because `rollout_logprobs` are the exact logprobs from generation, matching the exact `response_ids` used for training.

### Batch Expansion

A batch of N trajectories with M average turns produces N×M training samples:

```
Input:  4 prompts × 2 samples = 8 trajectories
Output: 8 trajectories × ~3 turns avg = ~24 step-samples
```

The trainer handles this transparently — `mini_batch_size` and `micro_train_batch_size_per_gpu` control memory as before. Step-wise only increases the number of gradient accumulation steps per optimizer update, not peak GPU memory.

---

## Comparison with Other Frameworks

### SkyRL (SkyRLGymGenerator)

SkyRL's built-in `SkyRLGymGenerator` already supports step-wise via `generator.step_wise_trajectories=True`. It uses token-in-token-out: the generator directly controls tokenization at each turn, so there's no retokenization problem. Harbor's case is different because Harbor runs an external agent loop (Terminus 2) that returns strings, requiring either retokenization or rollout_details.

Key code: `skyrl_gym_generator.py:353-371` (per-step output), `skyrl_gym_generator.py:704-774` (flattening).

### SkyRL-Agent (MemAgent)

SkyRL-Agent explicitly supports step-wise training for agents like MemAgent that process documents chunk-by-chunk. Each chunk interaction is a separate trainable step. Uses the same `step_wise_trajectories` flag and advantage broadcast mechanism. See `skyrl-agent/skyrl_agent/integrations/tinker/tinker_train.py`.

### SLIME

- **Multi-turn**: Yes, via async rollout loops (`examples/geo3k_vlm_multi_turn/rollout.py`)
- **TIS**: Full support via `--use-tis` flag, with custom TIS functions loadable via `--custom-tis-function-path`
- **Advantage**: Trajectory-level (not per-step broadcast). Single advantage per trajectory replicated across all tokens.
- **Key difference**: Accumulates tokens in a single sequence with loss masking (observation tokens masked out). Does NOT decompose into separate per-step (prompt, response) pairs.

### veRL

- **Multi-turn**: Experimental support via `examples/data_preprocess/multiturn.py`
- **Step-wise advantages**: Experimental `_stepwise_advantage_broadcast` in `rllm/experimental/verl/verl_advantage.py` — same pattern as SkyRL (compute at last step, broadcast back)
- **TIS**: No explicit TIS; uses response mask for selective loss computation
- **Key difference**: Step-wise is experimental/external (in rllm integration), not in core veRL

### rLLM

- **Multi-turn**: Dedicated `MultiTurnWorkflow` class (`rllm/workflows/multi_turn_workflow.py`) with step-by-step environment interaction
- **Step-wise tracking**: Full trajectory metadata including `step_nums`, `episode_ids`, `trajectory_ids`
- **Advantage broadcasting**: Leverages veRL's experimental `_stepwise_advantage_broadcast`
- **TIS**: Indirectly through veRL backend
- **Key difference**: Most mature multi-turn workflow abstraction, but relies on veRL backend for training

### tinker-cookbook

- **Multi-turn**: Yes, via `Transition` and `Trajectory` abstractions (`tinker_cookbook/rl/types.py`)
- **Per-step rewards**: Each `Transition` has an immediate reward
- **Advantage**: Trajectory-level, group-centered (within-group normalization). Replicated across all action tokens.
- **TIS**: No explicit importance sampling
- **Key difference**: `Transition` is the closest conceptual match to our per-step approach. Clean abstraction but no TIS support. Prefix-aware sequence merging handles observation tokens efficiently.

### Summary Table

| Framework | Step-Wise Decomposition | TIS | Advantage Broadcast | Re-Tokenization Avoidance |
|-----------|------------------------|-----|--------------------|-|
| **SkyRL (this)** | Yes (per-turn separate samples) | Yes (token/seq) | Last-step → all steps | Yes (rollout_details) |
| **SkyRL-Agent** | Yes (same mechanism) | Yes | Same | Yes (token-in-token-out) |
| **SLIME** | No (single sequence, loss mask) | Yes (`--use-tis`) | Trajectory-level | N/A (single-turn focus) |
| **veRL** | Experimental | No | Experimental broadcast | N/A |
| **rLLM** | Yes (MultiTurnWorkflow) | Via veRL | Via veRL experimental | N/A |
| **tinker-cookbook** | Yes (Transition objects) | No | Trajectory-level, group-centered | N/A |

### Mini-Batch Size with Step-Wise Expansion

All frameworks that do step-wise decomposition face the same question: the effective number of training samples grows by the average number of turns. In SkyRL (and our implementation), the `policy_mini_batch_size` is computed as `policy_mini_batch_size * n_samples`, which doesn't account for step expansion. This means more optimizer steps per batch (e.g., 10× for 10-turn trajectories).

- **SkyRL/SkyRL-Agent**: Keep as-is (more optimizer steps). This is the current behavior.
- **veRL/rLLM**: Optional `normalize_by_steps` flag to divide advantage by step count.
- **tinker-cookbook**: Group-centered advantages naturally handle this via within-group normalization.

---

## Implementation

### Architecture

```
Harbor Trial (async)
    ↓ results.agent_result.rollout_details (per-turn token IDs + logprobs)
HarborGenerator._build_step_wise_output()
    ↓ HarborStepWiseOutput (list of per-step HarborAgentOutput)
HarborGenerator._build_step_wise_generator_output()
    ↓ Flatten to GeneratorOutput with is_last_step, trajectory_ids, rollout_logprobs
SkyRL Trainer (unchanged)
    ↓ Advantage broadcast, TIS correction, PPO update
```

### Key Data Structures

```python
@dataclass
class HarborAgentOutput:
    response_ids: List[int]              # Completion token IDs for this turn
    reward: Union[float, List[float]]    # Per-token rewards (list for step-wise)
    stop_reason: str
    loss_mask: List[int]                 # 1 for generated tokens, 0 for masked
    prompt_ids: List[int]                # Full prompt including chat history
    trajectory_id: TrajectoryID
    rollout_logprobs: Optional[List[float]]  # Per-token logprobs from vLLM
    summarization_count: Optional[int]
    num_turns: Optional[int]

@dataclass
class HarborStepWiseOutput:
    step_outputs: List[HarborAgentOutput]  # One per agent turn
    trajectory_id: Optional[TrajectoryID]
    summarization_count: Optional[int]
    num_turns: Optional[int]
```

### GeneratorOutput Format (Step-Wise)

When `step_wise_trajectories=True`, the generator returns:

```python
{
    "prompt_token_ids": [[turn1_prompt], [turn2_prompt], ...],  # Per-step prompts
    "response_ids": [[turn1_completion], [turn2_completion], ...],  # Per-step completions
    "rewards": [[0,0,...,0], [0,0,...,0], ..., [0,0,...,R]],  # Per-token, reward at last token of last step
    "loss_masks": [[1,1,...,1], [1,1,...,1], ...],  # All completion tokens trainable
    "rollout_logprobs": [[lp1], [lp2], ...],  # Per-token logprobs from vLLM
    "is_last_step": [False, False, ..., True, False, ..., True],  # Marks final step per trajectory
    "trajectory_ids": [tid1, tid1, ..., tid1, tid2, ..., tid2],  # Same ID for all steps of a trajectory
    "stop_reasons": [...],
    "rollout_metrics": {...},
}
```

The trainer already handles this format (checks for `is_last_step` and `trajectory_ids` in `trainer.py:642-658, 784-815`).

### Non-Step-Wise Path (Unchanged)

When `step_wise_trajectories=False` (default), behavior is identical to the original implementation: re-tokenize chat history via `get_response_ids_and_loss_mask_from_messages()`, return single trajectory per prompt, `rollout_logprobs=None`, no `is_last_step`.

---

## Caveats and Design Decisions

### 1. Prompt Left-Truncation for Padding OOM Prevention

**Problem**: In step-wise mode, different steps have very different prompt/response length ratios. Early turns have short prompts (~100 tokens) but potentially long completions (~20K tokens with thinking). Late turns have long prompts (~30K tokens of chat history) but short completions (~200 tokens). The padding function (`convert_prompts_responses_to_batch_tensors`) pads ALL samples to `max(all_prompts) + max(all_responses)`, creating padded sequences far exceeding `max_seq_len`:

```
Step 1: prompt=100,  response=20000 → actual total = 20100
Step 5: prompt=30000, response=200  → actual total = 30200
Padded: every sample = 30000 + 20000 = 50000 tokens  ← OOM!
```

**Solution**: After flattening all steps, compute `max_prompt_budget = max_seq_len - max(all_response_lengths)` and truncate all prompts from the LEFT to fit within this budget. Left-truncation preserves the most recent context (which is most relevant for the model's generation).

```python
# In _build_step_wise_generator_output():
max_response_len = max(len(r) for r in responses)
max_prompt_budget = max(0, self.max_seq_len - max_response_len)
for i in range(len(prompt_token_ids)):
    if len(prompt_token_ids[i]) > max_prompt_budget:
        excess = len(prompt_token_ids[i]) - max_prompt_budget
        prompt_token_ids[i] = prompt_token_ids[i][excess:]  # Truncate from left
```

**Trade-off**: Truncated prompts mean the model trains on slightly different context than what it saw during generation. This could affect TIS ratios for the truncated steps. In practice, truncation primarily affects later turns (which have long prompts from accumulated chat history), and the truncated portion is early context that has diminishing influence.

### 2. Per-Step Response Truncation

Each step's completion is independently truncated to fit within `max_seq_len - len(prompt_ids)`:

```python
max_response_for_step = max(0, self.max_seq_len - len(turn_prompt_ids))
if len(completion_ids) > max_response_for_step:
    completion_ids = completion_ids[:max_response_for_step]
```

This mirrors the non-step-wise path's truncation behavior.

### 3. Summarization Not Supported

When `step_wise_trajectories=True`, context summarization (`enable_summarize=True`) is not supported. Summarization causes Harbor to split rollout_details into multiple segments (main + subagent), making per-turn alignment ambiguous. An assertion enforces this:

```python
if len(rollout_details_list) > 1:
    assert summarization_count == 0, "step_wise + summarization not supported"
```

The default Harbor config already has `enable_summarize: false`.

### 4. `collect_rollout_details` Must Be Enabled

Step-wise training requires Harbor's `collect_rollout_details=True` in the agent kwargs. This tells Terminus 2 to request `logprobs=True` and `return_token_ids=True` from vLLM via LiteLLM's `extra_body`. The generator auto-enables this if not set:

```python
if self.step_wise:
    if not agent_kwargs.get("collect_rollout_details", False):
        self._harbor_trial_config_template["agent"]["kwargs"]["collect_rollout_details"] = True
```

### 5. Loss Mask: All Completion Tokens Are Trainable

In step-wise mode, each step's `loss_mask = [1] * len(completion_ids)`. This is because the response consists ONLY of the model's completion tokens (no interleaved observation/user tokens). The prompt already contains the full chat history including previous observations.

This differs from the non-step-wise path where `get_response_ids_and_loss_mask_from_messages()` interleaves assistant and user/observation tokens in a single response, with loss_mask=0 for non-assistant tokens.

### 6. Failed Trajectories in Step-Wise Mode

Failed trajectories (timeout, error, missing rollout_details) are emitted as a single zeroed-out step:

```python
HarborStepWiseOutput(
    step_outputs=[HarborAgentOutput(response_ids=[0], reward=[0.0], loss_mask=[0], ...)],
    trajectory_id=trajectory_id,
)
```

Instance-level masking still applies: if any trajectory for a prompt fails, all trajectories for that prompt are zeroed out.

---

## Configuration

### Enable Step-Wise Training

```bash
generator.step_wise_trajectories=true
```

### Enable TIS

```bash
trainer.algorithm.off_policy_correction.tis_ratio_type=token  # or "sequence"
trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high=2.0  # recommended: 1.5-5.0
```

### Harbor Config (default.yaml)

Required settings for step-wise:
```yaml
agent:
  kwargs:
    collect_rollout_details: true   # Get per-turn token IDs + logprobs from vLLM
    enable_summarize: false          # Required: summarization breaks rollout_details
    store_all_messages: true         # Required: for chat history extraction
```

### Full Example

```bash
bash examples/train_integrations/harbor/run_codecontest_comparison.sh stepwise-tis
```

---

## Running

### Prerequisites

```bash
# Prepare dataset
uv run --isolated --extra harbor python examples/train_integrations/harbor/prepare_harbor_dataset.py --dataset open-thoughts/CodeContests

# Ensure DAYTONA_API_KEY and WANDB_API_KEY are exported
```

### Dev Validation (Small Batch)

```bash
bash examples/train_integrations/harbor/run_codecontest_stepwise_dev.sh
# batch=4, n_samples=2, step_wise=true, TIS=token, 10 steps
```

### Production Three-Way Comparison

```bash
bash examples/train_integrations/harbor/run_all_comparison.sh
# Runs sequentially: baseline → stepwise → stepwise-tis
# Each: batch=32, n_samples=8, 10 steps
# Kills Daytona sandboxes between runs
```

### Individual Runs

```bash
bash examples/train_integrations/harbor/run_codecontest_comparison.sh baseline
bash examples/train_integrations/harbor/run_codecontest_comparison.sh stepwise
bash examples/train_integrations/harbor/run_codecontest_comparison.sh stepwise-tis
```

### Cleanup Sandboxes

```bash
uv run --isolated --extra harbor python examples/train_integrations/harbor/kill_daytona_sandboxes.py
```

---

## Files Changed

| File | Change |
|------|--------|
| `pyproject.toml` (line 253) | Harbor dependency → local path `/home/ray/default/harbor` |
| `harbor_trial_config/default.yaml` | Added `collect_rollout_details: true` |
| `harbor_generator.py` | Main implementation: `HarborStepWiseOutput`, `_build_step_wise_output()`, `_build_step_wise_generator_output()`, prompt normalization |
| `run_codecontest_stepwise_dev.sh` | Dev run script (batch=4, n_samples=2) |
| `run_codecontest_comparison.sh` | Parameterized production script (baseline/stepwise/stepwise-tis) |
| `run_all_comparison.sh` | Chains all three comparison runs |
| `kill_daytona_sandboxes.py` | Daytona sandbox cleanup utility |

No changes to the SkyRL training loop (`trainer.py`) — the existing step-wise and TIS infrastructure handles everything.

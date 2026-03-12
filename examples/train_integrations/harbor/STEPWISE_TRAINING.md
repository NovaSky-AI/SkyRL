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

### Limitation: Step-Wise Rewards Are Dropped

SkyRL's `SkyRLGymGenerator` supports per-step rewards — `env.step()` can return a non-zero `step_reward` at each turn. In the **non-step-wise** path, these are correctly placed as per-token rewards at turn boundaries (`_build_per_token_rewards()`), and the advantage estimator (GRPO or GAE) sees all of them.

However, in the **step-wise** path, intermediate step rewards are **silently dropped**. The trainer filters to last steps only before computing advantages:

```python
# trainer.py:794 — only last-step rewards are used
last_step_rewards = token_level_rewards[is_last_step]
last_step_advantages, last_step_returns = compute_advantages_and_returns(
    token_level_rewards=last_step_rewards, ...
)
# Broadcast back: all steps get the same advantage from last step's reward
advantages = last_step_advantages[traj_ids]
```

This means if `env.step()` returns `reward=0.5` at step 2 and `reward=1.0` at the final step, only `reward=1.0` contributes to the advantage. The `0.5` is placed in the per-token reward list for step 2 but never read during advantage computation.

**No framework supports both step-wise decomposition AND per-step advantages:**

| Framework | Per-step rewards in return? | Advantage granularity |
|-----------|---------------------------|----------------------|
| **SkyRL (step-wise)** | No — dropped by `[is_last_step]` filter | Scalar per trajectory, broadcast |
| **SkyRL-Agent** | No — `Transition.reward=0.0` always | Scalar per trajectory, broadcast |
| **Prime-RL** | N/A — no per-step reward concept | Scalar per rollout, broadcast |
| **veRL/rLLM** | No — `assert mode == "broadcast"` | Scalar per trajectory, broadcast |
| **tinker-cookbook** | **Yes** — `get_total_rewards()` sums all `transition.reward` + `final_reward` | But still scalar per trajectory, broadcast |
| **SLIME** | No step-wise decomposition | Trajectory-level |

tinker-cookbook comes closest: it correctly **sums** per-step rewards into the total return before computing group-centered advantages. But the advantage is still one scalar per trajectory, broadcast to all action tokens. There is no per-step advantage.

The fundamental reason: **GRPO has no natural per-step formulation.** GRPO groups trajectories by prompt and computes `advantage = reward - mean(group_rewards)`. This is inherently trajectory-level — intermediate steps from different trajectories of the same prompt aren't directly comparable.

**GAE could in principle do per-step advantages** — it uses a value function `V(s)` to estimate advantage at each token: `δ_t = r_t + γV(s_{t+1}) - V(s_t)`. SkyRL's non-step-wise path already supports GAE with per-token rewards at turn boundaries. But no one has combined GAE with step-wise decomposition — the `[is_last_step]` filter is applied regardless of which advantage estimator is configured.

### Step-Wise Training Is NOT Mathematically Equivalent to Non-Step-Wise

Step-wise decomposition changes the loss in ways that depend on the loss reduction method. Consider a trajectory with 3 turns:

- **Non-step-wise**: 1 training sample. `response = [A1, O2, A2, O3, A3]` with `loss_mask = [1,1,0,0,1,1,0,0,1,1]`.
- **Step-wise**: 3 training samples, each with pure completion tokens and `loss_mask = [1,1]`.

The per-token loss values (PPO surrogate) are also not identical because the model processes different contexts — step-wise has shorter sequences per forward pass, and prompt left-truncation can alter conditioning. But even assuming identical per-token losses, the reduction differs:

**`token_mean`**: `masked_mean(loss, loss_mask)` — sum of valid token losses / count of valid tokens. This is the closest to equivalent because both approaches weight each valid token equally. But the batch composition differs: step-wise has N×M step-samples per mini-batch vs N trajectories, and `mini_batch_size` doesn't account for step expansion. **Approximately equivalent.**

**`sequence_mean`**: `masked_mean(loss, loss_mask, dim=-1).mean()` — per-sequence token-mean, then batch-mean. **NOT equivalent.** Each step-sample is a separate "sequence" getting equal weight. A trajectory with 10 turns produces 10 sequences and gets 10× the gradient contribution of a 1-turn trajectory. In non-step-wise, every trajectory is 1 sequence regardless of turn count.

**`seq_mean_token_sum_norm`** (Dr. GRPO): `sum(loss * mask, dim=-1) / max_seq_len`, then `.mean()`. **NOT equivalent.** Same per-sequence weighting issue as `sequence_mean`. Additionally, short step-samples (common — early turns have short completions) get their token sum divided by a large `max_seq_len`, then receive equal weight in the `.mean()`.

#### Cross-Framework: Prefix Merging Preserves Loss Semantics

This weighting issue is not unique to SkyRL. The key distinction is between frameworks that **always decompose** every turn (always inequivalent) versus those that use **prefix-aware merging** (equivalent when prefixes hold):

| Framework | Decomposition strategy | Equivalent to non-step-wise? |
|-----------|----------------------|------------------------------|
| **SLIME** | No decomposition (single sequence, loss mask) | Always equivalent — it IS non-step-wise |
| **Prime-RL** | Prefix merging: merge when extension holds, split when breaks | **Equivalent when extension holds** (common case). Diverges only on context resets. |
| **Agent Lightning** (`trajectory`) | Prefix merging (same logic as Prime-RL) | **Equivalent when prefix holds.** Splits on mismatch (retoken/template/post-processing). |
| **Agent Lightning** (`transition`) | Always decomposes every turn | **Never equivalent** — same as SkyRL step-wise |
| **tinker-cookbook** | Prefix merging (same logic as Prime-RL) | **Equivalent when extension holds.** Diverges for MemAgent context resets. |
| **SkyRL-Agent** | Prefix merging via `transitions_to_training_data()` | **Equivalent for standard ReAct** (all turns merge into 1 datum). Diverges for MemAgent. |
| **veRL/rLLM** (experimental) | Always decomposes every turn | **Never equivalent** — same weighting issue |
| **SkyRL step-wise** | Always decomposes every turn | **Never equivalent** |
| **Harbor step-wise** (ours) | Always decomposes every turn | **Never equivalent** |

The insight: prefix-aware merging (Prime-RL, tinker-cookbook, SkyRL-Agent) isn't just a compute optimization (O(T) vs O(T²)) — it also **preserves loss reduction semantics**. When all turns merge into one sample, the loss reduction treats the trajectory as a single sequence, identical to non-step-wise.

SkyRL/Harbor's per-turn decomposition trades loss weighting consistency for the benefit of exact token IDs and logprobs (avoiding retokenization drift for TIS). Whether that trade-off is worth it depends on whether TIS correctness matters more than loss weighting equivalence.

### Full Equivalence Analysis: Beyond Loss Reduction

Loss reduction is the most visible difference, but step-wise decomposition also affects other components:

| Component | Equivalent? | Root cause | Fixable? |
|-----------|------------|------------|----------|
| Advantage computation (GRPO) | **Yes** | `[is_last_step]` filter ensures same inputs to GRPO | N/A |
| Per-token reward placement | **Doesn't matter for GRPO** | GRPO does `scores = token_level_rewards.sum(dim=-1)` — position is irrelevant. Only matters for GAE where `δ_t = r_t + γV(t+1) - V(t)` uses per-position rewards. | N/A |
| Loss reduction | **No** | Per-sample weighting (more steps = more weight) | **Yes** — per-trajectory weighting (see below) |
| `advantage_batch_normalize` | **No** | Mean/std computed across all step-samples equally | **Yes** — same per-trajectory weighting |
| Forward pass logprobs | **No** (tiny) | Each step is a separate shorter sequence. Logprobs differ from the single long-sequence non-step-wise forward pass due to different prompt/response boundary placement and prompt left-truncation. Numerically small (same model, same causal attention). | **No** — fundamental to decomposition |
| KL loss/penalty | **No** (tiny) | Inherits from logprob difference | **No** — inherits from above |
| Entropy | **No** (tiny) | Inherits from logprob difference | **No** — inherits from above |
| Metrics | **Mostly yes** | Filtered by `is_last_step` for reward metrics | N/A |

The logprob difference is fundamental and unfixable — it's inherent to processing shorter sequences independently vs one long sequence. But it's numerically tiny. The loss reduction and normalization differences are significant and fixable.

### Proposed Fix: Per-Trajectory Weighting

The loss reduction inequivalence can be fixed by weighting each step-sample by `1 / n_steps_in_its_trajectory`:

```python
# Current (inequivalent):
per_seq_loss = masked_mean(loss, loss_mask, dim=-1)  # [N_samples]
total_loss = per_seq_loss.mean()  # each sample weight = 1/N_samples

# Fixed:
per_seq_loss = masked_mean(loss, loss_mask, dim=-1)  # [N_samples]
weights = 1.0 / steps_per_trajectory[traj_ids]       # e.g. [1/3, 1/3, 1/3, 1/2, 1/2, 1, 1]
weights = weights / weights.sum()                      # normalize
total_loss = (per_seq_loss * weights).sum()
```

A 3-turn trajectory's 3 samples each get weight `1/3`, so their combined contribution equals one trajectory. Same idea applies to `seq_mean_token_sum_norm` and `advantage_batch_normalize`.

The information needed is already available in all frameworks:
- SkyRL step-wise: `is_last_step` → `cumsum` gives trajectory boundaries
- SkyRL-Agent: `episode_nums` directly
- Prime-RL / Agent Lightning: `len(merged_trace_idx)` per rollout

**No framework currently implements this fix.** For `token_mean`, the fix is less critical — it already weights each valid token equally regardless of grouping, so it's approximately equivalent.

### Policy Loss Functions: Step-Wise Equivalence

All policy loss functions compute per-token loss values then call `reduce_loss`. The per-token computation is element-wise (operates on `log_probs[i][t]`, `old_log_probs[i][t]`, `advantages[i][t]` independently) — **except GSPO**:

| Policy Loss | Per-token computation identical? | Additional step-wise issue? |
|---|---|---|
| **REGULAR** (PPO clip) | Yes (modulo tiny logprob diff) | Only `reduce_loss` |
| **DUAL_CLIP** | Yes | Only `reduce_loss` |
| **SAPO** | Yes | Only `reduce_loss` (recommends `sequence_mean`) |
| **GSPO** | **No** | Sequence-level IS weight + `reduce_loss` |
| **CISPO** | Yes | Only `reduce_loss` |
| **CLIP_COV** | Yes | Only `reduce_loss` |
| **KL_COV** | Yes | Only `reduce_loss` |

**GSPO** computes a **sequence-level importance weight**: `log_importance_weights = masked_mean(log_ratio, loss_mask, dim=-1)` — the mean log-ratio across all tokens in the same sample. In non-step-wise, "one sample" = full trajectory. In step-wise, "one sample" = one turn. The IS weight is computed over different scopes, producing different values and noisier estimates (fewer tokens to average over per turn).

### Off-Policy Correction: Step-Wise Equivalence

The off-policy correction utilities (`off_policy_correction_utils.py`) have multiple sequence-level operations that are affected by step-wise decomposition:

| Component | Affected? | Issue |
|---|---|---|
| **Token-level TIS** (`tis_ratio_type="token"`) | **No** | Purely per-token: `clamp(exp(old - rollout))` |
| **Sequence-level TIS** (`tis_ratio_type="sequence"`) | **Yes** | `sum(log_ratio, dim=-1)` — product of token IS ratios across the sequence. Different scope: full trajectory vs one turn. |
| **Outlier token mask** | **Yes** | Masks entire sequence if *any* token has outlier ratio (`.all(dim=-1)`). In non-step-wise, one bad token masks 100 tokens. In step-wise, it only masks that turn's ~5 tokens. More granular but different behavior. |
| **Geometric sequence mask** | **Yes** | Geometric mean of IS ratios per sequence: `exp(sum(log_ratio) / num_tokens)`. Different `num_tokens` (full trajectory vs one turn) gives different means. |
| **Product sequence mask** | **Yes** | Product of IS ratios: `sum(log_ratio, dim=-1)`. Same issue as sequence-level TIS. |

All sequence-level off-policy corrections that aggregate across `dim=-1` are affected — they compute statistics over a "sequence" which is the full trajectory in non-step-wise but a single turn in step-wise. Token-level TIS is the only mode that is fully equivalent.

---

## Comparison with Other Frameworks

### SkyRL (SkyRLGymGenerator)

SkyRL's built-in `SkyRLGymGenerator` already supports step-wise via `generator.step_wise_trajectories=True`. It uses token-in-token-out: the generator directly controls tokenization at each turn, so there's no retokenization problem. Harbor's case is different because Harbor runs an external agent loop (Terminus 2) that returns strings, requiring either retokenization or rollout_details.

Key code: `skyrl_gym_generator.py:353-371` (per-step output), `skyrl_gym_generator.py:704-774` (flattening).

### SkyRL-Agent (Full Flow)

SkyRL-Agent has explicit step-wise training built around three core abstractions: `Transition`, `transitions_to_training_data()`, and prefix-aware merging. Here is the full pipeline:

#### Step 1: Transition Recording (During Agent Execution)

Each LLM call is captured by the `@record_transition` decorator (`skyrl_agent/functional/utils.py:62`):

```python
@record_transition
async def _generate_with_recording(self, input_ids=[], ...):
    ...
```

This creates one `Transition` per LLM call:

```python
Transition(
    ob=Observation(input_ids=[...]),    # Full token IDs sent TO the LLM
    ac=TokensWithLogprobs(
        token_ids=[...],               # Tokens generated BY the LLM
        logprobs=[...],                # Per-token logprobs
        text="...",
    ),
    reward=0.0,                        # Placeholder — set later at trajectory level
    episode_done=False,
)
```

All transitions accumulate in `self.transitions: List[Transition]` during the agent run.

#### Step 2: `transitions_to_training_data()` — Prefix-Aware Merging

**Input**: `List[Transition]` from a **single trajectory** (one agent run).
**Output**: `List[TrainingDatum]` — potentially **fewer** items than input transitions.

The function (`utils.py:136-235`) maintains an accumulator and processes transitions one by one. The key logic: if the current transition's observation is a **prefix extension** of the accumulated sequence, the transition is **merged** into the current datum. Otherwise, a new datum starts:

```
Transition 1: ob=[O1],         ac=[A1]
Transition 2: ob=[O1,A1,O2],   ac=[A2]  ← ob extends full_sequence → MERGE
Transition 3: ob=[O3],         ac=[A3]  ← ob is NOT a prefix → FLUSH, start new datum
```

Result of merging transitions 1+2 into one `TrainingDatum`:

```python
TrainingDatum(
    input_tokens=[O1],                           # First observation = "prompt"
    response_tokens=[A1, O2, A2],                # Everything after = "response"
    response_logprobs=[lp1..., 0.0..., lp2...],  # Real logprobs for actions, 0.0 for obs
    response_mask=[1,1,..., 0,0,..., 1,1,...],     # 1 for action tokens, 0 for obs tokens
)
```

**When does merging happen?** In a standard ReAct agent, each LLM call receives the full conversation history. So `ob` for turn 2 = `[O1, A1, O2]`, which is a prefix extension of `[O1, A1]`. All turns merge into **one** `TrainingDatum` → the trajectory produces **1 step**.

**When does it NOT merge?** When context resets occur — e.g., MemAgent's `next_with_summary` tool replaces the context with a summary. The new observation has no prefix relationship with the previous sequence, so a new datum starts. This is how MemAgent produces **multiple** `TrainingDatum`s (= multiple steps) from one trajectory.

#### Step 3: Post-Processing (`AgentRunner._post_process_results()`, `base.py:406-418`)

Iterates over all trajectories in the batch:

```python
for result in matched_results:                         # For each trajectory
    transitions = result.get("transitions", [])
    data_list = transitions_to_training_data(transitions)  # → List[TrainingDatum]

    for data in data_list:                              # For each step (datum)
        prompt_input_ids.append(data.input_tokens)
        response_ids.append(data.response_tokens)
        logprobs.append(data.response_logprobs)
        response_assistant_mask.append(data.response_mask)
        is_last_episode_list.append(False)

    is_last_episode_list[-1] = True                     # Mark last step
    steps_per_trajectory.append(len(data_list))         # Track steps per traj

    # Broadcast trajectory-level reward to ALL steps (scalar, same value)
    reward_list.extend([result.get("reward", False)] * len(data_list))
```

**Reward handling**: The trajectory-level reward (scalar from the environment) is **replicated identically** to every step. All steps of the same trajectory get the same scalar reward. This is different from SkyRLGymGenerator/Harbor which use per-token reward lists with the reward placed at a specific token position.

#### Step 4: Output Format

```python
output = {
    "prompt_token_ids": prompt_input_ids,     # Per-step prompts
    "response_ids": response_ids,             # Per-step responses (interleaved obs+action tokens)
    "rewards": reward_list,                   # Per-step scalar rewards (same for all steps of a traj)
    "traj_rewards": traj_reward_list,         # Per-trajectory scalar rewards (not expanded)
    "loss_masks": loss_mask,                  # Per-step masks (0 for obs tokens, 1 for action tokens)
    "episode_nums": steps_per_trajectory,     # [3, 2, 4, ...] — num steps each trajectory produced
    "is_last_episode": is_last_episode_list,  # [F, F, T, F, T, F, F, F, T, ...]
    "traj_idx": traj_idx_list,                # Trajectory ID per step
    "rollout_logprobs": logprobs,             # Per-step logprobs (aligned with response_ids)
    "rollout_metrics": rollout_metrics,
}
```

#### Step 5: Tinker Integration Consumes This (`tinker_train.py:357-443`)

```python
rollouts = await agent_generator.run(input_batch)

# Use traj_rewards (one per trajectory, NOT step-expanded)
all_returns = [float(r) for r in rollouts["traj_rewards"]]

# Compute GRPO advantages at trajectory level
all_advantages = compute_advantages_grpo(all_returns, group_size=group_size)

# Broadcast advantages to steps using episode_nums
step_advantages = []
for idx, num_steps in enumerate(num_steps_per_trajectory):
    step_advantages.extend([all_advantages[idx]] * num_steps)
```

Then for each step, it builds a Tinker `Datum` with the full sequence (prompt + response), logprobs, and the broadcasted advantage value.

#### Key Differences: SkyRL-Agent vs Harbor Step-Wise

| Aspect | SkyRL-Agent | Harbor Step-Wise |
|--------|-------------|-----------------|
| **What is a "step"?** | A `TrainingDatum` from prefix-aware merging. For standard ReAct = 1 step (all turns merge). For MemAgent = N steps (context resets create breaks). | One LLM turn from `rollout_details.completion_token_ids[i]`. Every turn is always a separate step. |
| **Reward format** | Scalar per step; all steps get same trajectory reward | Per-token list; reward only at last token of last step |
| **Response content** | Interleaved obs+action tokens with `response_mask` distinguishing them | Pure completion tokens only (obs is in the prompt of the next step) |
| **Logprobs** | Aligned with response_tokens: real logprobs for actions, 0.0 for obs tokens | Aligned with completion tokens only |
| **Advantage computation** | Done in Tinker integration using `episode_nums` to broadcast | Done in SkyRL trainer using `is_last_step` to broadcast |
| **Why multiple steps?** | Context resets (MemAgent summarization) break prefix continuity | Every LLM turn is inherently a separate step |

#### Key Files

| File | Purpose |
|------|---------|
| `skyrl_agent/functional/utils.py:12-36` | `Transition`, `Observation`, `TokensWithLogprobs` dataclasses |
| `skyrl_agent/functional/utils.py:62-115` | `@record_transition` decorator |
| `skyrl_agent/functional/utils.py:136-235` | `transitions_to_training_data()` — prefix-aware merging |
| `skyrl_agent/agents/base.py:406-627` | `_post_process_results()` — flattening + reward broadcast |
| `skyrl_agent/integrations/tinker/tinker_train.py:357-443` | Tinker training loop — advantage broadcast + datum creation |

### Prime-RL

Prime-RL takes a fundamentally different approach: **prefix-aware trajectory merging with whole-sample scalar advantages** rather than per-step decomposition.

#### The Extension Property

Each multi-turn trajectory consists of steps with `(prompt_ids, completion_ids, completion_logprobs, completion_mask)`. The function `interleave_rollout()` (`prime_rl/orchestrator/trajectories.py:38-180`) processes them:

For each step, it checks if the step's `prompt_ids` is a **prefix extension** of any active sample's accumulated sequence. If yes → **merge** into that sample. If not → **start a new sample**.

```
5-step trajectory where extension breaks at step 4:

Steps 1-3: extension holds → merged into Sample 1
  completion_ids  = [A1,  delta_O2, A2,  delta_O3, A3]
  completion_mask = [1.., 0......., 1.., 0......., 1..]
  logprobs        = [lp1, 0.0....., lp2, 0.0....., lp3]

Step 4: extension breaks (e.g., thinking stripped by chat template)
Steps 4-5: merged into Sample 2
```

The `extend_sample()` function appends new prompt delta tokens with `mask=False, logprobs=0.0` (not trainable) and new completion tokens with `mask=True, logprobs=actual` (trainable). This is structurally identical to SkyRL-Agent's prefix-aware merging in `transitions_to_training_data()`.

**When does extension break?** Models like Qwen3 that strip `<think>` tags across turns, context compaction/summarization, sub-agent handoffs where context is discontinuous.

#### Advantages: One Scalar Per Rollout

Advantages are computed at the **rollout level**, not per-step (`advantage.py:65-91`):

```python
# GRPO: advantage = reward - mean(group_rewards)
# One scalar per rollout, broadcast to ALL tokens:
advantages = [training_example.advantage] * len(input_ids)
```

There is no `is_last_step` or `episode_nums` — every merged sample gets a single scalar advantage from the rollout-level reward.

#### Loss: IPO with Token-Level Importance Sampling

The loss (`loss.py:107-163`) implements IPO (INTELLECT Policy Optimization):

```python
log_importance_ratio = trainer_logprobs - inference_logprobs
importance_ratio = exp(log_importance_ratio)

# Trust region via probability difference masking (not ratio clipping like SkyRL's TIS)
probs_diff = exp(trainer_logprobs) - exp(inference_logprobs)
keep_mask = loss_mask & (|probs_diff| < threshold)

pg_loss = keep_mask * advantages * importance_ratio
kl_loss = loss_mask * log_importance_ratio²
loss = -pg_loss + kl_tau * kl_loss
```

The `inference_logprobs` are exact generation logprobs (stored per-step, aligned during merging). No re-tokenization needed because the extension property guarantees prefix alignment.

#### Key Differences from SkyRL Step-Wise

| Aspect | Prime-RL | SkyRL / Harbor Step-Wise |
|--------|----------|--------------------------|
| **When turns become separate samples** | Only when extension breaks | Always — every turn is separate |
| **Advantage** | One scalar per rollout | Computed on last step, broadcast |
| **Off-policy correction** | Probability-difference masking (IPO) | Ratio clipping (TIS) |
| **Compute scaling** | O(T) when extension holds | O(T²) always |
| **Philosophy** | Merge when possible, split when forced | Always decompose into per-turn units |

### Agent Lightning

Agent Lightning (Microsoft Research) supports two modes: **transition-level** (each turn = separate sample, like SkyRL step-wise) and **trajectory-level** (merge turns into one sample, like Prime-RL). Configured via `trace_aggregator.level: "trajectory"` or `"transition"`.

#### Trajectory-Level Aggregation

The trajectory path (`daemon.py:915-1023`) uses prefix matching identical in concept to Prime-RL:

```python
for turn_index, trace in enumerate(sample_info["trace_list"]):
    is_prefix, diagnostic = ids_startswith(
        trace["prompt_ids"] + trace["response_ids"],
        current_context, tokenizer, debug,
    )
    if is_prefix:
        current_context = trace["prompt_ids"] + trace["response_ids"]
        current_merged_trace_idx.append(turn_index)
    else:
        # Start new group — soft fallback, no retry
        merged_trace_idx.append(current_merged_trace_idx)
        current_merged_trace_idx = [turn_index]
        current_context = trace["prompt_ids"] + trace["response_ids"]
```

Merged turns produce one sample with interleaved response tokens and a `response_mask` (`1` for agent responses, `0` for prompt/observation delta tokens) — structurally identical to Prime-RL's `extend_sample()`.

#### Failure Handling — Same as Prime-RL (Soft Fallback)

When prefix matching fails, the trajectory is **split into multiple samples** starting at the mismatch point. No retry, no re-tokenization. This is exactly Prime-RL's behavior.

The blog post ([trajectory_level_aggregation](https://agent-lightning.github.io/posts/trajectory_level_aggregation/)) documents five failure modes:

1. **Retoken mismatch** (BPE artifacts): Same text tokenizes differently in generation vs re-tokenization. E.g., `<think>` → `["<", "think", ">"]` during generation but `["<th", "ink", ">"]` during template application.

2. **Template mismatch**: Chat templates insert/strip special tokens at turn boundaries. E.g., `<end_of_text>` generated explicitly by the model but stripped by the template on the next turn.

3. **Post-processing modifications**: If agents truncate outputs (e.g., removing chain-of-thought) before feeding history to subsequent turns, the stored rollout won't match the prompt prefix.

4. **Normalization artifacts**: Whitespace, escape character, or unicode normalization shifts token boundaries.

5. **Structural alignment**: Manual string concatenation bypasses chat templates, causing missing role headers.

A debug mode (`trace_aggregator.debug: true`) classifies mismatches into three categories and logs them:
- `template_mismatch`: Special token sequence differs
- `retoken_mismatch`: Token IDs differ but decoded text matches (BPE non-determinism)
- `others_mismatch`: Content itself differs

#### Rewards and Advantages

Scalar reward placed at the last token of the merged sample (`daemon.py:1070`):
```python
token_level_scores[torch.arange(n_transition), eos_mask_idx] = scores
```

Uses VERL's `compute_advantage()` with GRPO/GAE. One scalar advantage per merged sample, broadcast to all response tokens. Same pattern as Prime-RL.

#### Transition-Level (Alternative)

When `level: "transition"`, each turn becomes a separate sample with only its own response tokens — equivalent to SkyRL step-wise. No `response_mask` needed (all response tokens are trainable). Same per-sample weighting issues as SkyRL step-wise with `sequence_mean`.

#### Key Difference from Prime-RL

Structurally very similar to Prime-RL. The main addition is the **diagnostic system**: categorizing mismatches into template/retoken/others and logging them, which Prime-RL doesn't do. Both use the same soft-fallback strategy (split into new sample on mismatch).

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

| Framework | Step-Wise Decomposition | TIS / Off-Policy | Advantage | Re-Tokenization Avoidance |
|-----------|------------------------|------------------|-----------|---------------------------|
| **SkyRL (this)** | Yes (per-turn separate samples) | Yes, ratio clipping (token/seq) | Last-step → broadcast to all steps | Yes (rollout_details) |
| **SkyRL-Agent** | Yes (prefix-aware merging) | Yes | Scalar per traj → broadcast via `episode_nums` | Yes (token-in-token-out) |
| **Prime-RL** | Merge when prefix holds, split when breaks | Yes, probability-diff masking (IPO) | Scalar per rollout → broadcast to all tokens | Yes (exact prefix invariant) |
| **Agent Lightning** | Both modes: `trajectory` (merge) or `transition` (per-turn) | KL penalty (no explicit IS) | Scalar per sample → broadcast | Yes (vLLM `return_token_ids` + prefix matching) |
| **SLIME** | No (single sequence, loss mask) | Yes (`--use-tis`) | Trajectory-level | N/A |
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
harbor_agent_loop() (async, per-trajectory)
    ├─ Success → _build_step_wise_output() → HarborStepWiseOutput
    └─ Failure → HarborAgentOutput (same as non-step-wise failures)
           ↓
_build_step_wise_generator_output() (batch-level)
    ├─ _identify_masked_instances() (shared with non-step-wise path)
    └─ Flatten to GeneratorOutput with is_last_step, trajectory_ids, rollout_logprobs
           ↓
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

### 1. Padding OOM (Pending Fix in `convert_prompts_responses_to_batch_tensors`)

**Problem**: In step-wise mode, different steps have very different prompt/response length ratios. Early turns have short prompts (~100 tokens) but potentially long completions (~20K tokens with thinking). Late turns have long prompts (~30K tokens of chat history) but short completions (~200 tokens). The padding function (`convert_prompts_responses_to_batch_tensors`) pads ALL samples to `max(all_prompts) + max(all_responses)`, creating padded sequences far exceeding `max_seq_len`:

```
Step 1: prompt=100,  response=20000 → actual total = 20100
Step 5: prompt=30000, response=200  → actual total = 30200
Padded: every sample = 30000 + 20000 = 50000 tokens  ← OOM!
```

**Status**: This is a known issue. The fix should be in `convert_prompts_responses_to_batch_tensors` (in `skyrl/train/dataset/preprocess.py`) — each sequence should be capped at `max_seq_len` total length rather than taking `max(all_prompts) + max(all_responses)`. `HarborGenerator` intentionally does NOT truncate/pad prompts or responses itself; that responsibility belongs to the downstream batch tensor construction.

### 2. Summarization Not Supported

When `step_wise_trajectories=True`, context summarization (`enable_summarize=True`) is not supported. Summarization causes Harbor to split rollout_details into multiple segments (main + subagent), making per-turn alignment ambiguous. An assertion enforces this:

```python
if len(rollout_details_list) > 1:
    assert summarization_count == 0, "step_wise + summarization not supported"
```

The default Harbor config already has `enable_summarize: false`.

### 3. `collect_rollout_details` Must Be Enabled

Step-wise training requires Harbor's `collect_rollout_details=True` in the agent kwargs. This tells Terminus 2 to request `logprobs=True` and `return_token_ids=True` from vLLM via LiteLLM's `extra_body`. The generator auto-enables this if not set:

```python
if self.step_wise:
    if not agent_kwargs.get("collect_rollout_details", False):
        self._harbor_trial_config_template["agent"]["kwargs"]["collect_rollout_details"] = True
```

### 4. Loss Mask: All Completion Tokens Are Trainable

In step-wise mode, each step's `loss_mask = [1] * len(completion_ids)`. This is because the response consists ONLY of the model's completion tokens (no interleaved observation/user tokens). The prompt already contains the full chat history including previous observations.

This differs from the non-step-wise path where `get_response_ids_and_loss_mask_from_messages()` interleaves assistant and user/observation tokens in a single response, with loss_mask=0 for non-assistant tokens.

### 5. Failed Trajectories in Step-Wise Mode

Failed trajectories (timeout, error, missing rollout_details) return a plain `HarborAgentOutput` with zeroed fields (same as non-step-wise failures):

```python
HarborAgentOutput(response_ids=[0], reward=0, stop_reason="error", loss_mask=[0], prompt_ids=[0], ...)
```

The batch-level `_build_step_wise_generator_output()` identifies these via `_identify_masked_instances()` (shared with the non-step-wise path) and emits a single zeroed-out step for each masked instance. Instance-level masking still applies: if any trajectory for a prompt fails, all trajectories for that prompt are zeroed out.

### 6. Feature Compatibility

Step-wise training changes the batch structure: N trajectories become N×M step-samples. Several trainer features assume 1:1 correspondence between batch indices and trajectories.

| Feature | Compatible? | Issue |
|---------|------------|-------|
| `dynamic_sampling` (replace/filter) | **No** | Reward variance computation is polluted by intermediate zero-reward steps. Index-level replacement breaks trajectory integrity (replaces one step of trajectory A with a step from trajectory B, producing incoherent prompt/response). Filter sampling's reward grouping is wrong. |
| `zero_variance_filter` | **Accidentally bypassed** | Skipped because step-wise rewards are per-token lists (`isinstance(rewards[0], list)` → True). Would break with scalar rewards: groups by uid, computes `np.std(rewards)` on step-level entries mixing zeros and actual rewards. |
| `advantage_batch_normalize` | **Semantically different** | `normalize_advantages_dict` computes mean/std across all step-samples equally. Trajectories with more turns contribute more samples → more influence on normalization statistics. |
| `use_kl_in_reward` | **Partially broken** | `apply_reward_kl_penalty` adds per-token KL penalty to the rewards tensor. Works at the tensor level, but `compute_advantages_and_returns` then filters to `[is_last_step]` only — KL penalty on intermediate steps is dropped, same as intermediate step rewards. |
| `use_kl_loss` | **Yes** | Token-level regularizer: `masked_mean(kl, loss_mask)` per micro-batch. No trajectory structure dependency. |
| `use_entropy_loss` | **Yes** | Token-level: `masked_mean(entropy, loss_mask)`. No trajectory structure dependency. |
| `update_ref_every_epoch` | **Yes** | Weight sync only. No batch structure dependency. |
| `dump_data_batch` | **Yes** | Works, but dumped data has step-wise expanded structure (N×M) which may confuse analysis tools. |
| `batched=True` | **Blocked** | Explicit validation in `SkyRLGymGenerator.__init__`. |
| Custom chat templates | **Blocked** | Explicit validation in `SkyRLGymGenerator.__init__`. |
| `use_conversation_multi_turn=False` | **Blocked** | Explicit validation in `SkyRLGymGenerator.__init__`. |

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
| `harbor_generator.py` | Main implementation: `HarborStepWiseOutput`, `_build_step_wise_output()`, `_build_step_wise_generator_output()`, `_identify_masked_instances()` (shared masking logic) |
| `run_codecontest_stepwise_dev.sh` | Dev run script (batch=4, n_samples=2) |
| `run_codecontest_comparison.sh` | Parameterized production script (baseline/stepwise/stepwise-tis) |
| `run_all_comparison.sh` | Chains all three comparison runs |
| `kill_daytona_sandboxes.py` | Daytona sandbox cleanup utility |

No changes to the SkyRL training loop (`trainer.py`) — the existing step-wise and TIS infrastructure handles everything.

---

## Appendix: Trainer Internals Reference

Context for future iteration on the Harbor step-wise implementation.

### The 4 Sets of Logprobs

The training pipeline uses four distinct sets of log probabilities:

| Logprob | When computed | Where stored | Used for |
|---------|--------------|--------------|----------|
| `π_rollout` | During vLLM generation | `rollout_logprobs` in GeneratorOutput | TIS off-policy correction: `TIS_ratio = exp(π_old - π_rollout)` |
| `π_old` | Stage 3 (`fwd_logprobs_values_reward`), ONCE before training | `action_log_probs` in TrainingInputBatch | PPO ratio denominator: `ratio = exp(π_current - π_old)` |
| `π_ref` | Stage 3, from frozen reference model | `base_action_log_probs` in TrainingInputBatch | KL penalty: `KL(π_current, π_ref)` |
| `π_current` | Stage 5, RECOMPUTED each micro-batch with gradients | Local variable in `_forward_backward_micro` | PPO ratio numerator (this is what gets trained) |

### Training Call Chain

See `TRAIN_POLICY_CALL_CHAIN.md` for the full call chain. Key structure:

```
train_critic_and_policy(data)
  └─ _execute_training_step("policy", data)       ← trainer.py
     └─ for each epoch:
        └─ for each mini-batch (optimizer step):   ← trainer.py:1070-1082
           │  dispatch → Worker.forward_backward_from_staged()
           │  │  data sharded across DP workers (each gets mini_batch/dp_size samples)
           │  └─ Worker.forward_backward()         ← worker.py:675
           │     └─ for each micro-batch:           ← gradient accumulation
           │        └─ _forward_backward_micro()    ← worker.py:749, GPU hot path
           │           model.forward() → π_current
           │           loss_fn(π_current, π_old, advantages) → loss
           │           loss.backward() → accumulates .grad
           └─ Worker.optim_step()                   ← worker.py:939
              grad *= 1/N, optimizer.step(), zero_grad()
```

### Model Forward Pass

`HFModelWrapper.forward()` (`model_wrapper.py:261`):
- Input: `sequences [batch, max_prompt + max_response]` — the FULL padded sequence
- Runs the entire transformer on the full sequence length
- Slices output to response portion: `log_probs[:, -num_actions-1 : -1]`
- Returns: `action_log_probs [batch, max_response]`

Peak GPU memory is determined by `micro_batch_size × (max_prompt + max_response)`, not just response length.

### Padding: Full Batch, Not Per Mini-Batch

`convert_prompts_responses_to_batch_tensors()` (`preprocess.py:28`) pads ALL samples to the global max prompt and max response lengths across the ENTIRE batch. This happens once before training, not per mini-batch. Consequence for step-wise: if one step-sample has a 64K prompt (last turn), ALL step-samples are padded to 64K prompt length, including early turns with tiny prompts. This is a major source of memory waste and can cause OOM. **Pending fix**: each sequence should be capped at `max_seq_len` total length in `convert_prompts_responses_to_batch_tensors` rather than using `max(all_prompts) + max(all_responses)`.

### Three Masks

| Mask | Shape | What it marks |
|------|-------|---------------|
| `attention_mask` | `[batch, max_prompt + max_response]` | 0 for left-pad, 1 for real tokens. Used by transformer attention. |
| `response_mask` | `[batch, max_response]` | 1 for response tokens, 0 for right-pad. Used to slice logprobs from full-sequence model output. |
| `loss_mask` | `[batch, max_response]` | 1 for trainable tokens, 0 for obs/pad. Subset of response_mask. Used in loss computation. |

Relationship: `loss_mask ⊆ response_mask`. In non-step-wise, obs tokens have `response_mask=1` but `loss_mask=0`. In step-wise (Harbor), response is pure completion tokens, so `loss_mask = response_mask` (all 1s).

### GRPO Advantage Details

`compute_grpo_outcome_advantage()` (`ppo_utils.py:1132`):
1. `scores = token_level_rewards.sum(dim=-1)` — collapse to scalar (position doesn't matter)
2. Group by uid (prompt), compute per-group mean (and optionally std)
3. `advantage = (score - group_mean) / (group_std + ε)` (or just `score - group_mean` without std normalization)
4. `advantages = scores.unsqueeze(-1) × response_mask` — broadcast scalar to all response tokens
5. Singleton groups (1 sample): `mean=0, std=1` → advantage = raw score

### Advantage Estimator Comparison

| Estimator | Baseline | Normalization | Per-token variation | Needs critic |
|---|---|---|---|---|
| **GRPO** | mean(group) | optional std(group) | No (scalar broadcast) | No |
| **RLOO** | mean(group) × N/(N-1) leave-one-out | No | No (scalar broadcast) | No |
| **REINFORCE++** | None (batch whitening) | batch-level whiten | Yes (if γ<1) | No |
| **GAE** | V(s) from critic | batch-level whiten | Yes (always) | **Yes** |

For GRPO: reward token position doesn't matter — `sum()` collapses it. Only GAE cares about position (`δ_t = r_t + γV(t+1) - V(t)`).

### Memory Analysis

| Component | Determined by | GPU or CPU |
|---|---|---|
| Model forward/backward activations | `micro_batch_size × (max_prompt + max_response)` | GPU |
| Model parameters | Fixed (8B × 2 bytes / dp_size) | GPU |
| Gradients | Fixed (same as parameters, allocated once) | GPU |
| Optimizer states (Adam) | 2× parameters, offloaded after step | GPU → CPU |
| Full training batch (padded tensors) | `len(data) × (max_prompt + max_response)` | CPU (Ray object store) |
| Per-worker mini-batch slice | `(mini_batch_size / dp_size) × seq_len` | CPU → GPU |

Step-wise multiplies `len(data)` by avg turns (M), increasing CPU memory M×. GPU peak memory depends on the padded seq_len — currently `max(all_prompts) + max(all_responses)` which can cause OOM (see caveat #1). Once `convert_prompts_responses_to_batch_tensors` is fixed to cap at `max_seq_len`, GPU peak memory will be bounded.

### SkyRLGymGenerator vs Harbor Step-Wise: Structural Difference

| Aspect | SkyRLGymGenerator step-wise | Harbor step-wise |
|--------|---------------------------|-----------------|
| Response content | `action_tokens + obs_tokens` | Pure completion tokens only |
| Loss mask | `[1,1,...,0,0,...,1,1,...]` (action=1, obs=0) | `[1,1,...,1]` (all trainable) |
| Logprobs | Real for actions, 0.0 for obs | Real for all completion tokens |
| Obs tokens | In response, masked by loss_mask | In next step's prompt |
| Overlong filtering | **Broken** — intermediate steps don't end with EOS, so `apply_overlong_filtering` zeros their loss_mask | Works correctly — completion tokens may end with EOS |

**Potential fix for SkyRLGymGenerator**: Remove obs tokens from `turn_response_ids` (only include `output_ids`, not `output_ids + obs_ids`). This would make it structurally identical to Harbor step-wise, fix the overlong filtering bug, and reduce padding waste.

### Chunked MDP Perspective (ROLL Team / IPA)

The ROLL team's IPA algorithm argues that chunk-level (= per-turn) decomposition is **better** than both token-level and trajectory-level for agentic RL:

- **Token-level problem**: Most tokens don't change environment state. A 500-token thinking block + 10-token tool call all get same advantage, but only the tool call mattered.
- **Trajectory-level problem**: One scalar advantage for 30 turns. Turn 3 was the critical mistake, turns 4-30 were wasted, but all get identical gradient signal.
- **Chunk-level (step-wise)**: Each turn is the natural "decision unit." Enables per-chunk credit assignment, per-chunk IS masking, per-chunk returns.

With terminal-only reward (no per-step env rewards), chunk-level advantages still equal trajectory-level broadcast. The real benefit is chunk-level IS masking — selectively dropping turns where policy has drifted, rather than all-or-nothing. See: [ROLL Team IPA paper](https://arxiv.org/pdf/2512.24873).

### Contiguity No Longer Required

The trainer's advantage broadcast was updated to use trajectory-id-based mapping instead of the `cumsum(shifted_is_last_step)` trick. Steps from the same trajectory no longer need to be adjacent in the batch. See `feature/stepwise-traj-id-broadcast` branch in `/home/ray/default/SkyRL-stepwise-validation`.

### Inspection Scripts

All in `examples/train_integrations/harbor/`:

| Script | What it shows |
|--------|--------------|
| `inspect_trainer_dataflow.py` | Full trainer pipeline stages 0-5 with formulas, non-step-wise multi-turn |
| `inspect_stepwise_dataflow.py` | Harbor-style step-wise generator output with dummy data |
| `inspect_stepwise_skyrl_gym.py` | Actual SkyRLGymGenerator step-wise with mocked LLM/env |
| `inspect_stepwise_vs_nonstepwise.py` | Side-by-side comparison: same data through both paths, all stages |
| `TRAIN_POLICY_CALL_CHAIN.md` | Full call chain for the training phase |
`

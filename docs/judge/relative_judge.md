# Relative Judge: Group Evaluation & Training Integration
**17 task groups · 36 trajectories · Haiku & Sonnet · May 2026**

---

## What the relative judge does

The absolute judge scores each trajectory in isolation on a 1–5 rubric, then the score is compared against the verifier after the fact. The relative judge receives all rollouts for the same task at once and ranks them against each other. The scoring prompt explicitly instructs it to spread scores across the 1–5 range given the group it sees.

This matters because an absolute judge must simultaneously estimate task difficulty and agent quality — two confounded signals. The relative judge cancels out difficulty: it only answers "which rollout is better than which," so per-task hardness drops out of the comparison.

---

## The evaluation setup

The 200 Claude trajectories contain **17 task keys** with ≥ 2 rollouts (36 trajectories total). Two relative judge variants were evaluated:

| Judge | Model | n | Spearman ρ | AUC-ROC |
|---|---|---|---|---|
| `relative_haiku` | claude-haiku-4-5 | 36 | 0.600 | 0.864 |
| `relative_sonnet` | claude-sonnet-4-6 | 36 | **0.759** | **0.954** |

For comparison, the best absolute config on the full 200 trajectories:

| Judge | n | Spearman ρ | AUC-ROC |
|---|---|---|---|
| `screenshots_only` (abs) | 185 | 0.506 | 0.793 |

The relative judges improve on absolute by **+25pp ρ** (Sonnet) and **+16pp AUC-ROC**, measured on their common subset.

---

## The 17 groups in detail

![Group heatmap](figR1_group_heatmap.png)

Each row is one task group; each cell is one rollout. ✓ = verifier pass, ✗ = verifier fail. Scores are shown numerically.

**The fundamental problem with this evaluation set:**

| Group type | Count | Trajectories |
|---|---|---|
| All-fail groups | 9 | 18 |
| All-pass groups | 7 | 16 |
| **Mixed-outcome (pass + fail)** | **1** | **2** |

15 of 17 groups are **homogeneous** — all rollouts for the same task share the same binary verifier outcome. When all rollouts fail (or all pass), the relative judge has no discriminative signal to produce — it can only rank quality within the same outcome tier.

The impressive AUC-ROC numbers (0.864–0.954) are partly an artifact of this: the relative judge correctly assigns low scores to all-fail groups and high scores to all-pass groups, which reads as strong discrimination in aggregate but isn't really testing within-group ranking.

---

## The one real test: booking [pass, fail]

![Mixed group detail](figR3_mixed_group.png)

The single mixed-outcome group is a booking task with two rollouts — one that passed, one that failed. This is the only group where the judge's within-group ranking can be independently verified against a binary outcome.

- **Relative Haiku**: scores pass=0.85, fail=0.60 → correct ✓
- **Relative Sonnet**: scores pass=0.70, fail=0.36 → correct ✓, larger gap
- **Absolute Screenshots**: scores pass=0.20, fail=0.30 → **wrong ✗** (ranked the failure higher)

The absolute judge not only gets this group wrong but does so with high confidence — the fail trajectory happens to have a visually clean sequence that looked good to the judge in isolation. The relative judge, seeing both rollouts side by side, correctly identifies the one that actually succeeded.

---

## Within-group score spread

Even for homogeneous groups, the relative judge provides useful signal: it spreads scores within the group to reflect quality differences among rollouts that all happened to pass (or all fail).

![Group spread](figR2_group_spread.png)

The relative Sonnet judge produces meaningful within-group spread even when all rollouts share the same binary outcome. The absolute screenshots judge produces near-zero spread on most homogeneous groups (it assigns nearly identical scores to rollouts it sees as equivalent-difficulty). This is the behavioral quality signal — relevant for RL training even when the verifier provides no gradient.

---

## Translation to training

During RL training (GRPO/REINFORCE), the standard reward signal is binary: `reward = 1 if verifier passes else 0`. After group normalization (subtracting the group mean), each rollout gets an advantage in `[-1, +1]`. This is the baseline.

The taste judge adds a second signal layer. Here's what each judge produces as a training advantage for a simulated G=8 batch (5 fail, 3 pass):

![Training sim](figR4_training_sim.png)

- **Absolute judge (normalized)**: produces advantages roughly correlated with outcome, but with significant noise in the fail cluster — some failures get positive advantage because they had verbose, well-structured action text. The reward signal is diluted.
- **Relative judge (normalized)**: tighter separation. All failures get negative advantage; all passes get positive. The gradient is cleaner.
- **Binary verifier (baseline)**: perfectly clean signal, but zero gradient when all rollouts pass or all fail (no variance).

**The key training benefit of the relative judge over binary verifier:**

When a group is all-pass (common in curriculum training as the model improves), the binary verifier produces zero advantage for every rollout — no learning signal. The relative judge continues to rank quality within the passing group, pushing the model toward more efficient, coherent passes over time. This is the behavioral quality gradient that pure RL with a binary verifier misses.

### Practical wiring into the training loop

The `score_trajectory_group` and `score_trajectory_group_haiku` functions in [research/judge/judge.py](../../research/judge/judge.py) already support arbitrary group sizes. The interface maps directly onto GRPO's rollout structure:

```python
# Per training step, for each prompt p with G rollouts:
rollouts = [
    {"actions": extract_actions(traj), "outcome": verifier_passed(traj),
     "screenshots": load_screenshots(traj)}   # optional
    for traj in training_batch[p]
]

# Absolute judge (cheap, per-rollout, runs during generation)
for traj in training_batch[p]:
    abs_score = score_trajectory_haiku(task=p.task, actions=..., outcome=..., screenshots=...,
                                       blind_outcome=True)

# Relative judge (runs once per group, after all G rollouts are collected)
rel_scores = score_trajectory_group(task=p.task, rollouts=rollouts,
                                    model="claude-sonnet-4-6", blind_outcome=True)

# Combine: advantage = λ * (verifier_reward - group_mean) + (1-λ) * (rel_score - group_mean)
```

The `blind_outcome=True` flag is critical in both cases — the judge must not see the verifier result or it will short-circuit the rubric by anchoring on the binary outcome.

---

## Token Budget

![Token scaling](figT2_token_scaling.png)

Unlike absolute judges (which are called once per rollout in parallel), the relative judge is called once per group and receives all G rollouts in a single prompt. Token cost scales linearly with G.

| Config | G=2 | G=8 | G=16 | G=32 | 200k limit |
|---|---|---|---|---|---|
| `relative_haiku` (w/ screenshots) | 8,713 | 33,007 | 65,399 | 130,183 | **G ≈ 49** |
| `relative_sonnet` (text only) | 3,219 | 11,031 | 21,447 | 42,279 | G ≈ 7,600+ |

**`relative_haiku` hits the 200k context wall at G ≈ 49.** Standard GRPO runs at G=8 are fine (16.5% of context), but large-rollout experiments (G=32+) start consuming 65–130k tokens per call. At G=49+ the call would exceed the context window and fail.

`relative_sonnet` is text-only (no screenshots) and never approaches the context limit at any realistic G. For large-G training it's the safer choice.

**Comparison to absolute judge cost at G=8:**

The absolute `screenshots_only` judge at G=8 makes 8 parallel calls × 6,145 tokens = **49,160 total tokens** processed. The `relative_haiku` judge makes 1 call × 33,007 tokens = **33,007 tokens** — cheaper per batch despite having more tokens per call, because there's no redundant system prompt and task description repeated 8 times.

The stars (★) on the linear plot mark the actual measured values from the 17-group evaluation (G=2 groups), confirming the linear extrapolation model is accurate.

---

## Caveats and what would strengthen this

1. **One real discriminative test is not enough.** The 0.954 AUC comes mostly from correctly ranking all-fail < all-pass across groups — an easy task. We need a dataset where most groups have mixed outcomes (like a proper GRPO rollout batch during training, which will have natural 50/50 variance).

2. **Sonnet cost at training time.** At G=8 and ~$0.002/call, the relative Sonnet judge adds ~$0.00025/rollout. For large-scale runs (millions of rollouts), this becomes $250/M rollouts — non-trivial. Haiku relative at 0.864 AUC may be the right tradeoff.

3. **The within-group spread signal is unmeasured.** We can see that the relative judge spreads homogeneous groups better than absolute, but we haven't measured whether this spread correlates with any independent quality metric (e.g., human preference ratings, partial verifier credit). This is the key ablation to run next.

4. **Screenshot availability.** `relative_haiku` uses screenshots; `relative_sonnet` is text-only. For training rollouts where screenshots are available, `relative_haiku` is the cheaper vision-capable option (it achieved 0.864 AUC).

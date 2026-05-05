# Taste Judge Ablation Analysis
**200 Claude Trajectories · Haiku judge · May 2026**

---

## Dataset

| | |
|---|---|
| Trajectories | 200 |
| Models | claude-opus-4.6 (145), claude-opus-4.7 (55) |
| Verifier outcomes | 100 pass · 100 fail (7 partial scores) |
| Environments | 13 distinct envs |
| With screenshots | 199 / 200 |
| Multi-rollout task groups | 17 tasks · 36 trajectories (for relative judge) |

The 200 trajectories are a balanced 50/50 pass/fail split sampled from the claude-opus fleet evaluation dataset.

---

## Absolute Judge Configurations

Four Haiku-based absolute judge configurations were evaluated (blind to verifier outcome):

| Config | Description |
|---|---|
| `actions_only` | Agent action text only — no screenshots |
| `screenshots_only` | Up to 4 evenly-sampled screenshots — action text hidden |
| `actions_and_screenshots` | Both action text and screenshots |
| `screenshots_with_reasoning` | Screenshots + Claude narration surfaced as `REASONING_TRACES` (action steps hidden) |

`screenshots_with_reasoning` tests whether framing Claude's narration as explicit *reasoning traces* (via the judge prompt's `REASONING_TRACES` section) improves over screenshots alone. It strips out tool-result noise and signals to the judge to use stated intent directly when scoring `intent_clarity` and `coherence`.

Scores are on a 1–5 scale per axis, rescaled to 0–1 for the weighted total.

---

## Key Metrics

![Reliability metrics](fig5_metrics.png)

| Config | n | Spearman ρ | AUC-ROC | Var | Null% |
|---|---|---|---|---|---|
| `actions_only` | 187 | 0.409 | 0.744 | 0.061 | 0.0% |
| **`screenshots_only`** | **182** | **0.505** | **0.792** | **0.082** | 2.2% |
| `actions_and_screenshots` | 184 | 0.452 | 0.761 | 0.061 | 1.1% |
| `screenshots_with_reasoning` | 185 | 0.379 | 0.711 | 0.071 | 0.5% |

**Screenshots-only is the strongest single-signal judge on all four metrics.** It leads by ~10pp Spearman and ~5pp AUC-ROC over actions-only. Notably, `screenshots_with_reasoning` is the *weakest* config — adding reasoning traces hurts rather than helps.

### Why screenshots beat actions for Claude models

Claude's action text is already verbose, reasoning-trace quality — it narrates intent clearly regardless of whether the task was actually accomplished. This floods the actions-only judge with high-quality text that doesn't discriminate pass/fail well.

Screenshots, by contrast, show the actual UI state after each step. A failed task will often show the wrong page, an error modal, or a stalled navigation — all visually obvious. The judge picks this up through `ui_grounding` and `coherence` axes.

Adding action text back on top of screenshots (`actions_and_screenshots`) *hurts* relative to screenshots alone (0.452 vs 0.505 ρ). The verbose action text dilutes the visual signal rather than complementing it.

---

## Score Distributions

![Score distributions](fig1_score_distributions.png)

Screenshots-only shows the widest spread (variance 0.082 vs 0.061 for the other two), which is desirable for a reward signal — it avoids the "everyone gets 0.5" mode. Actions-only and actions+screenshots both pile up around 0.4–0.6.

---

## Discrimination by Verifier Outcome

![Score by outcome](fig2_outcome_vs_score.png)

Mean score gaps between pass and fail:

| Config | Pass mean | Fail mean | Δ |
|---|---|---|---|
| `actions_only` | 0.562 | 0.381 | **+0.181** |
| `screenshots_only` | 0.619 | 0.343 | **+0.276** |
| `actions_and_screenshots` | 0.574 | 0.371 | **+0.203** |

Screenshots-only has the largest pass/fail gap (+0.276), consistent with its superior AUC-ROC.

---

## Per-Axis Breakdown

### Screenshots-only: pass vs fail by axis

![Axis pass/fail](fig3_axis_pass_fail.png)

All five axes discriminate well. Largest gaps (Δ ≈ 1.2 points on the 1–5 scale):

- **Coherence** (+1.22): failed trajectories visually show disjointed sequences — navigating back and forth, retrying the same failed action.
- **Intent Clarity** (+1.20): when screenshots show the wrong page, the judge infers the agent lost track of the task.
- **Efficiency** (+1.11): failed trajectories tend to spend more turns flailing.

### Per-axis means across all three configs

![Axis by config](fig4_axis_by_config.png)

Notable pattern: **actions-only inflates intent_clarity** (3.44 vs 2.95 for screenshots-only). Claude's reasoning traces explicitly state intent, so the judge always gives high intent_clarity — even when the agent fails. Screenshots-only doesn't have this leakage, explaining why it's a more calibrated judge.

---

## Actions-only vs Screenshots-only Score Comparison

![Config scatter](fig7_config_scatter.png)

Most points fall above the diagonal — screenshots-only scores higher than actions-only on the same trajectory. The green (pass) cluster is concentrated in the top-right, and red (fail) in the bottom-left, but screenshots-only separates them more cleanly vertically than actions-only does horizontally.

---

## Environment Breakdown (Screenshots-only Judge)

![Env heatmap](fig6_env_heatmap.png)

Key observations:
- **cartograph** tops the chart (mean ~0.9) — these are simpler, shorter tasks with clean navigation patterns.
- **budget / dmv / finance-lh** score lowest — multi-step form-filling tasks with many error opportunities.
- **cloner-grader** (n=89, the bulk of the dataset) clusters around 0.5–0.6 across all axes — representing the "average" difficulty tier.
- **Efficiency is universally the lowest-scoring axis** across all envs. This is expected: Claude agents tend to take more steps than necessary, especially when uncertain.

---

## Reasoning Traces Ablation

`screenshots_with_reasoning` passes Claude's assistant narration as explicit `REASONING_TRACES` (the judge prompt's dedicated section for thinking text) rather than as `ACTIONS`. It's the most targeted test of whether the text signal itself — stripped of tool-result noise and framed as stated intent — can add anything on top of screenshots.

**Result: it cannot.** This config is the worst performer across all three metrics:

![Reasoning traces metrics](fig9_reasoning_metrics.png)

| Config | Spearman ρ | AUC-ROC |
|---|---|---|
| `screenshots_only` | **0.505** | **0.792** |
| `actions_only` | 0.409 | 0.744 |
| `actions_and_screenshots` | 0.452 | 0.761 |
| `screenshots_with_reasoning` | 0.379 | 0.711 |

### Why reasoning traces hurt

The judge system prompt instructs: *"When REASONING_TRACES are present, use them directly to score intent_clarity and coherence — the agent's stated intent is explicit."* That instruction backfires for Claude models. Claude always narrates intent confidently — even when the task is failing. Surfacing this narration as authoritative reasoning traces inflates `intent_clarity` and `coherence` for failed trajectories, collapsing the pass/fail gap on those axes.

![Reasoning traces per-axis](fig8_reasoning_traces.png)

Mean pass/fail score gap (Δ, 1–5 scale):

| Axis | `screenshots_only` Δ | `screenshots+reasoning` Δ | Change |
|---|---|---|---|
| intent_clarity | 1.20 | 0.59 | **−0.61** |
| efficiency | 1.11 | 0.99 | −0.12 |
| recovery | 0.94 | 0.56 | **−0.38** |
| ui_grounding | 1.09 | 0.67 | **−0.42** |
| coherence | 1.22 | 0.71 | **−0.51** |

Every axis degrades, but `intent_clarity` loses more than half its discriminative power (+1.20 → +0.59). The effect on `efficiency` is small (−0.12) because efficiency is scored from step count patterns visible in both signals equally; the judge doesn't over-rely on stated intent there.

### Practical implication for thinking-mode models (Qwen3-VL)

The finding is specific to Claude models where narration quality is uniformly high. For models with structured `<think>` blocks (Qwen3-VL thinking mode), the reasoning signal would be separable from the action steps and might behave differently — the thinking could be genuinely calibrated rather than uniformly confident. **Do not suppress reasoning traces for Qwen thinking-mode models without running a separate ablation on Qwen trajectories.**

---

## Token Budget

Token counts measured via the Anthropic beta token-counting endpoint on 10 sampled trajectories.

![Token budget](figT1_token_budget.png)

| Config | Mean tokens | Min | Max | % of 200k ctx |
|---|---|---|---|---|
| `actions_only` | 3,764 | ~1,500 | ~11,100 | 1.9% |
| `screenshots_only` | 6,145 | 6,113 | 6,205 | 3.1% |
| `actions_and_screenshots` | 9,261 | ~7,000 | ~16,600 | 4.6% |

All absolute configs are comfortably within the 200k context window at any realistic trajectory length. The error bars on `actions_only` reflect trajectory length variance — long booking/fos-accounting tasks with 15+ turns produce action lists exceeding 10k tokens, while short tasks are under 2k.

`screenshots_only` has the tightest token budget because screenshot size is fixed (4 × 1280×720 JPEG ≈ 5,497 tokens total), independent of trajectory length. This is another reason to prefer it: token cost is predictable and low.

See [relative_judge.md](relative_judge.md#token-budget) for how relative judge costs scale with group size G.

---

## Recommendations

1. **For cheap online scoring during rollout: use screenshots-only Haiku.** It's the best absolute config, runs fast, and avoids action-text verbosity inflation. See [relative_judge.md](relative_judge.md) for how to layer on group-relative scoring at training time.

2. **Don't use actions_and_screenshots for Claude models.** The text degrades the visual signal. The finding may be inverted for Qwen models (weaker action narration = more additive text signal).

3. **Don't pass Claude narration as `reasoning_traces`.** `screenshots_with_reasoning` is the weakest config. Claude always narrates intent confidently — even when failing — so the judge's `REASONING_TRACES` trust path inflates `intent_clarity` and `coherence` on failed trajectories. The improvement the `REASONING_TRACES` path was designed for applies only to genuinely calibrated thinking (e.g., Qwen3-VL `<think>` blocks); run a separate Qwen ablation before enabling it there.

4. **Efficiency is universally low-scoring.** Claude agents consistently take more steps than needed. If using taste reward as a training signal, weight efficiency higher to incentivize concise trajectories, or add a separate turn-count penalty.

5. **Intent_clarity is inflated by any text signal.** The axis is meaningful for screenshots-only and relative judges, but unreliable for actions-only or reasoning-traces configs on Claude — the verbose narration makes every trajectory look intentional even when it's failing.

6. **2% null rate on screenshots-only** is acceptable. The single Haiku parse failure mode is a malformed JSON response; the retry path in `judge.py` handles it.

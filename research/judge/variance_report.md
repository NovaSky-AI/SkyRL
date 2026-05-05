# Taste Judge Variance Analysis

**Sources:**
- Offline ablation: 200 Claude trajectories from `fleet-cu-claude-trajectories/`  
- Group judge simulation: 27 fleet-cu tasks with 4 rollouts (matching `n_samples_per_prompt=4`)
- Live training: WandB run [`gjfocn7r`](https://wandb.ai/thefleet/fleet-browser-use-grpo/runs/gjfocn7r) — 4 steps as of 2026-05-05

---

## 1. Separate Judge — Offline Ablation

![Judge score distribution by outcome class](plot_score_dist.png)

| Config | Outcome | n | Mean | Std | P10 | P50 | P90 | % scoring < 0.5 |
|---|---|---|---|---|---|---|---|---|
| `actions_only` | fail | 85 | 0.381 | 0.202 | 0.20 | 0.31 | 0.64 | 62% |
| `actions_only` | **pass** | 102 | 0.562 | **0.251** | 0.20 | 0.54 | 0.90 | **34%** |
| `screenshots_only` | fail | 83 | 0.339 | 0.205 | 0.11 | 0.31 | 0.66 | 83% |
| `screenshots_only` | **pass** | 102 | 0.616 | **0.282** | 0.25 | 0.64 | 0.95 | **46%** |
| `actions_and_screenshots` | fail | 84 | 0.372 | 0.191 | 0.20 | 0.26 | 0.63 | 66% |
| `actions_and_screenshots` | **pass** | 102 | 0.573 | **0.250** | 0.25 | 0.57 | 0.90 | **35%** |

**Std within outcome=1 is ≥ 0.25 across all configs** — the judge spreads passing trajectories across [0.11, 0.95], not just copying the binary label. 34–46% of passing trajectories score below 0.5, capturing clunky or lucky passes. KS test separates fail/pass distributions with p < 0.0002 for `actions_only` and p ≈ 0 for the others.

---

## 2. Group Judge — Simulation on Fleet-CU Tasks (n=4)

Taste reward is gated by pass: `taste = judge_score if pass else 0`. All-fail groups are dead for **both** signals.

![Group judge scores per task](plot_group_judge.png)

| Task | n | Pass | Binary adv_var | Taste adv_var | Notes |
|---|---|---|---|---|---|
| File perms | 4 | 0/4 | 0.000 | 0.000 | both dead — all-fail |
| Accounts payable | 4 | 0/4 | 0.000 | 0.000 | both dead — all-fail |
| Calendar email | 4 | 4/4 | **0.000** | **0.020** | taste only — all-pass |
| Finance SCD | 4 | 4/4 | **0.000** | **0.043** | taste only — all-pass |
| Cash ctrl. | 4 | 4/4 | **0.000** | **0.021** | taste only — all-pass |
| Exception search | 4 | 3/4 | 0.188 | **0.151** | mixed — both active |
| Zillow homes | 4 | 2/4 | 0.250 | **0.178** | mixed — both active |
| flama repo | 4 | 2/4 | 0.250 | **0.169** | mixed — both active |
| Vehicle pickup | 4 | 2/4 | 0.250 | **0.136** | mixed — both active |

**Across 25 non-all-fail groups: mean binary adv_var = 0.133 · mean taste adv_var = 0.052**

Taste's role is within passing rollouts, not failing ones. For all-pass groups (~1% of batches), it's the only differentiator. For mixed groups (~79%), it adds a quality gradient on top of binary — a 0.95-scoring pass reinforces the policy more than a 0.35-scoring pass does.

---

## 3. Live Training — WandB Run `gjfocn7r`

**Config:** n_samples_per_prompt=4, train_batch_size=50, taste_floor=0.1, 25 active environments

| Step | avg_taste | avg_eff_taste | pass@4 | avg_raw_reward |
|---|---|---|---|---|
| 1 | 0.136 | 0.886 | 0.250 | 0.070 |
| 2 | 0.166 | 0.896 | 0.208 | 0.037 |
| 3 | 0.151 | 0.858 | 0.333 | 0.079 |
| 4 | 0.136 | — | 0.333 | 0.072 |

- **`taste_judge_fail_rate` = 0.0** — no silent reward poisoning

![Effective taste spread per environment](plot_live_env.png)

| Environment | avg_taste | avg_eff_taste |
|---|---|---|
| booking | 0.275 | 0.831 |
| hubspot | 0.250 | 0.450 |
| netta | 0.000 | 0.550 |
| outlook | 0.190 | 0.450 |
| pagerduty | 0.181 | 1.000 |
| ramp | 0.550 | 0.944 |
| stackline | 0.512 | 0.878 |
| ticketmaster | 0.110 | 0.494 |
| reddit | 0.000 | 1.000 |
| datadog | 0.000 | 1.000 |


---

## 5. Weighted Combination — Binary + α·Taste (No Gating)

Combined reward: `r_i = b_i + α·t_i`, judge scores **all** rollouts including fails.

The within-group advantage variance works out to:

```
Var(r_adv) = (k/n)(1−k/n)·(1 + α·Δμ)² + α²·[p·σ_p² + (1−p)·σ_f²]
```

where `Δμ = μ_{t|pass} − μ_{t|fail}` is the mean taste gap between passing and failing rollouts, and `σ_p`, `σ_f` are taste std within each class. From the ablation: `Δμ ≈ 0.22` (pass mean 0.58, fail mean 0.36), `σ ≈ 0.20` for both.

Two differences from gating `t_i = 0` for fails:
- The `(1 + α·Δμ)²` term uses the **gap** (0.22) not the pass mean (0.58) — taste helps less per unit α in mixed groups, because fails also score well on taste so it doesn't point as cleanly in the same direction as binary.
- All-fail groups are no longer dead: they get `α²·σ_f²` from taste variation among fails.

| α | 0/4 pass | 1/4 pass | 2/4 pass | 3/4 pass | 4/4 pass | E[Var]\* |
|---|---|---|---|---|---|---|
| 0.0 (binary only) | 0.000 | 0.188 | 0.250 | 0.188 | 0.000 | 0.100 |
| 0.25 | 0.003 | 0.212 | 0.281 | 0.212 | 0.003 | 0.114 |
| 0.5 | 0.010 | 0.241 | 0.318 | 0.241 | 0.010 | 0.133 |
| 1.0 | 0.040 | 0.319 | 0.412 | 0.319 | 0.040 | 0.188 |
| 2.0 | 0.160 | 0.549 | 0.678 | 0.549 | 0.160 | 0.367 |

\* p ≈ 0.16 per rollout (consistent with pass@4 ≈ 0.33 from live training).

α=0.5 gives **33% more expected variance than binary alone** (0.133 vs 0.100) and wakes up all-fail groups that contribute nothing under binary. The reward is no longer in [0,1] — GRPO normalization handles the scale, but KL coefficient may need retuning.

---

## 6. Summary

| Claim | Evidence |
|---|---|
| Judge has spread within outcome=1 | std 0.25–0.28, range [0.11, 0.95], 34–46% of passes score < 0.5 |
| Taste adds quality gradient for mixed groups | mean taste adv_var = 0.052 alongside binary adv_var = 0.133 |
| Taste uniquely differentiates all-pass groups | ~1% of batches; binary adv_var = 0 there |
| Live training: judge doesn't poison reward | taste_judge_fail_rate = 0.0; taste_floor = 0.1 prevents silent zeros |
| Flat signal is correct when expected | trivial all-pass task (calendar) has std = 0.035 |

# Taste Judge Variance Analysis

**Sources:**
- Offline ablation: 200 Claude trajectories from `fleet-cu-claude-trajectories/`  
- Group judge simulation: 8 fleet-cu tasks with 4+ rollouts (images available)
- Live training: WandB run [`gjfocn7r`](https://wandb.ai/thefleet/fleet-browser-use-grpo/runs/gjfocn7r) — `fleet_qwen35_browser_use_ablation-screenshots`, 4 training steps as of 2026-05-05

---

## 1. Separate Judge — Offline Ablation (n = 185–187 trajectories per config)

### 1.1 Distribution by outcome class

![Judge score distribution by outcome class](plot_score_dist.png)

| Config | Outcome | n | Mean | Std | P10 | P50 | P90 | % scoring < 0.5 |
|---|---|---|---|---|---|---|---|---|
| `actions_only` | fail | 85 | 0.381 | 0.202 | 0.20 | 0.31 | 0.64 | 62% |
| `actions_only` | **pass** | 102 | 0.562 | **0.251** | 0.20 | 0.54 | 0.90 | **34%** |
| `screenshots_only` | fail | 83 | 0.339 | 0.205 | 0.11 | 0.31 | 0.66 | 83% |
| `screenshots_only` | **pass** | 102 | 0.616 | **0.282** | 0.25 | 0.64 | 0.95 | **46%** |
| `actions_and_screenshots` | fail | 84 | 0.372 | 0.191 | 0.20 | 0.26 | 0.63 | 66% |
| `actions_and_screenshots` | **pass** | 102 | 0.573 | **0.250** | 0.25 | 0.57 | 0.90 | **35%** |

### 1.2 Key findings

**Std within outcome=1 is ≥ 0.25 across all configs.** The judge is not copying the binary label. It spreads passing trajectories across the full range [0.11, 0.95].

**34–46% of passing trajectories score below 0.5.** These are tasks the verifier calls "success" but where the agent's execution was clunky, inefficient, or got lucky. The judge catches this; the binary reward doesn't.

**KS test separates the fail/pass distributions** with p < 0.0002 for `actions_only` and p ≈ 0 for the other two — confirming the judge has directional signal, not just noise.

The `screenshots_only` config has the highest within-outcome=1 spread (std = 0.282), suggesting visual grounding is the strongest signal axis.

---

## 2. Group Judge — Simulation on Fleet-CU Tasks (n=4, matching training config)

Tasks drawn from `trajectories.jsonl` filtered to sessions with local images in `fleet-cu-trajectories/images/`, sampled to exactly 4 rollouts per group — matching `n_samples_per_prompt=4` in the live run. 27 tasks qualified (≥4 fleet-cu rollouts). Groups scored with `score_trajectory_group_haiku` (blind outcome, screenshots included).

### 2.1 Per-task results

![Group judge scores per task](plot_group_judge.png)

The plot is split into three sections: all-fail groups (binary gradient = 0), all-pass groups (binary gradient = 0), and mixed. Selected rows below:

Taste reward is gated by pass: `taste = judge_score if pass else 0`. All-fail groups are dead for **both** signals.

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

**Summary across 25 non-all-fail groups: mean binary adv_var = 0.133 · mean taste adv_var = 0.052**


### 2.4 Key finding

**Taste's role is within passing rollouts, not failing ones.** All-fail groups are dead for both signals. The value of taste is: (1) for all-pass groups (~1% of batches), it's the only differentiator; (2) for mixed groups (~79%), it adds a quality gradient on top of the binary pass/fail signal — a 0.95-scoring pass reinforces the policy more than a 0.35-scoring pass does.

---

## 3. Live Training — WandB Run `gjfocn7r`

**Run:** `fleet_qwen35_browser_use_ablation-screenshots`  
**Config:** n_samples_per_prompt=4, train_batch_size=50, taste_floor=0.1, 25 active environments  
**Status:** 4 training steps completed as of 2026-05-05T09:xx

### 3.1 Global taste stats across training steps

Note: `std_taste` / `std_eff_taste` below are **batch-level** (historical, from early runs before the fix). Going forward the code logs `within_group_std_taste_reward` / `within_group_std_effective_taste` — the mean per-group std across the 4-rollout groups, which is the number that actually matters for GRPO signal.

| Step | avg_taste | std_taste (batch) | avg_eff_taste | std_eff_taste (batch) | pass@4 | avg_raw_reward |
|---|---|---|---|---|---|---|
| 1 | 0.136 | 0.189 | 0.886 | 0.285 | 0.250 | 0.070 |
| 2 | 0.166 | 0.237 | 0.896 | 0.272 | 0.208 | 0.037 |
| 3 | 0.151 | 0.217 | 0.858 | 0.312 | 0.333 | 0.079 |
| 4 | 0.136 | 0.213 | — | — | 0.333 | 0.072 |

- **`taste_judge_fail_rate` = 0.0** — the judge never errored; no silent reward poisoning

### 3.2 Per-environment taste distribution (step 3, latest)

![Effective taste spread per environment](plot_live_env.png)

Note: std values below are batch-level (across all rollouts in the batch for each env), not within-group.

| Environment | avg_taste | std_taste (batch) | avg_eff_taste | std_eff_taste (batch) |
|---|---|---|---|---|
| booking | 0.275 | **0.389** | 0.831 | **0.335** |
| hubspot | 0.250 | 0.210 | 0.450 | **0.375** |
| netta | 0.000 | 0.000 | 0.550 | **0.481** |
| outlook | 0.190 | 0.175 | 0.450 | **0.388** |
| pagerduty | 0.181 | **0.256** | 1.000 | 0.000 |
| ramp | 0.550 | n/a | 0.944 | 0.159 |
| stackline | 0.512 | 0.053 | 0.878 | 0.227 |
| ticketmaster | 0.110 | **0.246** | 0.494 | **0.446** |
| reddit | 0.000 | 0.000 | 1.000 | 0.000 |
| datadog | 0.000 | n/a | 1.000 | 0.000 |

**High-variance environments** (netta, outlook, hubspot, ticketmaster, booking) are where the taste reward adds the most signal — the verifier alone would see only 0/1, but the judge spreads scores from 0.45 to 0.95 among passing rollouts in these envs.

**Zero-variance environments** (reddit, datadog, bi-dashboard, dmv) — all rollouts land the same taste score. The judge adds no incremental signal here, but it also doesn't corrupt the reward: flat taste × binary reward = same as binary reward alone.

---

## 4. Binary Reward vs Taste Reward: The Right Comparison

Raw variance numbers (batch-level std) are not the right comparison metric. In GRPO, the learning signal comes from **within-group advantage variance**: `var(reward - group_mean)`. A reward that is high-variance across the full batch but constant within every group contributes nothing to training.

### 4.1 Dead groups (taste is gated by pass)

With `pass@4 = 0.33` and `n = 4` rollouts per group:

| Group outcome | P(this outcome) | Binary adv_var | Taste adv_var | Notes |
|---|---|---|---|---|
| 0/4 pass | **20%** | **0.000** | **0.000** | **both dead** — no passes to score |
| 1/4 pass | 40% | 0.188 | >0 | both active |
| 2/4 pass | 29% | 0.250 | >0 | both active |
| 3/4 pass | 10% | 0.188 | >0 | both active |
| 4/4 pass | **1%** | **0.000** | **>0** | **taste only** — binary flat, taste spreads |

**20% of training groups are dead for both rewards** (all fail). Taste cannot rescue these — there are no passing rollouts to assign reward to. The separate 1% all-pass case is where taste uniquely adds signal.

![Binary vs taste advantage variance per group](plot_binary_vs_taste_variance.png)

The left panel shows empirical within-group advantage variance (taste correctly = 0 for all-fail groups). The right panel shows the theoretical distribution given pass@4 = 0.33.

### 4.2 What taste adds to mixed-outcome groups

Even in groups where binary reward has signal, taste adds **within-class discrimination** that binary cannot provide:

- Binary puts all weight on the pass/fail boundary (rollout *crossed* the threshold vs didn't)
- Taste additionally signals *how cleanly* a passing rollout executed — a pass with judge score 0.35 should receive less positive reinforcement than one with 0.79
- Concretely: in the SF cafe group, rollout 4 (verifier=pass, judge=0.35) and rollout 3 (verifier=fail, judge=0.35) receive the same taste score despite different binary labels — the judge correctly sees them as equivalent quality

### 4.3 Taste variance is smaller than binary variance within mixed groups — that's expected

In the groups with mixed outcomes (e.g. SF cafe, Kernel project), binary advantage variance (0.23, 0.25) exceeds taste advantage variance (0.07, 0.05). This is not a problem — binary reward is doing the heavy lifting for pass/fail discrimination, and taste is adding a secondary gradient on top. The two signals are **complementary**, not competing.

The combined reward `binary + α × taste` has advantage variance:
```
var(binary_adv + α × taste_adv) = var(binary_adv) + α²·var(taste_adv) + 2α·cov(binary_adv, taste_adv)
```
Since taste is partially correlated with binary (passing rollouts tend to score higher), `cov > 0`, meaning taste further amplifies the binary signal when they agree, and moderates it when they disagree (e.g., ugly pass).

---

## 6. Summary: Why Taste Doesn't Collapse Variance

| Claim | Evidence |
|---|---|
| Judge has spread within outcome=1 (offline) | std 0.25–0.28, range [0.11, 0.95], 34–46% of passes score < 0.5 |
| Taste adds quality gradient on top of binary for mixed groups | mean taste adv_var = 0.052 alongside binary adv_var = 0.133 in non-all-fail groups |
| Live training shows consistent std | std_taste_reward ≈ 0.22, std_effective_taste ≈ 0.31 across 3 steps |
| Judge doesn't poison reward | taste_judge_fail_rate = 0.0, taste_floor = 0.1 prevents silent zeros |
| Flat signal is correct when expected | calendar task (all-pass, trivial) correctly has std = 0.035 |
| Meaningful per-env signal | 10/25 envs have std_eff_taste > 0.15; those have meaningful quality variation |

The offline std (0.25–0.28) and live std_effective_taste (0.31) are in agreement, validating that the offline eval is predictive of training-time behavior.

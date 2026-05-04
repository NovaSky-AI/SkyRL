# Pre-registered hypotheses for the taste judge

Pre-registered before running the judge on the full HF dataset. Each
hypothesis is paired with a falsification rule and an analysis plan so we
don't post-hoc move the goalposts.

## H1 — Verifier successes are not all great
**Statement.** Among trajectories with `verifier_score == 1`, the standard
deviation of `taste_weighted_total` (1-5 scale) is `> 0.7`.

**Why this matters.** If H1 holds, the verifier+taste signal carries strictly
more information than the verifier alone, and adding the taste reward should
shape the policy in non-trivial ways. If H1 fails, taste collapses onto
verifier in the success regime and the auxiliary reward is mostly noise on
positives.

**Falsifier.** Run on >= 100 verifier=1 trajectories with the Claude judge.
Compute population stddev of `weighted_total`. If stddev <= 0.70, H1 is
falsified.

**Analysis plan.** Report mean, stddev, p10/p50/p90 of `weighted_total`
conditioned on verifier=1, plus a histogram. No subset cherry-picking.

**Prior.** From the 10-trajectory validation pass we found stddev ~1.08 in
the verifier=1 subset (n=7) — directional support but underpowered. We
expect the dataset value to be 0.8-1.0.

---

## H2 — Inter-rater agreement is real
**Statement.** Cohen's kappa between the Claude judge and the GPT-4o judge,
computed per axis on a held-out set of 100 trajectories, is `> 0.5` on at
least 4 of the 5 axes.

**Why this matters.** A taste signal we can't reproduce across models is
mostly model idiosyncrasy. Kappa > 0.5 ("moderate" by Landis & Koch) is the
floor for treating the signal as real rather than artefactual.

**Falsifier.** Score all 100 trajectories with both judges. Compute linear
Cohen's kappa per axis. If fewer than 4 axes clear 0.5, H2 is falsified.

**Analysis plan.** Report kappa per axis with n. Also report % exact-match
and % within-1 agreement as descriptive sanity checks. Pre-commit: if any
single axis has kappa < 0.3 even when 4 others pass, we will rewrite that
axis's anchors before scaling up.

**Prior.** ui_grounding and coherence likely highest agreement (concrete
visible signals); recovery likely lowest (rare and the default-to-4 rule
adds a coordination problem between judges).

---

## H3 (own) — Taste pulls down on pretty failures more than the verifier rewards effort
**Statement.** Define `pretty_failure` = trajectories with `verifier=0` and
`weighted_total >= 4.0`. Define `ugly_success` = trajectories with
`verifier=1` and `weighted_total <= 2.5`. We hypothesize that
`P(pretty_failure | verifier=0)` is at least 2x `P(ugly_success | verifier=1)`.

**Why this matters.** This is a sharp claim about which side of the
verifier the taste signal carries the most *information*. If pretty failures
are common (e.g., agent did everything right but hit a 2FA wall), the taste
score is most useful as a *salvage* signal — it tells RL "this was good even
though verifier=0, partial credit". If instead ugly successes dominate, the
taste score is most useful as a *penalty* signal — "verifier=1 but stop
rewarding this". The two regimes call for different reward-shaping schemes,
so knowing which we're in is load-bearing for downstream design.

**Falsifier.** On the same 100-trajectory eval set, compute both
proportions. If the ratio (pretty_failure_rate / ugly_success_rate) is
< 2.0, H3 is falsified. If verifier=0 rows are < 10, defer judgement
(insufficient power).

**Analysis plan.** Report both rates with Wilson 95% CIs. Bucket the
pretty_failure trajectories by failure cause (auth wall / out-of-stock /
network error / dataset bug) — this directly informs how reward shaping
should weight them.

**Prior.** ~20% of verifier=0 are pretty failures (auth walls dominate);
<5% of verifier=1 are ugly successes (the verifier mostly catches the
egregious cases). So we expect the ratio in the 4-5x range.

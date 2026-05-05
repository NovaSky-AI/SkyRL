# Judge Calibration Eval — Global Step 5

**Date:** 2026-05-05  
**Splits:** `baseline_global_step_5/` (no taste) vs `exp_global_step_5/` (screenshots-only Haiku taste, `SKYRL_TASTE_BLIND_ACTIONS=1`)  
**Script:** `research/judge/training_traj_eval.py` — run with `--workers 16 --stability-n 20`

**Too early to conclude anything.** Step 5 is ~5% of a typical training run. The metrics below are baselines to compare against at steps 20–30+, not evidence for or against the judge.

---

## FP Rate (key metric to watch)

Passing trajectories (verifier reward > 0) where the judge scores low (< 0.4) — the judge thinks the model got lucky or solved it clunkily.

| Step | Baseline FP rate | Exp FP rate |
|---|---|---|
| 5 | 61.5% (16/26) | 60.0% (15/25) |

Identical at step 5. If taste training is working, exp FP rate should fall while baseline stays flat. Re-run at steps 20, 50, 100.

## Judge Noise Floor

RMSD 0.148 on 0–1 scale (18 cache-busted re-scores, Spearman 0.63). ~15% per-call noise on absolute scores; rankings moderately stable. This sets the ceiling on how clean the quality gradient can be during training regardless of calibration.

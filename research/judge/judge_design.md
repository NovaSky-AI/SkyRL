# Taste Judge — Design Notes

## 1. Rubric summary

Five 1-5 axes scored independently, then combined with fixed weights:

| axis            | weight | what it captures                                  |
| --------------- | ------ | ------------------------------------------------- |
| intent_clarity  | 0.20   | each action has an obvious purpose                |
| efficiency      | 0.20   | action count near optimal; no wasted motion       |
| recovery        | 0.20   | diagnoses unexpected events instead of retrying   |
| ui_grounding    | 0.25   | clicks/typing land on right elements              |
| coherence       | 0.15   | one mind, one plan from start to finish           |

`weighted_total = 0.20 IC + 0.20 EFF + 0.20 REC + 0.25 UI + 0.15 COH`,
range 1.00-5.00. ui_grounding is upweighted because mis-grounding invalidates
every other axis; coherence is downweighted because it correlates ~0.6 with
intent_clarity. `recovery` defaults to 4 when no unexpected event occurred,
otherwise the two judges fight over an undefined variable.

## 2. Prompt template (excerpt)

```
You are a strict but fair "taste judge" for computer-use agent trajectories.
You will receive: TASK, ACTIONS, OUTCOME, SCREENSHOTS (up to 4).

Score 1-5 (integers only) on FIVE independent axes:
  intent_clarity, efficiency, recovery, ui_grounding, coherence
Anchors: 1=clearly bad, 3=mediocre, 5=excellent.
Do NOT let OUTCOME inflate or deflate scores.
When uncertain between adjacent scores, pick the LOWER one.

Return STRICT JSON:
{ "scores": { ... five integers ... }, "rationale": "<2-4 sentences>" }
```

Full prompt in `judge.py:SYSTEM_PROMPT`. Screenshots, if provided, are
sampled to 4 evenly-spaced frames (first/last guaranteed) and attached as
vision blocks.

## 3. Calibration notes (validation pass)

Scored 10 hand-constructed trajectories across the realistic distribution
(synthetic; the HF dataset was unreachable from the sandbox — flagged in
`sample_scores.json`).

- **mean = 3.76, stddev = 1.15** — good spread.
- **verifier=1 subset (n=7): mean 3.99, stddev 1.08** — exceeds the H1
  threshold of 0.70 directionally, though underpowered.
- **verifier=0 subset (n=3): mean 3.22** — includes one pretty failure
  (`syn_007`, total 4.55, halted at a 2FA wall) the verifier misclassifies.
- **Binding axis:** ui_grounding pulled the total down most often (3/10),
  validating the 0.25 weight.
- **"Pick the lower one"** anchor mattered: without it, my own scores
  drifted +0.5 on borderline cases.

## 4. Known failure modes

1. **Outcome bleed.** Even with the explicit instruction, judges round
   generously when verifier=1 is in context. Ablation: hide the outcome on
   50% of calls and check for a discontinuity at the boundary.
2. **Recovery underdetermined.** Most trajectories see no unexpected event,
   so the axis defaults to 4 and carries little signal except on the
   10-20% with real surprises. Consider only emitting recovery on those.
3. **Long-tail action vocab.** Embedded screenshots inside actions (as
   base64) can blow context; truncate fields > 2 KB before judging.
4. **Length bias.** Longer trajectories tend to score lower on efficiency
   purely from surface area. Watch corr(n_actions, total); if < -0.5,
   re-anchor efficiency.
5. **Inter-rater on coherence.** Coherence is the most subjective axis;
   expect kappa < 0.5. If so, rewrite anchors with textual examples.
6. **Prompt injection via OCR.** Hostile screenshot text could steer the
   judge. Not yet hardened.

## 5. Scaling to 10k trajectories

**Cost.** ~3k input + ~250 output tokens per call. Claude Sonnet at
$3/$15 per MTok = ~$0.013/trajectory; 10k = **~$130** per judge. Adding
GPT-4o for inter-rater roughly doubles it to **~$260**.

**Batching.** Each call is independent; use the existing ThreadPoolExecutor
in `score_dataset.py` with workers=16-32. Both APIs serve >50 RPS at our
tier — that stays inside our concurrent-request budget. Don't switch to
async unless RPM-limited.

**Caching.** SHA-256 store keyed on `(task, actions, outcome, model)`
already deduplicates re-runs (~10 MB at 10k). Mostly helps on crash
retries; for live RL the same rollout rarely repeats.

**Sampling.** Don't judge every rollout. Score 1-in-K (K=4-8) and use the
moving average as the auxiliary reward; verifier still fires every step.
Cuts cost ~4x.

**Robustness.** `judge.py` fallback returns a `None`-shaped result on any
exception. The training loop must treat `None` as "no taste signal, use
verifier only" — never 0 — or a rate-limit storm becomes phantom negative
reward.

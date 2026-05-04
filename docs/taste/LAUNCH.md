# Taste-Judge GRPO Launch Recipe

Wires `research/judge/judge.py` into the SkyRL Fleet GRPO training loop.
Reward shape is **GATED TASTE**:

```
effective_taste = max(taste_floor, taste_score)   # 1.0 if judge fails / None
reward          = verifier_reward * effective_taste
```

Blended only on the terminal step of each rollout, with a 10s judge timeout
and verifier-only fallback (`effective_taste = 1.0`, so reward collapses to
`verifier_reward`) on timeout/exception/None.

### Why gated > additive

The previous additive shape `R = alpha * verifier + (1-alpha) * taste`
rewarded "pretty failures" — a trajectory that fails the verifier (v=0)
but narrates clean intent (t high) earned `(1-alpha) * t > 0`, which
incentivized the policy to learn good-looking failure modes. Gated taste
closes this hack: `verifier=0` forces `reward=0` regardless of taste, so
there is zero gradient toward pretty-failure mimicry. Among successes,
ugly successes still earn `floor * verifier` (default `floor=0.1`) so GRPO
sees within-group taste variance and can prefer pretty successes; setting
`floor=1.0` collapses the shape to pure verifier and serves as a clean
ablation baseline. **The floor is set to 0.1 (not 0.3) because offline
analysis showed mean rescaled taste of verifier=1 trajectories is ~0.13;
floor=0.3 would clip nearly all successes and kill within-group variance.
Re-tune floor after a 50-100 step pilot using the empirical effective_taste
P25 logged in WandB.**

## One-block launch

```bash
# 0. From your machine:
cd /tmp && rm -rf skyrl-fleet && git clone https://github.com/fleet-ai/skyrl-fleet.git
cd /tmp/skyrl-fleet

# 1. Apply the env patch (adds taste_floor config, _apply_taste_reward helper,
#    and updates the three terminal returns + get_metrics).
git apply /Users/alliegu/Desktop/fleet/integration/env.py.diff

# 2. Vendor the taste-judge package into the workdir Python path.
cp -r /Users/alliegu/Desktop/fleet/integration/skyrl_taste skyrl-gym/skyrl_taste
cp -r /Users/alliegu/Desktop/fleet/research/judge research/judge

# 3. Drop the new YAML config into tasks/.
cp /Users/alliegu/Desktop/fleet/integration/configs/openenv-fleet-grpo-vl-taste.yaml \
   tasks/openenv-fleet-grpo-vl-taste.yaml

# 4. Sky launch with the new yaml + new env vars (judge keys are NEW; the rest
#    are unchanged from the existing VL launch).
sky launch tasks/openenv-fleet-grpo-vl-taste.yaml \
  --env FLEET_API_KEY="$FLEET_API_KEY" \
  --env WANDB_API_KEY="$WANDB_API_KEY" \
  --env AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
  --env AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
  --env ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
  --env OPENAI_API_KEY="$OPENAI_API_KEY"
```

## Required env vars

- `ANTHROPIC_API_KEY` — **required**. Default judge backend (Claude via
  `research/judge/judge.py`). Without it the judge import fails and the env
  silently falls back to verifier-only reward (you'll see
  `taste_judge_failed=True` in WandB).
- `OPENAI_API_KEY` — **only required if running inter-rater agreement
  passes** (GPT-4o judge for cross-checking Claude scores during eval). Not
  needed for the standard training run.
- `FLEET_API_KEY`, `WANDB_API_KEY`, `AWS_ACCESS_KEY_ID`,
  `AWS_SECRET_ACCESS_KEY` — same as the upstream VL launch.

**Important:** Invoke `judge.py` with `blind_outcome=True` at training time
to suppress outcome bleed (Stream 4 finding — when the judge sees the
verifier outcome, taste scores correlate ~0.7 with verifier and the
shaping signal collapses to a noisy duplicate of the binary reward). The
async wrapper in `skyrl_taste/judge.py` handles this; double-check the
flag is forwarded if you swap the wrapper.

## WandB metrics to watch

- `reward/train/mean` — gated reward; bounded above by verifier mean.
- `env/taste_reward` — judge's [0,1] raw score per trajectory.
- `env/effective_taste` — `max(floor, taste_reward)`; what actually
  multiplies the verifier.
- `env/verifier_reward` — raw binary verifier per trajectory.
- `env/taste_floor` — the configured floor; sanity-check.
- `env/taste_judge_failed` — should stay near 0; spikes mean Anthropic
  outage or judge parse failures (auto-fallback to pure verifier engaged).
- **Cross-check**: in within-group runs, plot Pearson(`taste_reward`,
  `verifier_reward`). If correlation collapses below ~0.3, the judge is
  scoring a different signal than the verifier — that's the expected case
  and where the shaped-reward gradient comes from. If it climbs above
  ~0.7, suspect outcome bleed (re-verify `blind_outcome=True`).
- `reward/train/variance_per_prompt` and `signal_ratio` (from
  `integrations/fleet/reward_metrics.py`) should *increase* relative to a
  verifier-only baseline on groups with mixed pretty/ugly successes.

## Rollback

**Runtime kill switch (no redeploy):**
```bash
sky exec <cluster> "echo SKYRL_TASTE_DISABLED=1 >> ~/.bashrc && pkill -HUP -f main_fleet"
# or update the SkyPilot env block and re-launch with --env SKYRL_TASTE_DISABLED=1
```
This makes `score_trajectory_async` return `None`, the env's
`effective_taste` becomes `1.0`, and reward collapses to pure verifier.

**Full revert (uncheck-out the patch):**
```bash
cd /tmp/skyrl-fleet
git apply -R /Users/alliegu/Desktop/fleet/integration/env.py.diff
rm -rf skyrl-gym/skyrl_taste research/judge
```

## Two-knob ablation (floor x grpo_norm_by_std)

| floor \ grpo_norm_by_std | true (default)                                                  | false (recommended w/ gated taste)                                  |
|---|---|---|
| 0.0 (pure multiplicative) | Ugly successes get R=0; group std collapses on all-ugly groups. Heavy gradient damping; expect slow learning. | Same dynamics, undamped; risk of policy ignoring ugly successes entirely. |
| 0.1                       | Tiny within-success variance; std-norm wipes most of the gradient. | Tight bonus for pretty successes; conservative shaping.            |
| 0.1 (default)             | Tiny within-success variance from floor itself; std-norm still wipes most of the gradient. | **Headline candidate.** Multiplicative-with-cushion; closes hack and matches the empirical taste distribution. |
| 0.3                       | Within-success std damped; offline data shows nearly all successes clip to floor — kills the signal. | Heavier shaping; only sensible if live taste distribution skews high. |
| 0.5                       | Floor close to pretty-mid; less taste differentiation among successes. | Shallower shaping; useful as sensitivity check. |
| 1.0 (pure verifier)       | **Identical to upstream baseline.** A/B control, no taste in std. | Identical to upstream too (no taste in std).                       |

Recommended order: run cell `(0.1, false)` first as the headline candidate,
then `(0.1, true)` to measure the std-norm effect, then `(1.0, true)` as
the upstream baseline. `(0.0, false)` is a diagnostic: confirms the gate
itself bites (ugly successes get zero) without floor compensation.

## Risks / gotchas

- **Judge latency budget**: 10s timeout x `n_samples_per_prompt=4` at
  `train_batch_size=50` = ~200 concurrent judge calls per training step.
  Anthropic rate limits will throttle you before the GPU does. Watch
  `taste_judge_failed` — sustained >10% means raise the limit or batch.
- **Reward range**: gated reward is in `[0, 1]` — same as verifier — so
  pass@n threshold (`reward >= 1.0` in `reward_metrics.py:79-82`) only
  triggers on `(verifier=1, taste=1.0)`. With `floor=0.1` and `verifier=1`,
  blended max is 1.0 only when `taste_score=1.0`. **Pass@n will look
  worse than verifier-only**; report it alongside the new gated-reward
  mean, and consider plotting `verifier_reward >= 1.0` as a separate
  pass@n line for direct comparison to the baseline.
- **Outcome bleed**: confirmed Stream 4 risk if the judge ever sees the
  verifier outcome. Keep `blind_outcome=True` in `score_trajectory_async`.

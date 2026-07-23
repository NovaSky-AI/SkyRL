# REPO-R: Entropy-Aware Advantage Scaling

REPO-R rescales per-token advantages using the latest policy's log-probabilities, boosting rare
correct actions and attenuating common ones (and softening penalties on rare incorrect actions).
It is derived from the general REPO advantage by setting the per-token penalty coefficient to
`beta = zeta * |A(s, a)|`. See the [REPO paper (ICLR 2026), Appendix D.2](https://arxiv.org/pdf/2603.11682).

## Formula

Starting from `A_REPO = A - beta * L`, REPO-R uses the raw log-prob `logp = log pi_theta(a | s)`
(a practical stand-in for the centered `L`, which would require recomputing the full expected
log-prob after every gradient step):

- `A > 0`:  `A * (1 - zeta * logp)`, then clamped to `>= 0`
- `A < 0`:  `A * (1 + zeta * logp)`, then clamped to `<= 0`

`logp` is detached (stop-grad) and taken from the current forward pass, so the rescaling only
reshapes the advantage magnitude — the gradient still flows through the configured policy loss
applied to the rescaled advantage.

## Usage

REPO-R is a **standalone advantage transform** applied before the configured policy loss, so it
composes with any `policy_loss_type` (`regular`, `dual_clip`, `gspo`, `rollout_is`, ...). Enable it
independently of the loss:

```bash
trainer.algorithm.repo.enabled=true
trainer.algorithm.repo.zeta=0.001
# optionally combine with a specific loss, e.g.:
# trainer.algorithm.policy_loss_type="rollout_is"
```

### Adaptive controller (optional)

Set `trainer.algorithm.repo.target_entropy` to a float to enable the once-per-iteration adaptive
controller. It halves/doubles `|zeta|` (flipping the sign at the `zeta_min` boundary) to drive the
measured `policy/policy_entropy` toward the target:

```bash
trainer.algorithm.repo.target_entropy=0.3
trainer.algorithm.repo.zeta_min=1e-4
trainer.algorithm.repo.zeta_max=0.05
```

The REPO paper clips `|zeta|` to `[1e-4, 0.05]` for REPO-R (these are the defaults).

Leave `target_entropy` unset (`null`) to keep `zeta` fixed. The current coefficient is logged as
`policy/repo_r_zeta` each step.

See `run_repo_r_gsm8k.sh` for a full colocated GSM8K example.

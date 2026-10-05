"""megatron_loss_output gives the same gradient in both Megatron normalization modes.

Simulates Megatron's fixed factors on CPU for every (DP rank, CP rank, microbatch):
- default mode: schedule multiplies a 2-tuple loss by cp / num_microbatches, DDP averages over dp * cp;
- per-token mode (calculate_per_token_loss): 3-tuple loss left unscaled, DDP sums over dp * cp,
  and SkyRL runs finalize_model_grads without a token count (its loss is normalized through the
  pre-scaled advantages), so there is no final division.
CP ranks compute the full-sequence loss but only their own tokens' gradient flows back, as with
SkyRL's all-gathered CP log-probs.
"""

import pytest
import torch

from skyrl.backends.skyrl_train.distributed.megatron.loss_scaling import (
    megatron_loss_output,
)

DIM = 4
REG_COEF = 0.05


def _make_data(dp, n_mb, seed=0):
    gen = torch.Generator().manual_seed(seed)
    lens = [3, 7, 4, 10, 2, 6, 5, 9, 8, 3, 6, 4]
    seqs = [(torch.randn(n, DIM, generator=gen), torch.randn(n, generator=gen)) for n in lens[: dp * n_mb * 2]]
    n_total = sum(len(a) for _, a in seqs)
    # (dp rank, microbatch) -> sequences; advantages pre-scaled by the mini-batch token count (token_mean)
    placement = {(d, m): [] for d in range(dp) for m in range(n_mb)}
    for i, (x, a) in enumerate(seqs):
        placement[(i % dp, (i // dp) % n_mb)].append((x, a / n_total))
    return seqs, n_total, placement


def _cp_local(n, cp, rank):
    idx = torch.arange(n)
    return (idx * cp) // n == rank


def _terms(w, mb_seqs, cp, cp_rank):
    """Pre-scaled policy sum and per-microbatch mean regularizer, local-gradient CP semantics."""
    policy, reg_sum, count = 0.0, 0.0, 0
    for x, a in mb_seqs:
        mask = _cp_local(len(a), cp, cp_rank)
        tok = -a * torch.nn.functional.logsigmoid(x @ w)
        reg = REG_COEF * (x @ w) ** 2
        policy = policy + torch.where(mask, tok, tok.detach()).sum()
        reg_sum = reg_sum + torch.where(mask, reg, reg.detach()).sum()
        count += len(a)
    return (
        policy,
        (reg_sum / count if count else torch.zeros(())),
        int(sum(_cp_local(len(a), cp, cp_rank).sum() for _, a in mb_seqs)),
    )


def _simulate(w0, placement, dp, cp, n_mb, n_padding_mb, per_token):
    n_total_mb = n_mb + n_padding_mb
    n_real = n_mb
    grad_sum = torch.zeros(DIM)
    for d in range(dp):
        for c in range(cp):
            for m in range(n_total_mb):
                w = w0.clone().requires_grad_(True)
                if m < n_mb:
                    policy, reg, _ = _terms(w, placement[(d, m)], cp, c)
                else:  # padding microbatch: loss_mask is zero, so both terms are zero
                    policy, reg = (w * 0).sum(), (w * 0).sum()
                out = megatron_loss_output(
                    policy,
                    reg,
                    {},
                    num_microbatches=n_total_mb,
                    num_real_microbatches=n_real,
                    dp_size=dp,
                    per_token_loss=per_token,
                    num_tokens=7,
                )
                if per_token:
                    loss, num_tokens, _ = out  # schedule leaves a 3-tuple loss unscaled
                    assert int(num_tokens) == 7
                else:
                    loss, _ = out
                    loss = loss * cp / n_total_mb  # schedule, 2-tuple path
                loss.backward()
                grad_sum += w.grad
    # per-token: DDP sums, no final division; default: DDP averages over dp * cp
    return grad_sum if per_token else grad_sum / (dp * cp)


def _reference(w0, placement, dp, n_mb):
    w = w0.clone().requires_grad_(True)
    total = 0.0
    for d in range(dp):
        for m in range(n_mb):
            policy, reg, _ = _terms(w, placement[(d, m)], cp=1, cp_rank=0)
            total = total + policy + reg / (dp * n_mb)
    total.backward()
    return w.grad


@pytest.mark.parametrize(
    "dp,cp,n_mb,n_padding_mb", [(1, 1, 1, 0), (1, 1, 4, 0), (2, 1, 2, 0), (1, 2, 2, 0), (2, 2, 3, 1)]
)
def test_both_modes_match_reference(dp, cp, n_mb, n_padding_mb):
    _, _, placement = _make_data(dp, n_mb)
    w0 = torch.randn(DIM, generator=torch.Generator().manual_seed(1))
    ref = _reference(w0, placement, dp, n_mb)
    default = _simulate(w0, placement, dp, cp, n_mb, n_padding_mb, per_token=False)
    per_token = _simulate(w0, placement, dp, cp, n_mb, n_padding_mb, per_token=True)
    torch.testing.assert_close(default, ref, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(per_token, ref, rtol=1e-5, atol=1e-7)


def test_two_tuple_under_per_token_mode_is_off_by_dp_times_cp():
    """Why per-token mode needs its own branch: the default-mode 2-tuple with the flag on (DDP sums)."""
    dp, cp, n_mb = 2, 2, 2
    _, _, placement = _make_data(dp, n_mb)
    w0 = torch.randn(DIM, generator=torch.Generator().manual_seed(1))
    grad_sum = torch.zeros(DIM)
    for d in range(dp):
        for c in range(cp):
            for m in range(n_mb):
                w = w0.clone().requires_grad_(True)
                policy, _, _ = _terms(w, placement[(d, m)], cp, c)
                loss, _ = megatron_loss_output(
                    policy, 0.0, {}, num_microbatches=n_mb, num_real_microbatches=n_mb, dp_size=dp, per_token_loss=False
                )
                (loss * cp / n_mb).backward()
                grad_sum += w.grad
    w = w0.clone().requires_grad_(True)
    sum(_terms(w, placement[(d, m)], 1, 0)[0] for d in range(dp) for m in range(n_mb)).backward()
    torch.testing.assert_close(grad_sum, w.grad * dp * cp, rtol=1e-5, atol=1e-7)

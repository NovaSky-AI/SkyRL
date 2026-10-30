"""SDPO over vocabulary shards; never gather the full vocabulary.

The scalar loss is replicated across TP ranks. SUM therefore has an identity
backward, as in Megatron's reduce-from-model-parallel-region primitive.
"""

import math

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint


class _Sum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group):
        result = value.clone()
        dist.all_reduce(result, group=group)
        return result

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class _LogSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, valid, group, temperature):
        scaled = (logits.float() / temperature).masked_fill(~valid, -torch.inf)
        maximum = scaled.amax(-1, keepdim=True)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
        denominator = (scaled - maximum).exp().sum(-1, keepdim=True)
        dist.all_reduce(denominator, group=group)
        log_probs = scaled - maximum - denominator.log()
        ctx.save_for_backward(log_probs.exp(), valid)
        ctx.group, ctx.temperature = group, temperature
        return log_probs

    @staticmethod
    def backward(ctx, grad):
        probability, valid = ctx.saved_tensors
        grad = grad.masked_fill(~valid, 0)
        # Each shard contributes part of dL/d(log p). The softmax Jacobian
        # couples every vocabulary entry, including entries on other ranks.
        total = grad.sum(-1, keepdim=True)
        dist.all_reduce(total, group=ctx.group)
        return (grad - probability * total) / ctx.temperature, None, None, None


def _log_probs(logits, valid, group, temperature):
    return _LogSoftmax.apply(logits, valid, group, temperature)


def _select(log_probs, indices, start, group):
    local = indices - start
    owned = (local >= 0) & (local < log_probs.shape[-1])
    values = log_probs.gather(-1, local.clamp(0, log_probs.shape[-1] - 1))
    return _Sum.apply(torch.where(owned, values, 0.0), group)


@torch.no_grad()
def vocab_parallel_action_logprobs(logits, token_ids, *, vocab_size, group, temperature=1.0, chunk_size=128):
    """Sampled-token logprobs without gathering vocabulary shards or retaining a graph."""
    if logits.shape[:-1] != token_ids.shape or temperature <= 0 or chunk_size <= 0:
        raise ValueError("Invalid action-logprob shapes, temperature, or chunk size")
    width = logits.shape[-1]
    start = dist.get_rank(group) * width
    if not 0 < vocab_size <= width * dist.get_world_size(group):
        raise ValueError("Invalid unpadded vocabulary size")
    valid = torch.arange(width, device=logits.device) + start < vocab_size
    flat = logits.detach().reshape(-1, width)
    ids = token_ids.reshape(-1, 1)
    result = torch.empty(ids.shape[0], device=logits.device, dtype=torch.float32)
    for offset in range(0, len(ids), chunk_size):
        lp = _log_probs(flat[offset : offset + chunk_size], valid, group, temperature)
        result[offset : offset + chunk_size] = _select(lp, ids[offset : offset + chunk_size], start, group).squeeze(-1)
    return result.view_as(token_ids)


def _tail(log_probs, valid, indices, start, group):
    """Log mass of the unselected vocabulary, summed in log space across shards."""
    width = log_probs.shape[-1]
    local = indices - start
    owned = ((local >= 0) & (local < width)).int()
    selected = torch.zeros_like(log_probs, dtype=torch.int32).scatter_add_(-1, local.clamp(0, width - 1), owned) > 0
    rest = log_probs.masked_fill(selected | ~valid, -torch.inf)
    maximum = rest.detach().amax(-1, keepdim=True)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
    return maximum + _Sum.apply((rest - maximum).exp().sum(-1, keepdim=True), group).log()


def _compress(log_probs, tail=None):
    if tail is None:
        return log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)
    return torch.cat((log_probs, tail), dim=-1)


def _divergence(student, teacher, alpha):
    if alpha == 0:
        return teacher.exp() * (teacher - student)
    if alpha == 1:
        return student.exp() * (student - teacher)
    mixture = torch.logaddexp(student + math.log(1 - alpha), teacher + math.log(alpha))
    return (1 - alpha) * student.exp() * (student - mixture) + alpha * teacher.exp() * (teacher - mixture)


@torch.no_grad()
def vocab_parallel_entropy_weights(logits, *, vocab_size, group, device, temperature=1.0, beta=1.0, chunk_size=128):
    """Unnormalized exp(-beta H(teacher)); CPU logits are streamed in chunks."""
    width = logits.shape[-1]
    start = dist.get_rank(group) * width
    valid = torch.arange(width, device=device) + start < vocab_size
    flat = logits.reshape(-1, width)
    result = torch.empty(flat.shape[0], device=device, dtype=torch.float32)
    for offset in range(0, len(flat), chunk_size):
        lp = _log_probs(flat[offset : offset + chunk_size].to(device), valid, group, temperature)
        entropy = -(lp.exp() * lp.masked_fill(~valid, 0)).sum(-1)
        dist.all_reduce(entropy, group=group)
        result[offset : offset + chunk_size] = (-beta * entropy).exp()
    return result.reshape(logits.shape[:-1])


def vocab_parallel_sdpo_loss(
    student,
    teacher,
    mask,
    config,
    *,
    vocab_size,
    group,
    temperature=1.0,
    response_ids=None,
    rollout_logprobs=None,
    advantage_clip=None,
    return_stats=False,
):
    """Masked loss SUM, with the same student-top-k/tail semantics as sdpo_loss.

    Inputs are [batch, response, local_vocab]. Teacher may reside on CPU.
    Padding vocabulary entries are excluded before normalization/top-k.
    """
    if student.shape != teacher.shape or student.shape[:-1] != mask.shape:
        raise ValueError("SDPO student, teacher, and response mask shapes must agree")
    if config.loss != "distribution" or temperature <= 0:
        raise ValueError("Megatron SDPO requires distribution loss and positive temperature")
    if advantage_clip is not None and config.alpha != 1:
        raise ValueError("Distribution advantage clipping requires alpha=1")
    if config.rollout_is_clip is not None and (
        response_ids is None
        or rollout_logprobs is None
        or response_ids.shape != mask.shape
        or rollout_logprobs.shape != mask.shape
    ):
        raise ValueError("Rollout importance clipping requires aligned response IDs and rollout logprobs")
    width = student.shape[-1]
    world = dist.get_world_size(group)
    start = dist.get_rank(group) * width
    if not 0 < vocab_size <= width * world:
        raise ValueError("Invalid unpadded vocabulary size")
    valid = torch.arange(width, device=student.device) + start < vocab_size
    s, t, weights = (
        student.reshape(-1, width),
        teacher.reshape(-1, width),
        mask.reshape(-1),
    )

    ids = response_ids.reshape(-1) if response_ids is not None else None
    behavior = rollout_logprobs.reshape(-1) if rollout_logprobs is not None else None

    def chunk_loss(s_chunk, t_chunk, weight, action_ids, behavior_lp):
        t_chunk = t_chunk.detach().to(s_chunk.device)
        sp = _log_probs(s_chunk, valid, group, temperature)
        with torch.no_grad():
            tp = _log_probs(t_chunk, valid, group, temperature)
        # Stats: |advantage| mass, valid tokens, clipped advantage mass,
        # clipped importance-ratio sum, importance-ratio clipped tokens.
        stats = s_chunk.new_zeros(5, dtype=torch.float32)
        eligible = weight > 0
        ratio = torch.ones_like(weight, dtype=torch.float32)
        if config.rollout_is_clip is not None:
            with torch.no_grad():
                action_lp = _select(sp.detach(), action_ids[:, None], start, group).squeeze(-1)
                log_ratio = action_lp - behavior_lp
                cap = math.log(config.rollout_is_clip)
                ratio = log_ratio.clamp(max=cap).exp()
                stats[3] = (ratio * eligible).sum()
                stats[4] = ((log_ratio > cap) & eligible).sum()
        compressed = config.topk is not None and config.topk < vocab_size
        if compressed:
            k = min(config.topk, width)
            values, indices = sp.detach().topk(k, dim=-1)
            gathered_values = [torch.empty_like(values) for _ in range(world)]
            gathered_indices = [torch.empty_like(indices) for _ in range(world)]
            dist.all_gather(gathered_values, values.contiguous(), group=group)
            dist.all_gather(gathered_indices, (indices + start).contiguous(), group=group)
            order = torch.cat(gathered_values, -1).topk(config.topk, dim=-1).indices
            indices = torch.cat(gathered_indices, -1).gather(-1, order)
            # Summing 1 - selected mass would lose the tail of a peaked distribution in FP32.
            s_tail = _tail(sp, valid, indices, start, group) if config.add_tail else None
            t_tail = _tail(tp, valid, indices, start, group) if config.add_tail else None
            sp = _compress(_select(sp, indices, start, group), s_tail)
            tp = _compress(_select(tp, indices, start, group), t_tail)
        else:
            # Avoid 0 * (inf - inf) for padded vocabulary entries.
            sp, tp = sp.masked_fill(~valid, 0), tp.masked_fill(~valid, 0)
        terms = _divergence(sp, tp, config.alpha)
        losses = terms.sum(-1) if compressed else _Sum.apply(terms.masked_fill(~valid, 0).sum(-1), group)
        if config.alpha == 1 and (return_stats or advantage_clip is not None):
            delta = (sp - tp).detach()
            probability = sp.exp()
            if not compressed:
                probability = probability.masked_fill(~valid, 0)
            mass = probability.detach() * eligible[:, None]
            stats[0] = (mass * delta.abs()).sum()
            stats[1] = eligible.sum()
            if advantage_clip is not None:
                stats[2] = (mass * (delta.abs() > advantage_clip)).sum()
                surrogate = (probability * delta.clamp(-advantage_clip, advantage_clip)).sum(-1)
                if not compressed:
                    surrogate = _Sum.apply(surrogate, group)
                losses = losses.detach() + surrogate - surrogate.detach()
            if not compressed:
                # Counts and importance ratios are replicated; only vocabulary masses are sharded.
                masses = stats[[0, 2]].contiguous()
                dist.all_reduce(masses, group=group)
                stats[0], stats[2] = masses.unbind()
        return (losses * weight * ratio).sum(), stats.detach()

    result = student.new_zeros((), dtype=torch.float32)
    stats = student.new_zeros(5, dtype=torch.float32)
    for i in range(0, s.shape[0], config.loss_chunk_size):
        args = (
            s[i : i + config.loss_chunk_size],
            t[i : i + config.loss_chunk_size],
            weights[i : i + config.loss_chunk_size],
            ids[i : i + config.loss_chunk_size] if ids is not None else None,
            behavior[i : i + config.loss_chunk_size] if behavior is not None else None,
        )
        chunk_result, chunk_stats = checkpoint(chunk_loss, *args, use_reentrant=False)
        result = result + chunk_result
        stats = stats + chunk_stats
    return (result, stats) if return_stats else result

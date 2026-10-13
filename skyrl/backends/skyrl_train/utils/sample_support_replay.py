"""Support-conditioned scoring for bounded sampler replay."""

from dataclasses import dataclass, replace
from enum import StrEnum

import torch

from skyrl.backends.skyrl_train.distributed.megatron.token_metadata import (
    TokenMetadataLayout,
    align_token_metadata,
    scatter_packed_token_values_to_batch,
)
from skyrl.backends.skyrl_train.utils.off_policy_correction_utils import MemberWeight
from skyrl.backends.skyrl_train.utils.packed_ragged_tensor import PackedRaggedTensor
from skyrl.backends.skyrl_train.utils.sample_support import (
    SAMPLE_SUPPORT_FIELD,
    SAMPLE_SUPPORT_NO_ROW,
    SAMPLE_SUPPORT_TORCH_DTYPE,
    PackedSampleSupport,
    SampleSupportColumn,
    align_sample_support_row_ids,
    packed_support_logprobs,
)


def missing_sample_support_message(backend: str) -> str:
    """Describe how to enable and forward a missing support payload."""
    return (
        f"sample-support replay is enabled but the {backend} forward received no "
        f"{SAMPLE_SUPPORT_FIELD!r}. Set generator.inference_engine.enable_return_sample_support_set=true "
        "so the sampler records it; if it is already set, the generator is dropping the field before "
        "the trainer sees it -- SkyRLGymGenerator forwards it, and a custom generator must too."
    )


@dataclass(frozen=True)
class SampleSupportScores:
    """Support-conditioned scores and the rows backed by recorded support.

    The score-centering fields are set when the scorer is given a ``centering_weight``:
    ``centering_term`` is Σ_v sg[q̃_v f(π_v / q̃_v)] log π_v over each row, ``sampler_logprobs`` is
    log q̃ of the sampled token, ``support_tv`` the total variation between q̃ and π, and
    ``sampler_mass`` the sampler's mass on the recorded row before renormalizing it into q̃. When
    ``f`` depends on the advantage's sign, ``centering_term`` is the term for positive advantages
    and ``centering_term_negative`` the one for negative advantages; otherwise the latter is None.
    The sampler fields are 0 where no recorded row backs a position; the centering terms and
    ``support_tv`` are also 0 on single-member rows, where log π = 0.
    """

    logprobs: torch.Tensor
    entropy: torch.Tensor | None
    valid_mask: torch.Tensor
    centering_term: torch.Tensor | None = None
    centering_term_negative: torch.Tensor | None = None
    sampler_logprobs: torch.Tensor | None = None
    support_tv: torch.Tensor | None = None
    sampler_mass: torch.Tensor | None = None


class ScoreCenteringMetric(StrEnum):
    """Per-microbatch score-centering diagnostics; ``_min`` keys reduce by minimum."""

    TERM_ABS_MEAN = "score_centering/term_abs_mean"
    SUPPORT_TV_MEAN = "score_centering/support_tv_mean"
    SAMPLER_ROW_MASS_MIN = "score_centering/sampler_row_mass_min"


@dataclass(frozen=True)
class ScoreCenteringLossInputs:
    """What a loss needs at its action positions: the centering term for each token's advantage
    sign, and the sampler logprobs its ratio must use, log q̃ of the sampled token."""

    term: torch.Tensor
    sampler_logprobs: torch.Tensor

    def loss_kwargs(self) -> dict[str, torch.Tensor]:
        return {"score_centering_term": self.term}


def score_centering_loss_inputs(
    scores: SampleSupportScores,
    positions: slice,
    rollout_logprobs: torch.Tensor | None,
    advantages: torch.Tensor,
) -> ScoreCenteringLossInputs:
    """The score-centering inputs at ``positions``.

    The loss's ratio uses log q̃, renormalized over the recorded row like the trainer's logprob,
    so a row the sampler gave mass below 1 (ties, the sampled-token repair) leaves no drift.
    Positions without a recorded row carry no loss; they keep the rollout logprob (or 0) and a
    zero centering term.
    """
    term = scores.centering_term[:, positions]
    if scores.centering_term_negative is not None:
        term = torch.where(advantages < 0, scores.centering_term_negative[:, positions], term)
    sampler_logprobs = scores.sampler_logprobs[:, positions]
    fallback = rollout_logprobs if rollout_logprobs is not None else torch.zeros_like(sampler_logprobs)
    valid = scores.valid_mask[:, positions]
    return ScoreCenteringLossInputs(
        term=term, sampler_logprobs=torch.where(valid, sampler_logprobs, fallback.to(sampler_logprobs.dtype))
    )


def score_centering_metrics(
    scores: SampleSupportScores, inputs: ScoreCenteringLossInputs, positions: slice, loss_mask: torch.Tensor
) -> dict[str, float]:
    """Diagnostics over the loss-bearing positions that have a recorded support row, in one host sync."""
    mask = scores.valid_mask[:, positions] & loss_mask.to(torch.bool)
    count = mask.sum().clamp(min=1)
    mass = scores.sampler_mass[:, positions]
    stats = torch.stack(
        [
            (inputs.term.detach().abs() * mask).sum() / count,
            (scores.support_tv[:, positions] * mask).sum() / count,
            # A microbatch with no such position (token-batching padding) reports the neutral 1.0.
            torch.cat([torch.where(mask, mass, 1.0).flatten(), mass.new_ones(1)]).min(),
        ]
    ).tolist()
    return {metric.value: value for metric, value in zip(ScoreCenteringMetric, stats, strict=True)}


def reject_unsupported_sample_support_packing(sub_seq_lengths: list[list[int]] | None) -> None:
    """Require every packed row to contain one whole trajectory."""
    if sub_seq_lengths is None:
        return
    if any(len(row_lengths) > 1 for row_lengths in sub_seq_lengths):
        raise ValueError(
            "sample-support replay does not support controller-packed multi-subsequence rows, got "
            f"sub-sequence counts {[len(row_lengths) for row_lengths in sub_seq_lengths]}"
        )


def _project_pair_chunk(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    row_ids: torch.Tensor,
    token_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Score one bounded block of ``(token position, vocab row)`` candidate pairs."""
    selected_hidden = hidden.index_select(0, row_ids).to(weight.dtype)
    selected_weight = weight.index_select(0, token_ids)
    return (selected_hidden * selected_weight).sum(dim=-1) / temperature


class _ChunkedCandidateProjection(torch.autograd.Function):
    """Selected LM-head projection that never retains a ``[pairs, hidden]`` activation."""

    @staticmethod
    def forward(ctx, hidden, weight, row_ids, token_ids, temperature, chunk_size):
        ctx.save_for_backward(hidden, weight, row_ids, token_ids)
        ctx.temperature = temperature
        ctx.chunk_size = chunk_size
        output = torch.empty(token_ids.shape, dtype=torch.float32, device=hidden.device)
        for start in range(0, token_ids.numel(), chunk_size):
            end = min(start + chunk_size, token_ids.numel())
            projected = _project_pair_chunk(hidden, weight, row_ids[start:end], token_ids[start:end], temperature)
            output[start:end] = projected.to(torch.float32)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        hidden, weight, row_ids, token_ids = ctx.saved_tensors
        grad_hidden = torch.zeros_like(hidden) if ctx.needs_input_grad[0] else None
        grad_weight = torch.zeros_like(weight) if ctx.needs_input_grad[1] else None
        for start in range(0, token_ids.numel(), ctx.chunk_size):
            end = min(start + ctx.chunk_size, token_ids.numel())
            chunk_rows = row_ids[start:end]
            chunk_tokens = token_ids[start:end]
            chunk_grad = grad_output[start:end].to(weight.dtype) / ctx.temperature
            if grad_hidden is not None:
                hidden_contribution = chunk_grad.unsqueeze(1) * weight.index_select(0, chunk_tokens)
                grad_hidden.index_add_(0, chunk_rows, hidden_contribution.to(hidden.dtype))
            if grad_weight is not None:
                selected_hidden = hidden.index_select(0, chunk_rows).to(weight.dtype)
                grad_weight.index_add_(0, chunk_tokens, chunk_grad.unsqueeze(1) * selected_hidden)
        return grad_hidden, grad_weight, None, None, None, None


def _project_candidate_pairs(
    hidden: torch.Tensor,
    row_ids: torch.Tensor,
    token_ids: torch.Tensor,
    lm_head_weight: torch.Tensor,
    temperature: float,
    chunk_size: int | None,
) -> torch.Tensor:
    """Project selected ``(token position, vocab row)`` pairs in bounded chunks."""
    if token_ids.numel() == 0:
        return torch.empty(0, dtype=torch.float32, device=hidden.device)
    pair_chunk_size = token_ids.numel() if chunk_size is None else chunk_size
    if pair_chunk_size <= 0:
        raise ValueError(f"candidate projection chunk size must be positive, got {pair_chunk_size}")
    if torch.is_grad_enabled() and (hidden.requires_grad or lm_head_weight.requires_grad):
        # A plain loop retains every selected chunk for backward. The custom backward reselects
        # one chunk at a time, so peak candidate-pair storage stays at the chunk bound.
        return _ChunkedCandidateProjection.apply(
            hidden,
            lm_head_weight,
            row_ids,
            token_ids,
            temperature,
            pair_chunk_size,
        )

    output = torch.empty(token_ids.shape, dtype=torch.float32, device=hidden.device)
    for start in range(0, token_ids.numel(), pair_chunk_size):
        end = min(start + pair_chunk_size, token_ids.numel())
        projected = _project_pair_chunk(hidden, lm_head_weight, row_ids[start:end], token_ids[start:end], temperature)
        output[start:end] = projected.to(torch.float32)
    return output


def _invert_row_ids(row_ids: torch.Tensor, num_rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Map support rows to model positions and reject duplicate row claims."""
    flat_rows = row_ids.reshape(-1).long()
    named = (flat_rows >= 0) & (flat_rows < num_rows)
    # One slot past the last row absorbs every position that names none, so the scatters below
    # stay in range without a separate mask.
    safe_rows = torch.where(named, flat_rows, num_rows)
    row_position = torch.full((num_rows + 1,), -1, dtype=torch.long, device=flat_rows.device)
    row_position.scatter_(0, safe_rows, torch.arange(flat_rows.numel(), device=flat_rows.device))
    claims = torch.zeros(num_rows + 1, dtype=torch.long, device=flat_rows.device).scatter_add_(
        0, safe_rows, named.long()
    )
    contested = (claims[:-1] > 1).nonzero().flatten()
    if contested.numel():
        raise ValueError(
            f"support rows {contested.tolist()} are each named by more than one model position, "
            "so no single position's scores can stand for the row"
        )
    return row_position[:-1], safe_rows


def _tp_max(values: torch.Tensor, tp_group: torch.distributed.ProcessGroup | None) -> torch.Tensor:
    """Elementwise maximum across the tensor-parallel group, in place; ``values`` carry no gradient."""
    if tp_group is not None and torch.distributed.get_world_size(tp_group) > 1:
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.MAX, group=tp_group)
    return values


def _tp_sum(local: torch.Tensor, tp_group: torch.distributed.ProcessGroup | None) -> torch.Tensor:
    """Sum across the tensor-parallel group, keeping only this rank's share of the gradient.

    Each rank's local sum is differentiable in its own vocabulary shard, so the all-reduced value is
    used forward while the backward reaches only the local term.
    """
    total = local.detach().clone()
    if tp_group is not None and torch.distributed.get_world_size(tp_group) > 1:
        torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM, group=tp_group)
    return total + local - local.detach()


@dataclass(frozen=True)
class _SupportRows:
    """Which support rows a microbatch scores, and which of their members this vocabulary shard owns.

    Row tensors are indexed by packed support row, member tensors by packed member. The ``local_*``
    tensors list each (row, vocabulary id, model position) this shard projects: members of rows with
    more than one member, since a single-member row's logprob is 0 without a projection.
    """

    num_rows: int
    positions_shape: torch.Size
    # Model position scoring each row (-1 if none), and each position's row (``num_rows`` if none).
    row_position: torch.Tensor
    safe_rows: torch.Tensor
    valid_rows: torch.Tensor
    projected_rows: torch.Tensor
    member_rows: torch.Tensor
    member_vocab: torch.Tensor
    sampled_for_row: torch.Tensor
    local_members: torch.Tensor
    local_rows: torch.Tensor
    local_vocab: torch.Tensor
    local_positions: torch.Tensor

    @classmethod
    def build(
        cls,
        support: PackedRaggedTensor,
        support_ids: torch.Tensor,
        row_ids: torch.Tensor,
        sampled_ids: torch.Tensor,
        *,
        vocab_start_index: int,
        vocab_end_index: int,
    ) -> "_SupportRows":
        num_rows = support.num_rows
        row_position, safe_rows = _invert_row_ids(row_ids, num_rows)
        row_lengths = support.row_lengths.to(torch.long)
        scored_rows = row_position >= 0
        projected_rows = scored_rows & (row_lengths > 1)
        member_rows = torch.repeat_interleave(
            torch.arange(num_rows, device=row_ids.device), row_lengths, output_size=support.num_members
        )
        member_vocab = support_ids.long()
        local_members = (
            projected_rows[member_rows] & (member_vocab >= vocab_start_index) & (member_vocab < vocab_end_index)
        )
        local_rows = member_rows[local_members]
        return cls(
            num_rows=num_rows,
            positions_shape=sampled_ids.shape,
            row_position=row_position,
            safe_rows=safe_rows,
            valid_rows=scored_rows & (row_lengths > 0),
            projected_rows=projected_rows,
            member_rows=member_rows,
            member_vocab=member_vocab,
            sampled_for_row=sampled_ids.reshape(-1).long()[row_position.clamp(min=0)],
            local_members=local_members,
            local_rows=local_rows,
            local_vocab=member_vocab[local_members],
            local_positions=row_position[local_rows],
        )

    def row_sum(self, local_values: torch.Tensor) -> torch.Tensor:
        """Sum per-local-member values into their rows."""
        return local_values.new_zeros(self.num_rows).index_add(0, self.local_rows, local_values)

    def at_positions(self, row_values: torch.Tensor) -> torch.Tensor:
        """Read each model position's row, with the sentinel slot standing in for "no row"."""
        return torch.cat([row_values, row_values.new_zeros(1)])[self.safe_rows].reshape(self.positions_shape)


def _member_logits(
    logits_or_hidden: torch.Tensor,
    rows: _SupportRows,
    *,
    vocab_start_index: int,
    vocab_end_index: int,
    lm_head_weight: torch.Tensor | None,
    temperature: float,
    chunk_size: int | None,
) -> torch.Tensor:
    """This shard's member logits, gathered from the logits or projected from hidden states."""
    flat_source = logits_or_hidden.reshape(-1, logits_or_hidden.shape[-1])
    local_ids = rows.local_vocab - vocab_start_index
    if lm_head_weight is None:
        compute_dtype = (
            torch.float32 if logits_or_hidden.dtype in (torch.float16, torch.bfloat16) else logits_or_hidden.dtype
        )
        local_values = flat_source[rows.local_positions, local_ids].to(compute_dtype)
    else:
        if lm_head_weight.shape[0] != vocab_end_index - vocab_start_index:
            raise ValueError(
                f"lm_head_weight holds {lm_head_weight.shape[0]} rows for vocabulary shard "
                f"[{vocab_start_index}, {vocab_end_index})"
            )
        local_values = _project_candidate_pairs(
            flat_source, rows.local_positions, local_ids, lm_head_weight, temperature, chunk_size
        )
    # A vocabulary shard that owns no member of this microbatch would otherwise leave the backward
    # graph entirely -- `_project_candidate_pairs` short-circuits an empty pair list -- and both
    # Megatron and FSDP expect a gradient buffer for the hidden states and the LM head on every
    # rank. A zero-weighted touch of each input restores the edge and adds 0.0 to the values.
    connect = flat_source.reshape(-1)[:1].sum() * 0.0
    if lm_head_weight is not None:
        connect = connect + lm_head_weight.reshape(-1)[:1].sum() * 0.0
    return local_values + connect.to(local_values.dtype)


@dataclass(frozen=True)
class _RowNormalizers:
    """The renormalization over each projected row: ``log π^S_v = shifted_v - log_denominator[row]``."""

    shifted: torch.Tensor
    log_denominator: torch.Tensor
    row_logprobs: torch.Tensor
    row_entropy: torch.Tensor | None


def _row_normalizers(
    local_values: torch.Tensor,
    rows: _SupportRows,
    tp_group: torch.distributed.ProcessGroup | None,
    *,
    compute_entropy: bool,
    entropy_requires_grad: bool,
) -> _RowNormalizers:
    """Reduce each row's max, normalizer and sampled-member logit across the vocabulary shards.

    One MAX and one stacked SUM collective; entropy, H = log Z - E_p[l - max], is another row of the
    same SUM because both terms are sums over the members Z already sums over.
    """
    local_max = torch.full((rows.num_rows,), float("-inf"), dtype=local_values.dtype, device=local_values.device)
    local_max.scatter_reduce_(0, rows.local_rows, local_values.detach(), reduce="amax", include_self=True)
    safe_max = torch.where(rows.projected_rows, _tp_max(local_max, tp_group), 0.0)

    shifted = local_values - safe_max[rows.local_rows]
    local_exp = shifted.exp()
    sampled_member = rows.local_vocab == rows.sampled_for_row[rows.local_rows]
    local_stats = [rows.row_sum(local_exp), rows.row_sum(torch.where(sampled_member, local_values, 0.0))]
    if compute_entropy:
        weights = local_exp if entropy_requires_grad else local_exp.detach()
        local_stats.append(rows.row_sum(weights * (shifted if entropy_requires_grad else shifted.detach())))
    global_stats = _tp_sum(torch.stack(local_stats), tp_group)
    denominator, sampled_score = global_stats[0], global_stats[1]
    log_denominator = torch.where(rows.projected_rows, denominator, 1.0).log()
    row_logprobs = torch.where(rows.projected_rows, sampled_score - safe_max - log_denominator, 0.0)

    row_entropy = None
    if compute_entropy:
        weighted_sum = global_stats[2] if entropy_requires_grad else global_stats[2].detach()
        safe_denominator = torch.where(
            rows.projected_rows, denominator if entropy_requires_grad else denominator.detach(), 1.0
        )
        row_entropy = torch.where(
            rows.projected_rows, safe_denominator.log() - weighted_sum / safe_denominator, 0.0
        ).clamp_min(0.0)
    return _RowNormalizers(
        shifted=shifted, log_denominator=log_denominator, row_logprobs=row_logprobs, row_entropy=row_entropy
    )


@dataclass(frozen=True)
class _SamplerRows:
    """The sampler's processed distribution over each recorded row, renormalized into q̃."""

    # log q̃ of every member, and of each row's sampled member.
    member_log_q: torch.Tensor
    row_logprobs: torch.Tensor
    # log of the sampler's mass on the row before renormalizing.
    log_mass: torch.Tensor


def _sampler_rows(rows: _SupportRows, member_logprobs: torch.Tensor | None, dtype: torch.dtype) -> _SamplerRows:
    """Renormalize the sampler's member logprobs over each row.

    Every rank holds the whole support, so this needs no communication and must stay out of the
    vocabulary-sharded sums. A support with no members (placeholder rows) may carry no logprob column.
    """
    member_q = (member_logprobs if member_logprobs is not None else rows.member_vocab.new_zeros(0)).to(dtype)
    q_max = torch.full((rows.num_rows,), float("-inf"), dtype=dtype, device=member_q.device)
    q_max.scatter_reduce_(0, rows.member_rows, member_q, reduce="amax", include_self=True)
    q_max = torch.where(rows.valid_rows, q_max, 0.0)
    q_shifted_mass = member_q.new_zeros(rows.num_rows).index_add(
        0, rows.member_rows, (member_q - q_max[rows.member_rows]).exp()
    )
    log_mass = torch.where(rows.valid_rows, q_max + torch.where(rows.valid_rows, q_shifted_mass, 1.0).log(), 0.0)
    member_log_q = member_q - log_mass[rows.member_rows]
    sampled_members = rows.member_vocab == rows.sampled_for_row[rows.member_rows]
    row_logprobs = member_q.new_zeros(rows.num_rows).index_add(
        0, rows.member_rows, torch.where(sampled_members, member_log_q, 0.0)
    )
    return _SamplerRows(member_log_q=member_log_q, row_logprobs=row_logprobs, log_mass=log_mass)


def _with_score_centering(
    scores: SampleSupportScores,
    rows: _SupportRows,
    normalizers: _RowNormalizers,
    member_logprobs: torch.Tensor | None,
    centering_weight: MemberWeight,
    tp_group: torch.distributed.ProcessGroup | None,
) -> SampleSupportScores:
    """Add the score-centering fields to ``scores``.

    Over this shard's members, c_v = sg[q̃_v f(π^S_v / q̃_v)], and the term
    Σ c_v log π^S_v = Σ c_v (l_v - max) - (Σ c_v) log Z needs two row sums per advantage sign; they
    share one TP SUM with the total-variation sum Σ |q̃_v - π^S_v|.
    """
    sampler = _sampler_rows(rows, member_logprobs, normalizers.shifted.dtype)
    local_log_q = sampler.member_log_q[rows.local_members]
    local_q = local_log_q.exp()
    local_logprobs = (normalizers.shifted - normalizers.log_denominator[rows.local_rows]).detach()
    signs = (False, True) if centering_weight.depends_on_advantage_sign else (False,)
    sums = []
    for negative_advantage in signs:
        weights = centering_weight.weights(local_logprobs, local_log_q, negative_advantage=negative_advantage)
        coefficients = (local_q * weights).detach()
        sums.extend([rows.row_sum(coefficients * normalizers.shifted), rows.row_sum(coefficients)])
    sums.append(rows.row_sum((local_q - local_logprobs.exp()).abs()))
    stats = _tp_sum(torch.stack(sums), tp_group)
    terms = [
        rows.at_positions(
            torch.where(
                rows.projected_rows,
                stats[2 * i] - stats[2 * i + 1].detach() * normalizers.log_denominator,
                0.0,
            )
        )
        for i in range(len(signs))
    ]
    return replace(
        scores,
        centering_term=terms[0],
        centering_term_negative=terms[1] if len(terms) > 1 else None,
        sampler_logprobs=rows.at_positions(torch.where(rows.valid_rows, sampler.row_logprobs, 0.0)),
        support_tv=rows.at_positions(torch.where(rows.projected_rows, 0.5 * stats[-1].detach(), 0.0)),
        sampler_mass=rows.at_positions(torch.where(rows.valid_rows, sampler.log_mass.exp(), 0.0)),
    )


def sample_support_scores(
    logits_or_hidden: torch.Tensor,
    sampled_ids: torch.Tensor,
    support: PackedRaggedTensor,
    row_ids: torch.Tensor,
    *,
    vocab_start_index: int,
    vocab_end_index: int,
    tp_group: torch.distributed.ProcessGroup | None,
    compute_entropy: bool = False,
    entropy_requires_grad: bool = False,
    lm_head_weight: torch.Tensor | None = None,
    temperature: float = 1.0,
    chunk_size: int | None = None,
    centering_weight: MemberWeight | None = None,
) -> SampleSupportScores:
    """Renormalize over each token's recorded support members.

    Each support row must contain distinct members, including the sampled token. With
    ``centering_weight``, also score-center against the support's sampler member logprobs.
    """
    if logits_or_hidden.shape[:-1] != sampled_ids.shape or row_ids.shape != sampled_ids.shape:
        raise ValueError(
            "logits, sampled_ids, and row_ids must have matching prefix shapes, got "
            f"{logits_or_hidden.shape[:-1]}, {sampled_ids.shape}, and {row_ids.shape}"
        )
    support_ids = support.columns[SampleSupportColumn.TOKEN_IDS]
    if support_ids.dtype != SAMPLE_SUPPORT_TORCH_DTYPE:
        raise ValueError(f"sample support must use {SAMPLE_SUPPORT_TORCH_DTYPE} vocab ids, got {support_ids.dtype}")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if entropy_requires_grad and not compute_entropy:
        raise ValueError("entropy gradients require compute_entropy=True")
    member_logprobs = packed_support_logprobs(support)
    if centering_weight is not None and member_logprobs is None and support.num_members:
        raise ValueError(
            "score centering needs the sampler's logprob of every support member, but the support "
            "carries none. SkyRL's inference client requests them with the support when "
            "trainer.algorithm.enable_score_centering is set; a generator must keep each "
            "CompactSampleSupport it gets back, or build its own with logprobs."
        )

    rows = _SupportRows.build(
        support,
        support_ids,
        row_ids,
        sampled_ids,
        vocab_start_index=vocab_start_index,
        vocab_end_index=vocab_end_index,
    )
    local_values = _member_logits(
        logits_or_hidden,
        rows,
        vocab_start_index=vocab_start_index,
        vocab_end_index=vocab_end_index,
        lm_head_weight=lm_head_weight,
        temperature=temperature,
        chunk_size=chunk_size,
    )
    normalizers = _row_normalizers(
        local_values,
        rows,
        tp_group,
        compute_entropy=compute_entropy,
        entropy_requires_grad=entropy_requires_grad,
    )
    scores = SampleSupportScores(
        logprobs=rows.at_positions(normalizers.row_logprobs),
        entropy=None if normalizers.row_entropy is None else rows.at_positions(normalizers.row_entropy),
        valid_mask=rows.at_positions(rows.valid_rows),
    )
    if centering_weight is None:
        return scores
    return _with_score_centering(scores, rows, normalizers, member_logprobs, centering_weight, tp_group)


def sample_support_row_ids_in_batch_positions(
    sample_support: PackedSampleSupport,
    layout: TokenMetadataLayout,
) -> torch.Tensor:
    """Map support rows into canonical left-padded batch coordinates."""
    row_ids = align_sample_support_row_ids(sample_support, layout)
    sequence_length = layout.attention_mask.shape[1]
    padding_widths = sequence_length - layout.attention_mask.sum(dim=1).to(torch.long)
    positions = torch.arange(sequence_length, device=row_ids.device).unsqueeze(0) - padding_widths.unsqueeze(1)
    return torch.where(positions >= 0, row_ids.gather(1, positions.clamp(min=0)), SAMPLE_SUPPORT_NO_ROW)


def score_aligned_sample_support(
    aligned_source: torch.Tensor,
    aligned_sampled_ids: torch.Tensor,
    aligned_row_ids: torch.Tensor,
    aligned_loss_mask: torch.Tensor,
    sample_support: PackedSampleSupport,
    *,
    vocab_start_index: int,
    vocab_end_index: int,
    tp_group: torch.distributed.ProcessGroup | None,
    compute_entropy: bool = False,
    entropy_requires_grad: bool = False,
    lm_head_weight: torch.Tensor | None = None,
    temperature: float = 1.0,
    chunk_size: int | None = None,
    centering_weight: MemberWeight | None = None,
) -> SampleSupportScores:
    """Score token-aligned recorded support rows."""
    scores = sample_support_scores(
        aligned_source,
        aligned_sampled_ids,
        sample_support,
        aligned_row_ids,
        vocab_start_index=vocab_start_index,
        vocab_end_index=vocab_end_index,
        tp_group=tp_group,
        compute_entropy=compute_entropy,
        entropy_requires_grad=entropy_requires_grad,
        lm_head_weight=lm_head_weight,
        temperature=temperature if lm_head_weight is not None else 1.0,
        chunk_size=chunk_size,
        centering_weight=centering_weight,
    )
    unsupported_loss_active = aligned_loss_mask & ~scores.valid_mask
    if unsupported_loss_active.any():
        raise ValueError("sample-support replay requires captured support for every loss-active token")
    return scores


def compute_sample_support_scores(
    logits_or_hidden: torch.Tensor,
    sequences: torch.Tensor,
    loss_mask: torch.Tensor | None,
    sample_support: PackedSampleSupport | None,
    num_actions: int,
    *,
    packed: bool,
    metadata_layout: TokenMetadataLayout | None,
    vocab_start_index: int,
    vocab_end_index: int,
    tp_group: torch.distributed.ProcessGroup | None,
    lm_head_weight: torch.Tensor | None,
    temperature: float,
    chunk_size: int | None,
    compute_entropy: bool = False,
    entropy_requires_grad: bool = False,
    centering_weight: MemberWeight | None = None,
) -> SampleSupportScores:
    """Score support-conditioned logprobs in canonical trainer layout."""
    if sample_support is None:
        raise ValueError(missing_sample_support_message("Megatron"))
    if loss_mask is None:
        raise ValueError("sample-support replay requires the response loss mask")
    if metadata_layout is None:
        raise ValueError("sample-support replay requires the shared token metadata layout")

    target_loss_mask = torch.zeros_like(sequences, dtype=torch.bool)
    target_loss_mask[:, -num_actions:] = loss_mask.to(torch.bool)
    if packed:
        row_ids = align_sample_support_row_ids(sample_support, metadata_layout)
        aligned_sampled_ids = align_token_metadata(sequences, metadata_layout, 0, next_token=True)
        aligned_loss_mask = align_token_metadata(target_loss_mask, metadata_layout, False, next_token=True)
        aligned_source = logits_or_hidden
    else:
        # The domain ends at real token L_i - 2, so dropping the last column loses no row.
        row_ids = sample_support_row_ids_in_batch_positions(sample_support, metadata_layout)[:, :-1]
        aligned_sampled_ids = sequences[:, 1:]
        aligned_loss_mask = target_loss_mask[:, 1:]
        aligned_source = logits_or_hidden[:, :-1]

    scores = score_aligned_sample_support(
        aligned_source,
        aligned_sampled_ids,
        row_ids,
        aligned_loss_mask,
        sample_support,
        vocab_start_index=vocab_start_index,
        vocab_end_index=vocab_end_index,
        tp_group=tp_group,
        compute_entropy=compute_entropy,
        entropy_requires_grad=entropy_requires_grad,
        lm_head_weight=lm_head_weight,
        temperature=temperature,
        chunk_size=chunk_size,
        centering_weight=centering_weight,
    )
    if not packed:
        return scores

    def to_batch(values: torch.Tensor | None) -> torch.Tensor | None:
        return None if values is None else scatter_packed_token_values_to_batch(values, metadata_layout, 0)

    return SampleSupportScores(
        logprobs=to_batch(scores.logprobs),
        entropy=to_batch(scores.entropy),
        valid_mask=scatter_packed_token_values_to_batch(scores.valid_mask, metadata_layout, False),
        centering_term=to_batch(scores.centering_term),
        centering_term_negative=to_batch(scores.centering_term_negative),
        sampler_logprobs=to_batch(scores.sampler_logprobs),
        support_tv=to_batch(scores.support_tv),
        sampler_mass=to_batch(scores.sampler_mass),
    )

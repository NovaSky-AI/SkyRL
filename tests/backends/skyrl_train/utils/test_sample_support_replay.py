"""Tests for support-conditioned scoring."""

import sys
import threading
import types
from collections import Counter
from typing import List, Tuple

import numpy as np
import pytest
import torch

from skyrl.backends.skyrl_train.distributed.megatron.token_metadata import (
    TokenMetadataLayout,
)
from skyrl.backends.skyrl_train.utils import sample_support_replay
from skyrl.backends.skyrl_train.utils.off_policy_correction_utils import (
    DPPOTrustRegion,
    DPPOType,
    RatioBand,
)
from skyrl.backends.skyrl_train.utils.packed_ragged_tensor import PackedRaggedTensor
from skyrl.backends.skyrl_train.utils.packed_tensor import (
    PackedTensor,
    cu_seqlens_from_lengths,
)
from skyrl.backends.skyrl_train.utils.ppo_utils import (
    PolicyLossRegistry,
    score_centering_weight,
)
from skyrl.backends.skyrl_train.utils.sample_support import (
    SAMPLE_SUPPORT_NO_ROW,
    SAMPLE_SUPPORT_TORCH_DTYPE,
)
from skyrl.backends.skyrl_train.utils.sample_support_replay import (
    SampleSupportScores,
    compute_sample_support_scores,
    reject_unsupported_sample_support_packing,
    sample_support_scores,
    score_centering_loss_inputs,
)
from skyrl.train.config import AlgorithmConfig, CISPOConfig, DPPOConfig
from tests.backends.skyrl_train.sample_support_helpers import ragged_support

VOCAB = 11
TOP_K = 4
DENSE_SCORER_KWARGS = dict(
    vocab_start_index=0,
    vocab_end_index=VOCAB,
    tp_group=None,
    lm_head_weight=None,
    temperature=1.0,
    chunk_size=None,
)


def _from_grid(support_ids):
    """CSR support and row ids for a ``[..., top_k]`` grid, one row per position; negative entries are absent."""
    support = ragged_support([support_ids.reshape(-1, support_ids.shape[-1])])
    return support, torch.arange(support.num_rows).reshape(support_ids.shape[:-1])


def _reference_support_logprobs(logits, sampled_ids, support_ids):
    """Renormalize over each row's recorded members with a dense full-vocabulary softmax."""
    outputs = []
    for row_logits, sampled_id, support in zip(
        logits.reshape(-1, logits.shape[-1]),
        sampled_ids.reshape(-1),
        support_ids.reshape(-1, support_ids.shape[-1]),
        strict=True,
    ):
        members = support[support >= 0].long()
        outputs.append(
            row_logits.new_zeros(())
            if members.numel() == 0
            else row_logits[sampled_id] - torch.logsumexp(row_logits[members], dim=0)
        )
    return torch.stack(outputs).reshape(sampled_ids.shape)


def test_support_scores_match_a_dense_reference_in_value_and_gradient():
    logits = torch.randn(2, 3, VOCAB, dtype=torch.float64, requires_grad=True)
    sampled_ids = torch.tensor([[2, 5, 1], [8, 3, 7]])
    support_ids = torch.tensor(
        [
            [[2, 4, 6, -1], [5, -1, -1, -1], [-1, -1, -1, -1]],
            [[8, 0, 9, 4], [3, 2, -1, -1], [7, 1, 5, -1]],
        ],
        dtype=SAMPLE_SUPPORT_TORCH_DTYPE,
    )

    scores = sample_support_scores(
        logits,
        sampled_ids,
        *_from_grid(support_ids),
        vocab_start_index=0,
        vocab_end_index=VOCAB,
        tp_group=None,
    )

    assert scores.entropy is None
    assert scores.valid_mask.tolist() == [[True, True, False], [True, True, True]]
    torch.testing.assert_close(scores.logprobs, _reference_support_logprobs(logits, sampled_ids, support_ids))

    scores.logprobs.sum().backward()
    reference_logits = logits.detach().clone().requires_grad_(True)
    _reference_support_logprobs(reference_logits, sampled_ids, support_ids).sum().backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad)


def _reference_support_entropy(logits, support_ids):
    """Compute entropy over each row's recorded support."""
    outputs = []
    for row_logits, support in zip(
        logits.reshape(-1, logits.shape[-1]),
        support_ids.reshape(-1, support_ids.shape[-1]),
        strict=True,
    ):
        members = support[support >= 0].long()
        if members.numel() == 0:
            outputs.append(row_logits.new_zeros(()))
        else:
            member_logprobs = torch.log_softmax(row_logits[members], dim=0)
            outputs.append(-(member_logprobs.exp() * member_logprobs).sum())
    return torch.stack(outputs).reshape(support_ids.shape[:-1])


@pytest.mark.parametrize("entropy_requires_grad", [False, True])
def test_support_entropy_matches_a_dense_reference(entropy_requires_grad):
    logits = torch.randn(2, 3, VOCAB, dtype=torch.float64, requires_grad=True)
    sampled_ids = torch.tensor([[2, 5, 1], [8, 3, 7]])
    support_ids = torch.tensor(
        [
            [[2, 4, 6, -1], [5, -1, -1, -1], [-1, -1, -1, -1]],
            [[8, 0, 9, 4], [3, 2, -1, -1], [7, 1, 5, -1]],
        ],
        dtype=SAMPLE_SUPPORT_TORCH_DTYPE,
    )

    scores = sample_support_scores(
        logits,
        sampled_ids,
        *_from_grid(support_ids),
        vocab_start_index=0,
        vocab_end_index=VOCAB,
        tp_group=None,
        compute_entropy=True,
        entropy_requires_grad=entropy_requires_grad,
    )

    assert scores.entropy is not None
    assert scores.entropy.requires_grad == entropy_requires_grad
    torch.testing.assert_close(scores.entropy, _reference_support_entropy(logits, support_ids))

    loss = scores.logprobs.sum()
    reference_logits = logits.detach().clone().requires_grad_(True)
    reference_loss = _reference_support_logprobs(reference_logits, sampled_ids, support_ids).sum()
    if entropy_requires_grad:
        loss = loss + scores.entropy.sum()
        reference_loss = reference_loss + _reference_support_entropy(reference_logits, support_ids).sum()
    loss.backward()
    reference_loss.backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad)


def test_entropy_gradients_require_computing_the_entropy():
    with pytest.raises(ValueError, match="compute_entropy=True"):
        sample_support_scores(
            torch.randn(1, VOCAB),
            torch.tensor([1]),
            *_from_grid(torch.tensor([[1, 2]], dtype=SAMPLE_SUPPORT_TORCH_DTYPE)),
            vocab_start_index=0,
            vocab_end_index=VOCAB,
            tp_group=None,
            entropy_requires_grad=True,
        )


TP_SIZE = 2


class _FakeTensorParallel:
    """Run vocabulary shards concurrently and reduce them at a barrier."""

    def __init__(self, world_size: int = TP_SIZE):
        self.world_size = world_size
        self.calls: List[Tuple[object, Tuple[int, ...]]] = []
        self._barrier = threading.Barrier(world_size, timeout=60)
        self._slots: List[torch.Tensor | None] = [None] * world_size
        self._lock = threading.Lock()
        self._local = threading.local()

    def set_rank(self, rank: int) -> None:
        self._local.rank = rank

    def all_reduce(self, tensor, op=None, group=None):
        with self._lock:
            self.calls.append((op, tuple(tensor.shape)))
        self._slots[self._local.rank] = tensor.clone()
        self._barrier.wait()
        combined = self._slots[0]
        for other in self._slots[1:]:
            combined = torch.maximum(combined, other) if op == torch.distributed.ReduceOp.MAX else combined + other
        self._barrier.wait()
        tensor.copy_(combined)


@pytest.fixture
def tensor_parallel(monkeypatch):
    fake = _FakeTensorParallel()
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: fake.world_size)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake.all_reduce)
    return fake


def _tensor_parallel_scores(fake, logits, sampled_ids, support, row_ids, *, all_ranks=False, **entropy_kwargs):
    """Score the same rows on every vocabulary shard and return rank 0's result, or every rank's."""
    width = VOCAB // fake.world_size
    results: List[SampleSupportScores | None] = [None] * fake.world_size
    errors: List[BaseException] = []

    def run(rank: int) -> None:
        fake.set_rank(rank)
        start = rank * width
        end = VOCAB if rank == fake.world_size - 1 else start + width
        try:
            results[rank] = sample_support_scores(
                logits[..., start:end],
                sampled_ids,
                support,
                row_ids,
                vocab_start_index=start,
                vocab_end_index=end,
                tp_group=object(),
                **entropy_kwargs,
            )
        except BaseException as error:
            errors.append(error)
            fake._barrier.abort()

    threads = [threading.Thread(target=run, args=(rank,)) for rank in range(fake.world_size)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    assert all(scores is not None for scores in results)
    return results if all_ranks else results[0]


TP_SAMPLED_IDS = torch.tensor([[2, 8, 1], [9, 3, 6]])
TP_SUPPORT_IDS = torch.tensor(
    [
        [[2, 7, 9, -1], [8, 1, -1, -1], [1, 6, 10, 4]],
        [[9, 0, 5, 3], [3, 8, -1, -1], [6, 2, -1, -1]],
    ],
    dtype=SAMPLE_SUPPORT_TORCH_DTYPE,
)


def test_entropy_rides_the_existing_tensor_parallel_reduction(tensor_parallel):
    """Entropy widens the SUM payload without adding a collective."""
    logits = torch.randn(2, 3, VOCAB, dtype=torch.float64)
    rows = TP_SAMPLED_IDS.numel()

    _tensor_parallel_scores(
        tensor_parallel,
        logits,
        TP_SAMPLED_IDS,
        *_from_grid(TP_SUPPORT_IDS),
        compute_entropy=False,
        entropy_requires_grad=False,
    )
    without_entropy = list(tensor_parallel.calls)
    tensor_parallel.calls.clear()
    _tensor_parallel_scores(
        tensor_parallel,
        logits,
        TP_SAMPLED_IDS,
        *_from_grid(TP_SUPPORT_IDS),
        compute_entropy=True,
        entropy_requires_grad=True,
    )
    with_entropy = list(tensor_parallel.calls)

    max_op, sum_op = torch.distributed.ReduceOp.MAX, torch.distributed.ReduceOp.SUM
    assert len(with_entropy) == len(without_entropy) == 2 * TP_SIZE
    assert Counter(op for op, _ in with_entropy) == Counter(op for op, _ in without_entropy)
    assert Counter(without_entropy) == Counter({(max_op, (rows,)): TP_SIZE, (sum_op, (2, rows)): TP_SIZE})
    assert Counter(with_entropy) == Counter({(max_op, (rows,)): TP_SIZE, (sum_op, (3, rows)): TP_SIZE})


def test_tensor_parallel_shards_reduce_to_the_unsharded_entropy(tensor_parallel):
    logits = torch.randn(2, 3, VOCAB, dtype=torch.float64)

    sharded = _tensor_parallel_scores(
        tensor_parallel,
        logits,
        TP_SAMPLED_IDS,
        *_from_grid(TP_SUPPORT_IDS),
        compute_entropy=True,
        entropy_requires_grad=False,
    )

    assert sharded.entropy is not None
    torch.testing.assert_close(sharded.entropy, _reference_support_entropy(logits, TP_SUPPORT_IDS))
    torch.testing.assert_close(
        sharded.logprobs,
        _reference_support_logprobs(logits, TP_SAMPLED_IDS, TP_SUPPORT_IDS),
    )


def test_support_scores_renormalize_over_the_support_not_the_vocabulary():
    """A row whose support omits most of the vocabulary must not match a full-vocab logprob."""
    logits = torch.randn(1, 1, VOCAB, dtype=torch.float64)

    scores = sample_support_scores(
        logits,
        torch.tensor([[3]]),
        *_from_grid(torch.tensor([[[3, 7, -1, -1]]], dtype=SAMPLE_SUPPORT_TORCH_DTYPE)),
        vocab_start_index=0,
        vocab_end_index=VOCAB,
        tp_group=None,
    )

    expected = logits[0, 0, 3] - torch.logsumexp(logits[0, 0, [3, 7]], dim=0)
    torch.testing.assert_close(scores.logprobs[0, 0], expected)
    assert scores.logprobs[0, 0] > logits[0, 0].log_softmax(dim=-1)[3]


def test_fused_projection_matches_explicit_logits_in_value_and_gradient():
    temperature = 0.7
    hidden = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(9, 5, dtype=torch.float64, requires_grad=True)
    sampled_ids = torch.tensor([[1, 4, 7], [2, 5, 8]])
    support_ids = torch.tensor(
        [
            [[1, 0, 3], [4, 6, -1], [7, -1, -1]],
            [[2, 1, 8], [5, 4, -1], [8, 0, 6]],
        ],
        dtype=SAMPLE_SUPPORT_TORCH_DTYPE,
    )

    fused = sample_support_scores(
        hidden,
        sampled_ids,
        *_from_grid(support_ids),
        vocab_start_index=0,
        vocab_end_index=weight.shape[0],
        tp_group=None,
        lm_head_weight=weight,
        temperature=temperature,
        chunk_size=4,
    ).logprobs
    fused.sum().backward()

    explicit_hidden = hidden.detach().clone().requires_grad_(True)
    explicit_weight = weight.detach().clone().requires_grad_(True)
    explicit = sample_support_scores(
        (explicit_hidden @ explicit_weight.T) / temperature,
        sampled_ids,
        *_from_grid(support_ids),
        vocab_start_index=0,
        vocab_end_index=weight.shape[0],
        tp_group=None,
    ).logprobs
    explicit.sum().backward()

    torch.testing.assert_close(fused, explicit, check_dtype=False)
    torch.testing.assert_close(hidden.grad, explicit_hidden.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(weight.grad, explicit_weight.grad, rtol=1e-5, atol=1e-6)


def test_fused_projection_bounds_the_candidate_pair_temporary(monkeypatch):
    """No projection sees more than ``chunk_size`` candidate pairs at once."""
    chunk_size = 3
    widths: List[int] = []
    original = sample_support_replay._project_pair_chunk

    def spy(hidden, weight, row_ids, token_ids, temperature):
        widths.append(int(token_ids.numel()))
        return original(hidden, weight, row_ids, token_ids, temperature)

    monkeypatch.setattr(sample_support_replay, "_project_pair_chunk", spy)
    hidden = torch.randn(2, 4, 5, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(9, 5, dtype=torch.float64, requires_grad=True)

    # Distinct members per row, led by the sampled token.
    support_ids = (torch.arange(2 * 4 * TOP_K) % 9).reshape(2, 4, TOP_K).to(SAMPLE_SUPPORT_TORCH_DTYPE)
    scores = sample_support_scores(
        hidden,
        support_ids[..., 0].long(),
        *_from_grid(support_ids),
        vocab_start_index=0,
        vocab_end_index=weight.shape[0],
        tp_group=None,
        lm_head_weight=weight,
        chunk_size=chunk_size,
    )
    scores.logprobs.sum().backward()

    assert max(widths) <= chunk_size
    assert len(widths) > 1
    # Every one of the 2 * 4 * TOP_K support pairs, chunked.
    assert sum(widths) == 2 * 4 * TOP_K
    assert hidden.grad is not None and weight.grad is not None


@pytest.fixture
def megatron_parallel_state(monkeypatch):
    """Let ``model_utils`` import for its pure-torch packed-index helpers."""
    try:
        import megatron.core.parallel_state as mpu
    except ModuleNotFoundError:
        megatron = types.ModuleType("megatron")
        core = types.ModuleType("megatron.core")
        mpu = types.ModuleType("megatron.core.parallel_state")
        megatron.core = core
        core.parallel_state = mpu
        monkeypatch.setitem(sys.modules, "megatron", megatron)
        monkeypatch.setitem(sys.modules, "megatron.core", core)
        monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", mpu)
    return mpu


def test_multi_subsequence_rows_are_rejected():
    reject_unsupported_sample_support_packing(None)
    reject_unsupported_sample_support_packing([[7], [9]])

    with pytest.raises(ValueError, match="multi-subsequence"):
        reject_unsupported_sample_support_packing([[7], [4, 5]])


# (prompt_len, response_len) per trajectory, including unequal prompt lengths.
JOIN_LENGTHS: List[Tuple[int, int]] = [(1, 3), (2, 3)]


def _attention_mask(lengths: List[Tuple[int, int]]) -> torch.Tensor:
    """Left-padded, as ``convert_prompts_responses_to_batch_tensors`` builds it."""
    totals = [prompt + response for prompt, response in lengths]
    mask = torch.zeros((len(totals), max(totals)), dtype=torch.bool)
    for row, total in enumerate(totals):
        mask[row, max(totals) - total :] = True
    return mask


def _layout(lengths: List[Tuple[int, int]], *, packed: bool = False) -> TokenMetadataLayout:
    mask = _attention_mask(lengths)
    totals = [prompt + response for prompt, response in lengths]
    if not packed:
        return TokenMetadataLayout(
            attention_mask=mask,
            sequence_lengths=totals,
            aligned_sequence_length=max(totals),
        )
    return TokenMetadataLayout(
        attention_mask=mask,
        sequence_lengths=totals,
        aligned_sequence_length=sum(totals),
        padded_sequence_lengths=totals,
        cu_seqlens_padded=cu_seqlens_from_lengths(totals),
    )


def _support(lengths: List[Tuple[int, int]]) -> PackedTensor:
    """Disjoint support sets per row, so a misplaced join changes the renormalizer."""
    response_lens = [response for _, response in lengths]
    rows = [[(row_index * 3 + offset) % VOCAB for offset in range(3)] for row_index in range(sum(response_lens))]
    return PackedTensor(
        torch.tensor(rows, dtype=SAMPLE_SUPPORT_TORCH_DTYPE),
        cu_seqlens_from_lengths(response_lens),
    )


def _batch_tensors(lengths: List[Tuple[int, int]]):
    mask = _attention_mask(lengths)
    generator = torch.Generator().manual_seed(0)
    sequences = torch.randint(0, VOCAB, mask.shape, generator=generator)
    support_row = 0
    for index, (_, response) in enumerate(lengths):
        for offset in range(response):
            sequences[index, mask.shape[1] - response + offset] = (support_row * 3) % VOCAB
            support_row += 1
    num_actions = max(response for _, response in lengths)
    loss_mask = torch.zeros((mask.shape[0], num_actions), dtype=torch.bool)
    for row, (_, response) in enumerate(lengths):
        loss_mask[row, num_actions - response :] = True
    logits = torch.randn((*mask.shape, VOCAB), dtype=torch.float64, generator=generator)
    return sequences, mask, loss_mask, num_actions, logits


def _reference_from_canonical_positions(logits, sequences, lengths, support: PackedTensor):
    """Score each response token at the canonical position whose logit predicts it."""
    expected = torch.zeros((logits.shape[0], logits.shape[1] - 1), dtype=logits.dtype)
    sequence_length = logits.shape[1]
    for row, (prompt, response) in enumerate(lengths):
        padding = sequence_length - prompt - response
        for offset in range(response):
            position = padding + prompt + offset - 1
            members = support.segment(row)[offset].long()
            expected[row, position] = logits[row, position, sequences[row, position + 1]] - torch.logsumexp(
                logits[row, position, members], dim=0
            )
    return expected


def _scores(lengths, *, packed=False, support=None, **overrides):
    """Score one batch through CSR support; the inputs are seeded alike."""
    sequences, mask, loss_mask, num_actions, logits = _batch_tensors(lengths)
    if packed:
        logits = torch.cat([logits[row, mask[row]] for row in range(logits.shape[0])], dim=0).unsqueeze(0)
    field = _support(lengths) if support is None else support
    return compute_sample_support_scores(
        logits,
        sequences,
        loss_mask,
        ragged_support(field),
        num_actions,
        packed=packed,
        metadata_layout=_layout(lengths, packed=packed),
        **DENSE_SCORER_KWARGS,
    )


def test_row_id_join_scores_the_response_suffix_domain():
    sequences, _, _, _, logits = _batch_tensors(JOIN_LENGTHS)

    actual = _scores(JOIN_LENGTHS).logprobs

    expected = _reference_from_canonical_positions(logits, sequences, JOIN_LENGTHS, _support(JOIN_LENGTHS))
    torch.testing.assert_close(actual, expected)


def test_packed_and_unpacked_joins_score_the_same_tokens(megatron_parallel_state):
    lengths = [(3, 3), (4, 2)]

    unpacked = _scores(lengths).logprobs
    packed = _scores(lengths, packed=True).logprobs

    torch.testing.assert_close(packed, unpacked)


def test_loss_active_token_without_support_is_rejected():
    lengths = [(3, 3)]
    support = _support(lengths)
    support.values[-1] = -1

    with pytest.raises(ValueError, match="every loss-active token"):
        _scores(lengths, support=support)


def test_multiple_loss_active_tokens_without_support_are_rejected():
    lengths = [(3, 3)]
    support = _support(lengths)
    support.values[-2:] = -1

    with pytest.raises(ValueError, match="every loss-active token"):
        _scores(lengths, support=support)


def test_replay_requires_support_a_loss_mask_and_a_layout():
    lengths = [(3, 3)]
    sequences, _, loss_mask, num_actions, logits = _batch_tensors(lengths)

    with pytest.raises(ValueError) as missing_support:
        compute_sample_support_scores(
            logits,
            sequences,
            loss_mask,
            None,
            num_actions,
            packed=False,
            metadata_layout=_layout(lengths),
            **DENSE_SCORER_KWARGS,
        )
    assert "generator.inference_engine.enable_return_sample_support_set" in str(missing_support.value)
    assert "SkyRLGymGenerator" in str(missing_support.value)
    with pytest.raises(ValueError, match="response loss mask"):
        compute_sample_support_scores(
            logits,
            sequences,
            None,
            ragged_support(_support(lengths)),
            num_actions,
            packed=False,
            metadata_layout=_layout(lengths),
            **DENSE_SCORER_KWARGS,
        )
    with pytest.raises(ValueError, match="token metadata layout"):
        compute_sample_support_scores(
            logits,
            sequences,
            loss_mask,
            ragged_support(_support(lengths)),
            num_actions,
            packed=False,
            metadata_layout=None,
            **DENSE_SCORER_KWARGS,
        )


# Include empty, singleton, partially filled, and full rows.
RAGGED_LENGTHS: List[Tuple[int, int]] = [(2, 4), (3, 3)]
RAGGED_MEMBER_COUNTS = [TOP_K, 0, 2, 1, TOP_K, 1, 0]


def _sized_support(lengths, member_counts) -> PackedTensor:
    """Fixed-width rows with the stated member counts and a distinct member set per row."""
    response_lens = [response for _, response in lengths]
    assert len(member_counts) == sum(response_lens)
    rows = torch.full((sum(response_lens), TOP_K), -1, dtype=SAMPLE_SUPPORT_TORCH_DTYPE)
    for row_index, count in enumerate(member_counts):
        for slot in range(count):
            rows[row_index, slot] = (row_index * TOP_K + slot) % VOCAB
    return PackedTensor(rows, cu_seqlens_from_lengths(response_lens))


def _sized_batch(lengths, member_counts):
    """Build a batch whose sampled tokens belong to their support rows."""
    support = _sized_support(lengths, member_counts)
    mask = _attention_mask(lengths)
    sequence_length = mask.shape[1]
    generator = torch.Generator().manual_seed(7)
    sequences = torch.randint(0, VOCAB, mask.shape, generator=generator)
    logits = torch.randn((*mask.shape, VOCAB), dtype=torch.float64, generator=generator)
    num_actions = max(response for _, response in lengths)
    loss_mask = torch.zeros((mask.shape[0], num_actions), dtype=torch.bool)
    row = 0
    for index, (_, response) in enumerate(lengths):
        loss_mask[index, num_actions - response :] = True
        for offset in range(response):
            members = support.values[row][support.values[row] >= 0]
            loss_mask[index, num_actions - response + offset] = members.numel() > 0
            if members.numel():
                # The weakest member, so the sampled token is not also the row's maximum.
                sequences[index, sequence_length - response + offset] = int(members[-1])
            row += 1
    return support, sequences, mask, loss_mask, num_actions, logits


def _sized_scores(lengths, member_counts, *, packed=False, **overrides):
    support, sequences, mask, loss_mask, num_actions, logits = _sized_batch(lengths, member_counts)
    if packed:
        logits = torch.cat([logits[row, mask[row]] for row in range(logits.shape[0])], dim=0).unsqueeze(0)
    return compute_sample_support_scores(
        logits,
        sequences,
        loss_mask,
        ragged_support(support),
        num_actions,
        packed=packed,
        metadata_layout=_layout(lengths, packed=packed),
        **{**DENSE_SCORER_KWARGS, **overrides},
    )


@pytest.mark.parametrize("packed", [False, True])
def test_csr_scores_logprobs_entropy_and_masks(megatron_parallel_state, packed):
    entropy_kwargs = dict(compute_entropy=True, entropy_requires_grad=False)
    ragged = _sized_scores(RAGGED_LENGTHS, RAGGED_MEMBER_COUNTS, packed=packed, **entropy_kwargs)

    assert ragged.entropy is not None
    assert int(ragged.valid_mask.sum()) == sum(1 for count in RAGGED_MEMBER_COUNTS if count)
    assert int((ragged.logprobs != 0).sum()) > 0


def test_csr_scores_have_gradients():
    support, sequences, _, loss_mask, num_actions, logits = _sized_batch(RAGGED_LENGTHS, RAGGED_MEMBER_COUNTS)
    source = logits.detach().clone().requires_grad_(True)
    scores = compute_sample_support_scores(
        source,
        sequences,
        loss_mask,
        ragged_support(support),
        num_actions,
        packed=False,
        metadata_layout=_layout(RAGGED_LENGTHS),
        **{
            **DENSE_SCORER_KWARGS,
            "compute_entropy": True,
            "entropy_requires_grad": True,
        },
    )
    assert scores.entropy is not None
    (scores.logprobs + scores.entropy).sum().backward()
    assert float(source.grad.abs().sum()) > 0


def test_a_single_member_row_scores_exactly_zero():
    lengths = [(2, 2)]
    entropy_kwargs = dict(compute_entropy=True, entropy_requires_grad=False)
    ragged = _sized_scores(lengths, [1, 1], **entropy_kwargs)

    scored = ragged.valid_mask
    assert int(scored.sum()) == 2
    assert torch.all(ragged.logprobs[scored] == 0.0)
    assert torch.all(ragged.entropy[scored] == 0.0)


def test_a_masked_row_with_no_members_scores_zero():
    lengths = [(3, 3)]
    scores = _sized_scores(lengths, [2, 2, 0])

    eos_position = _attention_mask(lengths).shape[1] - 2
    assert not scores.valid_mask[0, eos_position]
    assert scores.logprobs[0, eos_position] == 0


@pytest.mark.parametrize("member_counts", [[2, 2, 0], [2, 0, 0]])
def test_loss_active_empty_rows_are_rejected(member_counts):
    lengths = [(3, 3)]
    support, sequences, _, loss_mask, num_actions, logits = _sized_batch(lengths, member_counts)
    loss_mask[:] = True
    with pytest.raises(ValueError, match="every loss-active token"):
        compute_sample_support_scores(
            logits,
            sequences,
            loss_mask,
            ragged_support(support),
            num_actions,
            packed=False,
            metadata_layout=_layout(lengths),
            **DENSE_SCORER_KWARGS,
        )


def test_an_all_padding_microbatch_scores_nothing_in_the_ragged_form():
    support = PackedRaggedTensor(
        [torch.empty(0, dtype=SAMPLE_SUPPORT_TORCH_DTYPE)],
        cu_seqlens_from_lengths([]),
        cu_seqlens_from_lengths([0]),
    )
    mask = torch.zeros((1, 4), dtype=torch.bool)
    mask[0, 0] = True

    scores = compute_sample_support_scores(
        torch.randn((1, 4, VOCAB), dtype=torch.float64),
        torch.zeros((1, 4), dtype=torch.long),
        torch.zeros((1, 2), dtype=torch.bool),
        support,
        2,
        packed=False,
        metadata_layout=TokenMetadataLayout(attention_mask=mask, sequence_lengths=[1], aligned_sequence_length=4),
        **DENSE_SCORER_KWARGS,
    )

    assert not scores.valid_mask.any()
    assert torch.all(scores.logprobs == 0)


CSR_UNIT_SUPPORT = PackedRaggedTensor(
    [torch.tensor([1, 2, 3, 4, 5, 6], dtype=SAMPLE_SUPPORT_TORCH_DTYPE)],
    cu_seqlens_from_lengths([3, 0, 1, 2]),
    cu_seqlens_from_lengths([4]),
)
CSR_UNIT_ROW_IDS = torch.tensor([[SAMPLE_SUPPORT_NO_ROW, 0, 1, 2, 3]])
CSR_UNIT_SAMPLED = torch.tensor([[0, 3, 0, 4, 6]])
CSR_UNIT_KWARGS = dict(
    vocab_start_index=0,
    vocab_end_index=VOCAB,
    tp_group=None,
    compute_entropy=False,
    entropy_requires_grad=False,
)


def test_only_multi_member_rows_reach_the_projection(monkeypatch):
    widths: List[int] = []
    original = sample_support_replay._project_candidate_pairs

    def spy(hidden, row_ids, token_ids, weight, temperature, chunk_size):
        widths.append(int(token_ids.numel()))
        return original(hidden, row_ids, token_ids, weight, temperature, chunk_size)

    monkeypatch.setattr(sample_support_replay, "_project_candidate_pairs", spy)
    hidden = torch.randn(1, 5, 3, dtype=torch.float64)
    weight = torch.randn(VOCAB, 3, dtype=torch.float64)

    ragged = sample_support_scores(
        hidden,
        CSR_UNIT_SAMPLED,
        CSR_UNIT_SUPPORT,
        CSR_UNIT_ROW_IDS,
        lm_head_weight=weight,
        **CSR_UNIT_KWARGS,
    )
    # Rows 0 and 3 are the only ones with more than one member: 3 + 2 pairs, in one chunk.
    assert widths == [5]
    assert ragged.valid_mask.tolist() == [[False, True, False, True, True]]
    assert float(ragged.logprobs[0, 3]) == 0.0


def test_two_positions_naming_one_support_row_are_refused():
    with pytest.raises(ValueError, match="named by more than one model position"):
        sample_support_scores(
            torch.randn(1, 2, VOCAB, dtype=torch.float64),
            torch.tensor([[1, 1]]),
            CSR_UNIT_SUPPORT,
            torch.tensor([[0, 0]]),
            **CSR_UNIT_KWARGS,
        )


def test_ragged_support_ids_must_use_the_canonical_dtype():
    support = PackedRaggedTensor(
        [torch.tensor([1, 2], dtype=torch.int64)],
        cu_seqlens_from_lengths([2]),
        cu_seqlens_from_lengths([1]),
    )

    with pytest.raises(ValueError, match=str(SAMPLE_SUPPORT_TORCH_DTYPE)):
        sample_support_scores(
            torch.randn(1, 1, VOCAB, dtype=torch.float64),
            torch.tensor([[1]]),
            support,
            torch.tensor([[0]]),
            **CSR_UNIT_KWARGS,
        )


def test_ragged_tensor_parallel_shards_reduce_to_the_unsharded_scores(tensor_parallel):
    logits = torch.randn(1, 5, VOCAB, dtype=torch.float64)
    entropy_kwargs = dict(compute_entropy=True, entropy_requires_grad=False)

    sharded = _tensor_parallel_scores(
        tensor_parallel,
        logits,
        CSR_UNIT_SAMPLED,
        CSR_UNIT_SUPPORT,
        CSR_UNIT_ROW_IDS,
        **entropy_kwargs,
    )

    unsharded = sample_support_scores(
        logits,
        CSR_UNIT_SAMPLED,
        CSR_UNIT_SUPPORT,
        CSR_UNIT_ROW_IDS,
        **{**CSR_UNIT_KWARGS, **entropy_kwargs},
    )
    torch.testing.assert_close(sharded.logprobs, unsharded.logprobs)
    torch.testing.assert_close(sharded.entropy, unsharded.entropy)
    assert torch.equal(sharded.valid_mask, unsharded.valid_mask)


def test_the_ragged_reduction_is_sized_by_support_rows_not_model_positions(
    tensor_parallel,
):
    logits = torch.randn(1, 5, VOCAB, dtype=torch.float64)
    entropy_kwargs = dict(compute_entropy=True, entropy_requires_grad=False)
    rows, positions = CSR_UNIT_SUPPORT.num_rows, CSR_UNIT_SAMPLED.numel()
    assert rows < positions

    _tensor_parallel_scores(
        tensor_parallel,
        logits,
        CSR_UNIT_SAMPLED,
        CSR_UNIT_SUPPORT,
        CSR_UNIT_ROW_IDS,
        **entropy_kwargs,
    )

    max_op, sum_op = torch.distributed.ReduceOp.MAX, torch.distributed.ReduceOp.SUM
    assert Counter(tensor_parallel.calls) == Counter({(max_op, (rows,)): TP_SIZE, (sum_op, (3, rows)): TP_SIZE})


def _tensor_parallel_fused_csr(fake, hidden, weight, sampled_ids, support, row_ids):
    """Run every vocabulary shard concurrently, keeping each rank's own differentiable inputs."""
    width = VOCAB // fake.world_size
    outputs: List[tuple | None] = [None] * fake.world_size
    errors: List[BaseException] = []

    def run(rank: int) -> None:
        fake.set_rank(rank)
        start = rank * width
        end = VOCAB if rank == fake.world_size - 1 else start + width
        shard_hidden = hidden.detach().clone().requires_grad_(True)
        shard_weight = weight[start:end].detach().clone().requires_grad_(True)
        try:
            scores = sample_support_scores(
                shard_hidden,
                sampled_ids,
                support,
                row_ids,
                vocab_start_index=start,
                vocab_end_index=end,
                tp_group=object(),
                compute_entropy=False,
                entropy_requires_grad=False,
                lm_head_weight=shard_weight,
            )
            outputs[rank] = (scores, shard_hidden, shard_weight)
        except BaseException as error:
            errors.append(error)
            fake._barrier.abort()

    threads = [threading.Thread(target=run, args=(rank,)) for rank in range(fake.world_size)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return outputs


def test_a_vocabulary_shard_owning_no_member_still_reaches_the_backward_graph(
    tensor_parallel,
):
    """The projection short-circuits an empty pair list, and every rank still needs a grad buffer."""
    # Every member lands in the low shard, so the high shard owns none of them.
    support = PackedRaggedTensor(
        [torch.tensor([0, 1, 2, 3], dtype=SAMPLE_SUPPORT_TORCH_DTYPE)],
        cu_seqlens_from_lengths([2, 2]),
        cu_seqlens_from_lengths([2]),
    )
    hidden = torch.randn(1, 2, 3, dtype=torch.float64)
    weight = torch.randn(VOCAB, 3, dtype=torch.float64)

    outputs = _tensor_parallel_fused_csr(
        tensor_parallel,
        hidden,
        weight,
        torch.tensor([[1, 3]]),
        support,
        torch.tensor([[0, 1]]),
    )

    for scores, shard_hidden, shard_weight in outputs:
        scores.logprobs.sum().backward()
        assert shard_hidden.grad is not None
        assert shard_weight.grad is not None
    _, empty_hidden, empty_weight = outputs[1]
    assert torch.all(empty_weight.grad == 0)
    assert torch.all(empty_hidden.grad == 0)
    # And the shard that owns everything still produced the scores.
    assert outputs[0][0].valid_mask.all()


# Score centering over the recorded support.

SC_SUPPORT_IDS = torch.tensor(
    [[[2, 7, 9, -1], [8, 1, -1, -1], [1, 6, 10, 4], [5, -1, -1, -1], [-1, -1, -1, -1]]],
    dtype=SAMPLE_SUPPORT_TORCH_DTYPE,
)
SC_SAMPLED_IDS = torch.tensor([[7, 1, 10, 5, 3]])
# Rows 0 and 2 carry less than all of the sampler's mass, as a repaired row or a tie at the
# k-th logit does.
SC_ROW_MASS = torch.tensor([0.9, 1.0, 0.7, 1.0, 1.0], dtype=torch.float64)
SC_BAND = RatioBand(low=0.8, high=1.25)


def _sampler_scores(seed: int = 0) -> torch.Tensor:
    """Sampler logprobs on the ``SC_SUPPORT_IDS`` grid, each row's members summing to ``SC_ROW_MASS``."""
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(SC_SUPPORT_IDS.shape, generator=generator, dtype=torch.float64)
    present = SC_SUPPORT_IDS >= 0
    log_q = torch.where(present, raw, float("-inf")).log_softmax(dim=-1) + SC_ROW_MASS.log().reshape(1, -1, 1)
    return torch.where(present, log_q, 0.0).float()


def _centering_support(member_logprobs: torch.Tensor):
    support = ragged_support([SC_SUPPORT_IDS[0].numpy()], logprobs=[member_logprobs[0].numpy()])
    return support, torch.arange(support.num_rows).reshape(1, -1)


def _reference_centering(logits, sampled_ids, member_logprobs, weight, negative_advantage=False):
    """Per position: T = Σ sg[q̃ f(π / q̃)] log π and log q̃ of the sampled token, densely."""
    terms, sampler_logprobs = [], []
    for row_logits, sampled_id, ids, scores in zip(
        logits.reshape(-1, VOCAB),
        sampled_ids.reshape(-1),
        SC_SUPPORT_IDS.reshape(-1, TOP_K),
        member_logprobs.reshape(-1, TOP_K),
    ):
        present = ids >= 0
        members = ids[present].long()
        if members.numel() < 2:
            terms.append(row_logits.new_zeros(()))
            sampler_logprobs.append(row_logits.new_zeros(()))
            continue
        log_q = scores[present].to(row_logits.dtype)
        log_q = log_q - torch.logsumexp(log_q, dim=0)
        log_pi = row_logits[members] - torch.logsumexp(row_logits[members], dim=0)
        weights = weight.weights(log_pi.detach(), log_q, negative_advantage=negative_advantage)
        coefficients = (log_q.exp() * weights).detach()
        terms.append((coefficients * log_pi).sum())
        sampler_logprobs.append(log_q[members == sampled_id][0])
    return torch.stack(terms).reshape(sampled_ids.shape), torch.stack(sampler_logprobs).reshape(sampled_ids.shape)


def test_centering_term_matches_a_dense_reference_in_value_and_gradient():
    logits = torch.randn(1, 5, VOCAB, dtype=torch.float64, requires_grad=True)
    member_logprobs = _sampler_scores()
    support, row_ids = _centering_support(member_logprobs)

    scores = sample_support_scores(
        logits,
        SC_SAMPLED_IDS,
        support,
        row_ids,
        vocab_start_index=0,
        vocab_end_index=VOCAB,
        tp_group=None,
        centering_weight=SC_BAND,
    )
    expected_term, expected_sampler_logprobs = _reference_centering(logits, SC_SAMPLED_IDS, member_logprobs, SC_BAND)

    torch.testing.assert_close(scores.centering_term, expected_term)
    torch.testing.assert_close(scores.sampler_logprobs, expected_sampler_logprobs)
    (grad,) = torch.autograd.grad(scores.centering_term.sum(), logits)
    (expected_grad,) = torch.autograd.grad(expected_term.sum(), logits)
    torch.testing.assert_close(grad, expected_grad)
    # Singleton and empty rows are exact: log π = 0 there.
    assert scores.centering_term[0, 3] == 0 and scores.centering_term[0, 4] == 0
    torch.testing.assert_close(scores.sampler_mass[0, :4], SC_ROW_MASS[:4].float(), check_dtype=False)


def test_centering_through_the_fused_lm_head_matches_the_logit_path():
    hidden = torch.randn(1, 5, 3, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(VOCAB, 3, dtype=torch.float64, requires_grad=True)
    member_logprobs = _sampler_scores()
    support, row_ids = _centering_support(member_logprobs)
    kwargs = dict(
        vocab_start_index=0,
        vocab_end_index=VOCAB,
        tp_group=None,
        centering_weight=SC_BAND,
    )

    fused = sample_support_scores(
        hidden,
        SC_SAMPLED_IDS,
        support,
        row_ids,
        lm_head_weight=weight,
        chunk_size=4,
        **kwargs,
    )
    (fused.logprobs + fused.centering_term).sum().backward()
    explicit_hidden = hidden.detach().clone().requires_grad_(True)
    explicit_weight = weight.detach().clone().requires_grad_(True)
    explicit = sample_support_scores(explicit_hidden @ explicit_weight.T, SC_SAMPLED_IDS, support, row_ids, **kwargs)
    (explicit.logprobs + explicit.centering_term).sum().backward()

    # The fused projection computes in float32.
    close = dict(check_dtype=False, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(fused.centering_term, explicit.centering_term, **close)
    torch.testing.assert_close(fused.sampler_logprobs, explicit.sampler_logprobs, **close)
    torch.testing.assert_close(hidden.grad, explicit_hidden.grad, **close)
    torch.testing.assert_close(weight.grad, explicit_weight.grad, **close)


def test_an_in_band_support_has_a_gradient_free_centering_term():
    """With every ratio inside the band, c = π and Σ π ∇log π = ∇Σ π = 0."""
    logits = torch.randn(1, 5, VOCAB, dtype=torch.float64, requires_grad=True)
    support, row_ids = _centering_support(_sampler_scores())

    scores = sample_support_scores(
        logits,
        SC_SAMPLED_IDS,
        support,
        row_ids,
        vocab_start_index=0,
        vocab_end_index=VOCAB,
        tp_group=None,
        centering_weight=RatioBand(low=0.0, high=float("inf")),
    )

    (grad,) = torch.autograd.grad(scores.centering_term.sum(), logits)
    torch.testing.assert_close(grad, torch.zeros_like(grad))


SC_DRIFT_CASES = {
    "rollout_is": AlgorithmConfig(policy_loss_type="rollout_is", eps_clip_low=0.2, eps_clip_high=0.25),
    "rollout_is_wide": AlgorithmConfig(policy_loss_type="rollout_is", eps_clip_low=0.5, eps_clip_high=1.0),
    "cispo_rollout": AlgorithmConfig(
        policy_loss_type="cispo",
        cispo=CISPOConfig(cispo_eps_clip_low=0.1, cispo_eps_clip_high=0.1, cispo_anchor="rollout"),
    ),
    "dppo_tv": AlgorithmConfig(
        policy_loss_type="dppo", dppo=DPPOConfig(dppo_type="binary_tv", delta_low=0.02, delta_high=0.02)
    ),
    "dppo_kl": AlgorithmConfig(
        policy_loss_type="dppo", dppo=DPPOConfig(dppo_type="binary_kl", delta_low=0.002, delta_high=0.002)
    ),
    "reinforce": AlgorithmConfig(policy_loss_type="reinforce"),
}


@pytest.mark.parametrize("past_the_ratio_clip", [False, True])
@pytest.mark.parametrize("advantage", [1.0, -1.0])
@pytest.mark.parametrize("case", sorted(SC_DRIFT_CASES))
def test_centered_losses_have_no_drift_under_a_constant_advantage(case, advantage, past_the_ratio_clip):
    """Σ_y q̃_y ∇L_y = 0 at every row, through each production loss and the raw rollout logprobs.

    The trainer sits near the sampler, so each loss's weight keeps some members and drops or clamps
    others, and the rows whose sampler mass is below 1 separate q̃ from the raw rollout logprob.
    ``past_the_ratio_clip`` gives one member a log ratio beyond ``SAMPLER_LOG_RATIO_CLIP``.
    """
    config = SC_DRIFT_CASES[case]
    loss_fn = PolicyLossRegistry.get(config.policy_loss_type)
    weight = score_centering_weight(config.policy_loss_type, config)
    member_logprobs = _sampler_scores(seed=3)
    if past_the_ratio_clip:
        member_logprobs[0, 2, 3] = -30.0
    generator = torch.Generator().manual_seed(4)
    present = SC_SUPPORT_IDS >= 0
    logits = torch.full((1, 5, VOCAB), -4.0, dtype=torch.float64)
    near_sampler = member_logprobs.double() + 0.2 * torch.randn(
        member_logprobs.shape, generator=generator, dtype=torch.float64
    )
    if past_the_ratio_clip:
        # π^S ≈ 0.005 against q̃ ≈ e^-29: small enough that DPPO's trust region keeps it.
        near_sampler[0, 2, 3] = -5.0
    logits.scatter_(-1, SC_SUPPORT_IDS.clamp(min=0).long(), torch.where(present, near_sampler, -4.0))
    support, row_ids = _centering_support(member_logprobs)
    advantages = torch.full((1, 1), advantage, dtype=torch.float64)

    for position, ids in enumerate(SC_SUPPORT_IDS[0]):
        members = ids[ids >= 0].tolist()
        if len(members) < 2:
            continue
        log_q = member_logprobs[0, position, : len(members)].double()
        q_tilde = (log_q - torch.logsumexp(log_q, dim=0)).exp()
        expected_update = torch.zeros(VOCAB, dtype=torch.float64)
        for member, member_q in zip(members, q_tilde, strict=True):
            row_logits = logits[:, position : position + 1].clone().requires_grad_(True)
            scores = sample_support_scores(
                row_logits,
                torch.tensor([[member]]),
                support,
                row_ids[:, position : position + 1],
                vocab_start_index=0,
                vocab_end_index=VOCAB,
                tp_group=None,
                centering_weight=weight,
            )
            rollout_logprobs = log_q[members.index(member)].reshape(1, 1)
            inputs = score_centering_loss_inputs(scores, slice(None), rollout_logprobs, advantages)
            loss, _ = loss_fn(
                scores.logprobs,
                None,
                advantages,
                config,
                loss_mask=torch.ones(1, 1, dtype=torch.float64),
                rollout_logprobs=inputs.sampler_logprobs,
                **inputs.loss_kwargs(),
            )
            (grad,) = torch.autograd.grad(loss, row_logits)
            expected_update += member_q * grad.reshape(-1)
        torch.testing.assert_close(expected_update, torch.zeros_like(expected_update), atol=1e-10, rtol=0)


def test_a_sign_dependent_weight_scores_both_signs_against_a_dense_reference():
    logits = torch.randn(1, 5, VOCAB, dtype=torch.float64, requires_grad=True)
    member_logprobs = _sampler_scores()
    support, row_ids = _centering_support(member_logprobs)
    weight = DPPOTrustRegion(DPPOType.BINARY_TV, delta_low=0.05, delta_high=0.05)

    scores = sample_support_scores(
        logits,
        SC_SAMPLED_IDS,
        support,
        row_ids,
        vocab_start_index=0,
        vocab_end_index=VOCAB,
        tp_group=None,
        centering_weight=weight,
    )

    for term, negative in ((scores.centering_term, False), (scores.centering_term_negative, True)):
        expected, _ = _reference_centering(logits, SC_SAMPLED_IDS, member_logprobs, weight, negative_advantage=negative)
        torch.testing.assert_close(term, expected)
    assert not torch.allclose(scores.centering_term, scores.centering_term_negative)


def test_centering_shards_reduce_to_the_unsharded_term_and_gradient(tensor_parallel):
    logits = torch.randn(1, 5, VOCAB, dtype=torch.float64, requires_grad=True)
    support, row_ids = _centering_support(_sampler_scores())
    centering_kwargs = dict(centering_weight=SC_BAND)

    sharded = _tensor_parallel_scores(
        tensor_parallel,
        logits,
        SC_SAMPLED_IDS,
        support,
        row_ids,
        all_ranks=True,
        **centering_kwargs,
    )
    unsharded = sample_support_scores(
        logits,
        SC_SAMPLED_IDS,
        support,
        row_ids,
        vocab_start_index=0,
        vocab_end_index=VOCAB,
        tp_group=None,
        **centering_kwargs,
    )

    for field in ("centering_term", "sampler_logprobs", "support_tv", "sampler_mass"):
        torch.testing.assert_close(getattr(sharded[0], field), getattr(unsharded, field))
    # Each rank's backward reaches only its own vocabulary shard; together they make the whole gradient.
    (sharded_grad,) = torch.autograd.grad(sum(scores.centering_term.sum() for scores in sharded), logits)
    (unsharded_grad,) = torch.autograd.grad(unsharded.centering_term.sum(), logits)
    torch.testing.assert_close(sharded_grad, unsharded_grad)


def test_centering_requires_the_sampler_member_logprobs():
    support, row_ids = _from_grid(SC_SUPPORT_IDS)
    with pytest.raises(ValueError, match="sampler's logprob of every support member"):
        sample_support_scores(
            torch.randn(1, 5, VOCAB),
            SC_SAMPLED_IDS,
            support,
            row_ids,
            vocab_start_index=0,
            vocab_end_index=VOCAB,
            tp_group=None,
            centering_weight=SC_BAND,
        )


def _centering_scores(lengths, *, packed):
    """Score one seeded batch with score centering; each member's sampler score is distinct."""
    sequences, mask, loss_mask, num_actions, logits = _batch_tensors(lengths)
    if packed:
        logits = torch.cat([logits[row, mask[row]] for row in range(logits.shape[0])], dim=0).unsqueeze(0)
    field = _support(lengths)
    grids = [field.segment(index).numpy() for index in range(len(field))]
    member_logprobs = [-0.5 - 0.1 * np.arange(grid.size, dtype=np.float32).reshape(grid.shape) for grid in grids]
    return compute_sample_support_scores(
        logits,
        sequences,
        loss_mask,
        ragged_support(grids, logprobs=member_logprobs),
        num_actions,
        packed=packed,
        metadata_layout=_layout(lengths, packed=packed),
        **DENSE_SCORER_KWARGS,
        centering_weight=SC_BAND,
    )


def test_packed_and_unpacked_centering_land_in_the_same_canonical_positions(
    megatron_parallel_state,
):
    unpacked = _centering_scores(JOIN_LENGTHS, packed=False)
    packed = _centering_scores(JOIN_LENGTHS, packed=True)

    for field in ("centering_term", "sampler_logprobs", "support_tv", "sampler_mass"):
        torch.testing.assert_close(getattr(packed, field), getattr(unpacked, field))
    assert unpacked.centering_term.abs().sum() > 0

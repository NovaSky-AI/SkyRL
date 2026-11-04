"""Sample-support replay through the FSDP forward."""

import math
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch import nn

from skyrl.backends.skyrl_train.distributed.ulysses import utils as ulysses_utils
from skyrl.backends.skyrl_train.utils.off_policy_correction_utils import RatioBand
from skyrl.backends.skyrl_train.utils.packed_ragged_tensor import PackedRaggedTensor
from skyrl.backends.skyrl_train.utils.packed_tensor import (
    PackedTensor,
    cu_seqlens_from_lengths,
)
from skyrl.backends.skyrl_train.utils.sample_support import (
    SAMPLE_SUPPORT_FIELD,
    SAMPLE_SUPPORT_NO_ROW,
    SAMPLE_SUPPORT_TORCH_DTYPE,
    SCORE_CENTERING_SCORES_KEY,
    SUPPORT_LOGPROB_TERM_KEY,
    CompactSampleSupport,
)
from skyrl.backends.skyrl_train.utils.sample_support_replay import (
    SampleSupportScores,
    ScoreCenteringMetric,
    SupportLogprobTerm,
)
from skyrl.backends.skyrl_train.workers.model_wrapper import (
    HFModelWrapper,
    _SampleSupportChannels,
)
from skyrl.backends.skyrl_train.workers.worker import PolicyWorkerBase
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.dataset.replay_buffer import Experience
from tests.backends.skyrl_train.sample_support_helpers import ragged_support

VOCAB = 12


class _TokenIndexedLM(nn.Module):
    """A position-independent model for comparing packed and unpacked layouts."""

    def __init__(self, vocab_size: int = VOCAB):
        super().__init__()
        self.table = nn.Parameter(torch.randn(vocab_size, vocab_size, dtype=torch.float64))

    def forward(self, input_ids, **kwargs):
        return {"logits": self.table[input_ids]}


def _support(segments: list[list[list[int]]]) -> PackedTensor:
    """Pack one ``[response_tokens, top_k]`` reference grid per trajectory; ``-1`` marks no member."""
    return PackedTensor(
        torch.tensor([row for segment in segments for row in segment], dtype=SAMPLE_SUPPORT_TORCH_DTYPE),
        cu_seqlens_from_lengths([len(segment) for segment in segments]),
    )


def _loss_mask(response_lengths: list[int], num_actions: int) -> torch.Tensor:
    """Right-aligned, as ``convert_prompts_responses_to_batch_tensors`` builds it."""
    mask = torch.zeros((len(response_lengths), num_actions), dtype=torch.bool)
    for row, length in enumerate(response_lengths):
        mask[row, num_actions - length :] = True
    return mask


def _reference(table, sequences, support: PackedTensor, num_actions: int) -> torch.Tensor:
    """Renormalize each response token over its recorded support row."""
    sequence_length = sequences.shape[1]
    expected = torch.zeros((sequences.shape[0], num_actions), dtype=table.dtype)
    for row in range(sequences.shape[0]):
        segment = support.segment(row)
        for offset in range(segment.shape[0]):
            position = sequence_length - segment.shape[0] + offset - 1
            members = segment[offset]
            members = members[members >= 0].long()
            logits = table[sequences[row, position]]
            expected[row, num_actions - segment.shape[0] + offset] = logits[
                sequences[row, position + 1]
            ] - torch.logsumexp(logits[members], dim=0)
    return expected


def _reference_entropy(table, sequences, support: PackedTensor, num_actions: int) -> torch.Tensor:
    """Compute entropy over each response token's recorded support."""
    sequence_length = sequences.shape[1]
    expected = torch.zeros((sequences.shape[0], num_actions), dtype=table.dtype)
    for row in range(sequences.shape[0]):
        segment = support.segment(row)
        for offset in range(segment.shape[0]):
            position = sequence_length - segment.shape[0] + offset - 1
            members = segment[offset][segment[offset] >= 0].long()
            member_logprobs = torch.log_softmax(table[sequences[row, position]][members], dim=0)
            expected[row, num_actions - segment.shape[0] + offset] = -(member_logprobs.exp() * member_logprobs).sum()
    return expected


def _wrapper(model: nn.Module, *, packed: bool = False) -> HFModelWrapper:
    return HFModelWrapper(
        model,
        bf16=False,
        use_flash_attention_2=packed,
        remove_microbatch_padding=packed,
    )


def _forward(wrapper, sequences, attention_mask, support, response_lengths, num_actions):
    """Run the forward on the CSR form of the ``support`` reference grid."""
    return wrapper(
        sequences,
        num_actions,
        attention_mask=attention_mask,
        sample_support=ragged_support(support),
        loss_mask=_loss_mask(response_lengths, num_actions),
        enable_sample_support_replay=True,
    )


def _forward_entropy(wrapper, sequences, attention_mask, support, response_lengths, num_actions, **kwargs):
    _, output = wrapper(
        sequences,
        num_actions,
        attention_mask=attention_mask,
        sample_support=ragged_support(support),
        loss_mask=_loss_mask(response_lengths, num_actions),
        enable_sample_support_replay=True,
        return_output=True,
        compute_entropy=True,
        **kwargs,
    )
    return output["entropy"][:, -num_actions - 1 : -1]


# (prompt_len, response_len) per trajectory. Row 1 is shorter in both, so it carries left
# padding and its support lands at a different canonical position than row 0's -- a channel
# placed without the per-row shift agrees with row 0 and is wrong for row 1.
RAGGED = [(2, 2), (1, 1)]


def _ragged_batch():
    totals = [prompt + response for prompt, response in RAGGED]
    sequence_length = max(totals)
    # Distinct ids everywhere, so a swapped row or position changes the logits it reads.
    sequences = torch.zeros((len(totals), sequence_length), dtype=torch.long)
    attention_mask = torch.zeros((len(totals), sequence_length), dtype=torch.long)
    next_id = 1
    for row, total in enumerate(totals):
        attention_mask[row, sequence_length - total :] = 1
        for column in range(sequence_length - total, sequence_length):
            sequences[row, column] = next_id
            next_id += 1
    return sequences, attention_mask


def _ragged_support(sequences) -> PackedTensor:
    """Member 0 is the token the row scores, as the sampler recorded it; member 1 is a decoy."""
    segments = []
    for row, (_prompt, response) in enumerate(RAGGED):
        segment = []
        for offset in range(response):
            token = int(sequences[row, sequences.shape[1] - response + offset])
            segment.append([token, (token + 5) % VOCAB])
        segments.append(segment)
    return _support(segments)


def test_forward_scores_the_response_suffix_domain():
    """One trajectory, no padding: every response token renormalizes over its recorded row."""
    sequences = torch.tensor([[1, 2, 3, 4]])
    support = _support([[[3, 8], [4, 0]]])
    model = _TokenIndexedLM()

    actual = _forward(_wrapper(model), sequences, torch.ones_like(sequences), support, [2], 2)
    actual.sum().backward()

    reference_table = model.table.detach().clone().requires_grad_(True)
    expected = _reference(reference_table, sequences, support, 2)
    expected.sum().backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(model.table.grad, reference_table.grad)


def test_left_padded_rows_shift_the_row_id_channel_by_their_own_padding_width():
    sequences, attention_mask = _ragged_batch()
    support = _ragged_support(sequences)
    model = _TokenIndexedLM()

    actual = _forward(_wrapper(model), sequences, attention_mask, support, [2, 1], 2)

    expected = _reference(model.table.detach(), sequences, support, 2)
    torch.testing.assert_close(actual, expected)
    # Row 1 generated one token, so its first response slot is never scored.
    assert actual[1, 0] == 0


def test_packed_microbatch_matches_the_padded_rectangle():
    """``remove_microbatch_padding`` reroutes every channel through ``nnz_indices``."""
    sequences, attention_mask = _ragged_batch()
    support = _ragged_support(sequences)
    model = _TokenIndexedLM()
    packed_model = _TokenIndexedLM()
    packed_model.table.data.copy_(model.table.data)

    unpacked = _forward(_wrapper(model), sequences, attention_mask, support, [2, 1], 2)
    packed = _forward(_wrapper(packed_model, packed=True), sequences, attention_mask, support, [2, 1], 2)
    unpacked.sum().backward()
    packed.sum().backward()

    torch.testing.assert_close(packed, unpacked)
    torch.testing.assert_close(packed_model.table.grad, model.table.grad)


@pytest.mark.parametrize("packed", [False, True])
def test_entropy_comes_from_the_recorded_support_not_the_vocabulary(monkeypatch, packed):
    sequences, attention_mask = _ragged_batch()
    support = _ragged_support(sequences)
    model = _TokenIndexedLM()
    wrapper = _wrapper(model, packed=packed)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("full-vocabulary entropy should not run during support replay")

    monkeypatch.setattr(wrapper, "chunked_entropy_from_logits_fn", fail_if_called)
    entropy = _forward_entropy(wrapper, sequences, attention_mask, support, [2, 1], 2)

    torch.testing.assert_close(entropy, _reference_entropy(model.table.detach(), sequences, support, 2))


@pytest.mark.parametrize("entropy_requires_grad", [False, True])
def test_entropy_carries_gradients_only_when_asked(entropy_requires_grad):
    sequences = torch.tensor([[1, 2, 3, 4]])
    support = _support([[[3, 8], [4, 0]]])
    model = _TokenIndexedLM()

    entropy = _forward_entropy(
        _wrapper(model),
        sequences,
        torch.ones_like(sequences),
        support,
        [2],
        2,
        entropy_requires_grad=entropy_requires_grad,
    )

    assert entropy.requires_grad == entropy_requires_grad
    if not entropy_requires_grad:
        return
    entropy.sum().backward()
    reference_table = model.table.detach().clone().requires_grad_(True)
    _reference_entropy(reference_table, sequences, support, 2).sum().backward()
    torch.testing.assert_close(model.table.grad, reference_table.grad)


def test_replay_never_scores_every_position_over_the_full_vocabulary(monkeypatch):
    """The ordinary full-vocabulary path stays off during support replay."""

    def fail_if_called(*args, **kwargs):
        raise AssertionError("full-sequence vocabulary logprobs should not run during support replay")

    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.workers.model_wrapper.logprobs_from_logits",
        fail_if_called,
    )
    sequences, attention_mask = _ragged_batch()
    support = _ragged_support(sequences)

    actual = _forward(_wrapper(_TokenIndexedLM()), sequences, attention_mask, support, [2, 1], 2)

    assert torch.isfinite(actual).all()


def test_an_unsupported_loss_active_token_is_rejected():
    sequences, attention_mask = _ragged_batch()
    support = _ragged_support(sequences)
    support.values[1] = -1

    with pytest.raises(ValueError, match="requires captured support for every loss-active token"):
        _forward(_wrapper(_TokenIndexedLM()), sequences, attention_mask, support, [2, 1], 2)


def test_replay_disabled_takes_the_ordinary_full_vocabulary_path():
    """Also the CPU cover for the ``logits.is_cuda`` gate on the flash-attn cross entropy."""
    sequences, attention_mask = _ragged_batch()
    model = _TokenIndexedLM()
    wrapper = _wrapper(model)

    disabled = wrapper(sequences, 2, attention_mask=attention_mask)
    ignored = wrapper(
        sequences,
        2,
        attention_mask=attention_mask,
        sample_support=ragged_support(_ragged_support(sequences)),
        loss_mask=_loss_mask([2, 1], 2),
        enable_sample_support_replay=False,
    )

    torch.testing.assert_close(ignored, disabled)
    table = model.table.detach()
    for row in range(sequences.shape[0]):
        for slot in range(2):
            position = sequences.shape[1] - 3 + slot
            logits = table[sequences[row, position]]
            expected = logits[sequences[row, position + 1]] - torch.logsumexp(logits, dim=0)
            torch.testing.assert_close(disabled[row, slot], expected)


@pytest.mark.parametrize(
    ("omitted", "message"),
    [("sample_support", f"received no {SAMPLE_SUPPORT_FIELD!r}"), ("loss_mask", "no loss mask")],
)
def test_replay_requires_both_the_support_and_the_loss_mask(omitted, message):
    sequences, attention_mask = _ragged_batch()
    kwargs = {
        "sample_support": ragged_support(_ragged_support(sequences)),
        "loss_mask": _loss_mask([2, 1], 2),
        "enable_sample_support_replay": True,
    }
    kwargs[omitted] = None

    with pytest.raises(ValueError, match=message):
        _wrapper(_TokenIndexedLM())(sequences, 2, attention_mask=attention_mask, **kwargs)


def test_sequence_parallel_slice_pads_row_ids_with_a_sentinel(monkeypatch):
    """Ulysses padding must not turn a missing row into row zero."""
    sp_size = 2
    monkeypatch.setattr(ulysses_utils, "get_ulysses_sequence_parallel_group", lambda: object())
    monkeypatch.setattr(
        ulysses_utils,
        "slice_input_tensor",
        lambda tensor, dim, padding=True, group=None: tensor.chunk(sp_size, dim=dim)[1],
    )
    channels = _SampleSupportChannels(
        row_ids=torch.tensor([[0, 1, 2]]),
        loss_mask=torch.tensor([[True, True, True]]),
    )

    tail = channels.slice_for_sequence_parallel(sp_size)

    assert tail.row_ids.tolist() == [[2, SAMPLE_SUPPORT_NO_ROW]]
    assert tail.loss_mask.tolist() == [[True, False]]


def _scored_forward(wrapper, sequences, attention_mask, support: PackedTensor, response_lengths, num_actions):
    """Score-centered forward over ``support`` whose members' sampler logprobs are distinct."""
    grids = [support.segment(index).numpy() for index in range(len(support))]
    member_logprobs = [-0.3 - 0.1 * np.arange(grid.size, dtype=np.float32).reshape(grid.shape) for grid in grids]
    _, output = wrapper(
        sequences,
        num_actions,
        attention_mask=attention_mask,
        sample_support=ragged_support(grids, logprobs=member_logprobs),
        loss_mask=_loss_mask(response_lengths, num_actions),
        enable_sample_support_replay=True,
        centering_weight=RatioBand(low=0.5, high=2.0),
        return_output=True,
    )
    return output[SCORE_CENTERING_SCORES_KEY]


def test_centering_scores_land_in_canonical_positions_packed_or_not():
    sequences, attention_mask = _ragged_batch()
    support = _ragged_support(sequences)
    model = _TokenIndexedLM()
    packed_model = _TokenIndexedLM()
    packed_model.table.data.copy_(model.table.data)

    unpacked = _scored_forward(_wrapper(model), sequences, attention_mask, support, [2, 1], 2)
    packed = _scored_forward(_wrapper(packed_model, packed=True), sequences, attention_mask, support, [2, 1], 2)
    unpacked.centering_term.sum().backward()
    packed.centering_term.sum().backward()

    for field in ("centering_term", "sampler_logprobs", "support_tv", "sampler_mass"):
        torch.testing.assert_close(getattr(packed, field), getattr(unpacked, field))
    assert torch.equal(packed.valid_mask, unpacked.valid_mask)
    torch.testing.assert_close(packed_model.table.grad, model.table.grad)
    # Only the three response positions predict a supported token.
    assert int(unpacked.valid_mask.sum()) == 3


def test_centering_requires_sample_support_replay():
    sequences, attention_mask = _ragged_batch()
    with pytest.raises(ValueError, match="score centering requires sample-support replay"):
        _wrapper(_TokenIndexedLM())(
            sequences, 2, attention_mask=attention_mask, centering_weight=RatioBand(low=0.5, high=2.0)
        )


def test_the_worker_centers_rollout_is_with_the_post_override_band():
    cfg = SkyRLTrainConfig()
    cfg.trainer.micro_train_batch_size_per_gpu = 2
    cfg.trainer.remove_microbatch_padding = False
    cfg.trainer.algorithm.policy_loss_type = "rollout_is"
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.use_entropy_loss = False
    cfg.trainer.algorithm.enable_sample_support_replay = True
    cfg.trainer.algorithm.enable_score_centering = True

    worker = PolicyWorkerBase(
        cfg=cfg.trainer,
        world_size=1,
        rank=0,
        local_rank=0,
        master_addr="localhost",
        master_port=12345,
        sequence_parallel_size=1,
    )
    worker.strategy = MagicMock()
    worker.device_mesh = MagicMock()
    worker.mesh_rank = SimpleNamespace(dp_size=1)
    worker.optimizer = None
    worker.scheduler = MagicMock(get_last_lr=MagicMock(return_value=[0.0]))
    canonical = torch.zeros((2, 3))
    scores = SampleSupportScores(
        logprobs=canonical,
        entropy=None,
        valid_mask=torch.tensor([[True, True, False], [True, True, False]]),
        centering_term=torch.tensor([[0.5, 1.0, 0.0], [-1.0, 2.0, 0.0]]),
        sampler_logprobs=torch.full((2, 3), -0.5),
        support_tv=torch.full((2, 3), 0.25),
        sampler_mass=torch.ones((2, 3)),
    )
    worker.model = MagicMock(
        return_value=(
            torch.zeros((2, 2), requires_grad=True),
            {
                "entropy": canonical,
                SCORE_CENTERING_SCORES_KEY: scores,
            },
        )
    )
    experience = Experience(
        sequences=torch.zeros((2, 3), dtype=torch.long),
        action_log_probs=None,
        base_action_log_probs=None,
        values=None,
        returns=None,
        advantages=torch.ones((2, 2)),
        attention_mask=torch.ones((2, 3), dtype=torch.long),
        loss_mask=torch.ones((2, 2)),
        response_mask=torch.ones((2, 2)),
        rollout_logprobs=torch.full((2, 2), -0.5),
        rollout_expert_indices=None,
        num_actions=2,
        info=None,
        rollout_sample_support=ragged_support(_support([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])),
    )

    with (
        patch("torch.cuda.current_device", return_value=torch.device("cpu")),
        patch("torch.autocast", side_effect=lambda *args, **kwargs: nullcontext()),
    ):
        status = worker._forward_backward_micro(
            experience,
            microbatch_weight=1.0,
            loss_fn_config={"eps_clip_low": 0.1, "eps_clip_high": 0.3},
        )

    assert worker.model.call_args.kwargs["centering_weight"] == RatioBand(low=0.9, high=1.3)
    # log π = 0 and log q̃ = -0.5 put every ratio at e^0.5 ≈ 1.65, outside the band, so only A·T remains.
    assert status["policy_loss"] == pytest.approx(0.5 + 1.0 - 1.0 + 2.0)
    assert status[ScoreCenteringMetric.TERM_ABS_MEAN.value] == pytest.approx(1.125)


def _weighted(support: PackedTensor, weights: list[list[list[float]]]) -> PackedRaggedTensor:
    """CSR support with a member-value column, one weight grid per trajectory matching ``support``'s grid."""
    rows = []
    for index in range(len(support)):
        grid = support.segment(index).numpy()
        rows.append(
            CompactSampleSupport.from_masked_rows(
                grid,
                grid >= 0,
                np.zeros(grid.shape, dtype=np.float32),
                member_values=np.asarray(weights[index], dtype=np.float32),
            )
        )
    return PackedRaggedTensor.from_rows(rows)


def _logprob_term_forward(wrapper, sequences, attention_mask, weighted, response_lengths, num_actions):
    """``sum_v w_v log pi(v)`` at the action positions, combined as the worker combines it."""
    action_log_probs, output = wrapper(
        sequences,
        num_actions,
        attention_mask=attention_mask,
        loss_mask=_loss_mask(response_lengths, num_actions),
        weighted_support=weighted,
        return_output=True,
    )
    return output[SUPPORT_LOGPROB_TERM_KEY].value(slice(-num_actions - 1, -1), action_log_probs)


def test_the_logprob_term_lands_on_the_logit_that_predicts_each_token_packed_or_not():
    sequences, attention_mask = _ragged_batch()
    support = _ragged_support(sequences)
    weights = [[[0.3, 0.5], [-0.2, 0.25]], [[0.1, 1.0]]]
    model = _TokenIndexedLM()
    packed_model = _TokenIndexedLM()
    packed_model.table.data.copy_(model.table.data)

    unpacked = _logprob_term_forward(_wrapper(model), sequences, attention_mask, _weighted(support, weights), [2, 1], 2)
    packed = _logprob_term_forward(
        _wrapper(packed_model, packed=True), sequences, attention_mask, _weighted(support, weights), [2, 1], 2
    )
    unpacked.sum().backward()
    packed.sum().backward()

    table = model.table.detach()
    expected = torch.zeros((2, 2), dtype=table.dtype)
    for row, (_prompt, response) in enumerate(RAGGED):
        for offset in range(response):
            position = sequences.shape[1] - response + offset - 1
            members = support.segment(row)[offset].long()
            values = torch.tensor(weights[row][offset], dtype=table.dtype)
            log_probs = torch.log_softmax(table[sequences[row, position]], dim=-1)
            expected[row, 2 - response + offset] = (values * log_probs[members]).sum()
    torch.testing.assert_close(unpacked, expected)
    torch.testing.assert_close(packed, unpacked)
    torch.testing.assert_close(packed_model.table.grad, model.table.grad)


def test_the_logprob_term_needs_the_loss_mask():
    sequences, attention_mask = _ragged_batch()
    weighted = _weighted(_ragged_support(sequences), [[[-0.5, 0.5], [0.25, -0.25]], [[-1.0, 1.0]]])
    with pytest.raises(ValueError, match="needs the microbatch's loss mask"):
        _wrapper(_TokenIndexedLM())(sequences, 2, attention_mask=attention_mask, weighted_support=weighted)


def test_the_worker_adds_the_logprob_term_without_the_importance_ratio():
    cfg = SkyRLTrainConfig()
    cfg.trainer.micro_train_batch_size_per_gpu = 2
    cfg.trainer.remove_microbatch_padding = False
    cfg.trainer.algorithm.policy_loss_type = "importance_sampling"
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.use_entropy_loss = False

    worker = PolicyWorkerBase(
        cfg=cfg.trainer,
        world_size=1,
        rank=0,
        local_rank=0,
        master_addr="localhost",
        master_port=12345,
        sequence_parallel_size=1,
    )
    worker.strategy = MagicMock()
    worker.device_mesh = MagicMock()
    worker.mesh_rank = SimpleNamespace(dp_size=1)
    worker.optimizer = None
    worker.scheduler = MagicMock(get_last_lr=MagicMock(return_value=[0.0]))
    # Canonical logit positions: the two action logits are columns 0 and 1, the last column predicts nothing.
    term = SupportLogprobTerm(
        anchored=torch.tensor([[0.5, 1.0, 9.0], [-1.0, 2.0, 9.0]], requires_grad=True),
        weight_sum=torch.tensor([[0.2, -0.4, 9.0], [0.3, 1.0, 9.0]]),
    )
    # The policy has moved since the rollout, whose logprobs were 0, so every importance ratio differs from 1.
    current = torch.tensor([[-0.5, -1.0], [-2.0, -0.2]], requires_grad=True)
    worker.model = MagicMock(return_value=(current, {"entropy": torch.zeros((2, 3)), SUPPORT_LOGPROB_TERM_KEY: term}))
    weighted = _weighted(_support([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), [[[-1.0, 1.0]] * 2, [[-1.0, 1.0]] * 2])
    experience = Experience(
        sequences=torch.zeros((2, 3), dtype=torch.long),
        action_log_probs=torch.zeros((2, 2)),
        base_action_log_probs=None,
        values=None,
        returns=None,
        advantages=torch.ones((2, 2)),
        attention_mask=torch.ones((2, 3), dtype=torch.long),
        loss_mask=torch.tensor([[1.0, 1.0], [1.0, 0.0]]),
        response_mask=torch.ones((2, 2)),
        rollout_logprobs=None,
        rollout_expert_indices=None,
        num_actions=2,
        info=None,
        rollout_sample_support=weighted,
    )

    with (
        patch("torch.cuda.current_device", return_value=torch.device("cpu")),
        patch("torch.autocast", side_effect=lambda *args, **kwargs: nullcontext()),
    ):
        status = worker._forward_backward_micro(experience, microbatch_weight=1.0)

    assert worker.model.call_args.kwargs["weighted_support"] == weighted
    assert worker.model.call_args.kwargs["sample_support"] is None
    assert torch.equal(worker.model.call_args.kwargs["loss_mask"], experience.loss_mask)
    # The term is anchored + weight_sum * log pi(y) at the current logprobs, with no ratio:
    # (0.5 + 0.2 * -0.5) + (1.0 - 0.4 * -1.0) + (-1.0 + 0.3 * -2.0) = 0.2 under the mask.
    assert status["loss_metrics/topk_policy_loss"] == pytest.approx(-0.2, abs=1e-6)
    # importance_sampling: -sum(exp(log pi - log pi_old) * A * mask).
    assert status["policy_loss"] == pytest.approx(-(math.exp(-0.5) + math.exp(-1.0) + math.exp(-2.0)) - 0.2, abs=1e-6)

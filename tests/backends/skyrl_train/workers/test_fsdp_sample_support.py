"""Sample-support replay through the FSDP forward."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from skyrl.backends.skyrl_train.distributed.ulysses import utils as ulysses_utils
from skyrl.backends.skyrl_train.utils.packed_tensor import (
    PackedTensor,
    cu_seqlens_from_lengths,
)
from skyrl.backends.skyrl_train.utils.sample_support import (
    SAMPLE_SUPPORT_ENTROPY_MASK_KEY,
    SAMPLE_SUPPORT_FIELD,
    SAMPLE_SUPPORT_NO_ROW,
    SAMPLE_SUPPORT_PADDING,
    SAMPLE_SUPPORT_TORCH_DTYPE,
)
from skyrl.backends.skyrl_train.workers.model_wrapper import (
    _NO_TRAJECTORY,
    HFModelWrapper,
    _SampleSupportChannels,
)
from skyrl.backends.skyrl_train.workers.worker import PolicyWorkerBase
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.dataset.replay_buffer import Experience

VOCAB = 12


class _TokenIndexedLM(nn.Module):
    """A position-independent model for comparing packed and unpacked layouts."""

    def __init__(self, vocab_size: int = VOCAB):
        super().__init__()
        self.table = nn.Parameter(torch.randn(vocab_size, vocab_size, dtype=torch.float64))

    def forward(self, input_ids, **kwargs):
        return {"logits": self.table[input_ids]}


def _support(segments: list[list[list[int]]]) -> PackedTensor:
    """Pack one ``[response_tokens, top_k]`` block per trajectory."""
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


def _reference_entropy(table, sequences, support: PackedTensor, num_actions: int) -> torch.Tensor:
    """Entropy over each response token's support at the logit that predicts it."""
    sequence_length = sequences.shape[1]
    expected = torch.zeros((sequences.shape[0], num_actions), dtype=table.dtype)
    for row in range(sequences.shape[0]):
        segment = support.segment(row)
        for offset in range(segment.shape[0]):
            members = segment[offset]
            members = members[members >= 0].long()
            if members.numel() == 0:
                continue
            position = sequence_length - segment.shape[0] + offset - 1
            member_logprobs = torch.log_softmax(table[sequences[row, position]][members], dim=0)
            expected[row, num_actions - segment.shape[0] + offset] = -(member_logprobs.exp() * member_logprobs).sum()
    return expected


def _reference(table, sequences, support: PackedTensor, num_actions: int) -> torch.Tensor:
    """Compute dense support-conditioned log probabilities."""
    sequence_length = sequences.shape[1]
    expected = torch.zeros((sequences.shape[0], num_actions), dtype=table.dtype)
    for row in range(sequences.shape[0]):
        segment = support.segment(row)
        for offset in range(segment.shape[0]):
            position = sequence_length - segment.shape[0] + offset - 1
            members = segment[offset]
            members = members[members >= 0].long()
            logits = table[sequences[row, position]]
            denominator = logits if members.numel() == 0 else logits[members]
            expected[row, num_actions - segment.shape[0] + offset] = logits[
                sequences[row, position + 1]
            ] - torch.logsumexp(denominator, dim=0)
    return expected


def _wrapper(model: nn.Module, *, packed: bool = False) -> HFModelWrapper:
    return HFModelWrapper(
        model,
        bf16=False,
        use_flash_attention_2=packed,
        remove_microbatch_padding=packed,
    )


def _forward(wrapper, sequences, attention_mask, support, response_lengths, num_actions, **kwargs):
    return wrapper(
        sequences,
        num_actions,
        attention_mask=attention_mask,
        sample_support=support,
        loss_mask=_loss_mask(response_lengths, num_actions),
        enable_sample_support_replay=True,
        **kwargs,
    )


def _forward_entropy(wrapper, sequences, attention_mask, support, response_lengths, num_actions, **kwargs):
    """Return entropy and its mask over response slots."""
    _, output = _forward(
        wrapper,
        sequences,
        attention_mask,
        support,
        response_lengths,
        num_actions,
        return_output=True,
        compute_entropy=True,
        **kwargs,
    )
    response = slice(-num_actions - 1, -1)
    return output["entropy"][:, response], output[SAMPLE_SUPPORT_ENTROPY_MASK_KEY][:, response]


# (prompt length, response length) for a ragged pair of trajectories.
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


def _ragged_support(sequences, *, unsupported_rows: tuple[int, ...] = ()) -> PackedTensor:
    """Build support rows containing the sampled token and a decoy."""
    segments = []
    for row, (_prompt, response) in enumerate(RAGGED):
        segment = []
        for offset in range(response):
            token = int(sequences[row, sequences.shape[1] - response + offset])
            segment.append([token, (token + 5) % VOCAB])
        segments.append(segment)
    support = _support(segments)
    for row_index in unsupported_rows:
        support.values[row_index] = SAMPLE_SUPPORT_PADDING
    return support


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
def test_the_appended_eos_falls_back_to_the_full_vocabulary(packed):
    """A loss-bearing token with an all-padding support row must not score 0.0."""
    sequences, attention_mask = _ragged_batch()
    # Row 0's last response token and row 1's only one: one fallback per trajectory.
    support = _ragged_support(sequences, unsupported_rows=(1, 2))
    model = _TokenIndexedLM()

    actual = _forward(_wrapper(model, packed=packed), sequences, attention_mask, support, [2, 1], 2)

    expected = _reference(model.table.detach(), sequences, support, 2)
    torch.testing.assert_close(actual, expected)
    assert (actual != 0).sum() == 3


@pytest.mark.parametrize("packed", [False, True])
def test_entropy_comes_from_the_recorded_support_not_the_vocabulary(monkeypatch, packed):
    """Use recorded support without running full-vocabulary entropy."""
    sequences, attention_mask = _ragged_batch()
    support = _ragged_support(sequences)
    model = _TokenIndexedLM()
    wrapper = _wrapper(model, packed=packed)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("full-vocabulary entropy should not run during support replay")

    monkeypatch.setattr(wrapper, "chunked_entropy_from_logits_fn", fail_if_called)
    entropy, mask = _forward_entropy(wrapper, sequences, attention_mask, support, [2, 1], 2)

    torch.testing.assert_close(entropy, _reference_entropy(model.table.detach(), sequences, support, 2))
    # Row 1 generated one token, so its first response slot has no support row and no entropy.
    assert mask.tolist() == [[True, True], [False, True]]


def test_the_appended_eos_position_is_outside_the_entropy_mask():
    """It is scored over the whole vocabulary, so averaging its 0.0 in would understate entropy."""
    sequences, attention_mask = _ragged_batch()
    support = _ragged_support(sequences, unsupported_rows=(1,))
    model = _TokenIndexedLM()

    entropy, mask = _forward_entropy(_wrapper(model), sequences, attention_mask, support, [2, 1], 2)

    torch.testing.assert_close(entropy, _reference_entropy(model.table.detach(), sequences, support, 2))
    assert mask.tolist() == [[True, False], [False, True]]
    assert entropy[0, 1] == 0


@pytest.mark.parametrize("entropy_requires_grad", [False, True])
def test_entropy_carries_gradients_only_when_asked(entropy_requires_grad):
    """A metric must stay out of the backward graph; a loss term must reach the parameters."""
    sequences = torch.tensor([[1, 2, 3, 4]])
    support = _support([[[3, 8], [4, 0]]])
    model = _TokenIndexedLM()

    entropy, _ = _forward_entropy(
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


def test_the_policy_entropy_metric_excludes_the_unsupported_row():
    """``PolicyWorkerBase`` must intersect the entropy mask with the loss mask, not ignore it."""
    entropy = torch.tensor([[0.5, 1.5], [2.5, 3.5]])
    entropy_mask = torch.tensor([[True, False], [True, True]])
    loss_mask = torch.ones((2, 2))

    cfg = SkyRLTrainConfig()
    cfg.trainer.micro_train_batch_size_per_gpu = 2
    cfg.trainer.remove_microbatch_padding = False
    cfg.trainer.algorithm.policy_loss_type = "dual_clip"
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.use_entropy_loss = False
    cfg.trainer.algorithm.enable_sample_support_replay = True

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
    # Match the (B, S) shape sliced by the worker.
    trailing = torch.zeros((2, 1))
    worker.model = MagicMock(
        return_value=(
            torch.zeros((2, 2), requires_grad=True),
            {
                "entropy": torch.cat([entropy, trailing], dim=1),
                SAMPLE_SUPPORT_ENTROPY_MASK_KEY: torch.cat([entropy_mask, trailing.bool()], dim=1),
            },
        )
    )

    experience = Experience(
        sequences=torch.zeros((2, 3), dtype=torch.long),
        action_log_probs=torch.zeros((2, 2)),
        base_action_log_probs=None,
        values=None,
        returns=None,
        advantages=torch.zeros((2, 2)),
        attention_mask=torch.ones((2, 3), dtype=torch.long),
        loss_mask=loss_mask,
        response_mask=torch.ones((2, 2)),
        rollout_logprobs=None,
        rollout_expert_indices=None,
        num_actions=2,
        info=None,
        rollout_sample_support=_support([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
    )

    with (
        patch("torch.cuda.current_device", return_value=torch.device("cpu")),
        patch("torch.autocast", side_effect=lambda *args, **kwargs: nullcontext()),
    ):
        status = worker._forward_backward_micro(experience, microbatch_weight=1.0)

    expected = float((entropy * entropy_mask).sum() / entropy_mask.sum())
    assert status["policy_entropy"] == pytest.approx(expected)


def test_replay_never_scores_every_position_over_the_full_vocabulary(monkeypatch):
    """The fallback softmaxes one row per trajectory; the ordinary ``B * S`` path must stay off."""

    def fail_if_called(*args, **kwargs):
        raise AssertionError("full-sequence vocabulary logprobs should not run during support replay")

    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.workers.model_wrapper.logprobs_from_logits",
        fail_if_called,
    )
    sequences, attention_mask = _ragged_batch()
    support = _ragged_support(sequences, unsupported_rows=(2,))

    actual = _forward(_wrapper(_TokenIndexedLM()), sequences, attention_mask, support, [2, 1], 2)

    assert torch.isfinite(actual).all()


def test_a_second_unsupported_token_in_one_trajectory_is_rejected():
    sequences, attention_mask = _ragged_batch()
    support = _ragged_support(sequences, unsupported_rows=(0, 1))

    with pytest.raises(ValueError, match="at most one loss-bearing token"):
        _forward(_wrapper(_TokenIndexedLM()), sequences, attention_mask, support, [2, 1], 2)


def test_replay_disabled_takes_the_ordinary_full_vocabulary_path():
    sequences, attention_mask = _ragged_batch()
    model = _TokenIndexedLM()
    wrapper = _wrapper(model)

    disabled = wrapper(sequences, 2, attention_mask=attention_mask)
    ignored = wrapper(
        sequences,
        2,
        attention_mask=attention_mask,
        sample_support=_ragged_support(sequences),
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
        "sample_support": _ragged_support(sequences),
        "loss_mask": _loss_mask([2, 1], 2),
        "enable_sample_support_replay": True,
    }
    kwargs[omitted] = None

    with pytest.raises(ValueError, match=message):
        _wrapper(_TokenIndexedLM())(sequences, 2, attention_mask=attention_mask, **kwargs)


def test_sequence_parallel_slice_pads_each_channel_with_its_own_sentinel(monkeypatch):
    """The Ulysses pad is invisible to the tokens but not to a channel whose 0 means something."""
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
        trajectory_ids=torch.tensor([[0, 0, 0]]),
    )

    tail = channels.slice_for_sequence_parallel(sp_size)

    assert tail.row_ids.tolist() == [[2, SAMPLE_SUPPORT_NO_ROW]]
    assert tail.loss_mask.tolist() == [[True, False]]
    assert tail.trajectory_ids.tolist() == [[0, _NO_TRAJECTORY]]

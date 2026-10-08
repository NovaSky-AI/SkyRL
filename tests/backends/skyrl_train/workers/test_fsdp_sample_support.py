"""Sample-support replay through the FSDP forward."""

import pytest
import torch
from torch import nn

from skyrl.backends.skyrl_train.distributed.ulysses import utils as ulysses_utils
from skyrl.backends.skyrl_train.utils.packed_tensor import (
    PackedTensor,
    cu_seqlens_from_lengths,
)
from skyrl.backends.skyrl_train.utils.sample_support import (
    SAMPLE_SUPPORT_FIELD,
    SAMPLE_SUPPORT_NO_ROW,
    SAMPLE_SUPPORT_PADDING,
    SAMPLE_SUPPORT_TORCH_DTYPE,
)
from skyrl.backends.skyrl_train.workers.model_wrapper import (
    HFModelWrapper,
    _SampleSupportChannels,
)

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
    return wrapper(
        sequences,
        num_actions,
        attention_mask=attention_mask,
        sample_support=support,
        loss_mask=_loss_mask(response_lengths, num_actions),
        enable_sample_support_replay=True,
    )


def _forward_entropy(wrapper, sequences, attention_mask, support, response_lengths, num_actions, **kwargs):
    _, output = wrapper(
        sequences,
        num_actions,
        attention_mask=attention_mask,
        sample_support=support,
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
    support.values[1] = SAMPLE_SUPPORT_PADDING

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

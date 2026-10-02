"""Score-centering head scoring through the FSDP forward.

Run with:
    uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/workers/test_fsdp_score_centering_head.py
"""

import pytest
import torch
from torch import nn

from skyrl.backends.skyrl_train.utils.packed_tensor import (
    PackedTensor,
    cu_seqlens_from_lengths,
)
from skyrl.backends.skyrl_train.utils.sample_support import (
    SAMPLE_SUPPORT_PADDING,
    SAMPLE_SUPPORT_TORCH_DTYPE,
)
from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper

VOCAB = 12


class _TokenIndexedLM(nn.Module):
    """A position-independent model: the logits at a position depend only on the token there."""

    def __init__(self, vocab_size: int = VOCAB):
        super().__init__()
        self.table = nn.Parameter(torch.randn(vocab_size, vocab_size, dtype=torch.float64))

    def forward(self, input_ids, **kwargs):
        return {"logits": self.table[input_ids]}


# (prompt_len, response_len) per trajectory; row 1 is shorter so it carries left padding.
RAGGED = [(2, 3), (1, 2)]
K = 3


def _ragged_batch():
    totals = [prompt + response for prompt, response in RAGGED]
    sequence_length = max(totals)
    sequences = torch.zeros((len(totals), sequence_length), dtype=torch.long)
    attention_mask = torch.zeros((len(totals), sequence_length), dtype=torch.long)
    next_id = 1
    for row, total in enumerate(totals):
        attention_mask[row, sequence_length - total :] = 1
        for column in range(sequence_length - total, sequence_length):
            sequences[row, column] = next_id
            next_id += 1
    return sequences, attention_mask


def _ragged_head(sequences) -> PackedTensor:
    """Member 0 is the sampled token; the other members are decoys, with one short row per trajectory."""
    rows = []
    for row, (_prompt, response) in enumerate(RAGGED):
        for offset in range(response):
            token = int(sequences[row, sequences.shape[1] - response + offset])
            members = [token, (token + 5) % VOCAB, (token + 7) % VOCAB]
            if offset == 0:
                members[-1] = SAMPLE_SUPPORT_PADDING
            rows.append(members)
    return PackedTensor(
        torch.tensor(rows, dtype=SAMPLE_SUPPORT_TORCH_DTYPE),
        cu_seqlens_from_lengths([response for _, response in RAGGED]),
    )


def _loss_mask(num_actions: int) -> torch.Tensor:
    mask = torch.zeros((len(RAGGED), num_actions), dtype=torch.bool)
    for row, (_prompt, response) in enumerate(RAGGED):
        mask[row, num_actions - response :] = True
    return mask


def _reference(table, sequences, head: PackedTensor, num_actions: int, renormalize: bool) -> torch.Tensor:
    """Score each response token's head members under the logits that predict that token."""
    sequence_length = sequences.shape[1]
    expected = torch.full((sequences.shape[0], num_actions, K), float("-inf"), dtype=table.dtype)
    for row in range(sequences.shape[0]):
        segment = head.segment(row)
        for offset in range(segment.shape[0]):
            position = sequence_length - segment.shape[0] + offset - 1
            logits = table[sequences[row, position]]
            members = segment[offset].long()
            valid = members >= 0
            if renormalize:
                log_z = torch.logsumexp(logits[members[valid]], dim=0)
            else:
                log_z = torch.logsumexp(logits, dim=0)
            expected[row, num_actions - segment.shape[0] + offset, valid] = logits[members[valid]] - log_z
    return expected


def _forward(wrapper, sequences, attention_mask, head, num_actions, *, replay=False):
    _, output = wrapper(
        sequences,
        num_actions,
        attention_mask=attention_mask,
        sample_support=head,
        loss_mask=_loss_mask(num_actions),
        enable_sample_support_replay=replay,
        score_centering_head=True,
        return_output=True,
    )
    return output["score_centering_head_logprobs"]


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("replay", [False, True])
def test_head_logprobs_match_reference_and_backprop(packed, replay):
    sequences, attention_mask = _ragged_batch()
    head = _ragged_head(sequences)
    model = _TokenIndexedLM()
    wrapper = HFModelWrapper(model, bf16=False, use_flash_attention_2=packed, remove_microbatch_padding=packed)
    num_actions = max(response for _, response in RAGGED)

    actual = _forward(wrapper, sequences, attention_mask, head, num_actions, replay=replay)
    assert actual.shape == (len(RAGGED), num_actions, K)
    valid = torch.isfinite(actual)
    actual[valid].sum().backward()

    reference_table = model.table.detach().clone().requires_grad_(True)
    expected = _reference(reference_table, sequences, head, num_actions, renormalize=replay)
    expected[torch.isfinite(expected)].sum().backward()

    # The head is scored in float32 regardless of the model dtype.
    torch.testing.assert_close(actual, expected.to(actual.dtype), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(model.table.grad, reference_table.grad, atol=1e-5, rtol=1e-5)
    # Row 1 generated two tokens, so its first response slot has no head.
    assert torch.isneginf(actual[1, 0]).all()


def test_head_scoring_is_off_by_default():
    sequences, attention_mask = _ragged_batch()
    wrapper = HFModelWrapper(_TokenIndexedLM(), bf16=False)
    _, output = wrapper(sequences, 3, attention_mask=attention_mask, return_output=True)
    assert "score_centering_head_logprobs" not in output

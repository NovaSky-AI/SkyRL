"""The vendored ``pad_input`` / ``unpad_input`` round-trip and match the flash-attn contract."""

import torch

from skyrl.backends.skyrl_train.utils.bert_padding import pad_input, unpad_input


def test_unpad_then_pad_round_trip():
    torch.manual_seed(0)
    batch, seqlen, hidden = 3, 7, 5
    hidden_states = torch.randn(batch, seqlen, hidden)
    mask = torch.tensor(
        [[1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 0]],
        dtype=torch.int32,
    )
    out = unpad_input(hidden_states, mask)
    unpadded, indices, cu_seqlens, max_seqlen = out[0], out[1], out[2], out[3]

    assert unpadded.shape == (int(mask.sum()), hidden)
    assert cu_seqlens.tolist() == [0, 3, 10, 14]
    assert int(max_seqlen) == 7
    assert torch.equal(unpadded, hidden_states[mask.bool()])

    repadded = pad_input(unpadded, indices, batch, seqlen)
    assert torch.equal(repadded[mask.bool()], hidden_states[mask.bool()])
    assert torch.count_nonzero(repadded[~mask.bool()]) == 0

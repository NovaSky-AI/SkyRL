"""ML4 through ``HFModelWrapper.forward`` on SkyRL's padded batch layout.

ML4 runs with ``remove_microbatch_padding=false`` (recipe). SkyRL then passes
``position_ids = cumsum(mask) - 1`` (pads = 1) and the attention mask; ML4 derives
sequence boundaries from position_ids and masks padded keys (attention and indexer
top-k). Each row's action logprobs must equal an unpadded single-sequence forward.
"""

import torch


def test_padded_batch_matches_unpadded(ml4_model_dir):
    from conftest import load_ml4_wrapper
    from fake_ckpt import TINY_PARAMS

    wrapper, _ = load_ml4_wrapper(ml4_model_dir)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():  # non-zero adapters so LoRA is part of the forward
        for n, p in wrapper.model.named_parameters():
            if ".lora_B." in n:
                p.copy_(0.05 * torch.randn(p.shape, generator=g))
    wrapper.eval()

    P, R = 12, 10  # prompt slot (left-padded), response slot (right-padded)
    rows = [(12, 10), (5, 10), (9, 3), (1, 7)]  # (prompt len, response len)
    vocab = TINY_PARAMS["vocab_size"]
    seqs = torch.zeros(len(rows), P + R, dtype=torch.long)
    mask = torch.zeros_like(seqs)
    real = []
    for i, (p, r) in enumerate(rows):
        toks = torch.randint(1, vocab, (p + r,), generator=g)
        seqs[i, P - p : P + r] = toks
        mask[i, P - p : P + r] = 1
        real.append(toks)

    with torch.no_grad():
        got = wrapper(seqs, num_actions=R, attention_mask=mask)
        for i, (p, r) in enumerate(rows):
            logits = wrapper.model(real[i][None]).logits[0].float()
            ref = torch.log_softmax(logits, -1)[:-1].gather(-1, real[i][1:, None])[:, 0]
            # action slot k scores the token at P + k given everything before it
            torch.testing.assert_close(got[i, :r], ref[p - 1 : p - 1 + r], atol=1e-5, rtol=1e-5)

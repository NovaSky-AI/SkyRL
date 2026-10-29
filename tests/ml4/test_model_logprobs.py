"""``HFModelWrapper.forward`` with a model that computes chunked token logprobs
(ML4 ``config.logprob_chunk_size > 0``) equals the full-logits path: action logprobs,
masked entropy and LoRA gradients, on SkyRL's padded batch layout."""

import pytest
import torch


def _wrapper(model_dir, chunk):
    from conftest import LORA_ALPHA, LORA_RANK, SEED
    from ml4_train.loading import load_sharded
    from ml4_train.lora import LORA_TARGET_REGEX

    from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper

    torch.manual_seed(SEED)
    kw = {"attn_implementation": "eager"}
    if chunk:
        kw["logprob_chunk_size"] = chunk
    w = HFModelWrapper(
        model_dir,
        bf16=False,
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_REGEX,
        model_config_kwargs=kw,
        meta_init=True,
        meta_init_adapters=True,
    )
    load_sharded(w.model, model_dir)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for n, p in w.model.named_parameters():
            if "lora_" in n:
                p.copy_(0.05 * torch.randn(p.shape, generator=g))
    return w


@pytest.mark.parametrize("chunk,temperature", [(7, 1.0), (5, 0.6)], ids=["chunk7", "chunk5-temp"])
def test_chunked_model_logprobs_match_full_logits(ml4_model_dir, chunk, temperature):
    from fake_ckpt import TINY_PARAMS

    g = torch.Generator().manual_seed(1)
    rows, P, R = [(6, 5), (3, 5), (6, 2)], 6, 5
    seqs = torch.zeros(len(rows), P + R, dtype=torch.long)
    mask = torch.zeros_like(seqs)
    for i, (p, r) in enumerate(rows):
        seqs[i, P - p : P + r] = torch.randint(1, TINY_PARAMS["vocab_size"], (p + r,), generator=g)
        mask[i, P - p : P + r] = 1
    res = []
    for c in (0, chunk):
        w = _wrapper(ml4_model_dir, c)
        w.train()
        lp, out = w(
            seqs,
            num_actions=R,
            attention_mask=mask,
            temperature=temperature,
            return_output=True,
            compute_entropy=True,
            entropy_requires_grad=False,
        )
        assert (out.get("logits") is None) == bool(c)
        (lp * torch.linspace(-1, 1, lp.numel()).view_as(lp)).sum().backward()
        grads = {n: p.grad.clone() for n, p in w.model.named_parameters() if p.grad is not None}
        res.append((lp.detach(), out["entropy"].detach(), grads))
    (lp0, e0, g0), (lp1, e1, g1) = res
    torch.testing.assert_close(lp1, lp0, rtol=0, atol=1e-5)
    torch.testing.assert_close(e1, e0, rtol=0, atol=1e-5)
    assert (e1[mask == 0] == 0).all()
    assert g0.keys() == g1.keys() and g0
    worst = max(((g1[n] - g0[n]).abs().max() / g0[n].abs().max().clamp_min(1e-12)).item() for n in g0)
    assert worst < 1e-4, worst

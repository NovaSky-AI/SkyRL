"""GLM-5.3-Flash DSA k-pool indexer checks against the HF reference semantics.

megatron-core gained the pooled indexer in NVIDIA/Megatron-LM#7054, which is what lets the
Megatron path train on sequences longer than ``dsa_indexer_topk`` (2048 for this checkpoint).
Before that, ``glm5_next/dsa.py`` refused them rather than silently attending to a different
subset than the real model.

Two properties are checked:

- **Pooling math.** ``_kpool_compress_keys`` must match HF's
  ``Glm5NextTextIndexer.compress_keys``: a per-dimension softmax over the pool's slots of
  ``gate_score + ape``, used to weight the keys. Transcribed here independently from
  ``modeling_glm5_next`` rather than imported, so a change on either side shows up.
- **Selection behaviour.** Below ``index_topk`` the pooled selection
  covers every causally visible token (every pool is selectable), so it must reduce exactly to
  dense causal attention -- the regime the old ceiling allowed, which the working GSM8K/DAPO
  runs already relied on. Above it, selection genuinely drops tokens, and what must still hold
  is that it stays causal, respects the budget, and always keeps the query's own trailing pool.

Run with:
uv run --isolated --extra dev --extra megatron -- pytest -s \
    tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_glm5_next_kpool.py
"""

import pytest
import torch

pytestmark = pytest.mark.megatron

POOL_SIZE = 4
HEAD_DIM = 128
INDEX_TOPK = 64  # small stand-in for the checkpoint's 2048; the invariant is topk/pool_size pools
N_HEADS = 4


def _hf_reference_pool(k: torch.Tensor, gate_score: torch.Tensor, ape: torch.Tensor, pool_size: int):
    """Independent transcription of HF ``Glm5NextTextIndexer`` pooling.

    HF works in [batch, seq, dim] and does::

        logits        = grouped_gate_scores.float() + ape.float()[None, None]
        probabilities = logits.softmax(dim=2)          # over the pool's slots
        pool_keys     = (probabilities * grouped_keys).sum(dim=2)

    megatron-core works in [tokens, batch, dim] and only compresses complete pools.
    """
    tokens, batch, dim = k.shape
    n_pools = tokens // pool_size
    trimmed = n_pools * pool_size
    grouped_k = k[:trimmed].reshape(n_pools, pool_size, batch, dim)
    grouped_gate = gate_score[:trimmed].reshape(n_pools, pool_size, batch, dim)

    logits = grouped_gate.float() + ape.float()[None, :, None, :]
    probs = logits.softmax(dim=1)
    return (probs * grouped_k.float()).sum(dim=1)


def _make_inputs(seqlen: int, batch: int = 1, device="cuda", dtype=torch.bfloat16, seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)
    k = torch.randn(seqlen, batch, HEAD_DIM, device=device, dtype=dtype, generator=gen)
    gate = torch.randn(seqlen, batch, HEAD_DIM, device=device, dtype=dtype, generator=gen)
    ape = torch.randn(POOL_SIZE, HEAD_DIM, device=device, dtype=torch.float32, generator=gen)
    q = torch.randn(seqlen, batch, N_HEADS, HEAD_DIM, device=device, dtype=dtype, generator=gen)
    weights = torch.randn(seqlen, batch, N_HEADS, device=device, dtype=torch.float32, generator=gen)
    return q, k, weights, gate, ape


@pytest.mark.parametrize("seqlen", [64, 256, 1024])
def test_kpool_compress_keys_matches_hf_reference(seqlen):
    """The softmax-weighted pooling itself, against HF's formulation."""
    from megatron.core.transformer.experimental_attention_variant.dsa import (
        _kpool_compress_keys,
    )

    _, k, _, gate, ape = _make_inputs(seqlen)

    got = _kpool_compress_keys(k, gate, ape, POOL_SIZE)
    want = _hf_reference_pool(k, gate, ape, POOL_SIZE)

    assert got.shape[0] == seqlen // POOL_SIZE, f"expected {seqlen // POOL_SIZE} pools, got {got.shape[0]}"
    assert got.shape[1:] == k.shape[1:]
    # megatron accumulates in fp32 and returns bf16; compare at bf16 resolution.
    torch.testing.assert_close(got.float(), want.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("seqlen", [32, 64, 250])
def test_kpool_selects_every_visible_token_below_topk(seqlen):
    """At or below ``index_topk`` the pooled path must cover the full causal prefix.

    Every pool is selectable in that regime, so sparse selection degenerates to dense causal
    attention -- the property SkyRL's old ``dsa.py`` guard relied on when it reused megatron's
    token-level indexer for short sequences. Checked at the pool size GLM-5.3-Flash actually
    ships (``index_kpool=4``, ``index_head_dim=128``), so a change to either constant in the
    checkpoint surfaces here rather than in a training run.
    """
    from megatron.core.transformer.experimental_attention_variant.dsa import (
        fused_qk_topk_kpool,
    )
    from megatron.core.transformer.experimental_attention_variant.dsa_masking import (
        generate_varlen_mask_params_for_positions,
    )

    device = "cuda"
    cu = torch.tensor([0, seqlen], device=device)
    positions = torch.arange(seqlen, device=device)
    starts, ends = generate_varlen_mask_params_for_positions(cu, positions)

    q, k, weights, gate, ape = _make_inputs(seqlen, device=device)

    _, indices = fused_qk_topk_kpool(
        q,
        k,
        weights,
        index_topk=INDEX_TOPK,
        pool_size=POOL_SIZE,
        gate_score=gate,
        ape=ape,
        varlen_starts=starts,
        varlen_ends=ends,
        cu_seqlens_kv=cu,
        always_select_tail=True,
    )

    for query, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
        selected = indices[0][query]
        got = selected[selected >= 0].sort().values
        prefix_len = end - start

        if prefix_len <= INDEX_TOPK:
            # Under the budget every pool is selectable, so this must be exactly dense causal
            # attention -- the regime SkyRL's old guard relied on.
            want = torch.arange(start, end, device=device, dtype=got.dtype)
            assert got.numel() == want.numel(), (
                f"query {query}: selected {got.numel()} tokens, expected the full causal " f"prefix of {want.numel()}"
            )
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        else:
            # Past the budget selection actually drops tokens. This is the regime the old
            # ceiling refused, so pin the guarantees that still have to hold: stay causal,
            # respect the budget, and always keep the query's own trailing pool.
            assert got.numel() <= INDEX_TOPK + POOL_SIZE - 1, (
                f"query {query}: selected {got.numel()} tokens, over the " f"{INDEX_TOPK} + {POOL_SIZE - 1} budget"
            )
            assert int(got[0]) >= start and int(got[-1]) < end, (
                f"query {query}: selected outside its own sequence [{start}, {end}): "
                f"[{int(got[0])}, {int(got[-1])}]"
            )
            # ``always_select_tail`` force-keeps the *incomplete* trailing pool, not the most
            # recent tokens unconditionally: when the prefix is an exact multiple of the pool
            # size there is no partial pool, and the final complete pool competes on score like
            # any other.
            if tail_count := prefix_len % POOL_SIZE:
                tail = set(range(end - tail_count, end))
                assert tail <= set(got.tolist()), (
                    f"query {query}: always_select_tail dropped part of the incomplete pool "
                    f"{sorted(tail)}; missing {sorted(tail - set(got.tolist()))}"
                )

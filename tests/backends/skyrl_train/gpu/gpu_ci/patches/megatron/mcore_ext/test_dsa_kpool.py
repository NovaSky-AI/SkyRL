"""Contract test for megatron-core's DSA k-pool *selection*, on GPU.

Exercises SkyRL's vendored ``fused_qk_topk_kpool`` (``mcore_ext/dsa_kpool.py``, from
NVIDIA/Megatron-LM#7522), which the pinned megatron-core does not have. GLM-5.3-Flash trains
on sequences past ``dsa_indexer_topk``, so a regression in the selection kernel would
otherwise only surface as a quality drop in a training run.

Below ``index_topk`` every pool is selectable, so the pooled selection must reduce exactly to
dense causal attention -- the regime SkyRL's ``glm5_next/dsa.py`` guard relies on when it
falls back to the token-level indexer. Above it, selection genuinely drops tokens, and what
must still hold is that it stays causal, respects the budget, and keeps the query's own
trailing pool.

The pooling-math half is pure tensor math and runs on CPU, in
``tests/backends/skyrl_train/patches/megatron/mcore_ext/test_dsa_kpool_math.py``.

Run with:
uv run --isolated --extra dev --extra megatron pytest -s \
    tests/backends/skyrl_train/gpu/gpu_ci/patches/megatron/mcore_ext/test_dsa_kpool.py
"""

import pytest
import torch

pytestmark = pytest.mark.megatron

POOL_SIZE = 4
HEAD_DIM = 128
INDEX_TOPK = 64  # small stand-in for the checkpoint's 2048; the invariant is topk/pool_size pools
N_HEADS = 4


def _make_inputs(seqlen: int, batch: int = 1, device="cuda", dtype=torch.bfloat16, seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)
    k = torch.randn(seqlen, batch, HEAD_DIM, device=device, dtype=dtype, generator=gen)
    gate = torch.randn(seqlen, batch, HEAD_DIM, device=device, dtype=dtype, generator=gen)
    ape = torch.randn(POOL_SIZE, HEAD_DIM, device=device, dtype=torch.float32, generator=gen)
    q = torch.randn(seqlen, batch, N_HEADS, HEAD_DIM, device=device, dtype=dtype, generator=gen)
    weights = torch.randn(seqlen, batch, N_HEADS, device=device, dtype=torch.float32, generator=gen)
    return q, k, weights, gate, ape


@pytest.mark.parametrize("seqlen", [32, 64, 250])
def test_kpool_selects_every_visible_token_below_topk(seqlen):
    """At or below ``index_topk`` the pooled path must cover the full causal prefix.

    Every pool is selectable in that regime, so sparse selection degenerates to dense causal
    attention -- the property SkyRL's old ``dsa.py`` guard relied on when it reused megatron's
    token-level indexer for short sequences. Checked at the pool size GLM-5.3-Flash actually
    ships (``index_kpool=4``, ``index_head_dim=128``), so a change to either constant in the
    checkpoint surfaces here rather than in a training run.
    """
    from megatron.core.transformer.experimental_attention_variant.dsa_masking import (
        generate_varlen_mask_params_for_positions,
    )

    from skyrl.backends.skyrl_train.patches.megatron.mcore_ext.dsa_kpool import (
        fused_qk_topk_kpool,
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

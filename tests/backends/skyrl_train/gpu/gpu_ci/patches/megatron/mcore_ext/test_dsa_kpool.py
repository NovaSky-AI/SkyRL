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


@pytest.mark.parametrize("seq_lens", [[1000], [300, 517, 183]])
def test_kpool_query_chunking_is_exact(seq_lens, monkeypatch):
    """Scoring in query chunks (SkyRL's deviation from #7522) must select the same pools.

    The verbatim path materializes O(sq^2) per-head FP32 scores; the chunked path keeps only a
    chunk of them. Top-k is per query row, so forcing many small chunks -- including ones that
    straddle packed-document boundaries -- must reproduce the single-chunk indices exactly.
    """
    from megatron.core.transformer.experimental_attention_variant.dsa_masking import (
        generate_varlen_mask_params_for_positions,
    )

    from skyrl.backends.skyrl_train.patches.megatron.mcore_ext import dsa_kpool

    device = "cuda"
    seqlen = sum(seq_lens)
    cu = torch.tensor([0] + seq_lens, device=device).cumsum(0)
    # Global packed-row positions, as DSAttention passes them (``torch.arange(row_start, ...)``);
    # the bounds helper finds each row's document in ``cu`` from its global position.
    positions = torch.arange(seqlen, device=device)
    starts, ends = generate_varlen_mask_params_for_positions(cu, positions)
    q, k, weights, gate, ape = _make_inputs(seqlen, device=device)

    def run():
        return dsa_kpool.fused_qk_topk_kpool(
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

    full_scores, full_indices = run()
    assert full_scores is not None, "one chunk should cover every query at this size"

    num_pools = sum(n // POOL_SIZE for n in seq_lens)
    # 37 queries per chunk: not a divisor of any segment length or of the pool size.
    monkeypatch.setattr(dsa_kpool, "_KPOOL_SCORE_CHUNK_ELEMS", 37 * N_HEADS * num_pools)
    chunked_scores, chunked_indices = run()

    assert chunked_scores is None
    assert torch.equal(chunked_indices, full_indices)


def _reference_kpool_selection(q, weights, k_pooled, pool_token_base, seq_lens, use_relu):
    """Brute-force k-pool selection from its definition, in FP64 on CPU.

    Query ``i`` of a packed document ``[s, e)`` sees the complete pools that start at or after
    ``s`` and end at or before ``i`` (pools never span documents). A pool's score is
    ``sum_h w[i, h] * f(q[i, h] . k_pool)``, ``f`` = ReLU or identity; the selection is the
    ``INDEX_TOPK / POOL_SIZE`` best visible pools, and the incomplete tail ``[i + 1 - (i + 1 - s)
    % POOL_SIZE, i + 1)`` is always kept. Returns per-query scores over all pools, the visible
    mask and the expected tail tokens.
    """
    qd, wd, kd = q[:, 0].double().cpu(), weights[:, 0].double().cpu(), k_pooled[:, 0].double().cpu()
    dots = torch.einsum("shd,pd->shp", qd, kd)
    if use_relu:
        dots = dots.relu()
    scores = (dots * wd.unsqueeze(-1)).sum(dim=1)  # [sq, num_pools]
    base = pool_token_base.cpu()
    doc_start = torch.cat([torch.full((n,), s) for s, n in zip(torch.tensor([0] + seq_lens).cumsum(0), seq_lens)])
    query = torch.arange(sum(seq_lens))
    visible = (base[None, :] >= doc_start[:, None]) & (base[None, :] + POOL_SIZE - 1 <= query[:, None])
    tails = [list(range(i + 1 - (i + 1 - int(s)) % POOL_SIZE, i + 1)) for i, s in enumerate(doc_start.tolist())]
    return scores, visible, tails


@pytest.mark.parametrize("use_relu", [False, True])
@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("seq_lens", [[1000], [300, 517, 183], [130, 2, 261]])
def test_kpool_selection_matches_reference(seq_lens, chunked, use_relu, monkeypatch):
    """Selection past ``index_topk`` matches a brute-force reference, one-shot and chunked.

    ``test_kpool_query_chunking_is_exact`` only shows the two paths agree; this pins both to the
    definition. It checks the properties that define top-k rather than an exact ordering, so FP32
    vs FP64 rounding at a near-tie can't fail it: every selected pool is complete, causal and in
    the query's document; there are ``min(budget, visible)`` of them; no unselected visible pool
    scores above a selected one (up to rounding); and the tail tokens are exact.
    """
    from megatron.core.transformer.experimental_attention_variant.dsa_masking import (
        generate_varlen_mask_params_for_positions,
    )

    from skyrl.backends.skyrl_train.patches.megatron.mcore_ext import dsa_kpool

    device = "cuda"
    seqlen = sum(seq_lens)
    cu = torch.tensor([0] + seq_lens, device=device).cumsum(0)
    # Global packed-row positions, as DSAttention passes them (``torch.arange(row_start, ...)``);
    # the bounds helper finds each row's document in ``cu`` from its global position.
    positions = torch.arange(seqlen, device=device)
    starts, ends = generate_varlen_mask_params_for_positions(cu, positions)
    q, k, weights, gate, ape = _make_inputs(seqlen, device=device, seed=len(seq_lens) + 7 * use_relu)

    k_pooled, pool_token_base = dsa_kpool._kpool_compress_keys_per_seg(k, gate, ape, POOL_SIZE, cu)
    num_pools = k_pooled.size(0)
    if chunked:
        # 37 queries per chunk: not a divisor of any segment length or of the pool size.
        monkeypatch.setattr(dsa_kpool, "_KPOOL_SCORE_CHUNK_ELEMS", 37 * N_HEADS * num_pools)
    scores_out, indices = dsa_kpool.fused_qk_topk_kpool(
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
        use_relu=use_relu,
        always_select_tail=True,
    )
    assert (scores_out is None) == chunked

    scores, visible, tails = _reference_kpool_selection(q, weights, k_pooled, pool_token_base, seq_lens, use_relu)
    pool_of_start = {int(b): p for p, b in enumerate(pool_token_base.tolist())}
    budget = INDEX_TOPK // POOL_SIZE
    indices = indices[0].cpu()
    past_budget = 0
    for i in range(seqlen):
        blocks = indices[i, :INDEX_TOPK].view(budget, POOL_SIZE)
        blocks = blocks[blocks[:, 0] >= 0]
        offsets = torch.arange(POOL_SIZE, dtype=blocks.dtype)
        assert torch.equal(
            blocks, blocks[:, :1] + offsets
        ), f"query {i}: a selected pool is not {POOL_SIZE} contiguous tokens"
        chosen = [pool_of_start[int(b)] for b in blocks[:, 0]]
        assert len(set(chosen)) == len(chosen), f"query {i}: pool selected twice"
        vis = visible[i].nonzero().flatten().tolist()
        assert set(chosen) <= set(
            vis
        ), f"query {i}: selected pools outside the visible set: {sorted(set(chosen) - set(vis))}"
        assert len(chosen) == min(budget, len(vis)), f"query {i}: {len(chosen)} pools, expected {min(budget, len(vis))}"
        rest = sorted(set(vis) - set(chosen))
        if rest:
            past_budget += 1
            worst_in, best_out = scores[i, chosen].min().item(), scores[i, rest].max().item()
            tol = 1e-4 * max(1.0, abs(worst_in))
            assert (
                worst_in >= best_out - tol
            ), f"query {i}: pool scoring {best_out:.6f} was left out while one scoring {worst_in:.6f} was kept"
        got_tail = [t for t in indices[i, INDEX_TOPK:].tolist() if t >= 0]
        assert got_tail == tails[i], f"query {i}: tail {got_tail}, expected {tails[i]}"
    # The interesting regime must actually be exercised.
    assert past_budget > seqlen // 4, f"only {past_budget} queries had more visible pools than the budget"

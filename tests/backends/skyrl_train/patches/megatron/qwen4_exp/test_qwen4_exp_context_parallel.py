"""Qwen4Exp PLE and QSA indexer under context parallelism match CP=1.

Each gloo process is one CP rank holding its zigzag chunks of a packed stream (as megatron-core /
SkyRL split it): PLE outputs, input grads and parameter grads (summed over CP), and the QSA
selection of every local query must equal the single-rank result. The pack has EOS inside a
chunk, sequences of several lengths, and one sequence whose chunks are shorter than PLE's
9-token lookback.

Tiny configs, fp32, CPU.
"""

import types

import pytest
import torch

pytest.importorskip("megatron.core", reason="requires the megatron extra")
hf = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")

pytestmark = pytest.mark.megatron

H, N, EOS = 64, 4, 7
BUDGET, RATIO = 16, 4


def _ple_hf_config():
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    return Qwen4ExpTextConfig(
        hidden_size=H,
        hc_count=N,
        hc_lowrank=16,
        vocab_size=500,
        ngram_vocab_size_base=1000,
        ngram_size=3,
        heads_per_ngram=8,
        ple_embed_dim=64,
        ple_conv_kernel_size=4,
        ple_layer_ids=[2],
        eos_token_id=EOS,
        num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        rms_norm_eps=1e-6,
    )


def _qsa_hf_config():
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    return Qwen4ExpTextConfig(
        hidden_size=H,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=32,
        indexer_budget=BUDGET,
        indexer_compress_ratio=RATIO,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "mrope_section": [3, 3, 2],
            "mrope_interleaved": True,
        },
        vocab_size=500,
        num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        rms_norm_eps=1e-6,
    )


def _config(c):
    return types.SimpleNamespace(
        sequence_parallel=False,
        hidden_size=c.hidden_size,
        layernorm_epsilon=c.rms_norm_eps,
        qwen4_hc_count=c.hc_count,
        qwen4_hc_lowrank=c.hc_lowrank,
        qwen4_ple_embed_dim=c.ple_embed_dim,
        qwen4_ngram_size=c.ngram_size,
        qwen4_heads_per_ngram=c.heads_per_ngram,
        qwen4_eos_token_id=EOS,
        qwen4_ple_conv_kernel_size=c.ple_conv_kernel_size,
        qwen4_hf_path=None,
        qwen4_ple_hf_prefix=None,
        qwen4_indexer_n_heads=c.indexer_n_heads,
        qwen4_indexer_kv_heads=c.indexer_kv_heads,
        qwen4_indexer_head_dim=c.indexer_head_dim,
        qwen4_indexer_budget=c.indexer_budget,
        qwen4_indexer_compress_ratio=c.indexer_compress_ratio,
    )


def _pack(cp):
    """Padded lengths (multiples of 8 * cp, as packing_utils aligns them) and cu_seqlens."""
    unit = 8 * cp
    lengths = [unit, 3 * unit, 5 * unit, unit * 2]  # the first has chunks of 4 tokens (< 9)
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    return lengths, cu


def _init(rank, world_size, init_file):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    # megatron-core's gather / reduce-scatter allocate on torch.cuda.current_device().
    torch.cuda.current_device = lambda: "cpu"
    return dist


def _ple_worker(rank, world_size, init_file):
    from skyrl.backends.skyrl_train.patches.megatron.qwen4_exp.context_parallel import (
        CPLayout,
    )
    from skyrl.backends.skyrl_train.patches.megatron.qwen4_exp.modules import (
        Qwen4ExpPLE,
    )

    dist = _init(rank, world_size, init_file)
    try:
        c = _ple_hf_config()
        ref = hf.Qwen4ExpTextPLELayer(c, layer_idx=1, ple_layer_index=0)
        torch.manual_seed(0)
        with torch.no_grad():
            for p in ref.parameters():
                p.normal_(0, 0.3)
        emb = ref.ple_embedding
        weights = {k: v for k, v in ref.state_dict().items() if not k.startswith("ple_embedding.")}
        consts = (emb.layer_multipliers.clone(), emb.ngram_heads_vocab_sizes.clone(), emb.ngram_heads_offsets.clone())
        ples = []
        for _ in range(2):
            ple = Qwen4ExpPLE(_config(c), ple_layer_index=0)
            ple.load_state_dict(weights)
            ple.set_table(emb.ngram_embedding.weight.detach().clone(), 0, *consts)
            ples.append(ple)
        full, cp = ples

        lengths, cu = _pack(world_size)
        t = int(cu[-1])
        torch.manual_seed(1)
        tokens = torch.randint(10, 500, (t,))
        h = lengths[1] // (2 * world_size)
        tokens[[int(cu[1]) + h - 1, int(cu[1]) + h + 1, int(cu[2]) + 2]] = EOS  # EOS near / inside chunks
        x = torch.randn(t, 1, N * H)
        grad = torch.randn(t, 1, N * H)

        full.set_tokens(tokens[None], cu)
        x_full = x.clone().requires_grad_()
        out_full = full(x_full)
        (out_full * grad).sum().backward()

        layout = CPLayout.build(cu, dist.group.WORLD)
        pos = layout.positions
        cp.set_tokens(tokens[pos][None], cu, layout)
        x_local = x[pos].clone().requires_grad_()
        out_local = cp(x_local)
        (out_local * grad[pos]).sum().backward()

        # Same math; GEMMs over a different number of rows may round an element differently.
        torch.testing.assert_close(out_local, out_full[pos], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(x_local.grad, x_full.grad[pos], rtol=1e-5, atol=1e-5)
        for (name, p_cp), p_full in zip(cp.named_parameters(), full.parameters()):
            g = p_cp.grad.clone()
            dist.all_reduce(g)
            # fp32 sums in a different order: ~1e-6 of the gradient scale.
            torch.testing.assert_close(g, p_full.grad, rtol=1e-5, atol=1e-4, msg=name)
    finally:
        dist.destroy_process_group()


def _indexer_worker(rank, world_size, init_file):
    from megatron.core.models.common.embeddings.rotary_pos_embedding import (
        RotaryEmbedding,
    )

    from skyrl.backends.skyrl_train.patches.megatron.qwen4_exp.context_parallel import (
        CPLayout,
    )
    from skyrl.backends.skyrl_train.patches.megatron.qwen4_exp.modules import (
        Qwen4ExpQSAIndexer,
    )

    dist = _init(rank, world_size, init_file)
    try:
        c = _qsa_hf_config()
        indexer = Qwen4ExpQSAIndexer(_config(c))
        torch.manual_seed(0)
        with torch.no_grad():
            for p in indexer.parameters():
                p.normal_(0, 0.3)
        lengths, cu = _pack(world_size)
        t = int(cu[-1])
        rope = RotaryEmbedding(
            kv_channels=c.head_dim,
            rotary_percent=c.rope_parameters["partial_rotary_factor"],
            rotary_base=c.rope_parameters["rope_theta"],
            use_cpu_initialization=True,
        )
        freqs = rope.get_freqs_non_repeated(max(lengths))
        table = torch.cat((freqs, freqs), dim=-1)[:, None, None, :]
        packed = types.SimpleNamespace(cu_seqlens_q=cu, cu_seqlens_q_padded=cu, qkv_format="thd")
        torch.manual_seed(2)
        x = torch.randn(t, 1, H)

        ref = indexer.select(x, table, packed)
        layout = CPLayout.build(cu, dist.group.WORLD)
        got = indexer.select(x[layout.positions], table, packed, layout)
        assert got.cp_layout is layout
        for seq_ref, seq, start, length in zip(ref.sequences, got.sequences, cu[:-1].tolist(), lengths):
            assert (seq.start, seq.length) == (start // world_size, length // world_size)
            rows = layout.positions[seq.start : seq.start + seq.length] - start
            if seq_ref.block_mask is None:  # within the exactly-dense bound
                assert seq.block_mask is None
            else:
                assert torch.equal(seq.block_mask, seq_ref.block_mask[rows])
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("cp", [2, 4])
def test_ple_context_parallel_matches_cp1(tmp_path, cp):
    """PLE on this rank's zigzag chunks (n-gram ids from the gathered tokens, the dilated conv's
    lookback from the gathered chunk tails) equals the single-rank PLE: outputs, input grads, and
    parameter grads summed over CP."""
    import torch.multiprocessing as mp

    mp.spawn(_ple_worker, args=(cp, str(tmp_path / "init")), nprocs=cp)


@pytest.mark.parametrize("cp", [2, 4])
def test_indexer_context_parallel_matches_cp1(tmp_path, cp):
    """The QSA selection of every local query (blocks pooled per chunk, gathered over CP) equals the
    single-rank selection of the same query, bit for bit."""
    import torch.multiprocessing as mp

    mp.spawn(_indexer_worker, args=(cp, str(tmp_path / "init")), nprocs=cp)


def test_cp_global_positions_zigzag():
    from skyrl.backends.skyrl_train.patches.megatron.qwen4_exp.context_parallel import (
        cp_global_positions,
    )

    cu = torch.tensor([0, 8, 24])  # chunks of 2 and 4 tokens at CP=2
    assert cp_global_positions(cu, 0, 2).tolist() == [0, 1, 6, 7, 8, 9, 10, 11, 20, 21, 22, 23]
    assert cp_global_positions(cu, 1, 2).tolist() == [2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19]


# --- Split-K QSA attention under CP (_CudnnQSAContextParallel): forward and backward vs CP=1 ---


def _qsa_allowed(sel, lo, hi, blk_base, tok_base, block_size):
    """[T, T] keys each query may attend to under the cuDNN QSA metadata rule (qsa.md)."""
    total, nb = sel.shape
    j = torch.arange(total, device=sel.device)[None]
    b = blk_base[:, None].long() + torch.div(j - tok_base[:, None], block_size, rounding_mode="floor")
    in_range = (lo[:, None] <= j) & (j <= hi[:, None]) & (b >= 0) & (b < nb)
    return in_range & (sel.gather(1, b.clamp(0, nb - 1)) != 0)


def _ref_scores(q, k, scale, allowed):
    k = k.float().repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    s = torch.einsum("qhd,khd->hqk", q.float(), k) * scale
    return s.masked_fill(~allowed[None], float("-inf")), k


def _ref_qsa_forward(q, k, v, sel, lo, hi, blk_base, tok_base, block_first, block_last, scale, block_size=4):
    """Torch stand-in for ``cudnn.ops.qsa_forward``. Queries with no allowed key get junk (out 1,
    LSE 0): the caller must mask them, not rely on the kernel's values."""
    allowed = _qsa_allowed(sel, lo, hi, blk_base, tok_base, block_size)
    s, _ = _ref_scores(q, k, scale, allowed)
    lse = torch.logsumexp(s, dim=-1)
    p = torch.where(allowed[None], torch.exp(s - lse[..., None]), 0.0)
    out = torch.einsum("hqk,khd->qhd", p, v.float().repeat_interleave(q.shape[1] // v.shape[1], dim=1))
    empty = ~allowed.any(dim=1)
    out, lse = out.masked_fill(empty[:, None, None], 1.0), lse.masked_fill(empty[None], 0.0)
    return {"out": out.to(q.dtype), "out_fp32": out, "lse": lse}


def _ref_qsa_backward(
    g, q, k, v, out, lse, sel, lo, hi, blk_base, tok_base, block_first, block_last, scale, block_size=4
):
    """Torch stand-in for ``cudnn.ops.qsa_backward``: P and rowsum(dO * O) come from the given O / LSE."""
    rep = q.shape[1] // k.shape[1]
    allowed = _qsa_allowed(sel, lo, hi, blk_base, tok_base, block_size)
    s, kr = _ref_scores(q, k, scale, allowed)
    p = torch.where(allowed[None], torch.exp(s - lse[..., None]), 0.0)
    g = g.float()
    dv = torch.einsum("hqk,qhd->khd", p, g)
    dp = torch.einsum("qhd,khd->hqk", g, v.float().repeat_interleave(rep, dim=1))
    ds = p * (dp - (g * out).sum(-1).T[..., None])
    dq = torch.einsum("hqk,khd->qhd", ds, kr) * scale
    dk = torch.einsum("hqk,qhd->khd", ds, q.float()) * scale

    def fold(t):  # query heads -> their key group
        return t.reshape(t.shape[0], k.shape[1], rep, t.shape[-1]).sum(2)

    return {"dq": dq.to(q.dtype), "dk": fold(dk).to(k.dtype), "dv": fold(dv).to(v.dtype)}


def _cp_attention_case(cp, scale_len, device):
    """A pack of sparse, dense and very sparse sequences (padded lengths, whole blocks per chunk) and
    random block masks. Rows whose own block is complete keep it, so every query has a key; in the
    very sparse sequence many queries select nothing in some key chunks (empty rows of a pair)."""
    r = 4
    unit = 2 * cp * r * scale_len
    lengths = [8 * unit, 2 * unit, 6 * unit]
    gen = torch.Generator().manual_seed(0)
    masks = []
    for n, p in zip(lengths, [0.3, None, 0.05]):
        if p is None:
            masks.append(None)
            continue
        t = torch.arange(n)
        m = (torch.rand(n, n // r, generator=gen) < p) & (torch.arange(n // r) < ((t + 1) // r)[:, None])
        full = (t + 1) % r == 0
        m[t[full], (t // r)[full]] = True
        masks.append(m.to(device))
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)
    return r, lengths, cu, masks


def _cp_attention_run(cp, q, k, v, go, scale, r, lengths, cu, masks):
    """Every CP rank's ``_CudnnQSAContextParallel`` in one process. The gathered keys are shared
    leaves, so their gradient is the sum over ranks (what ``CPLayout.gather``'s reduce-scatter
    hands each owner). Returns global-order out / dq / dk / dv and whether some pair had empty rows."""
    from skyrl.backends.skyrl_train.patches.megatron.qwen4_exp.context_parallel import (
        CPLayout,
        cp_global_positions,
    )
    from skyrl.backends.skyrl_train.patches.megatron.qwen4_exp.modules import (
        QSASelection,
        QSASequenceSelection,
        _CPPairs,
        _CudnnQSAContextParallel,
    )

    all_pos = torch.cat([cp_global_positions(cu, i, cp) for i in range(cp)])
    inverse = torch.empty_like(all_pos)
    inverse[all_pos] = torch.arange(all_pos.numel(), device=all_pos.device)
    k_all, v_all = (t.detach().clone().requires_grad_() for t in (k, v))
    out, dq = torch.zeros_like(q, dtype=torch.float32), torch.zeros_like(q, dtype=torch.float32)
    empty_rows = False
    n_local = all_pos.numel() // cp
    for rank in range(cp):
        pos = all_pos[rank * n_local : (rank + 1) * n_local]
        layout = CPLayout(None, cp, rank, cu, pos, all_pos, inverse)
        seqs = []
        for start, n, m in zip(cu[:-1].tolist(), lengths, masks):
            lo, nl = start // cp, n // cp
            seqs.append(QSASequenceSelection(lo, nl, None if m is None else m[pos[lo : lo + nl] - start]))
        selection = QSASelection(seqs, r, cp_layout=layout)
        pairs = _CPPairs(selection, q.device)
        for half in range(2):
            for j in range(pairs.chunk_index(half) + 1):
                meta, empty = pairs.metadata(half, j)
                empty_rows |= meta is not None and bool(empty.any())
        q_local = q[pos].detach().clone().requires_grad_()
        out_local = _CudnnQSAContextParallel.apply(q_local, k_all, v_all, scale, selection)
        (out_local.float() * go[pos]).sum().backward()
        out[pos], dq[pos] = out_local.detach().float(), q_local.grad.float()
    return out, dq, k_all.grad.float(), v_all.grad.float(), empty_rows


def _dense_reference(q, k, v, go, scale, r, lengths, cu, masks):
    """fp32 autograd attention of the whole pack under the per-sequence ``_qsa_token_mask``."""
    from skyrl.backends.skyrl_train.patches.megatron.qwen4_exp.modules import (
        _qsa_token_mask,
    )

    total = int(cu[-1])
    allowed = torch.zeros(total, total, dtype=torch.bool, device=q.device)
    for start, n, m in zip(cu[:-1].tolist(), lengths, masks):
        allowed[start : start + n, start : start + n] = _qsa_token_mask(m, 0, n, r, q.device)
    qf, kf, vf = (t.detach().float().requires_grad_() for t in (q, k, v))
    s, _ = _ref_scores(qf, kf, scale, allowed)
    rep = q.shape[1] // k.shape[1]
    out = torch.einsum("hqk,khd->qhd", torch.softmax(s, dim=-1), vf.repeat_interleave(rep, dim=1))
    (out * go).sum().backward()
    return out.detach(), qf.grad, kf.grad, vf.grad


@pytest.mark.parametrize("cp", [2, 4])
def test_cp_qsa_attention_matches_cp1(cp, monkeypatch):
    """The split-K schedule, LSE merge and backward replay of ``_CudnnQSAContextParallel`` (with a
    torch stand-in for the cuDNN kernels, fp32, CPU): outputs and dq / dk / dv over every CP rank
    equal single-rank attention, for sparse and dense sequences and rows without keys in a pair."""
    from skyrl.backends.skyrl_train.patches.megatron.qwen4_exp import modules

    monkeypatch.setattr(modules, "_cudnn_qsa_ops", (_ref_qsa_forward, _ref_qsa_backward))
    r, lengths, cu, masks = _cp_attention_case(cp, 1, "cpu")
    total, hq, hkv, d = int(cu[-1]), 4, 2, 16
    torch.manual_seed(2)
    q, k, v, go = (torch.randn(total, h, d) for h in (hq, hkv, hkv, hq))
    scale = d**-0.5
    got = _cp_attention_run(cp, q, k, v, go, scale, r, lengths, cu, masks)
    assert got[-1], "the case should have pairs with empty rows"
    for name, a, b in zip(
        ("out", "dq", "dk", "dv"), got[:4], _dense_reference(q, k, v, go, scale, r, lengths, cu, masks)
    ):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5, msg=name)


def test_cp_qsa_attention_cudnn_matches_cp1():
    """``_CudnnQSAContextParallel`` with the real cuDNN kernels (Blackwell, bf16, Qwen3.8 heads) at
    CP=4: each of out / dq / dk / dv is as close to an fp32 reference as the CP=1 cuDNN path is."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    from skyrl.backends.skyrl_train.patches.megatron.qwen4_exp.modules import (
        QSASelection,
        QSASequenceSelection,
        _cudnn_qsa_usable,
        _CudnnQSA,
        cudnn_qsa_metadata,
    )

    cp, hq, hkv, d = 4, 24, 2, 256
    r, lengths, cu, masks = _cp_attention_case(cp, 4, "cuda")
    total = int(cu[-1])
    torch.manual_seed(2)
    q, k, v = (torch.randn(total, h, d, device="cuda", dtype=torch.bfloat16) for h in (hq, hkv, hkv))
    go = torch.randn(total, hq, d, device="cuda")
    if not _cudnn_qsa_usable(q, k, v, r):
        pytest.skip("needs the cuDNN QSA kernels (Blackwell, cudnn-frontend >= 1.32)")
    scale = d**-0.5
    ref = _dense_reference(q, k, v, go, scale, r, lengths, cu, masks)
    got = _cp_attention_run(cp, q, k, v, go, scale, r, lengths, cu, masks)

    seqs = [QSASequenceSelection(s, n, m) for s, n, m in zip(cu[:-1].tolist(), lengths, masks)]
    q1, k1, v1 = (t.detach().clone().requires_grad_() for t in (q, k, v))
    out1 = _CudnnQSA.apply(q1, k1, v1, scale, *cudnn_qsa_metadata(QSASelection(seqs, r), total, q.device))
    (out1.float() * go).sum().backward()
    cp1 = (out1.detach().float(), q1.grad.float(), k1.grad.float(), v1.grad.float())

    for name, x, b, c in zip(("out", "dq", "dk", "dv"), got[:4], cp1, ref):
        err_cp, err_cp1 = ((x - c).norm() / c.norm()).item(), ((b - c).norm() / c.norm()).item()
        print(f"{name}: CP{cp} rel err {err_cp:.2e}, CP1 rel err {err_cp1:.2e}")
        assert err_cp <= 2 * err_cp1 + 1e-3, f"{name}: CP{cp} {err_cp:.2e} vs CP1 {err_cp1:.2e}"

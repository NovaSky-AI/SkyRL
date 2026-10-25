"""The Qwen4Exp QSA indexer and sparse attention match the HF reference.

A tiny config with ``indexer_budget=16`` and ``indexer_compress_ratio=4`` puts the sparse regime
(sequences over 16 + 4 - 1 = 19 tokens) at short lengths. Covers the per-query selection mask vs
HF ``Qwen4ExpTextQSAIndexer`` and the gated attention (output and gradients) vs HF
``Qwen4ExpTextAttention`` on a padded packed (THD) stream, with HF run on each sequence
separately. The q / k / v / gate projections are computed from the HF weights as HF does; the
Megatron projections and the TE dense path are covered by the GPU parity test.

Tiny configs, fp32, CPU.
"""

import types

import pytest
import torch

pytest.importorskip("megatron.core", reason="requires the megatron extra")
hf = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")

pytestmark = pytest.mark.megatron

H, BUDGET, RATIO = 64, 16, 4
MAX_DENSE = BUDGET + RATIO - 1


def _hf_config():
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    c = Qwen4ExpTextConfig(
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
    c._attn_implementation = "eager"
    return c


def _mcore_config(c):
    return types.SimpleNamespace(
        hidden_size=c.hidden_size,
        layernorm_epsilon=c.rms_norm_eps,
        qwen4_indexer_n_heads=c.indexer_n_heads,
        qwen4_indexer_kv_heads=c.indexer_kv_heads,
        qwen4_indexer_head_dim=c.indexer_head_dim,
        qwen4_indexer_budget=c.indexer_budget,
        qwen4_indexer_compress_ratio=c.indexer_compress_ratio,
    )


def _randomize(module, seed=0):
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.normal_(0, 0.3)


def _rope_table(c, length):
    """megatron-core's RoPE angle table ([L, 1, 1, rot]) for the main attention's head dim."""
    from megatron.core.models.common.embeddings.rotary_pos_embedding import (
        RotaryEmbedding,
    )

    rope = RotaryEmbedding(
        kv_channels=c.head_dim,
        rotary_percent=c.rope_parameters["partial_rotary_factor"],
        rotary_base=c.rope_parameters["rope_theta"],
        use_cpu_initialization=True,
    )
    # ``get_emb`` without its move to the GPU: non-interleaved [freqs, freqs].
    freqs = rope.get_freqs_non_repeated(length)
    return torch.cat((freqs, freqs), dim=-1)[:, None, None, :]


def _hf_rope(c, length):
    rotary = hf.Qwen4ExpTextRotaryEmbedding(c)
    x = torch.zeros(1, length, c.hidden_size)
    return rotary(x, torch.arange(length)[None])


def _hf_causal_mask(length, dtype=torch.float32):
    mask = torch.full((length, length), torch.finfo(dtype).min, dtype=dtype).triu(1)
    return mask[None, None]


def _packed(lengths, pad_to):
    """PackedSeqParams-like (only the fields the indexer reads) with per-sequence padding."""
    padded = [(n + pad_to - 1) // pad_to * pad_to for n in lengths]
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    cu_padded = torch.tensor([0, *torch.tensor(padded).cumsum(0).tolist()], dtype=torch.int32)
    return types.SimpleNamespace(cu_seqlens_q=cu, cu_seqlens_q_padded=cu_padded, qkv_format="thd"), cu_padded


@pytest.fixture(params=[(1024, 512), (8, 5)], ids=["default_chunks", "small_chunks"])
def chunks(request, monkeypatch):
    from skyrl.backends.skyrl_train.patches.megatron.qwen4_exp import modules

    monkeypatch.setattr(modules, "QSA_INDEXER_CHUNK", request.param[0])
    monkeypatch.setattr(modules, "QSA_ATTENTION_CHUNK", request.param[1])
    return request.param


@pytest.mark.parametrize("length", [3, 18, MAX_DENSE, MAX_DENSE + 1, 23, 24, 37, 64, 101])
def test_indexer_selection_matches_hf(length, chunks):
    from skyrl.backends.skyrl_train.patches.megatron.qwen4_exp.modules import (
        Qwen4ExpQSAIndexer,
        _qsa_token_mask,
    )

    c = _hf_config()
    ref = hf.Qwen4ExpTextQSAIndexer(c, layer_idx=3)
    _randomize(ref)
    ours = Qwen4ExpQSAIndexer(_mcore_config(c))
    ours.load_state_dict(ref.state_dict())

    torch.manual_seed(1)
    x = torch.randn(1, length, H)
    with torch.no_grad():
        expected = ref(x, _hf_rope(c, length), _hf_causal_mask(length), None)[0, 0] == 0

    packed, _ = _packed([length], pad_to=1)
    selection = ours.select(x.transpose(0, 1), _rope_table(c, length), packed)
    if length <= MAX_DENSE:
        assert selection is None  # exactly dense: HF selects every visible key
        assert torch.equal(expected, torch.ones(length, length, dtype=torch.bool).tril())
        return
    (seq,) = selection.sequences
    got = _qsa_token_mask(seq.block_mask, 0, length, RATIO, x.device)
    assert torch.equal(got, expected)
    # Sanity: the sparse regime really drops keys, and each query keeps budget + tail tokens.
    t = torch.arange(length)
    assert torch.equal(got.sum(-1), torch.where((t + 1) // RATIO > BUDGET // RATIO, BUDGET + (t + 1) % RATIO, t + 1))


def test_indexer_selection_matches_hf_bf16():
    """bf16 (training dtype): same casts as HF, so the selection is still identical."""
    from skyrl.backends.skyrl_train.patches.megatron.qwen4_exp.modules import (
        Qwen4ExpQSAIndexer,
        _qsa_token_mask,
    )

    c, length = _hf_config(), 400
    ref = hf.Qwen4ExpTextQSAIndexer(c, layer_idx=3)
    _randomize(ref)
    ours = Qwen4ExpQSAIndexer(_mcore_config(c))
    ours.load_state_dict(ref.state_dict())
    ref.to(torch.bfloat16)
    ours.to(torch.bfloat16)

    torch.manual_seed(1)
    x = torch.randn(1, length, H, dtype=torch.bfloat16)
    cos, sin = _hf_rope(c, length)
    with torch.no_grad():
        mask = _hf_causal_mask(length, torch.bfloat16)
        expected = ref(x, (cos.bfloat16(), sin.bfloat16()), mask, None)[0, 0] == 0
    packed, _ = _packed([length], pad_to=1)
    (seq,) = ours.select(x.transpose(0, 1), _rope_table(c, length), packed).sequences
    assert torch.equal(_qsa_token_mask(seq.block_mask, 0, length, RATIO, x.device), expected)


def _attention_inputs(ref, x, cos, sin):
    """HF ``Qwen4ExpTextAttention`` q / k / v / gate for x ([T, C]); q/k/v as [T, heads, d]."""
    d = ref.head_dim
    q, gate = torch.chunk(ref.q_proj(x).view(x.shape[0], -1, d * 2), 2, dim=-1)
    q = ref.q_norm(q)
    k = ref.k_norm(ref.k_proj(x).view(x.shape[0], -1, d))
    v = ref.v_proj(x).view(x.shape[0], -1, d)
    q, k = hf.apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
    return q, k, v, gate.reshape(x.shape[0], -1)


def test_sparse_attention_matches_hf_packed(chunks):
    """Gated attention on a padded packed stream mixing dense and sparse-regime sequences."""
    from skyrl.backends.skyrl_train.patches.megatron.qwen4_exp.modules import (
        Qwen4ExpQSAIndexer,
        qsa_sparse_attention,
    )

    c = _hf_config()
    ref = hf.Qwen4ExpTextAttention(c, layer_idx=3)
    _randomize(ref)
    ours = Qwen4ExpQSAIndexer(_mcore_config(c))
    ours.load_state_dict(ref.indexer.state_dict())

    lengths = [45, 12, 70, MAX_DENSE + 1]
    packed, cu_padded = _packed(lengths, pad_to=8)
    total = int(cu_padded[-1])
    torch.manual_seed(2)
    x = torch.randn(total, H, requires_grad=True)
    table = _rope_table(c, max(lengths))

    # Ours: the indexer on the packed stream, then the masked attention; per-sequence RoPE positions.
    positions = torch.zeros(total, dtype=torch.long)
    for start, n in zip(cu_padded[:-1].tolist(), lengths):
        positions[start : start + n] = torch.arange(n)
    angles = table.view(table.shape[0], -1)[positions]
    q, k, v, gate = _attention_inputs(ref, x, angles.cos(), angles.sin())
    selection = ours.select(x.unsqueeze(1), table, packed)
    assert selection is not None and [s.block_mask is None for s in selection.sequences] == [
        False,
        True,
        False,
        False,
    ]
    core = qsa_sparse_attention(q, k, v, selection, ref.scaling)
    out = ref.o_proj(core * torch.sigmoid(gate))

    # HF on each sequence separately.
    expected, padding = [], torch.ones(total, dtype=torch.bool)
    for start, n in zip(cu_padded[:-1].tolist(), lengths):
        padding[start : start + n] = False
        y, _ = ref(x[start : start + n][None], _hf_rope(c, n), _hf_causal_mask(n))
        expected.append(y[0])
        torch.testing.assert_close(out[start : start + n], y[0], rtol=1e-5, atol=1e-5)
    assert torch.equal(core[padding], torch.zeros_like(core[padding]))

    # Gradients through the attention over the selected keys (padding rows excluded).
    torch.manual_seed(3)
    w = torch.randn_like(out)
    w[padding] = 0
    grads_ours = torch.autograd.grad((out * w).sum(), [x, ref.q_proj.weight, ref.k_proj.weight, ref.v_proj.weight])
    loss_ref = sum((y * w[start : start + n]).sum() for y, start, n in zip(expected, cu_padded[:-1].tolist(), lengths))
    grads_ref = torch.autograd.grad(loss_ref, [x, ref.q_proj.weight, ref.k_proj.weight, ref.v_proj.weight])
    for g_ours, g_ref in zip(grads_ours, grads_ref):
        torch.testing.assert_close(g_ours, g_ref, rtol=1e-4, atol=1e-5)
    assert all(p.grad is None for p in ours.parameters())


def test_indexer_dense_for_short_packed_sequences():
    from skyrl.backends.skyrl_train.patches.megatron.qwen4_exp.modules import (
        Qwen4ExpQSAIndexer,
    )

    c = _hf_config()
    ours = Qwen4ExpQSAIndexer(_mcore_config(c))
    packed, cu_padded = _packed([MAX_DENSE, 5, MAX_DENSE], pad_to=8)
    x = torch.randn(int(cu_padded[-1]), 1, H)
    assert ours.select(x, _rope_table(c, MAX_DENSE), packed) is None
    assert all(not p.requires_grad for p in ours.parameters())


@pytest.mark.parametrize("lead", [0, 4], ids=["no_leading_gap", "leading_gap"])
def test_cudnn_qsa_metadata_matches_mask(lead):
    """The cuDNN QSA selection tensors allow exactly the keys of ``_qsa_token_mask``, per packed
    sequence, and nothing for padding queries (between sequences and before the first)."""
    from skyrl.backends.skyrl_train.patches.megatron.qwen4_exp.modules import (
        QSASelection,
        QSASequenceSelection,
        _qsa_token_mask,
        cudnn_qsa_metadata,
    )

    r = 4
    torch.manual_seed(0)
    lengths, pad_to = [50, 13, 35, 7, 37], 8  # sparse, dense, sparse, dense, sparse; lengths % 4 vary
    seqs, start = [], lead
    for n, sparse in zip(lengths, [True, False, True, False, True]):
        mask = None
        if sparse:
            visible = (torch.arange(n) + 1) // r
            mask = (torch.rand(n, n // r) < 0.5) & (torch.arange(n // r) < visible[:, None])
        seqs.append(QSASequenceSelection(start, n, mask))
        start += (n + pad_to - 1) // pad_to * pad_to
    total = start
    sel, lo, hi, blk_base, tok_base, block_first, block_last = cudnn_qsa_metadata(QSASelection(seqs, r), total, "cpu")

    # The cuDNN rule (qsa.md): b = blk_base[t] + (j - tok_base[t]) // r; allowed iff lo <= j <= hi and sel[t, b].
    nb = sel.shape[1]
    j = torch.arange(total)[None, :]
    b = blk_base[:, None] + torch.div(j - tok_base[:, None], r, rounding_mode="floor")
    in_range = (lo[:, None] <= j) & (j <= hi[:, None]) & (b >= 0) & (b < nb)
    allowed = in_range & (sel.gather(1, b.clamp(0, nb - 1)) != 0)

    expected = torch.zeros(total, total, dtype=torch.bool)
    for s in seqs:
        expected[s.start : s.start + s.length, s.start : s.start + s.length] = _qsa_token_mask(
            s.block_mask, 0, s.length, r, "cpu"
        )
    assert torch.equal(allowed, expected)
    # Block spans tile the stream, at most r tokens each.
    assert int(block_first[0]) == 0 and int(block_last[-1]) == total - 1
    assert torch.all(block_last - block_first < r) and torch.all(block_first[1:] == block_last[:-1] + 1)

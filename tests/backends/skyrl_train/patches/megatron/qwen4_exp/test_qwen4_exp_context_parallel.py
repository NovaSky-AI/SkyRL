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

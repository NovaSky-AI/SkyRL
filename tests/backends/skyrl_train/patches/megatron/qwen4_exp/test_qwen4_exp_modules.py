"""The Qwen4Exp Megatron modules match the HF reference modules.

Covers the gated low-rank hyper-connection (with and without inject, i.e. the final mixer) and
the PLE n-gram layer on a packed stream: two sequences with EOS tokens inside, compared with HF
run on each sequence separately (HF resets the n-gram context and the dilated conv per sequence).
Also checks PLE under the row-sharded table and under sequence parallelism (gloo processes).

Tiny configs, fp32, CPU.
"""

import types

import pytest
import torch

pytest.importorskip("megatron.core", reason="requires the megatron extra")
hf = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")

pytestmark = pytest.mark.megatron

H, N, EOS = 64, 4, 7


def _hf_config():
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


def _mcore_config(c, sequence_parallel=False):
    return types.SimpleNamespace(
        sequence_parallel=sequence_parallel,
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
    )


def _randomize(module):
    torch.manual_seed(0)
    with torch.no_grad():
        for p in module.parameters():
            p.normal_(0, 0.3)


@pytest.mark.parametrize("use_combine", [True, False])
def test_gated_residual_matches_hf(use_combine):
    from skyrl.backends.skyrl_train.patches.megatron.qwen4_exp.modules import (
        Qwen4ExpFinalMixer,
        Qwen4ExpGatedResidual,
    )

    c = _hf_config()
    ref = hf.Qwen4ExpTextGatedResidual(c, use_combine=use_combine)
    _randomize(ref)
    ours = Qwen4ExpGatedResidual(_mcore_config(c)) if use_combine else Qwen4ExpFinalMixer(_mcore_config(c))
    ours.load_state_dict(ref.state_dict())

    x = torch.randn(5, 3, N * H)
    if use_combine:
        mixed_ref, _, inject_ref = ref(x)
        mixed, inject = ours(x)
        torch.testing.assert_close(inject, inject_ref)
        y = torch.randn(5, 3, H)
        torch.testing.assert_close(
            Qwen4ExpGatedResidual.inject(x, y, inject), x + (y.unsqueeze(-2) * inject_ref.unsqueeze(-1)).flatten(-2)
        )
    else:
        mixed_ref, mixed = ref(x), ours(x)
    torch.testing.assert_close(mixed, mixed_ref)


def test_ple_packed_matches_hf_per_sequence():
    from skyrl.backends.skyrl_train.patches.megatron.qwen4_exp.modules import (
        Qwen4ExpPLE,
    )

    c = _hf_config()
    ref = hf.Qwen4ExpTextPLELayer(c, layer_idx=1, ple_layer_index=0)
    _randomize(ref)
    emb = ref.ple_embedding

    ours = Qwen4ExpPLE(_mcore_config(c), ple_layer_index=0)
    ours.load_state_dict({k: v for k, v in ref.state_dict().items() if not k.startswith("ple_embedding.")})
    ours.set_table(
        emb.ngram_embedding.weight.detach().clone(),
        0,
        emb.layer_multipliers.clone(),
        emb.ngram_heads_vocab_sizes.clone(),
        emb.ngram_heads_offsets.clone(),
    )

    torch.manual_seed(1)
    seqs = [torch.randint(10, 500, (23,)), torch.randint(10, 500, (17,))]
    seqs[0][[5, 6, 12]] = EOS  # EOS resets the n-gram context, including back-to-back EOS
    seqs[1][9] = EOS
    hidden = [torch.randn(1, len(s), N * H) for s in seqs]

    with torch.no_grad():
        expected = torch.cat([ref(h, s[None], past_key_values=None)[0] for h, s in zip(hidden, seqs)])
        tokens = torch.cat(seqs)
        cu = torch.tensor([0, len(seqs[0]), len(tokens)], dtype=torch.int32)
        ours.set_tokens(tokens[None], cu)
        got = ours(torch.cat(hidden, dim=1).transpose(0, 1))[:, 0]
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-5)


def test_bridge_provider_accepts_qwen_sparse_attention(tmp_path):
    """transformers loads ``full_attention`` as ``qwen_sparse_attention``; the bridge still maps the pattern."""
    from megatron.bridge import AutoBridge
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig

    import skyrl.backends.skyrl_train.patches.megatron.qwen4_exp.bridge  # noqa: F401

    text = _hf_config().to_dict()
    text.update(
        output_gate_type="sigmoid",
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
    )
    Qwen4ExpConfig(text_config=text, architectures=["Qwen4ExpForConditionalGeneration"]).save_pretrained(tmp_path)
    provider = AutoBridge.from_hf_pretrained(str(tmp_path)).to_megatron_provider(load_weights=False)
    assert list(provider.linear_attention_freq) == [1, 1, 1, 0]
    assert provider.qwen4_ple_hf_prefix == "model.language_model.layers.1.ple."


def _sharded_ple_worker(rank, world_size, init_file):
    import torch.distributed as dist

    from skyrl.backends.skyrl_train.patches.megatron.qwen4_exp.modules import (
        Qwen4ExpPLE,
        _row_bounds,
    )

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        c = _hf_config()
        ref = hf.Qwen4ExpTextPLELayer(c, layer_idx=1, ple_layer_index=0)
        _randomize(ref)
        emb = ref.ple_embedding
        weights = {k: v for k, v in ref.state_dict().items() if not k.startswith("ple_embedding.")}
        table = emb.ngram_embedding.weight.detach().clone()
        consts = (emb.layer_multipliers.clone(), emb.ngram_heads_vocab_sizes.clone(), emb.ngram_heads_offsets.clone())

        full = Qwen4ExpPLE(_mcore_config(c), ple_layer_index=0)
        full.load_state_dict(weights)
        full.set_table(table, 0, *consts)
        sharded = Qwen4ExpPLE(_mcore_config(c), ple_layer_index=0)
        sharded.load_state_dict(weights)
        # Shard the rows the heads address (HF pads the table past the last head).
        num_rows = int(consts[2][-1] + consts[1][-1])
        lo, hi = _row_bounds(num_rows, world_size)[rank : rank + 2].tolist()
        sharded.set_table(table[lo:hi].clone(), lo, *consts, group=dist.group.WORLD)

        # Different token counts per rank; EOS inside so the n-gram ids cover every head.
        torch.manual_seed(10 + rank)
        tokens = torch.randint(10, 500, (11 + 7 * rank,))
        tokens[3] = EOS
        hidden = torch.randn(len(tokens), 1, N * H)
        cu = torch.tensor([0, len(tokens)], dtype=torch.int32)
        with torch.no_grad():
            for ple in (full, sharded):
                ple.set_tokens(tokens[None], cu)
            torch.testing.assert_close(sharded(hidden), full(hidden))
    finally:
        dist.destroy_process_group()


def test_ple_row_sharded_matches_unsharded(tmp_path):
    """Row-sharding the n-gram table over a group (3 ranks: does not divide the 16 heads) gives the
    unsharded result on every rank, with a different number of tokens per rank."""
    import torch.multiprocessing as mp

    mp.spawn(_sharded_ple_worker, args=(3, str(tmp_path / "init")), nprocs=3)


def _sequence_parallel_ple_worker(rank, world_size, init_file):
    import torch.distributed as dist

    from skyrl.backends.skyrl_train.patches.megatron.qwen4_exp.modules import (
        Qwen4ExpPLE,
        sp_gather,
        sp_local,
    )

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    # megatron-core's SP gather / reduce-scatter allocate on torch.cuda.current_device().
    torch.cuda.current_device = lambda: "cpu"
    try:
        c = _hf_config()
        ref = hf.Qwen4ExpTextPLELayer(c, layer_idx=1, ple_layer_index=0)
        _randomize(ref)
        emb = ref.ple_embedding
        weights = {k: v for k, v in ref.state_dict().items() if not k.startswith("ple_embedding.")}
        consts = (emb.layer_multipliers.clone(), emb.ngram_heads_vocab_sizes.clone(), emb.ngram_heads_offsets.clone())
        ples = []
        for sp in (False, True):
            ple = Qwen4ExpPLE(_mcore_config(c, sequence_parallel=sp), ple_layer_index=0)
            ple.load_state_dict(weights)
            ple.set_table(emb.ngram_embedding.weight.detach().clone(), 0, *consts)
            ples.append(ple)
        full, sharded = ples
        assert all(p.sequence_parallel for p in sharded.parameters())

        # Same pack on every rank; a sequence boundary and EOS fall inside rank 0's tokens so the
        # n-gram context and the dilated conv cross the rank boundary.
        torch.manual_seed(0)
        t = 12 * world_size
        tokens = torch.randint(10, 500, (t,))
        tokens[5] = EOS
        cu = torch.tensor([0, 7, t], dtype=torch.int32)
        x = torch.randn(t, 1, N * H)
        grad = torch.randn(t, 1, N * H)
        for ple in ples:
            ple.set_tokens(tokens[None], cu)

        x_full = x.clone().requires_grad_()
        out_full = full(x_full)
        (out_full * grad).sum().backward()

        group = dist.group.WORLD
        n = t // world_size
        x_local = x[rank * n : (rank + 1) * n].clone().requires_grad_()
        out_local = sp_local(sharded(sp_gather(x_local, group)), group)
        (out_local * grad[rank * n : (rank + 1) * n]).sum().backward()

        torch.testing.assert_close(out_local, out_full[rank * n : (rank + 1) * n])
        torch.testing.assert_close(x_local.grad, x_full.grad[rank * n : (rank + 1) * n])
        # Each rank's parameter grads cover its own tokens; summed over TP they are the full grads.
        for (name, p_sp), p_full in zip(sharded.named_parameters(), full.parameters()):
            g = p_sp.grad.clone()
            dist.all_reduce(g, group=group)
            torch.testing.assert_close(g, p_full.grad, msg=name)
    finally:
        dist.destroy_process_group()


def test_ple_sequence_parallel_matches_full(tmp_path):
    """PLE on the TP-gathered sequence, keeping this rank's tokens (as ``Qwen4ExpTransformerLayer``
    runs it under SP), matches the full-sequence PLE: outputs, input grads, and parameter grads once
    summed over TP (the ``sequence_parallel`` reduction)."""
    import torch.multiprocessing as mp

    mp.spawn(_sequence_parallel_ple_worker, args=(2, str(tmp_path / "init")), nprocs=2)

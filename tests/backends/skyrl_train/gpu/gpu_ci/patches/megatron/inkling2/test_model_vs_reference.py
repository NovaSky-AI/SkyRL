"""Inkling2 Megatron model vs. the plain-PyTorch reference (``inkling2/reference.py``).

A tiny random-weight model built by ``Inkling2ModelProvider`` (7 layers: local and global
attention, attn-res blocks of 3 so the state crosses two block boundaries, a 16-token window
and relative extent, and a length-scaling floor of 20 tokens so tau != 1) is copied into the
reference, and next-token logprobs are compared on two packed sequences longer than the window
and on an unpacked batch. Every expert is selected (top-k = number of experts): with fewer, BF16
rounding flips near-tied expert choices and single tokens jump by O(1) without anything being
wrong. The thresholds sit a factor >= 3 below the error of any of these single-site bugs: a
flipped relative bias, a swapped q/k/v attn-res query, an off-by-one convolution lag, a wrong
block boundary.

Run with:
uv run --isolated --extra dev --extra megatron pytest -s tests/backends/skyrl_train/gpu/gpu_ci/patches/megatron/inkling2/test_model_vs_reference.py
"""

import os

import pytest
import ray
import torch

pytestmark = pytest.mark.megatron

L, H, HEADS, KV, HEAD_DIM, E, FFN, V, UNPADDED_V = 7, 256, 4, 2, 64, 8, 128, 512, 500
LOCAL = [i for i in range(L) if i % 3 != 2]
COMMON = dict(sliding_window_size=16, rel_extent=16, d_rel=8, log_scaling_n_floor=20, n_layers_per_attn_res_block=3)


def _init_single_rank_megatron():
    import torch.distributed as dist
    from megatron.core import parallel_state as mpu
    from megatron.core import tensor_parallel

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29533")
    dist.init_process_group(backend="nccl", world_size=1, rank=0)
    torch.cuda.set_device(0)
    mpu.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=1)
    tensor_parallel.model_parallel_cuda_manual_seed(0)


def _build_megatron(backend: str):
    from megatron.core.process_groups_config import ProcessGroupCollection

    from skyrl.backends.skyrl_train.patches.megatron.inkling2.provider import (
        Inkling2ModelProvider,
    )

    provider = Inkling2ModelProvider(
        num_layers=L,
        hidden_size=H,
        ffn_hidden_size=FFN,
        num_attention_heads=HEADS,
        num_query_groups=KV,
        kv_channels=HEAD_DIM,
        vocab_size=V,
        unpadded_vocab_size=UNPADDED_V,
        seq_length=128,
        num_moe_experts=E,
        moe_router_topk=E,
        moe_ffn_hidden_size=FFN,
        local_layer_ids=LOCAL,
        inkling2_attention_backend=backend,
        # Inkling2 does not use TE attention; set as SkyRL's worker does (the test env sets NVTE_FUSED_ATTN=0).
        attention_backend="flash",
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        sequence_parallel=False,
        gradient_accumulation_fusion=False,
        **COMMON,
    )
    provider.finalize()
    provider._pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    model = provider.provide().cuda()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.copy_(1 + 0.2 * torch.randn_like(p.float()))
            elif "query" in name:
                p.copy_(0.5 * torch.randn_like(p.float()))
            elif "clip_scale" in name:
                p.copy_(1 + 0.3 * torch.randn_like(p.float()))
            elif "sconv" in name or "rel_proj" in name:
                p.copy_(0.3 * torch.randn_like(p.float()))
            else:
                p.copy_(0.02 * torch.randn_like(p.float()))
        for layer in model.decoder.layers:
            layer.mlp.router.expert_bias.copy_(0.05 * torch.randn_like(layer.mlp.router.expert_bias))
    return model


def _build_reference(model):
    """The reference with ``model``'s weights (Megatron names -> checkpoint names)."""
    from skyrl.backends.skyrl_train.patches.megatron.inkling2.configuration import (
        Inkling2TextConfig,
    )
    from skyrl.backends.skyrl_train.patches.megatron.inkling2.reference import (
        Inkling2Reference,
    )

    cfg = Inkling2TextConfig(
        vocab_size=V,
        unpadded_vocab_size=UNPADDED_V,
        hidden_size=H,
        num_hidden_layers=L,
        num_attention_heads=HEADS,
        num_key_value_heads=KV,
        head_dim=HEAD_DIM,
        swa_num_attention_heads=HEADS,
        swa_num_key_value_heads=KV,
        swa_head_dim=HEAD_DIM,
        local_layer_ids=LOCAL,
        n_routed_experts=E,
        num_experts_per_tok=E,
        intermediate_size=FFN,
        **COMMON,
    )
    with torch.device("cuda"):
        ref = Inkling2Reference(cfg)
    sd = dict(model.named_parameters())
    sd.update({n: b for n, b in model.named_buffers() if "expert_bias" in n})

    def put(ref_name, tensor):
        parent, attr = ref_name.rsplit(".", 1)
        getattr(ref.get_submodule(parent), attr).data = tensor.detach().clone()

    put("embed.weight", sd["embedding.word_embeddings.weight"])
    put("embed_norm.weight", sd["embedding.embed_norm.weight"])
    put("norm.weight", sd["decoder.final_layernorm.norm.weight"])
    put("attn_res.output_query", sd["decoder.final_layernorm.output_query"])
    put("unembed.weight", sd["output_layer.weight"])
    for i in range(L):
        m, r, local = f"decoder.layers.{i}.", f"layers.{i}.", i in LOCAL
        put(r + ("attn_norm.weight" if local else "q_input_norm.weight"), sd[m + "input_layernorm.weight"])
        if not local:
            put(r + "k_input_norm.weight", sd[m + "k_input_layernorm.weight"])
            put(r + "v_input_norm.weight", sd[m + "v_input_layernorm.weight"])
        put(r + "mlp_norm.weight", sd[m + "pre_mlp_layernorm.weight"])
        if i > 0:
            put(f"attn_res.attn_queries.{i - 1}", sd[m + "attn_query"])
        put(f"attn_res.mlp_queries.{i}", sd[m + "mlp_query"])
        a = m + "self_attention."
        for meg, name in (
            ("linear_q", "wq_du"),
            ("linear_k", "wk_dv"),
            ("linear_v", "wv_dv"),
            ("linear_r", "wr_du"),
            ("linear_gate", "gate_attn"),
            ("linear_sink", "attn_sink_proj"),
            ("linear_proj", "wo_ud"),
        ):
            put(r + f"attn.{name}.weight", sd[a + meg + ".weight"])
        put(r + "attn.q_l2_clip.scale", sd[a + "q_l2_clip_scale"])
        put(r + "attn.rel_l2_clip.scale", sd[a + "rel_l2_clip_scale"])
        put(r + "attn.rel_logits_proj.proj", sd[a + "rel_proj"])
        for conv in ("k_sconv", "v_sconv"):
            if local:
                put(r + f"attn.{conv}.weight", sd[a + f"{conv}.weight"])
            else:
                put(r + f"attn.{conv}.static_w", sd[a + f"{conv}.static_w"].t())  # Megatron: channel-first
                put(r + f"attn.{conv}.weight_proj", sd[a + f"{conv}.weight_proj.weight"])
        put(r + "attn_sconv.weight", sd[m + "attn_sconv.weight"])
        put(r + "mlp_sconv.weight", sd[m + "mlp_sconv.weight"])
        put(r + "mlp.gate.weight", sd[m + "mlp.router.weight"])
        put(r + "mlp.gate.bias", sd[m + "mlp.router.expert_bias"].float())
        fc1 = torch.stack([sd[m + f"mlp.experts.linear_fc1.weight{e}"] for e in range(E)])  # [E, 2I, H]: [gate; up]
        w13 = torch.stack([fc1[:, :FFN], fc1[:, FFN:]], dim=2).reshape(E, 2 * FFN, H)  # rows interleaved
        ref.layers[i].mlp.experts.load("w13_weight", w13)
        w2 = torch.stack([sd[m + f"mlp.experts.linear_fc2.weight{e}"] for e in range(E)])
        ref.layers[i].mlp.experts.load("w2_weight", w2)
    return ref.eval()


@ray.remote(num_gpus=1)
def _parity(backend: str):
    from megatron.core.packed_seq_params import PackedSeqParams

    import skyrl.backends.skyrl_train.workers.megatron  # noqa: F401  (FA4 import guard)

    _init_single_rank_megatron()
    torch.manual_seed(0)
    model = _build_megatron(backend).eval()
    ref = _build_reference(model)

    def ref_logprobs(x):
        return torch.log_softmax(ref.logits(ref.hidden_states(x), torch.bfloat16), -1)

    lens = [23, 41]
    ids = [torch.randint(0, UNPADDED_V, (n,), device="cuda") for n in lens]
    cu = torch.tensor([0, lens[0], sum(lens)], dtype=torch.int32, device="cuda")
    psp = PackedSeqParams(
        qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=max(lens), max_seqlen_kv=max(lens)
    )
    stats = {"token_mean": 0.0, "token_max": 0.0, "vocab_max": 0.0, "padded_vocab_max": -float("inf")}
    with torch.no_grad():
        packed = torch.log_softmax(model(torch.cat(ids).view(1, -1), None, None, packed_seq_params=psp).float(), -1)[0]
        batch = torch.stack([ids[1][:30], ids[1][5:35]])
        unpacked = torch.log_softmax(model(batch, None, None).float(), -1)
        pairs = [(packed[cu[k] : cu[k + 1]], ids[k]) for k in range(2)]
        pairs += [(unpacked[k], batch[k]) for k in range(2)]
        for meg, x in pairs:
            ref_lp = ref_logprobs(x)
            tgt = x[1:, None]
            d = (meg[:-1].gather(-1, tgt) - ref_lp[:-1].gather(-1, tgt)).abs()
            stats["token_mean"] = max(stats["token_mean"], d.mean().item())
            stats["token_max"] = max(stats["token_max"], d.max().item())
            stats["vocab_max"] = max(
                stats["vocab_max"], (meg[:, :UNPADDED_V] - ref_lp[:, :UNPADDED_V]).abs().max().item()
            )
            stats["padded_vocab_max"] = max(stats["padded_vocab_max"], meg[:, UNPADDED_V:].max().item())

    # Backward through the packed forward (FlexAttention's captured relative bias included).
    logits = model(torch.cat(ids).view(1, -1), None, None, packed_seq_params=psp)
    torch.log_softmax(logits.float(), -1)[0, :-1].gather(-1, torch.cat(ids)[1:, None]).sum().backward()
    grads = {n: p.grad for n, p in model.named_parameters() if any(k in n for k in ("rel_proj", "query", "sconv"))}
    stats["grads_finite_nonzero"] = all(
        g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads.values()
    )
    return stats


@pytest.mark.parametrize(
    "backend, token_mean, token_max",
    [("dense", 0.004, 0.02), ("flex", 0.005, 0.025)],
)
def test_megatron_matches_reference(ray_init_fixture, backend, token_mean, token_max):
    stats = ray.get(_parity.remote(backend))
    print(backend, stats)
    assert stats["token_mean"] < token_mean
    assert stats["token_max"] < token_max
    assert stats["padded_vocab_max"] < -1e20  # padded ids are masked out of the softmax
    assert stats["grads_finite_nonzero"]

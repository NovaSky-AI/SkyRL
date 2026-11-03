"""Checks megatron-bridge's FLOPs formula, as SkyRL feeds it, against independent counts.

- Dense GQA (Qwen2.5-1.5B geometry): ``torch.utils.flop_counter.FlopCounterMode`` around an HF
  forward + backward on the meta device. Per-token FLOPs are ``a + b * s``; ``a`` (projections, MLP,
  logits) must match bridge, and ``b`` must be twice bridge's because the counter costs the full
  ``s x s`` score matrix while bridge costs the causal half.
- GLM-5.3-Flash (KDA + DSA hybrid MoE with a k-pool indexer): a hand-written per-layer reference,
  since the counter can't see the TE / FLA / DSA kernels.

Run with:
uv run --isolated --extra dev --extra megatron -- pytest -s tests/backends/skyrl_train/workers/megatron/test_model_flops.py
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron.bridge", reason="requires the megatron extra")

pytestmark = pytest.mark.megatron

from skyrl.backends.skyrl_train.workers.megatron.model_flops import (  # noqa: E402
    full_recompute_fraction,
    logits_flops,
    model_flops,
)

_PROVIDER_DEFAULTS = dict(
    make_vocab_size_divisible_by=128,
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    mtp_num_layers=None,
    seq_length=4096,
    num_moe_experts=None,
    moe_ffn_hidden_size=None,
    moe_shared_expert_intermediate_size=None,
    moe_router_topk=1,
    moe_layer_freq=1,
    gated_linear_unit=True,
    multi_latent_attention=False,
    num_query_groups=None,
    is_hybrid_model=False,
    hybrid_layer_pattern=None,
    recompute_granularity=None,
)


def _provider(**fields):
    return SimpleNamespace(**{**_PROVIDER_DEFAULTS, **fields})


def _flops_per_token(provider, lengths):
    total = model_flops(provider, len(lengths), sum(lengths), sum(s * s for s in lengths))["flops"]
    return total / sum(lengths)


# ---------------------------------------------------------------------------------------------
# Dense GQA vs FlopCounterMode
# ---------------------------------------------------------------------------------------------

QWEN25_1P5B = dict(
    hidden_size=1536,
    num_hidden_layers=28,
    num_attention_heads=12,
    num_key_value_heads=2,
    intermediate_size=8960,
    vocab_size=151936,
)


def _counted_flops_per_token(seq_len):
    from torch.utils.flop_counter import FlopCounterMode
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(**QWEN25_1P5B, tie_word_embeddings=True, attn_implementation="eager")
    with torch.device("meta"):
        model = Qwen2ForCausalLM(config)
        input_ids = torch.zeros(1, seq_len, dtype=torch.long)
    with FlopCounterMode(display=False) as counter:
        model(input_ids=input_ids).logits.float().sum().backward()
    return counter.get_total_flops() / seq_len


def test_dense_gqa_matches_flop_counter():
    provider = _provider(
        hidden_size=QWEN25_1P5B["hidden_size"],
        num_layers=QWEN25_1P5B["num_hidden_layers"],
        num_attention_heads=QWEN25_1P5B["num_attention_heads"],
        num_query_groups=QWEN25_1P5B["num_key_value_heads"],
        kv_channels=QWEN25_1P5B["hidden_size"] // QWEN25_1P5B["num_attention_heads"],
        ffn_hidden_size=QWEN25_1P5B["intermediate_size"],
        vocab_size=QWEN25_1P5B["vocab_size"],
    )
    s1, s2 = 256, 512
    counted = [_counted_flops_per_token(s) for s in (s1, s2)]
    bridge = [_flops_per_token(provider, [s]) for s in (s1, s2)]

    def fit(values):
        slope = (values[1] - values[0]) / (s2 - s1)
        return values[0] - slope * s1, slope

    counted_linear, counted_quadratic = fit(counted)
    bridge_linear, bridge_quadratic = fit(bridge)
    assert bridge_linear == pytest.approx(counted_linear, rel=5e-3)
    assert bridge_quadratic == pytest.approx(counted_quadratic / 2, rel=5e-3)


def test_full_recompute_hfu():
    provider = _provider(
        hidden_size=1536,
        num_layers=28,
        num_attention_heads=12,
        num_query_groups=2,
        kv_channels=128,
        ffn_hidden_size=8960,
        vocab_size=151936,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
    )
    result = model_flops(provider, 1, 4096, 4096**2)
    decoder = result["flops"] - logits_flops(provider, 4096)
    assert result["hfu_flops"] == pytest.approx(result["flops"] + decoder / 3)

    assert "hfu_flops" not in model_flops(
        _provider(**{**vars(provider), "recompute_granularity": "selective"}), 1, 4096, 4096**2
    )
    block = _provider(**{**vars(provider), "recompute_method": "block", "recompute_num_layers": 7})
    assert full_recompute_fraction(block) == pytest.approx(0.25)


# ---------------------------------------------------------------------------------------------
# GLM-5.3-Flash vs a hand-written reference
# ---------------------------------------------------------------------------------------------

# zai-org/GLM-5.3-Flash text config.
GLM53 = dict(
    hidden_size=4096,
    num_attention_heads=64,
    intermediate_size=12288,
    moe_intermediate_size=2048,
    n_routed_experts=288,
    num_experts_per_tok=8,
    n_shared_experts=1,
    q_lora_rank=1536,
    kv_lora_rank=512,
    qk_head_dim=256,
    v_head_dim=256,
    index_n_heads=32,
    index_head_dim=128,
    index_topk=2048,
    index_kpool=4,
    linear_heads=64,
    linear_head_dim=128,
    linear_conv=4,
    vocab_size=154880,
)
GLM53_LAYER_TYPES = ["kda"] * 3 + ["dsa"]


def _glm_provider(layer_types, mlp_types, with_kimi_fields=True):
    g = GLM53
    fields = dict(
        hidden_size=g["hidden_size"],
        num_layers=len(layer_types),
        num_attention_heads=g["num_attention_heads"],
        num_query_groups=g["num_attention_heads"],
        kv_channels=g["hidden_size"] // g["num_attention_heads"],
        ffn_hidden_size=g["intermediate_size"],
        vocab_size=g["vocab_size"],
        num_moe_experts=g["n_routed_experts"],
        moe_ffn_hidden_size=g["moe_intermediate_size"],
        moe_shared_expert_intermediate_size=g["moe_intermediate_size"] * g["n_shared_experts"],
        moe_router_topk=g["num_experts_per_tok"],
        moe_layer_freq=[1 if t == "sparse" else 0 for t in mlp_types],
        multi_latent_attention=True,
        q_lora_rank=g["q_lora_rank"],
        kv_lora_rank=g["kv_lora_rank"],
        qk_head_dim=g["qk_head_dim"],
        qk_pos_emb_head_dim=0,
        v_head_dim=g["v_head_dim"],
        experimental_attention_variant="dsa",
        dsa_indexer_n_heads=g["index_n_heads"],
        dsa_indexer_head_dim=g["index_head_dim"],
        dsa_indexer_topk=g["index_topk"],
        dsa_indexer_topk_freq=1,
        dsa_indexer_skip_topk_offset=0,
        dsa_indexer_loss_coeff=0.0,
        dsa_indexer_kpool=g["index_kpool"],
        linear_attention_freq=[1 if t == "kda" else 0 for t in layer_types],
        linear_num_key_heads=g["linear_heads"],
        linear_num_value_heads=g["linear_heads"],
        linear_key_head_dim=g["linear_head_dim"],
        linear_value_head_dim=g["linear_head_dim"],
        linear_conv_kernel_dim=g["linear_conv"],
    )
    if with_kimi_fields:
        fields.update(
            kimi_kda_layers=tuple(i + 1 for i, t in enumerate(layer_types) if t == "kda"),
            kimi_linear_head_dim=g["linear_head_dim"],
            kimi_linear_num_heads=g["linear_heads"],
            kimi_linear_conv_kernel_size=g["linear_conv"],
        )
    return _provider(**fields)


def _glm_reference_per_token(layer_types, mlp_types, s):
    """fwd+bwd FLOPs per token of one length-``s`` sequence, layer by layer."""
    g = GLM53
    h, heads = g["hidden_size"], g["num_attention_heads"]
    n_kda = layer_types.count("kda")
    n_dsa = layer_types.count("dsa")
    n_moe = mlp_types.count("sparse")
    n_dense = len(mlp_types) - n_moe
    vocab = -(-g["vocab_size"] // 128) * 128

    # Per-token multiply-accumulates of the trained matmuls.
    mla_projections = (
        g["q_lora_rank"] * (h + heads * g["qk_head_dim"] + 1)
        + g["kv_lora_rank"] * (h + heads * (g["qk_head_dim"] + g["v_head_dim"]) + 1)
        + heads * g["v_head_dim"] * h
    )
    kh, kd = g["linear_heads"], g["linear_head_dim"]
    kda = (
        5 * h * kh * kd  # q, k, v, gate and output projections
        + h * kh  # beta
        + h * kd
        + kd * kh * kd  # low-rank forget gate
        + g["linear_conv"] * 3 * kh * kd  # short convolutions
        + 4 * kh * kd * kd  # delta-rule state update
    )
    mlp = (
        3
        * h
        * (
            n_dense * g["intermediate_size"]
            + n_moe * g["moe_intermediate_size"] * (g["num_experts_per_tok"] + g["n_shared_experts"])
        )
    )
    per_token = 6 * (n_dsa * mla_projections + n_kda * kda + mlp + h * vocab)

    # Attention over each query's context, averaged per token.
    dense_context = (s + 1) / 2
    topk = g["index_topk"]
    sparse_context = dense_context if s <= topk else topk - topk * (topk - 1) / (2 * s)
    per_token += 6 * n_dsa * heads * 2 * g["kv_lora_rank"] * sparse_context  # absorbed QK + AV
    # The indexer runs under no_grad (no auxiliary loss): projections and scoring, forward only.
    # Scoring covers the pooled keys, 1/kpool of the causal context.
    ih, ihd = g["index_n_heads"], g["index_head_dim"]
    per_token += 2 * n_dsa * (g["q_lora_rank"] * ih * ihd + h * ihd + h * ih)
    per_token += 2 * n_dsa * ih * (ihd + 1) * dense_context / g["index_kpool"]
    return per_token


GLM_VARIANTS = {
    # 8-layer slices of GLM-5.3-Flash used for the GPU validation runs, plus the full model.
    "mixed": (GLM53_LAYER_TYPES * 2, ["dense"] + ["sparse"] * 7),
    "all_kda": (["kda"] * 8, ["dense"] + ["sparse"] * 7),
    "all_dsa": (["dsa"] * 8, ["dense"] + ["sparse"] * 7),
    "full_model": (GLM53_LAYER_TYPES * 11 + ["kda"], ["dense"] * 3 + ["sparse"] * 42),
}


@pytest.mark.parametrize("variant", sorted(GLM_VARIANTS))
@pytest.mark.parametrize("seq_len", [2048, 8192, 32768])
def test_glm53_matches_reference(variant, seq_len):
    layer_types, mlp_types = GLM_VARIANTS[variant]
    provider = _glm_provider(layer_types, mlp_types)
    expected = _glm_reference_per_token(layer_types, mlp_types, seq_len)
    assert _flops_per_token(provider, [seq_len]) == pytest.approx(expected, rel=5e-3)


def test_glm53_without_kimi_fields_costs_kda_as_dsa():
    # Negative control: without the Kimi KDA fields, bridge costs every KDA layer as a DSA layer
    # (about 1.17x here; MoE and logits dilute it on an 8-layer slice).
    layer_types, mlp_types = GLM_VARIANTS["all_kda"]
    seq_len = 32768
    expected = _glm_reference_per_token(layer_types, mlp_types, seq_len)
    wrong = _flops_per_token(_glm_provider(layer_types, mlp_types, with_kimi_fields=False), [seq_len])
    assert wrong > 1.1 * expected


def test_model_flops_skipped_with_mtp():
    from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
        MegatronWorker,
    )

    worker = SimpleNamespace(provider=SimpleNamespace(mtp_num_layers=1), _mfu_mtp_warned=False)
    assert MegatronWorker.model_flops(worker, num_seqs=1, seqlen_sum=10, seqlen_squared_sum=100) is None


def test_train_compute_seconds_accumulate_and_reset(monkeypatch):
    from skyrl.backends.skyrl_train.workers.megatron import megatron_worker

    monkeypatch.setattr(megatron_worker.torch.cuda, "synchronize", lambda: None)
    clock = iter([0.0, 2.0, 10.0, 10.5])
    monkeypatch.setattr(megatron_worker.time, "perf_counter", lambda: next(clock))

    class _Worker:
        cfg = SimpleNamespace(log_mfu=True)
        _train_compute_seconds = 0.0

        @megatron_worker._counts_train_compute
        def forward_backward(self):
            return "status"

        @megatron_worker._counts_train_compute
        def optim_step(self):
            raise RuntimeError("step failed")

    worker = _Worker()
    assert worker.forward_backward() == "status"
    with pytest.raises(RuntimeError):
        worker.optim_step()
    # 2 s in forward_backward plus 0.5 s in the failed optim_step.
    pop = megatron_worker.MegatronPolicyWorkerBase.pop_train_compute_seconds
    assert pop(worker) == pytest.approx(2.5)
    assert pop(worker) == 0.0


def test_train_compute_seconds_off_without_log_mfu(monkeypatch):
    from skyrl.backends.skyrl_train.workers.megatron import megatron_worker

    def fail():
        raise AssertionError("timed although log_mfu is False")

    monkeypatch.setattr(megatron_worker.torch.cuda, "synchronize", fail)

    class _Worker:
        cfg = SimpleNamespace(log_mfu=False)
        _train_compute_seconds = 0.0

        @megatron_worker._counts_train_compute
        def forward_backward(self):
            return "status"

    worker = _Worker()
    assert worker.forward_backward() == "status"
    assert worker._train_compute_seconds == 0.0

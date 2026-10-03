from types import SimpleNamespace

import pytest

from skyrl.backends.skyrl_train.weight_sync.fp8 import resolve_fp8_spec
from skyrl.backends.skyrl_train.weight_sync.fp8.models import GLM5_FP8_SPEC

_LINEAR = (2048, 6144)


def test_glm5_spec_resolves_for_glm_moe_dsa():
    assert resolve_fp8_spec(SimpleNamespace(model_type="glm_moe_dsa")) is GLM5_FP8_SPEC
    assert not GLM5_FP8_SPEC.matches(SimpleNamespace(model_type="glm5_next"))


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.self_attn.q_a_proj.weight",
        "model.layers.0.self_attn.q_b_proj.weight",
        "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
        "model.layers.0.self_attn.kv_b_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.self_attn.indexer.wq_b.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.3.mlp.shared_experts.up_proj.weight",
        "model.layers.3.mlp.experts.17.gate_proj.weight",
        "model.layers.3.mlp.experts.255.down_proj.weight",
    ],
)
def test_glm5_linears_serialize_as_fp8(name):
    assert GLM5_FP8_SPEC.should_quantize(name, _LINEAR)


@pytest.mark.parametrize(
    "name",
    [
        # vLLM fuses wk + weights_proj into an unquantized wk_weights_proj.
        "model.layers.0.self_attn.indexer.wk.weight",
        "model.layers.0.self_attn.indexer.weights_proj.weight",
        # Router, embeddings, head and norms stay BF16.
        "model.layers.3.mlp.gate.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.layers.0.input_layernorm.weight",
    ],
)
def test_glm5_bf16_weights_stay_unquantized(name):
    assert not GLM5_FP8_SPEC.should_quantize(name, _LINEAR)


def test_glm5_spec_has_no_batched_experts_or_ignored_layers():
    assert GLM5_FP8_SPEC.moe_expert_spec("model.layers.3.mlp.experts.gate_up_proj") is None
    assert GLM5_FP8_SPEC.ignored_layers(SimpleNamespace(model_type="glm_moe_dsa")) == []


def test_glm5_mxfp8_wire_requires_32_aligned_reduction_dim():
    from skyrl.backends.skyrl_train.weight_sync.fp8.models.base import MXFP8

    name = "model.layers.3.mlp.experts.7.down_proj.weight"
    assert GLM5_FP8_SPEC.should_quantize(name, (6144, 2048), MXFP8)
    assert not GLM5_FP8_SPEC.should_quantize(name, (6144, 2050), MXFP8)


def test_glm5_mxfp8_wire_keeps_kv_b_proj_bf16():
    """vLLM's MLA folds kv_b_proj into W_UK_T / W_UV through a generic identity-GEMM dequant on
    MXFP8, which broke after a level-2 sleep; the MXFP8 wire sends it BF16 and has vLLM build it
    unquantized. The blockwise wire keeps it FP8."""
    from skyrl.backends.skyrl_train.weight_sync.fp8.models.base import (
        BLOCKWISE_FP8,
        MXFP8,
    )

    name = "model.layers.3.self_attn.kv_b_proj.weight"
    hf_config = SimpleNamespace(model_type="glm_moe_dsa")
    assert not GLM5_FP8_SPEC.should_quantize(name, (28672, 512), MXFP8)
    assert GLM5_FP8_SPEC.should_quantize(name, (28672, 512), BLOCKWISE_FP8)
    assert GLM5_FP8_SPEC.ignored_layers(hf_config, MXFP8) == ["re:.*self_attn\\.kv_b_proj"]
    assert GLM5_FP8_SPEC.ignored_layers(hf_config, BLOCKWISE_FP8) == []


@pytest.mark.vllm
def test_glm5_mxfp8_ignore_matches_vllm_modules():
    """vLLM's compressed-tensors matcher must skip kv_b_proj with the spec's pattern (else the wire
    sends BF16 into a quantized layer) and leave every other linear, incl. fused ones, quantized."""
    pytest.importorskip("vllm")
    from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
        should_ignore_layer,
    )

    from skyrl.backends.skyrl_train.weight_sync.fp8.models.base import MXFP8

    ignore = GLM5_FP8_SPEC.ignored_layers(SimpleNamespace(model_type="glm_moe_dsa"), MXFP8)
    # vLLM's GLM-5 (DeepSeek-V2/V3 family) packed modules.
    fused = {"fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"], "gate_up_proj": ["gate_proj", "up_proj"]}
    assert should_ignore_layer("model.layers.3.self_attn.kv_b_proj", ignore, fused)
    for name in (
        "model.layers.3.self_attn.q_b_proj",
        "model.layers.3.self_attn.o_proj",
        "model.layers.3.self_attn.fused_qkv_a_proj",
        "model.layers.3.self_attn.indexer.wq_b",
        "model.layers.0.mlp.gate_up_proj",
        "model.layers.3.mlp.shared_experts.down_proj",
    ):
        assert not should_ignore_layer(name, ignore, fused), name

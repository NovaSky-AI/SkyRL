from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.skyrl_train.weight_sync.fp8 import (
    BLOCKWISE_FP8,
    MXFP8,
    engine_exclude_list,
    iter_serialized_fp8_tensors,
    resolve_fp8_spec,
    resolve_serialized_fp8_config,
    resolve_user_provided_exclude_list,
)
from skyrl.backends.skyrl_train.weight_sync.fp8.models import GLM5_FP8_SPEC

_LINEAR = (2048, 6144)


def _glm5_config():
    """A small GLM-5 layout: one dense layer, then two MoE layers of four routed experts."""
    return SimpleNamespace(model_type="glm_moe_dsa", num_hidden_layers=3, first_k_dense_replace=1, n_routed_experts=4)


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


def test_glm5_spec_has_no_stacked_experts_or_base_exclusions():
    assert GLM5_FP8_SPEC.stacked_experts("model.layers.3.mlp.experts.gate_up_proj") is None
    assert GLM5_FP8_SPEC.base_exclude_list(SimpleNamespace(model_type="glm_moe_dsa")) == []


def test_glm5_fp8_modules_list_every_routed_expert_tensor():
    attention = [
        f"self_attn.{proj}"
        for proj in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj", "indexer.wq_b")
    ]
    dense_mlp = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    moe_mlp = [
        *(f"mlp.experts.{expert}.{proj}" for expert in range(4) for proj in ("gate_proj", "up_proj", "down_proj")),
        "mlp.shared_experts.gate_proj",
        "mlp.shared_experts.up_proj",
        "mlp.shared_experts.down_proj",
    ]

    assert GLM5_FP8_SPEC.fp8_modules(_glm5_config()) == [
        *(f"model.layers.0.{module}" for module in attention + dense_mlp),
        *(f"model.layers.{layer}.{module}" for layer in (1, 2) for module in attention + moe_mlp),
    ]


@pytest.mark.parametrize("wire_format", [BLOCKWISE_FP8, MXFP8])
def test_glm5_every_fp8_module_is_a_tensor_the_sender_quantizes(wire_format):
    for module in GLM5_FP8_SPEC.fp8_modules(_glm5_config(), wire_format):
        assert GLM5_FP8_SPEC.should_quantize(f"{module}.weight", _LINEAR, wire_format), module


def test_glm5_kv_b_proj_is_excludable_only_on_the_wire_that_sends_it_fp8():
    hf_config = _glm5_config()

    excluded = resolve_user_provided_exclude_list(GLM5_FP8_SPEC, hf_config, BLOCKWISE_FP8, ["*.kv_b_proj"])
    assert sorted(excluded) == [f"model.layers.{layer}.self_attn.kv_b_proj" for layer in range(3)]
    with pytest.raises(ValueError, match="'\\*.kv_b_proj' matches none"):
        resolve_user_provided_exclude_list(GLM5_FP8_SPEC, hf_config, MXFP8, ["*.kv_b_proj"])


def test_glm5_excludes_a_layers_routed_experts_as_vllms_expert_0_names():
    hf_config = _glm5_config()

    excluded = resolve_user_provided_exclude_list(GLM5_FP8_SPEC, hf_config, BLOCKWISE_FP8, ["*.layers.1.mlp.experts.*"])

    assert len(excluded) == 4 * 3
    assert engine_exclude_list(GLM5_FP8_SPEC, hf_config, BLOCKWISE_FP8, excluded) == [
        "model.layers.1.mlp.experts.0.gate_proj",
        "model.layers.1.mlp.experts.0.up_proj",
        "model.layers.1.mlp.experts.0.down_proj",
    ]


@pytest.mark.parametrize(
    "pattern",
    [
        "*.layers.1.mlp.experts.2.*",  # one expert
        "*.layers.1.mlp.experts.*.down_proj",  # one projection of every expert
        "*.gate_proj",  # every expert's gate_proj, along with the dense and shared-expert ones
    ],
)
def test_glm5_rejects_excluding_part_of_a_layers_routed_experts(pattern):
    with pytest.raises(ValueError, match="routed-expert tensors in 'model.layers.1.mlp.experts', but not"):
        resolve_user_provided_exclude_list(GLM5_FP8_SPEC, _glm5_config(), BLOCKWISE_FP8, [pattern])


def test_glm5_sender_keeps_excluded_expert_tensors_in_model_dtype():
    config = resolve_serialized_fp8_config(BLOCKWISE_FP8, _glm5_config(), ["*.layers.1.mlp.experts.*"])
    tensor = torch.randn(256, 256, dtype=torch.bfloat16)

    name = "model.layers.1.mlp.experts.2.up_proj.weight"
    excluded = list(iter_serialized_fp8_tensors(name, tensor, torch.bfloat16, config))
    assert [(n, t.dtype) for n, t in excluded] == [(name, torch.bfloat16)]

    other_layer = list(
        iter_serialized_fp8_tensors("model.layers.2.mlp.experts.2.up_proj.weight", tensor, torch.bfloat16, config)
    )
    assert other_layer[0][1].dtype == torch.float8_e4m3fn


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
    assert GLM5_FP8_SPEC.base_exclude_list(hf_config, MXFP8) == ["re:.*self_attn\\.kv_b_proj"]
    assert GLM5_FP8_SPEC.base_exclude_list(hf_config, BLOCKWISE_FP8) == []


@pytest.mark.vllm
def test_glm5_mxfp8_ignore_matches_vllm_modules():
    """vLLM's compressed-tensors matcher must skip kv_b_proj with the spec's pattern (else the wire
    sends BF16 into a quantized layer) and leave every other linear, incl. fused ones, quantized."""
    pytest.importorskip("vllm")
    from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
        should_ignore_layer,
    )

    from skyrl.backends.skyrl_train.weight_sync.fp8.models.base import MXFP8

    ignore = GLM5_FP8_SPEC.base_exclude_list(SimpleNamespace(model_type="glm_moe_dsa"), MXFP8)
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

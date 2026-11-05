"""SkyRL's FP8 engine exclude list for Qwen3.5 and GLM-5, run through vLLM's own matching.

At engine init vLLM maps every ignore-list name through the model's
``hf_to_vllm_mapper``, then decides each module from that list, expanding fused
modules through ``packed_modules_mapping``: ``Fp8Config`` uses
``is_layer_skipped`` (blockwise wire) and ``CompressedTensorsConfig`` uses
``should_ignore_layer`` (MXFP8 wire), checking a MoE layer through expert 0's
projections. These tests drive those functions on the module names vLLM builds,
so a name SkyRL emits that vLLM would not honor fails here. They also pin that
vLLM itself rejects excluding only part of a fused module, which SkyRL leaves to
it, and that it does not reject excluding only part of a layer's routed experts,
which SkyRL therefore rejects.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional

import pytest

pytest.importorskip("vllm")

pytestmark = pytest.mark.vllm

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E402
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (  # noqa: E402
    should_ignore_layer,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config  # noqa: E402
from vllm.model_executor.layers.quantization.utils.quant_utils import (  # noqa: E402
    is_layer_skipped,
)
from vllm.model_executor.models.deepseek_v2 import (  # noqa: E402
    GlmMoeDsaForCausalLM,
)
from vllm.model_executor.models.qwen3_5 import (  # noqa: E402
    Qwen3_5MoeForConditionalGeneration,
)

from skyrl.backends.skyrl_train.weight_sync.fp8 import (  # noqa: E402
    BLOCKWISE_FP8,
    MXFP8,
    ModelFp8Spec,
    engine_exclude_list,
    get_serialized_fp8_quantization_config,
    resolve_user_provided_exclude_list,
)
from skyrl.backends.skyrl_train.weight_sync.fp8.models import (  # noqa: E402
    GLM5_FP8_SPEC,
    QWEN35_FP8_SPEC,
)


@dataclass(frozen=True)
class _Model:
    spec: ModelFp8Spec
    hf_config: Any
    packed_modules_mapping: dict
    hf_to_vllm_mapper: Optional[Any] = None


QWEN35 = _Model(
    spec=QWEN35_FP8_SPEC,
    hf_config=SimpleNamespace(
        model_type="qwen3_5_moe",
        text_config=SimpleNamespace(
            model_type="qwen3_5_moe_text",
            layer_types=["linear_attention"] * 3 + ["full_attention"] * 2,
            shared_expert_intermediate_size=512,
        ),
    ),
    packed_modules_mapping=Qwen3_5MoeForConditionalGeneration.packed_modules_mapping,
    hf_to_vllm_mapper=Qwen3_5MoeForConditionalGeneration.hf_to_vllm_mapper,
)
GLM5 = _Model(
    spec=GLM5_FP8_SPEC,
    # One dense layer, then two MoE layers of four routed experts.
    hf_config=SimpleNamespace(
        model_type="glm_moe_dsa", num_hidden_layers=3, first_k_dense_replace=1, n_routed_experts=4
    ),
    # GlmMoeDsaForCausalLM.__init__ adds fused_qkv_a_proj for configs with a q_lora_rank, as GLM-5's.
    packed_modules_mapping={
        **GlmMoeDsaForCausalLM.packed_modules_mapping,
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
    },
)

QWEN = "language_model.model.layers"
QWEN35_EXCLUSIONS = [
    "*.layers.0.mlp.experts.*",
    "*.layers.1.mlp.shared_expert.*",
    "*.layers.2.linear_attn.out_proj",
    "*.layers.3.*",
]
# vLLM module name -> expected to be built unquantized, for the exclusions above.
QWEN35_EXPECTED_UNQUANTIZED = {
    # layer 0 (GDN): only the routed experts are excluded
    f"{QWEN}.0.mlp.experts": True,
    f"{QWEN}.0.linear_attn.in_proj_qkvz": False,
    f"{QWEN}.0.linear_attn.in_proj_ba": True,  # the spec's own ignore list
    f"{QWEN}.0.mlp.shared_expert.gate_up_proj": False,
    # layer 1 (GDN): only the shared expert
    f"{QWEN}.1.mlp.shared_expert.gate_up_proj": True,
    f"{QWEN}.1.mlp.shared_expert.down_proj": True,
    f"{QWEN}.1.mlp.experts": False,
    # layer 2 (GDN): only out_proj
    f"{QWEN}.2.linear_attn.out_proj": True,
    f"{QWEN}.2.linear_attn.in_proj_qkvz": False,
    # layer 3 (full attention): everything
    f"{QWEN}.3.self_attn.qkv_proj": True,
    f"{QWEN}.3.self_attn.o_proj": True,
    f"{QWEN}.3.mlp.experts": True,
    f"{QWEN}.3.mlp.shared_expert.gate_up_proj": True,
    f"{QWEN}.3.mlp.shared_expert.down_proj": True,
    # layer 4 (full attention): nothing
    f"{QWEN}.4.self_attn.qkv_proj": False,
    f"{QWEN}.4.mlp.experts": False,
}

GLM5_EXCLUSIONS = ["*.layers.0.mlp.*", "*.layers.1.mlp.experts.*", "*.layers.2.self_attn.*"]
# kv_b_proj outside layer 2 is left out: the MXFP8 base list keeps it unquantized everywhere.
GLM5_EXPECTED_UNQUANTIZED = {
    # layer 0 (dense): only the MLP
    "model.layers.0.mlp.gate_up_proj": True,
    "model.layers.0.mlp.down_proj": True,
    "model.layers.0.self_attn.fused_qkv_a_proj": False,
    "model.layers.0.self_attn.o_proj": False,
    # layer 1 (MoE): only the routed experts
    "model.layers.1.mlp.experts": True,
    "model.layers.1.mlp.shared_experts.gate_up_proj": False,
    "model.layers.1.mlp.shared_experts.down_proj": False,
    "model.layers.1.self_attn.q_b_proj": False,
    # layer 2 (MoE): only the attention
    "model.layers.2.self_attn.fused_qkv_a_proj": True,
    "model.layers.2.self_attn.q_b_proj": True,
    "model.layers.2.self_attn.kv_b_proj": True,
    "model.layers.2.self_attn.o_proj": True,
    "model.layers.2.self_attn.indexer.wq_b": True,
    "model.layers.2.mlp.experts": False,
    "model.layers.2.mlp.shared_experts.gate_up_proj": False,
}

WIRE_FORMATS = pytest.mark.parametrize("wire_format", [BLOCKWISE_FP8, MXFP8])


def _vllm_unquantized(model, wire_format, exclude_list):
    """vLLM's decision for an engine exclude list: vLLM module name -> built unquantized?"""
    hf_quant_config = get_serialized_fp8_quantization_config(exclude_list=exclude_list, wire_format=wire_format)
    config_cls = Fp8Config if wire_format == BLOCKWISE_FP8 else CompressedTensorsConfig
    quant_config = config_cls.from_config(hf_quant_config)
    if model.hf_to_vllm_mapper is not None:
        quant_config.apply_vllm_mapper(model.hf_to_vllm_mapper)
    fused = model.packed_modules_mapping

    def unquantized(module):
        if wire_format == BLOCKWISE_FP8:
            return is_layer_skipped(
                module,
                quant_config.ignored_layers,
                fused_mapping=fused,
                match_mode=quant_config.ignored_layers_match_mode,
            )
        if module.endswith(".experts"):
            # CompressedTensorsMoEMethod.get_moe_method checks expert 0's projections.
            names = [f"{module}.0.{proj}" for proj in ("gate_proj", "up_proj", "down_proj")]
            return all(should_ignore_layer(name, quant_config.ignore, fused) for name in names)
        return should_ignore_layer(module, quant_config.ignore, fused)

    return unquantized


def _unquantized_for_patterns(model, wire_format, patterns):
    excluded = resolve_user_provided_exclude_list(model.spec, model.hf_config, wire_format, patterns)
    exclude_list = engine_exclude_list(model.spec, model.hf_config, wire_format, excluded)
    return _vllm_unquantized(model, wire_format, exclude_list)


@WIRE_FORMATS
@pytest.mark.parametrize(
    "model, patterns, expected",
    [
        (QWEN35, QWEN35_EXCLUSIONS, QWEN35_EXPECTED_UNQUANTIZED),
        (GLM5, GLM5_EXCLUSIONS, GLM5_EXPECTED_UNQUANTIZED),
    ],
    ids=["qwen3.5", "glm5"],
)
def test_engine_builds_exactly_the_excluded_modules_unquantized(wire_format, model, patterns, expected):
    unquantized = _unquantized_for_patterns(model, wire_format, patterns)

    assert {module: unquantized(module) for module in expected} == expected


@WIRE_FORMATS
@pytest.mark.parametrize(
    "model, pattern, fused_module",
    [
        (QWEN35, "*.layers.3.self_attn.q_proj", f"{QWEN}.3.self_attn.qkv_proj"),
        (QWEN35, "*.layers.0.linear_attn.in_proj_z", f"{QWEN}.0.linear_attn.in_proj_qkvz"),
        (QWEN35, "*.layers.0.mlp.shared_expert.up_proj", f"{QWEN}.0.mlp.shared_expert.gate_up_proj"),
        (GLM5, "*.layers.0.self_attn.q_a_proj", "model.layers.0.self_attn.fused_qkv_a_proj"),
        (GLM5, "*.layers.0.mlp.gate_proj", "model.layers.0.mlp.gate_up_proj"),
        (GLM5, "*.layers.1.mlp.shared_experts.up_proj", "model.layers.1.mlp.shared_experts.gate_up_proj"),
    ],
)
def test_engine_rejects_excluding_part_of_a_fused_module(wire_format, model, pattern, fused_module):
    unquantized = _unquantized_for_patterns(model, wire_format, [pattern])

    with pytest.raises(ValueError, match="shards of"):
        unquantized(fused_module)


@pytest.mark.parametrize("wire_format, layer_unquantized", [(BLOCKWISE_FP8, True), (MXFP8, False)])
def test_vllm_accepts_excluding_part_of_a_layers_routed_experts(wire_format, layer_unquantized):
    """Listing one expert's projections passes vLLM without an error, but it builds the whole
    layer at one precision: unquantized on blockwise (the other experts arrive FP8), quantized
    on MXFP8 (this expert arrives BF16). resolve_user_provided_exclude_list rejects the split."""
    one_expert = [f"model.layers.1.mlp.experts.2.{proj}" for proj in ("gate_proj", "up_proj", "down_proj")]

    unquantized = _vllm_unquantized(GLM5, wire_format, one_expert)

    assert unquantized("model.layers.1.mlp.experts") is layer_unquantized
    with pytest.raises(ValueError, match="routed-expert tensors"):
        resolve_user_provided_exclude_list(GLM5.spec, GLM5.hf_config, wire_format, ["*.layers.1.mlp.experts.2.*"])

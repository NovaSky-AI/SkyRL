"""SkyRL's FP8 engine exclude list for Qwen3.5, run through vLLM's own matching.

At engine init vLLM maps every ignore-list name through the model's
``hf_to_vllm_mapper``, then decides each module from that list, expanding fused
modules through ``packed_modules_mapping``: ``Fp8Config`` uses
``is_layer_skipped`` (blockwise wire) and ``CompressedTensorsConfig`` uses
``should_ignore_layer`` (MXFP8 wire), checking a MoE layer through expert 0's
projections. These tests drive those functions on the module names vLLM builds,
so a name SkyRL emits that vLLM would not honor fails here. They also pin that
vLLM itself rejects excluding only part of a fused module, which SkyRL leaves to it.
"""

from types import SimpleNamespace

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
from vllm.model_executor.models.qwen3_5 import (  # noqa: E402
    Qwen3_5MoeForConditionalGeneration,
)

from skyrl.backends.skyrl_train.weight_sync.fp8 import (  # noqa: E402
    BLOCKWISE_FP8,
    MXFP8,
    engine_exclude_list,
    get_serialized_fp8_quantization_config,
    resolve_user_provided_exclude_list,
)
from skyrl.backends.skyrl_train.weight_sync.fp8.models import (  # noqa: E402
    QWEN35_FP8_SPEC,
)

VLLM = "language_model.model.layers"

# vLLM module name -> expected to be built unquantized, for the exclusions below.
EXPECTED_UNQUANTIZED = {
    # layer 0 (GDN): only the routed experts are excluded
    f"{VLLM}.0.mlp.experts": True,
    f"{VLLM}.0.linear_attn.in_proj_qkvz": False,
    f"{VLLM}.0.linear_attn.in_proj_ba": True,  # the spec's own ignore list
    f"{VLLM}.0.mlp.shared_expert.gate_up_proj": False,
    # layer 1 (GDN): only the shared expert
    f"{VLLM}.1.mlp.shared_expert.gate_up_proj": True,
    f"{VLLM}.1.mlp.shared_expert.down_proj": True,
    f"{VLLM}.1.mlp.experts": False,
    # layer 2 (GDN): only out_proj
    f"{VLLM}.2.linear_attn.out_proj": True,
    f"{VLLM}.2.linear_attn.in_proj_qkvz": False,
    # layer 3 (full attention): everything
    f"{VLLM}.3.self_attn.qkv_proj": True,
    f"{VLLM}.3.self_attn.o_proj": True,
    f"{VLLM}.3.mlp.experts": True,
    f"{VLLM}.3.mlp.shared_expert.gate_up_proj": True,
    f"{VLLM}.3.mlp.shared_expert.down_proj": True,
    # layer 4 (full attention): nothing
    f"{VLLM}.4.self_attn.qkv_proj": False,
    f"{VLLM}.4.mlp.experts": False,
}


MODEL = Qwen3_5MoeForConditionalGeneration
HF_CONFIG = SimpleNamespace(
    model_type="qwen3_5_moe",
    text_config=SimpleNamespace(
        model_type="qwen3_5_moe_text",
        layer_types=["linear_attention"] * 3 + ["full_attention"] * 2,
        shared_expert_intermediate_size=512,
    ),
)
EXCLUSIONS = [
    "*.layers.0.mlp.experts",
    "*.layers.1.mlp.shared_expert.*",
    "*.layers.2.linear_attn.out_proj",
    "*.layers.3.*",
]


def _engine_exclude_list(wire_format, patterns):
    user_provided_exclude_list = resolve_user_provided_exclude_list(QWEN35_FP8_SPEC, HF_CONFIG, patterns)
    return engine_exclude_list(QWEN35_FP8_SPEC, HF_CONFIG, wire_format, user_provided_exclude_list)


def _blockwise_unquantized(patterns):
    """vLLM's blockwise decision: vLLM module name -> built unquantized?"""
    quant_config = Fp8Config.from_config(
        get_serialized_fp8_quantization_config(
            exclude_list=_engine_exclude_list(BLOCKWISE_FP8, patterns), wire_format=BLOCKWISE_FP8
        )
    )
    quant_config.apply_vllm_mapper(MODEL.hf_to_vllm_mapper)

    def unquantized(module):
        return is_layer_skipped(
            module,
            quant_config.ignored_layers,
            fused_mapping=MODEL.packed_modules_mapping,
            match_mode=quant_config.ignored_layers_match_mode,
        )

    return unquantized


def _mxfp8_unquantized(patterns):
    """vLLM's MXFP8 decision: vLLM module name -> built unquantized?"""
    quant_config = CompressedTensorsConfig.from_config(
        get_serialized_fp8_quantization_config(exclude_list=_engine_exclude_list(MXFP8, patterns), wire_format=MXFP8)
    )
    quant_config.apply_vllm_mapper(MODEL.hf_to_vllm_mapper)

    def unquantized(module):
        if module.endswith(".experts"):
            # CompressedTensorsMoEMethod.get_moe_method checks expert 0's projections.
            names = [f"{module}.0.{proj}" for proj in ("gate_proj", "up_proj", "down_proj")]
            return all(should_ignore_layer(name, quant_config.ignore, MODEL.packed_modules_mapping) for name in names)
        return should_ignore_layer(module, quant_config.ignore, MODEL.packed_modules_mapping)

    return unquantized


@pytest.mark.parametrize("unquantized_for", [_blockwise_unquantized, _mxfp8_unquantized], ids=["blockwise", "mxfp8"])
def test_engine_builds_exactly_the_excluded_modules_unquantized(unquantized_for):
    unquantized = unquantized_for(EXCLUSIONS)

    assert {module: unquantized(module) for module in EXPECTED_UNQUANTIZED} == EXPECTED_UNQUANTIZED


@pytest.mark.parametrize("unquantized_for", [_blockwise_unquantized, _mxfp8_unquantized], ids=["blockwise", "mxfp8"])
@pytest.mark.parametrize(
    "pattern, fused_module",
    [
        ("*.layers.3.self_attn.q_proj", f"{VLLM}.3.self_attn.qkv_proj"),
        ("*.layers.0.linear_attn.in_proj_z", f"{VLLM}.0.linear_attn.in_proj_qkvz"),
        ("*.layers.0.mlp.shared_expert.up_proj", f"{VLLM}.0.mlp.shared_expert.gate_up_proj"),
    ],
)
def test_engine_rejects_excluding_part_of_a_fused_module(unquantized_for, pattern, fused_module):
    unquantized = unquantized_for([pattern])

    with pytest.raises(ValueError, match="shards of"):
        unquantized(fused_module)

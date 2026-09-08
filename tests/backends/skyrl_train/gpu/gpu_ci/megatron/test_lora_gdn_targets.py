"""GDN LoRA target safety check against the real Megatron-Bridge classes.

Pins the two facts the ``all-linear`` remap in ``configure_lora`` relies on:

1. The bridges' GDN ``in_proj`` mappings keep their hf_param layouts
   (four separate tensors for Qwen3.5's ``GDNLinearMappingSeparate``, two
   fused tensors for Qwen3-Next's ``GDNLinearMapping``), and
   ``gdn_in_proj_lora_is_safe`` classifies them accordingly.
2. peft_bridge still cannot handle a fused-layout (Qwen3-Next) ``in_proj``
   adapter: the merged export shape-mismatches and the unmerged export has no
   fused-adapter split for it. If these assertions start failing after a
   Megatron-Bridge bump, the bridge learned the fused layout and ``in_proj``
   can stop being removed for it in ``configure_lora``.

Run with:
uv run --isolated --extra dev --extra megatron -- pytest -s tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_lora_gdn_targets.py
"""

from types import SimpleNamespace

import pytest
import torch
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.param_mapping import (
    GDNLinearMapping,
    GDNLinearMappingSeparate,
)
from megatron.bridge.models.conversion.peft_bridge import (
    AdapterWeight,
    MegatronPeftBridge,
)

from skyrl.backends.skyrl_train.distributed.megatron.megatron_utils import (
    gdn_in_proj_lora_is_safe,
)

pytestmark = pytest.mark.megatron

HIDDEN = 64
RANK = 8
# Tiny GDN geometry: qk_dim = 2 heads * 16, v_dim = 4 heads * 16.
QK_DIM, V_DIM, NUM_V_HEADS = 32, 64, 4
FUSED_DIM = 2 * QK_DIM + 2 * V_DIM + 2 * NUM_V_HEADS  # Megatron in_proj rows
QKVZ_DIM = 2 * QK_DIM + 2 * V_DIM  # HF in_proj_qkvz rows
BA_DIM = 2 * NUM_V_HEADS  # HF in_proj_ba rows

QWEN3_NEXT_NAMES = [
    "model.layers.0.linear_attn.in_proj_qkvz.weight",
    "model.layers.0.linear_attn.in_proj_ba.weight",
]
QWEN35_NAMES = [
    "model.layers.0.linear_attn.in_proj_qkv.weight",
    "model.layers.0.linear_attn.in_proj_z.weight",
    "model.layers.0.linear_attn.in_proj_b.weight",
    "model.layers.0.linear_attn.in_proj_a.weight",
]


def _bridge_with_registry(registry: MegatronMappingRegistry):
    # `gdn_in_proj_lora_is_safe` installs the bridge's `hf_pretrained` onto the
    # model bridge before the lookup, for the bridges whose `mapping_registry`
    # probes the checkpoint. These registries are built directly and never read it.
    return SimpleNamespace(
        hf_pretrained=None,
        _model_bridge=SimpleNamespace(mapping_registry=lambda: registry),
    )


def _fused_in_proj_adapter() -> AdapterWeight:
    return AdapterWeight(
        global_base_prefix="decoder.layers.0.self_attention.in_proj",
        adapter_key=None,
        alpha=16,
        dim=RANK,
        linear_in_weight=SimpleNamespace(weight=torch.randn(RANK, HIDDEN)),
        linear_out_weight=SimpleNamespace(weight=torch.randn(FUSED_DIM, RANK)),
    )


def test_in_proj_lora_unsafe_for_qwen3_next_style_registry():
    registry = MegatronMappingRegistry(
        GDNLinearMapping(
            megatron_param="decoder.layers.*.self_attention.in_proj.weight",
            qkvz="model.layers.*.linear_attn.in_proj_qkvz.weight",
            ba="model.layers.*.linear_attn.in_proj_ba.weight",
        ),
    )
    assert not gdn_in_proj_lora_is_safe(_bridge_with_registry(registry))


def test_in_proj_lora_safe_for_qwen35_style_registry():
    registry = MegatronMappingRegistry(
        GDNLinearMappingSeparate(
            megatron_param="decoder.layers.*.self_attention.in_proj.weight",
            qkv="model.layers.*.linear_attn.in_proj_qkv.weight",
            z="model.layers.*.linear_attn.in_proj_z.weight",
            b="model.layers.*.linear_attn.in_proj_b.weight",
            a="model.layers.*.linear_attn.in_proj_a.weight",
        ),
    )
    assert gdn_in_proj_lora_is_safe(_bridge_with_registry(registry))


def test_in_proj_lora_safe_for_non_gdn_registry():
    # No GDN layers: in_proj matches nothing, so keeping it in the target
    # list is harmless.
    assert gdn_in_proj_lora_is_safe(_bridge_with_registry(MegatronMappingRegistry()))


def test_peft_bridge_recognizes_only_the_separate_gdn_layout():
    pb = MegatronPeftBridge()
    assert pb._is_gdn_in_proj_split(QWEN35_NAMES)
    assert not pb._is_gdn_in_proj_split(QWEN3_NEXT_NAMES)


def test_fused_gdn_in_proj_adapter_has_no_export_split():
    # With no fused-adapter split for the qkvz/ba layout, the unmerged export
    # generator falls through to yielding the full fused lora_B under the
    # first HF name only, dropping the in_proj_ba contribution.
    pb = MegatronPeftBridge()
    adapter = _fused_in_proj_adapter()
    megatron_model = [SimpleNamespace(config=SimpleNamespace(num_moe_experts=None))]
    slices = pb._get_fused_adapter_linear_out_slices(megatron_model, QWEN3_NEXT_NAMES, adapter.linear_out_weight.weight)
    assert slices is None


def test_fused_gdn_in_proj_adapter_merge_shape_mismatches():
    pb = MegatronPeftBridge()
    adapter = _fused_in_proj_adapter()
    megatron_model = [SimpleNamespace(config=SimpleNamespace(num_moe_experts=None))]
    converted = {
        QWEN3_NEXT_NAMES[0]: torch.zeros(QKVZ_DIM, HIDDEN),
        QWEN3_NEXT_NAMES[1]: torch.zeros(BA_DIM, HIDDEN),
    }
    with pytest.raises(RuntimeError):
        pb._merge_lora_adapter_weights(megatron_model, converted, [adapter])

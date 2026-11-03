"""DeepSeek-V4.1 load / export use the released checkpoint's layout (32x32 FP8 blocks, MXFP4 experts).

V4's hooks assume 128x128 FP8 blocks: the export rejected V4.1's scales (``Unsupported FP8 scale
geometry``), and the load silently applied the wrong scales.
"""

from dataclasses import dataclass
from typing import Optional

import pytest
import torch

pytest.importorskip("megatron.bridge")

from megatron.bridge.models.conversion import quantization_utils  # noqa: E402

from skyrl.backends.skyrl_train.patches.megatron.deepseek_v41.bridge import (  # noqa: E402
    DeepSeekV41Bridge,
)

E8M0 = getattr(torch, "float8_e8m0fnu", None)


@dataclass(frozen=True)
class Task:
    """The one ``WeightConversionTask`` field the hook reads (the real task is a frozen dataclass too)."""

    weight_dtype: Optional[torch.dtype] = None


@pytest.mark.skipif(E8M0 is None, reason="needs torch.float8_e8m0fnu")
def test_v41_export_uses_32_wide_fp8_blocks_and_mxfp4_experts():
    torch.manual_seed(0)
    bridge = object.__new__(DeepSeekV41Bridge)
    attn = torch.randn(512, 1024)  # layers.N.attn.wkv: real (512, 5120) with a (16, 160) scale
    expert = torch.randn(256, 1024)  # routed expert: (rows, cols / 32) MXFP4 scale
    hf_state_dict = {
        "layers.0.attn.wkv.scale": torch.ones(16, 32).to(E8M0),
        "layers.0.ffn.experts.0.w1.scale": torch.ones(256, 32).to(E8M0),
    }
    converted = {
        "layers.0.attn.wkv.weight": attn,
        "layers.0.ffn.experts.0.w1.weight": expert,
        "layers.0.attn_norm.weight": torch.ones(8),
    }

    out = bridge.maybe_modify_converted_hf_weight(Task(), converted, hf_state_dict)

    assert out["layers.0.attn.wkv.weight"].dtype == torch.float8_e4m3fn
    assert out["layers.0.attn.wkv.scale"].shape == (16, 32)
    scale = out["layers.0.attn.wkv.scale"].float()
    deq = out["layers.0.attn.wkv.weight"].float() * scale.repeat_interleave(32, 0).repeat_interleave(32, 1)
    assert (deq - attn).abs().max() / attn.abs().max() < 0.07  # E4M3 with power-of-two block scales
    assert out["layers.0.ffn.experts.0.w1.scale"].shape == (256, 32)
    deq = quantization_utils.maybe_dequantize_hf_quantized_weight(
        "layers.0.ffn.experts.0.w1.weight", out, dtype=torch.float32
    )
    assert (deq - expert).abs().max() / expert.abs().max() < 0.15  # FP4 E2M1
    assert torch.equal(out["layers.0.attn_norm.weight"], torch.ones(8))

    # With an export dtype set, V4's hook returns the weights unquantized.
    plain = bridge.maybe_modify_converted_hf_weight(Task(torch.bfloat16), converted, hf_state_dict)
    assert plain["layers.0.attn.wkv.weight"] is attn


@pytest.mark.skipif(E8M0 is None, reason="needs torch.float8_e8m0fnu")
def test_v41_load_dequantizes_fp8_in_32_wide_blocks():
    torch.manual_seed(0)
    bridge = object.__new__(DeepSeekV41Bridge)
    # Distinct power-of-two scale per 32x32 block, so a 128-wide decode is visibly wrong.
    scale = (2.0 ** torch.randint(-4, 4, (8, 8)).float()).to(E8M0)
    weight = torch.randn(256, 256).clamp(-2, 2).to(torch.float8_e4m3fn)
    state = {"layers.0.attn.wq_b.weight": weight, "layers.0.attn.wq_b.scale": scale}

    out = bridge.maybe_modify_loaded_hf_weight("layers.0.attn.wq_b.weight", state)

    expected = weight.float() * scale.float().repeat_interleave(32, 0).repeat_interleave(32, 1)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), expected, rtol=1e-2, atol=1e-2)
    # A dict hf_param (gated MLP) decodes each entry.
    both = bridge.maybe_modify_loaded_hf_weight({"gate": "layers.0.attn.wq_b.weight"}, state)
    torch.testing.assert_close(both["gate"], out)

"""DeepSeek-V4.1 load dequantizes FP8 weights in the released checkpoint's 32x32 blocks.

V4's load hook assumes 128x128 blocks; on V4.1's scales it does not raise, it applies the wrong scales.
"""

import pytest
import torch

pytest.importorskip("megatron.bridge")

from skyrl.backends.skyrl_train.patches.megatron.deepseek_v41.bridge import (  # noqa: E402
    DeepSeekV41Bridge,
)

E8M0 = getattr(torch, "float8_e8m0fnu", None)


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

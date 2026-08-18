import pytest
import torch

pytest.importorskip("megatron.bridge")

from skyrl.backends.skyrl_train.workers.megatron.model_bridges import GLM5FP8Bridge


def test_glm5_fp8_weight_uses_block_scales():
    weight = torch.ones((256, 256), dtype=torch.float8_e4m3fn)
    scale = torch.tensor([[0.5, 1.0], [2.0, 4.0]])

    result = GLM5FP8Bridge._maybe_dequantize_fp8(
        weight,
        "model.weight",
        {
            "model.weight": weight,
            "model.weight_scale_inv": scale,
        },
    )

    expected = scale.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1).to(torch.bfloat16)
    assert result.dtype == torch.bfloat16
    assert torch.equal(result, expected)


def test_glm5_fp8_weight_requires_scale():
    weight = torch.ones((128, 128), dtype=torch.float8_e4m3fn)

    with pytest.raises(KeyError, match="model.weight_scale_inv"):
        GLM5FP8Bridge._maybe_dequantize_fp8(weight, "model.weight", {"model.weight": weight})

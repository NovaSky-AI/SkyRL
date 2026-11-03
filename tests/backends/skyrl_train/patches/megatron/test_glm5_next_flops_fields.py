"""``glm5_next`` provider: the KDA layout fields megatron-bridge's FLOPs formula reads.

uv run --isolated --extra dev --extra megatron -- pytest tests/backends/skyrl_train/patches/megatron/test_glm5_next_flops_fields.py
"""

import pytest

pytest.importorskip("megatron.bridge", reason="requires the megatron extra")

pytestmark = pytest.mark.megatron


def test_glm53_provider_has_kimi_fields():
    from skyrl.backends.skyrl_train.patches.megatron.glm5_next.provider import (
        Glm5NextModelProvider,
    )

    fields = Glm5NextModelProvider.__dataclass_fields__
    for name in ("kimi_kda_layers", "kimi_linear_head_dim", "kimi_linear_num_heads", "kimi_linear_conv_kernel_size"):
        assert name in fields

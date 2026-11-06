"""``glm5_next`` bridge: the KDA layout fields megatron-bridge's FLOPs formula reads.

Builds the provider for a 4-layer GLM-5.3-Flash slice from its HF config (no weights).

uv run --isolated --extra dev --extra megatron -- pytest tests/backends/skyrl_train/patches/megatron/glm5_next/test_glm5_next_flops_fields.py
"""

import pytest

pytest.importorskip("megatron.bridge", reason="requires the megatron extra")

pytestmark = pytest.mark.megatron

GLM53_4LAYER = "eatang/GLM-5.3-Flash-4layer"


@pytest.fixture(scope="module")
def glm53_provider():
    from megatron.bridge import AutoBridge

    import skyrl.backends.skyrl_train.workers.megatron.model_bridges  # noqa: F401

    provider = AutoBridge.from_hf_pretrained(GLM53_4LAYER, trust_remote_code=True).to_megatron_provider(
        load_weights=False
    )
    provider.finalize()
    return provider


def test_bridge_sets_kda_layout_fields(glm53_provider):
    # kimi_kda_layers is 1-indexed and lists exactly the linear-attention layers.
    kda_layers = tuple(i + 1 for i, kind in enumerate(glm53_provider.linear_attention_freq) if kind == 1)
    assert kda_layers
    assert glm53_provider.kimi_kda_layers == kda_layers
    assert glm53_provider.kimi_linear_head_dim == glm53_provider.linear_key_head_dim
    assert glm53_provider.kimi_linear_num_heads == glm53_provider.linear_num_key_heads
    assert glm53_provider.kimi_linear_conv_kernel_size == glm53_provider.linear_conv_kernel_dim


def test_model_flops_costs_kda_layers_as_kda(glm53_provider):
    import copy

    from skyrl.backends.skyrl_train.workers.megatron.model_flops import model_flops

    stats = dict(num_seqs=1, seqlen_sum=32768, seqlen_squared_sum=32768**2)
    with_kda = model_flops(glm53_provider, **stats)["flops"]
    without_fields = copy.copy(glm53_provider)
    without_fields.kimi_kda_layers = ()
    without_kda = model_flops(without_fields, **stats)["flops"]
    # Without the fields, bridge costs every KDA layer as a DSA layer.
    assert with_kda < without_kda

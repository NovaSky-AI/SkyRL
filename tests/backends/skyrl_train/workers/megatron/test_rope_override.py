"""A RoPE override on the Megatron backend reaches the provider Megatron-Bridge builds, or fails loudly.

Each case builds a real provider the way ``MegatronWorker.init_configs`` does: ``AutoBridge.from_hf_pretrained``
on a checkpoint directory (here a tiny config saved to ``tmp_path``, no weights), ``apply_rope_override``,
``to_megatron_provider``, then ``check_rope_override``.

Run with:
uv run --isolated --extra dev --extra megatron -- pytest -s tests/backends/skyrl_train/workers/megatron/test_rope_override.py
"""

import pytest

pytest.importorskip("megatron.bridge", reason="requires the megatron extra")

pytestmark = pytest.mark.megatron

from megatron.bridge import AutoBridge  # noqa: E402
from transformers import DeepseekV3Config, LlamaConfig, Qwen2Config  # noqa: E402

from skyrl.backends.skyrl_train.workers.megatron.model_bridges import (  # noqa: E402
    apply_rope_override,
    check_rope_override,
)

YARN = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 4096, "rope_theta": 1_000_000.0}
LINEAR = {"rope_type": "linear", "factor": 2.0, "rope_theta": 1_000_000.0}
# Megatron Core applies llama3 scaling with fixed low/high frequency factors and original length; only the factor
# is free (Megatron-Bridge's llama_bridge.py rejects anything else).
LLAMA3 = {
    "rope_type": "llama3",
    "factor": 16.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
    "rope_theta": 500_000.0,
}
_TINY = dict(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2, max_position_embeddings=4096)


def _qwen2():
    return Qwen2Config(num_attention_heads=4, num_key_value_heads=2, architectures=["Qwen2ForCausalLM"], **_TINY)


def _llama():
    return LlamaConfig(num_attention_heads=4, num_key_value_heads=2, architectures=["LlamaForCausalLM"], **_TINY)


def _deepseek_v3():
    # MLA attention without query compression (q_lora_rank=None, like Moonlight) and a minimal MoE.
    return DeepseekV3Config(
        num_attention_heads=4, num_key_value_heads=4, q_lora_rank=None, kv_lora_rank=16, qk_rope_head_dim=8,
        qk_nope_head_dim=8, v_head_dim=8, n_routed_experts=4, num_experts_per_tok=2, n_shared_experts=1,
        moe_intermediate_size=32, first_k_dense_replace=1, n_group=1, topk_group=1,
        architectures=["DeepseekV3ForCausalLM"], **_TINY,
    )  # fmt: skip


def _bridge(config, tmp_path):
    # The call init_configs makes; weights load lazily and to_megatron_provider(load_weights=False) never asks.
    config.save_pretrained(tmp_path)
    return AutoBridge.from_hf_pretrained(tmp_path)


@pytest.mark.parametrize(
    "make_config, override, match",
    [
        pytest.param(_qwen2, {"rope_type": "default", "rope_theta": 500_000.0}, None, id="qwen2-theta"),
        pytest.param(_qwen2, YARN, None, id="qwen2-yarn"),
        pytest.param(_deepseek_v3, YARN, None, id="deepseek-v3-mla-yarn"),
        pytest.param(_llama, LINEAR, None, id="llama-linear"),
        pytest.param(_llama, LLAMA3, None, id="llama-llama3"),
        pytest.param(_qwen2, LINEAR, "Qwen2Bridge does not translate it", id="qwen2-linear-dropped"),
        pytest.param(_qwen2, {"rope_type": "dynamic", "factor": 2.0}, "'dynamic' is not supported", id="unknown-type"),
        pytest.param(_qwen2, {"rope_type": "yarn", "rope_theta": 1_000_000.0}, "needs a 'factor'", id="no-factor"),
    ],
)
def test_rope_override_reaches_the_provider_or_raises(tmp_path, make_config, override, match):
    bridge = _bridge(make_config(), tmp_path)
    apply_rope_override(bridge, override)
    provider = bridge.to_megatron_provider(load_weights=False)
    bridge_name = type(bridge._model_bridge).__name__

    if match is None:
        check_rope_override(provider, override, bridge_name)
    else:
        with pytest.raises(ValueError, match=match):
            check_rope_override(provider, override, bridge_name)


def test_theta_written_only_into_rope_parameters_loses_to_a_top_level_rope_theta(tmp_path):
    # A top-level rope_theta (remote-code configs keep one) beats a theta written only into rope_parameters.
    config = _qwen2()
    config.rope_theta = 10_000.0
    bridge = _bridge(config, tmp_path)
    bridge.hf_pretrained.config.rope_parameters = dict(YARN)
    with pytest.raises(ValueError, match="Qwen2Bridge does not translate it"):
        check_rope_override(bridge.to_megatron_provider(load_weights=False), YARN, "Qwen2Bridge")

    bridge = _bridge(config, tmp_path)
    apply_rope_override(bridge, YARN)
    check_rope_override(bridge.to_megatron_provider(load_weights=False), YARN, "Qwen2Bridge")

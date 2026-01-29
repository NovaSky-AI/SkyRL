import os
import tempfile

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig

from tx.models.configs import Glm4Config
from tx.models.glm4 import Glm4ForCausalLM, Glm4MoE
from tx.utils.models import load_safetensors


@pytest.mark.parametrize("tp", [1, 2])
def test_glm4_moe(tp: int):
    """Test GLM4-MoE model against HuggingFace implementation."""
    if not jax._src.xla_bridge.backends_are_initialized():
        jax.config.update("jax_num_cpu_devices", 2)

    if tp > 1 and os.getenv("CI"):
        pytest.skip("TP > 1 currently runs out of memory in the CI")

    model_name = "yujiepan/glm-4-moe-tiny-random"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, attn_implementation="eager", use_safetensors=True, trust_remote_code=True
    )

    inputs = ["The capital of France is", "The most popular programming language is"]
    batch = tokenizer(inputs, return_tensors="pt", padding=True)
    with torch.no_grad():
        hf_outputs = hf_model(
            batch.input_ids,
            attention_mask=batch.attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    # Save the HF model checkpoint so we can load our model from it
    with tempfile.TemporaryDirectory() as tmp:
        hf_model.save_pretrained(tmp, safe_serialization=True)

        base_config = PretrainedConfig.from_pretrained(model_name, trust_remote_code=True)
        config = Glm4Config(base_config, max_lora_adapters=32, max_lora_rank=32, shard_attention_heads=True)
        mesh = jax.make_mesh(
            (1, tp),
            ("fsdp", "tp"),
            axis_types=(jax.sharding.AxisType.Auto, jax.sharding.AxisType.Auto),
        )
        with jax.set_mesh(mesh):
            model = Glm4ForCausalLM(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
        load_safetensors(tmp, config, model)

        outputs = model(batch.input_ids.numpy(), attention_mask=batch.attention_mask.numpy(), output_hidden_states=True)

        assert outputs.hidden_states is not None
        assert np.allclose(hf_outputs.hidden_states[0], outputs.hidden_states[0], rtol=1e-6)
        # Higher tolerance due to cross-platform BLAS differences
        assert np.allclose(hf_outputs.hidden_states[1], outputs.hidden_states[1], rtol=6e-3, atol=6e-3)
        assert np.allclose(hf_outputs.hidden_states[-1], outputs.hidden_states[-1], rtol=3e-2, atol=6e-2)


def load_moe_base_weights(jax_moe_layer: Glm4MoE, hf_moe_layer) -> None:
    """Load base weights from HF MoE layer to JAX MoE layer.

    The tiny random model uses separate experts (ModuleList), matching our implementation.
    """
    # Router weights
    jax_moe_layer.gate.weight[:] = hf_moe_layer.gate.weight.detach().numpy().T
    jax_moe_layer.gate.e_score_correction_bias[:] = hf_moe_layer.gate.e_score_correction_bias.detach().numpy()

    # Expert weights - The tiny model uses ModuleList with separate gate_proj, up_proj, down_proj
    hf_experts = hf_moe_layer.experts
    num_experts = len(hf_experts)

    for i, expert in enumerate(hf_experts):
        jax_moe_layer.experts.gate_proj.weight[i, :, :] = expert.gate_proj.weight.detach().numpy().T
        jax_moe_layer.experts.up_proj.weight[i, :, :] = expert.up_proj.weight.detach().numpy().T
        jax_moe_layer.experts.down_proj.weight[i, :, :] = expert.down_proj.weight.detach().numpy().T

    # Shared experts
    jax_moe_layer.shared_experts.gate_proj.kernel[:] = hf_moe_layer.shared_experts.gate_proj.weight.detach().numpy().T
    jax_moe_layer.shared_experts.up_proj.kernel[:] = hf_moe_layer.shared_experts.up_proj.weight.detach().numpy().T
    jax_moe_layer.shared_experts.down_proj.kernel[:] = hf_moe_layer.shared_experts.down_proj.weight.detach().numpy().T


def test_glm4_moe_layer():
    """Test GLM4 MoE layer against HuggingFace implementation."""
    model_name = "yujiepan/glm-4-moe-tiny-random"
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", use_safetensors=True)
    base_config = PretrainedConfig.from_pretrained(model_name)
    config = Glm4Config(base_config, max_lora_adapters=0, max_lora_rank=0, shard_attention_heads=True)

    # First layer uses dense MLP (first_k_dense_replace=1), so we test layer 1 which has MoE
    hf_moe_layer = hf_model.model.layers[1].mlp
    torch.manual_seed(42)
    x = torch.randn(4, 2, config.hidden_size)
    with torch.no_grad():
        hf_expert_output = hf_moe_layer.forward(x)

    mesh = jax.make_mesh(
        (1, 1),
        ("fsdp", "tp"),
        axis_types=(jax.sharding.AxisType.Auto, jax.sharding.AxisType.Auto),
    )
    with jax.set_mesh(mesh):
        moe_layer = Glm4MoE(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
        load_moe_base_weights(moe_layer, hf_moe_layer)

    jax_expert_output = moe_layer(x.numpy())

    # Higher tolerance due to cross-platform BLAS differences
    assert np.allclose(hf_expert_output.detach().numpy(), jax_expert_output, rtol=6e-3, atol=6e-3)


def test_glm4_model_instantiation():
    """Test that GLM4 model can be instantiated with both dense and MoE configs."""
    # Test dense config (no n_routed_experts)
    dense_config_dict = {
        "vocab_size": 1000,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "rms_norm_eps": 1e-5,
        "rope_parameters": {"rope_theta": 10000.0},
        "partial_rotary_factor": 0.5,
        "use_qk_norm": False,
        "attention_bias": False,
        "n_routed_experts": None,
        "first_k_dense_replace": 0,
        "architectures": ["Glm4ForCausalLM"],
    }
    dense_base_config = PretrainedConfig.from_dict(dense_config_dict)
    dense_config = Glm4Config(dense_base_config, max_lora_adapters=1, max_lora_rank=1, shard_attention_heads=False)

    mesh = jax.make_mesh(
        (1, 1),
        ("fsdp", "tp"),
        axis_types=(jax.sharding.AxisType.Auto, jax.sharding.AxisType.Auto),
    )
    with jax.set_mesh(mesh):
        dense_model = Glm4ForCausalLM(dense_config, dtype=jnp.float32, rngs=nnx.Rngs(0))

    # Verify dense model uses Glm4MLP in all layers
    for layer in dense_model.model.layers:
        from tx.models.glm4 import Glm4MLP
        assert isinstance(layer.mlp, Glm4MLP), f"Expected Glm4MLP, got {type(layer.mlp)}"

    # Test MoE config
    moe_config_dict = {
        "vocab_size": 1000,
        "hidden_size": 64,
        "intermediate_size": 128,
        "moe_intermediate_size": 64,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "rms_norm_eps": 1e-5,
        "rope_parameters": {"rope_theta": 10000.0},
        "partial_rotary_factor": 0.5,
        "use_qk_norm": False,
        "attention_bias": False,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "n_group": 1,
        "topk_group": 1,
        "first_k_dense_replace": 1,
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.0,
        "architectures": ["Glm4MoeForCausalLM"],
    }
    moe_base_config = PretrainedConfig.from_dict(moe_config_dict)
    moe_config = Glm4Config(moe_base_config, max_lora_adapters=1, max_lora_rank=1, shard_attention_heads=False)

    with jax.set_mesh(mesh):
        moe_model = Glm4ForCausalLM(moe_config, dtype=jnp.float32, rngs=nnx.Rngs(0))

    # Verify MoE model uses Glm4MLP for first layer and Glm4MoE for rest
    from tx.models.glm4 import Glm4MLP, Glm4MoE
    assert isinstance(moe_model.model.layers[0].mlp, Glm4MLP), "Layer 0 should use dense MLP"
    assert isinstance(moe_model.model.layers[1].mlp, Glm4MoE), "Layer 1 should use MoE"
    assert isinstance(moe_model.model.layers[2].mlp, Glm4MoE), "Layer 2 should use MoE"


def test_glm4_forward_pass():
    """Test that GLM4 model can run a forward pass."""
    config_dict = {
        "vocab_size": 1000,
        "hidden_size": 64,
        "intermediate_size": 128,
        "moe_intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "rms_norm_eps": 1e-5,
        "rope_parameters": {"rope_theta": 10000.0},
        "partial_rotary_factor": 0.5,
        "use_qk_norm": False,
        "attention_bias": False,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "n_group": 1,
        "topk_group": 1,
        "first_k_dense_replace": 1,
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.0,
        "architectures": ["Glm4ForCausalLM"],
    }
    base_config = PretrainedConfig.from_dict(config_dict)
    config = Glm4Config(base_config, max_lora_adapters=1, max_lora_rank=1, shard_attention_heads=False)

    mesh = jax.make_mesh(
        (1, 1),
        ("fsdp", "tp"),
        axis_types=(jax.sharding.AxisType.Auto, jax.sharding.AxisType.Auto),
    )
    with jax.set_mesh(mesh):
        model = Glm4ForCausalLM(config, dtype=jnp.float32, rngs=nnx.Rngs(0))

    # Create dummy input
    batch_size, seq_len = 2, 8
    input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

    # Run forward pass
    outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)

    assert outputs.last_hidden_state.shape == (batch_size, seq_len, config.hidden_size)
    assert outputs.hidden_states is not None
    assert len(outputs.hidden_states) == config.num_hidden_layers + 1

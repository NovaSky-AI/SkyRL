import os
import tempfile

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE as HFDeepseekV3MoE
from tx.layers.lora import LoRAMixin
from tx.models.configs import DeepseekV3Config
from tx.models.deepseekv3 import DeepseekV3ForCausalLM, DeepseekV3MoE
from tx.utils.models import load_safetensors


@pytest.mark.parametrize("tp", [1, 2])
def test_deepseekv3(tp: int):
    if tp > 1 and os.getenv("CI"):
        pytest.skip("TP > 1 currently runs out of memory in the CI")

    model_name = "yujiepan/deepseek-v3-tiny-random"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, attn_implementation="eager", use_safetensors=True, trust_remote_code=True
    )

    inputs = ["The capital of France is", "The most popular programming language is"]
    batch = tokenizer(inputs, return_tensors="pt", padding=True)
    with torch.no_grad():
        hf_outputs = hf_model(
            batch.input_ids, attention_mask=batch.attention_mask, output_hidden_states=True, use_cache=False
        )

    # Save the HF model checkpoint so we can load our model from it
    with tempfile.TemporaryDirectory() as tmp:
        hf_model.save_pretrained(tmp, safe_serialization=True)

        base_config = PretrainedConfig.from_pretrained(model_name, trust_remote_code=True)
        config = DeepseekV3Config(base_config, max_lora_adapters=32, max_lora_rank=32, shard_attention_heads=True)
        # EP axis required for MoE expert sharding
        mesh = jax.make_mesh((1, 1, tp), ("fsdp", "ep", "tp"), axis_types=(jax.sharding.AxisType.Auto,) * 3)
        with jax.set_mesh(mesh):
            model = DeepseekV3ForCausalLM(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
            load_safetensors(tmp, config, model)
            outputs = model(
                batch.input_ids.numpy(), attention_mask=batch.attention_mask.numpy(), output_hidden_states=True
            )

        assert outputs.hidden_states is not None
        assert np.allclose(hf_outputs.hidden_states[0], outputs.hidden_states[0], rtol=1e-6)
        assert np.allclose(hf_outputs.hidden_states[1], outputs.hidden_states[1], rtol=1e-3, atol=1e-3)
        assert np.allclose(hf_outputs.hidden_states[-1], outputs.hidden_states[-1], rtol=3e-2, atol=6e-2)


def load_moe_base_weights(jax_moe_layer: DeepseekV3MoE, hf_moe_layer: HFDeepseekV3MoE) -> None:
    """Load base weights from HF MoE layer to JAX MoE layer."""
    jax_moe_layer.gate.weight[:] = hf_moe_layer.gate.weight.detach().numpy().T
    jax_moe_layer.gate.e_score_correction_bias[:] = hf_moe_layer.gate.e_score_correction_bias.detach().numpy()

    for i, expert in enumerate(hf_moe_layer.experts):
        jax_moe_layer.experts.gate_proj.weight[i, :, :] = expert.gate_proj.weight.detach().numpy().T
        jax_moe_layer.experts.up_proj.weight[i, :, :] = expert.up_proj.weight.detach().numpy().T
        jax_moe_layer.experts.down_proj.weight[i, :, :] = expert.down_proj.weight.detach().numpy().T

    jax_moe_layer.shared_experts.gate_proj.kernel[:] = hf_moe_layer.shared_experts.gate_proj.weight.detach().numpy().T
    jax_moe_layer.shared_experts.up_proj.kernel[:] = hf_moe_layer.shared_experts.up_proj.weight.detach().numpy().T
    jax_moe_layer.shared_experts.down_proj.kernel[:] = hf_moe_layer.shared_experts.down_proj.weight.detach().numpy().T


@pytest.mark.parametrize("ep,tp", [(1, 1), (1, 2), (2, 1)])
def test_deepseekv3_moe_layer(ep: int, tp: int):
    model_name = "yujiepan/deepseek-v3-tiny-random"
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", use_safetensors=True)
    base_config = PretrainedConfig.from_pretrained(model_name)
    config = DeepseekV3Config(base_config, max_lora_adapters=0, max_lora_rank=0, shard_attention_heads=True)

    # Initial deepseek layers don't have MoE
    hf_moe_layer = hf_model.model.layers[1].mlp
    torch.manual_seed(42)
    x = torch.randn(4, 2, config.hidden_size)
    with torch.no_grad():
        hf_expert_output = hf_moe_layer.forward(x)

    mesh = jax.make_mesh((1, ep, tp), ("fsdp", "ep", "tp"), axis_types=(jax.sharding.AxisType.Auto,) * 3)
    with jax.set_mesh(mesh):
        moe_layer = DeepseekV3MoE(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
        load_moe_base_weights(moe_layer, hf_moe_layer)

        jax_expert_output = moe_layer(x.numpy())

        # Higher tolerance due to cross-platform BLAS differences
        assert np.allclose(hf_expert_output.detach().numpy(), jax_expert_output, rtol=6e-3, atol=6e-3)


def load_lora_weights(
    jax_module: LoRAMixin,
    adapter_idx: int,
    lora_A_weights: np.ndarray,
    lora_B_weights: np.ndarray,
    scaling: float,
    rank: int,
) -> None:
    """Load LoRA weights from numpy arrays to JAX module."""
    assert (
        jax_module.lora_A is not None
        and jax_module.lora_B is not None
        and jax_module.lora_scaling is not None
        and jax_module.lora_ranks is not None
    )
    jax_module.lora_A[...] = jax_module.lora_A[...].at[adapter_idx].set(jnp.array(lora_A_weights))
    jax_module.lora_B[...] = jax_module.lora_B[...].at[adapter_idx].set(jnp.array(lora_B_weights))
    jax_module.lora_scaling[...] = jax_module.lora_scaling[...].at[adapter_idx].set(scaling)
    jax_module.lora_ranks[...] = jax_module.lora_ranks[...].at[adapter_idx].set(rank)


@pytest.mark.parametrize("ep,tp", [(1, 1), (1, 2), (2, 1)])
def test_deepseekv3_moe_layer_lora(ep: int, tp: int):
    """Test MoE LoRA by merging adapter into base weights and comparing outputs."""
    model_name = "yujiepan/deepseek-v3-tiny-random"
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", use_safetensors=True)
    base_config = PretrainedConfig.from_pretrained(model_name)
    config = DeepseekV3Config(base_config, max_lora_adapters=3, max_lora_rank=4, shard_attention_heads=True)

    hf_moe_layer = hf_model.model.layers[1].mlp
    x = torch.randn(3, 4, config.hidden_size)

    mesh = jax.make_mesh(
        (1, ep, tp),
        ("fsdp", "ep", "tp"),
        axis_types=(jax.sharding.AxisType.Auto,) * 3,
    )
    with jax.set_mesh(mesh):
        moe_layer = DeepseekV3MoE(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
        load_moe_base_weights(moe_layer, hf_moe_layer)

        # Set LoRA weights for all adapters
        rng = np.random.default_rng(42)
        scaling = 2.0
        rank = config.max_lora_rank
        for adapter_idx in range(config.max_lora_adapters):
            for proj in [moe_layer.experts.gate_proj, moe_layer.experts.up_proj, moe_layer.experts.down_proj]:
                assert proj.lora_A is not None and proj.lora_B is not None
                lora_A = rng.normal(0, 1.0, proj.lora_A[...].shape[1:])
                lora_B = rng.normal(0, 1.0, proj.lora_B[...].shape[1:])
                load_lora_weights(proj, adapter_idx, lora_A, lora_B, scaling, rank)

        # Test with different adapters per sample
        adapter_indices = jnp.array([0, 2, 1])
        output_with_lora = moe_layer(x.numpy(), adapter_indices=adapter_indices)

        # Test each sample by comparing with merged weights for its adapter
        for sample_idx in range(len(adapter_indices)):
            adapter_idx = int(adapter_indices[sample_idx])

            # Create merged model by adding LoRA weights to base weights
            moe_layer_merged = DeepseekV3MoE(config, dtype=jnp.float32, rngs=nnx.Rngs(1 + adapter_idx))

            # Copy router weights
            moe_layer_merged.gate.weight[:] = moe_layer.gate.weight[:]
            moe_layer_merged.gate.e_score_correction_bias[:] = moe_layer.gate.e_score_correction_bias[:]

            # Copy shared experts weights
            moe_layer_merged.shared_experts.gate_proj.kernel[:] = moe_layer.shared_experts.gate_proj.kernel[:]
            moe_layer_merged.shared_experts.up_proj.kernel[:] = moe_layer.shared_experts.up_proj.kernel[:]
            moe_layer_merged.shared_experts.down_proj.kernel[:] = moe_layer.shared_experts.down_proj.kernel[:]

            for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                proj = getattr(moe_layer.experts, proj_name)
                proj_merged = getattr(moe_layer_merged.experts, proj_name)

                # For each expert, merge: base + scaling * (lora_A @ lora_B)
                for expert_idx in range(config.n_routed_experts):
                    lora_A = proj.lora_A[adapter_idx, expert_idx, :, :]
                    lora_B = proj.lora_B[adapter_idx, expert_idx, :, :]
                    lora_delta = scaling * (lora_A @ lora_B)

                    # Copy base weight AND add LoRA delta
                    base_weight = proj.weight[expert_idx, :, :]
                    merged_weight = base_weight + lora_delta
                    proj_merged.weight[...] = proj_merged.weight[...].at[expert_idx, :, :].set(merged_weight)

            # Run merged model on this sample
            x_sample = x[sample_idx : sample_idx + 1].numpy()
            output_merged = moe_layer_merged(x_sample)

            assert np.allclose(output_with_lora[sample_idx : sample_idx + 1], output_merged, rtol=1e-3, atol=1e-3)

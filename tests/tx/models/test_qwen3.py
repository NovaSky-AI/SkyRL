import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeSparseMoeBlock as HFQwen3MoeSparseMoeBlock,
)

from skyrl.tx.layers.lora import LoRAMixin
from skyrl.tx.models.configs import Qwen3Config
from skyrl.tx.models.qwen3 import Qwen3ForCausalLM, Qwen3MoeSparseMoeBlock
from tests.tx.models.conftest import load_model


@pytest.mark.parametrize("tp", [1, 2])
def test_qwen3(tp: int):
    if tp > 1 and os.getenv("CI"):
        pytest.skip("TP > 1 currently runs out of memory in the CI")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    hf_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", attn_implementation="eager", use_safetensors=True
    )

    inputs = ["The capital of France is", "The most popular programming language is"]
    batch = tokenizer(inputs, return_tensors="pt", padding=True)
    with torch.no_grad():
        hf_outputs = hf_model(
            batch.input_ids, attention_mask=batch.attention_mask, output_hidden_states=True, return_dict=True
        )
    del hf_model

    _, model = load_model(
        "Qwen/Qwen3-0.6B",
        Qwen3Config,
        Qwen3ForCausalLM,
        ("fsdp", "tp"),
        mesh_shape=(1, tp),
        max_lora_adapters=32,
        max_lora_rank=32,
    )

    outputs = model(batch.input_ids.numpy(), attention_mask=batch.attention_mask.numpy(), output_hidden_states=True)
    assert outputs.hidden_states is not None
    assert np.allclose(hf_outputs.hidden_states[0], outputs.hidden_states[0], rtol=1e-6)
    assert np.allclose(hf_outputs.hidden_states[1], outputs.hidden_states[1], rtol=1e-3, atol=1e-3)
    # The final hidden state accumulates modest framework drift beyond the first layer.
    assert np.allclose(hf_outputs.hidden_states[-1], outputs.hidden_states[-1], rtol=5e-3, atol=7e-2)


def load_moe_base_weights(jax_moe_layer: Qwen3MoeSparseMoeBlock, hf_moe_layer: HFQwen3MoeSparseMoeBlock) -> None:
    """Load base weights from HF MoE layer to JAX MoE layer."""
    jax_moe_layer.gate.kernel[:] = hf_moe_layer.gate.weight.detach().numpy().T
    for i, expert in enumerate(hf_moe_layer.experts):
        jax_moe_layer.experts.gate_proj.weight[i, :, :] = expert.gate_proj.weight.detach().numpy().T
        jax_moe_layer.experts.up_proj.weight[i, :, :] = expert.up_proj.weight.detach().numpy().T
        jax_moe_layer.experts.down_proj.weight[i, :, :] = expert.down_proj.weight.detach().numpy().T


@pytest.mark.parametrize("ep,tp", [(1, 1), (1, 2), (2, 1)])
def test_qwen3_moe_layer(ep: int, tp: int):
    model_name = "trl-internal-testing/tiny-Qwen3MoeForCausalLM"
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", use_safetensors=True)
    base_config = PretrainedConfig.from_pretrained(model_name)
    config = Qwen3Config(base_config, max_lora_adapters=0, max_lora_rank=0, shard_attention_heads=True)

    hf_moe_layer = hf_model.model.layers[0].mlp
    x = torch.randn(4, 2, config.hidden_size)
    with torch.no_grad():
        hf_final_hidden_states, hf_router_logits = hf_moe_layer.forward(x)

    mesh = jax.make_mesh((1, ep, tp), ("fsdp", "ep", "tp"), axis_types=(jax.sharding.AxisType.Auto,) * 3)
    with jax.set_mesh(mesh):
        moe_layer = Qwen3MoeSparseMoeBlock(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
        load_moe_base_weights(moe_layer, hf_moe_layer)

        final_hidden_states, router_logits = moe_layer(x.numpy(), return_router_logits=True)

        assert np.allclose(hf_router_logits, router_logits, rtol=1e-4)
        assert np.allclose(hf_final_hidden_states, final_hidden_states, rtol=5e-2, atol=1e-2)


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
def test_qwen3_moe_layer_lora(ep: int, tp: int):
    """Test MoE LoRA by checking adapter isolation within a shared batch."""
    model_name = "trl-internal-testing/tiny-Qwen3MoeForCausalLM"
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", use_safetensors=True)
    base_config = PretrainedConfig.from_pretrained(model_name)
    config = Qwen3Config(base_config, max_lora_adapters=3, max_lora_rank=4, shard_attention_heads=True)

    hf_moe_layer = hf_model.model.layers[0].mlp
    x = torch.randn(3, 4, config.hidden_size)

    mesh = jax.make_mesh((1, ep, tp), ("fsdp", "ep", "tp"), axis_types=(jax.sharding.AxisType.Auto,) * 3)
    with jax.set_mesh(mesh):
        moe_layer = Qwen3MoeSparseMoeBlock(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
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

        output_000, _ = moe_layer(x.numpy(), adapter_indices=jnp.array([0, 0, 0]), return_router_logits=True)
        output_020, _ = moe_layer(x.numpy(), adapter_indices=jnp.array([0, 2, 0]), return_router_logits=True)
        output_021, _ = moe_layer(x.numpy(), adapter_indices=jnp.array([0, 2, 1]), return_router_logits=True)
        output_221, _ = moe_layer(x.numpy(), adapter_indices=jnp.array([2, 2, 1]), return_router_logits=True)

        # Changing one sample's adapter should not perturb the others in the same batch.
        assert np.allclose(output_000[0], output_020[0], rtol=1e-3, atol=2e-2)
        assert np.allclose(output_020[0], output_021[0], rtol=1e-3, atol=2e-2)
        assert np.allclose(output_020[1], output_021[1], rtol=1e-3, atol=2e-2)
        assert np.allclose(output_021[1], output_221[1], rtol=1e-3, atol=2e-2)
        assert np.allclose(output_021[2], output_221[2], rtol=1e-3, atol=2e-2)

        # The sample whose adapter changes should produce a measurably different result.
        assert not np.allclose(output_000[1], output_020[1], rtol=1e-4, atol=1e-4)
        assert not np.allclose(output_020[2], output_021[2], rtol=1e-4, atol=1e-4)
        assert not np.allclose(output_021[0], output_221[0], rtol=1e-4, atol=1e-4)


def test_qwen3_lora():
    """Test that batched multi-LoRA inference matches per-adapter inference."""
    base_model_name = "Qwen/Qwen3-0.6B"
    supported_modules = {
        "self_attn": ["o_proj"],
        "mlp": ["down_proj"],
    }
    num_adapters = 2
    rank = 8
    scaling = 2.0

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    inputs = ["The capital of France is", "My name is"]
    batch = tokenizer(inputs, return_tensors="pt", padding=True)

    config, model = load_model(
        base_model_name,
        Qwen3Config,
        Qwen3ForCausalLM,
        ("fsdp", "tp"),
        max_lora_adapters=num_adapters,
        max_lora_rank=rank,
    )

    rng = np.random.default_rng(0)
    for adapter_idx in range(num_adapters):
        for i in range(config.num_hidden_layers):
            jax_layer = model.model.layers[i]
            for module_name, projections in supported_modules.items():
                for proj_name in projections:
                    jax_proj = getattr(getattr(jax_layer, module_name), proj_name)
                    load_lora_weights(
                        jax_proj,
                        adapter_idx=adapter_idx,
                        lora_A_weights=rng.normal(0.0, 1e-2, size=jax_proj.lora_A[...].shape[1:]),
                        lora_B_weights=rng.normal(0.0, 1e-2, size=jax_proj.lora_B[...].shape[1:]),
                        scaling=scaling,
                        rank=rank,
                    )

    outputs_00 = model(
        batch.input_ids.numpy(),
        attention_mask=batch.attention_mask.numpy(),
        output_hidden_states=True,
        adapter_indices=jnp.array([0, 0], dtype=jnp.int32),
    )
    outputs_01 = model(
        batch.input_ids.numpy(),
        attention_mask=batch.attention_mask.numpy(),
        output_hidden_states=True,
        adapter_indices=jnp.array([0, 1], dtype=jnp.int32),
    )
    outputs_11 = model(
        batch.input_ids.numpy(),
        attention_mask=batch.attention_mask.numpy(),
        output_hidden_states=True,
        adapter_indices=jnp.array([1, 1], dtype=jnp.int32),
    )

    assert np.allclose(outputs_00.last_hidden_state[0], outputs_01.last_hidden_state[0], rtol=1e-3, atol=2e-2)
    assert np.allclose(outputs_01.last_hidden_state[1], outputs_11.last_hidden_state[1], rtol=1e-3, atol=2e-2)
    assert not np.allclose(outputs_00.last_hidden_state[1], outputs_01.last_hidden_state[1], rtol=1e-4, atol=1e-4)
    assert not np.allclose(outputs_01.last_hidden_state[0], outputs_11.last_hidden_state[0], rtol=1e-4, atol=1e-4)

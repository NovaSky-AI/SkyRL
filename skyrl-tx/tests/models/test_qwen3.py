import os
import tempfile

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from peft import LoraConfig, get_peft_model
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock as HFQwen3MoeSparseMoeBlock

from tx.layers.lora import LoRAMixin
from tx.models.configs import Qwen3Config
from tx.models.qwen3 import Qwen3ForCausalLM, Qwen3MoeSparseMoeBlock
from tx.utils.models import load_safetensors


@pytest.mark.parametrize("tp", [1, 2])
def test_qwen3(tp: int):
    if not jax._src.xla_bridge.backends_are_initialized():  # ty: ignore
        jax.config.update("jax_num_cpu_devices", 2)

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

    # Save the HF model checkpoint so we can load our model from it
    with tempfile.TemporaryDirectory() as tmp:
        hf_model.save_pretrained(tmp, safe_serialization=True)

        base_config = PretrainedConfig.from_pretrained("Qwen/Qwen3-0.6B")
        config = Qwen3Config(base_config, max_lora_adapters=32, max_lora_rank=32, shard_attention_heads=True)
        mesh = jax.make_mesh((1, tp), ("dp", "tp"))
        with jax.set_mesh(mesh):
            model = Qwen3ForCausalLM(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
        load_safetensors(tmp, config, model)

        outputs = model(batch.input_ids.numpy(), attention_mask=batch.attention_mask.numpy(), output_hidden_states=True)
        assert outputs.hidden_states is not None
        assert np.allclose(hf_outputs.hidden_states[0], outputs.hidden_states[0], rtol=1e-6)
        assert np.allclose(hf_outputs.hidden_states[1], outputs.hidden_states[1], rtol=1e-3, atol=1e-3)
        assert np.allclose(hf_outputs.hidden_states[-1], outputs.hidden_states[-1], rtol=1e-3, atol=1e-3)


def load_moe_base_weights(jax_moe_layer: Qwen3MoeSparseMoeBlock, hf_moe_layer: HFQwen3MoeSparseMoeBlock) -> None:
    """Load base weights from HF MoE layer to JAX MoE layer."""
    jax_moe_layer.gate.kernel[:] = hf_moe_layer.gate.weight.detach().numpy().T
    for i, expert in enumerate(hf_moe_layer.experts):
        jax_moe_layer.experts.gate_proj.weight[i, :, :] = expert.gate_proj.weight.detach().numpy().T
        jax_moe_layer.experts.up_proj.weight[i, :, :] = expert.up_proj.weight.detach().numpy().T
        jax_moe_layer.experts.down_proj.weight[i, :, :] = expert.down_proj.weight.detach().numpy().T


def test_qwen3_moe_layer():
    model_name = "trl-internal-testing/tiny-Qwen3MoeForCausalLM"
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", use_safetensors=True)
    base_config = PretrainedConfig.from_pretrained(model_name)
    config = Qwen3Config(base_config, max_lora_adapters=0, max_lora_rank=0, shard_attention_heads=True)

    hf_moe_layer = hf_model.model.layers[0].mlp
    x = torch.randn(4, 2, config.hidden_size)
    with torch.no_grad():
        hf_final_hidden_states, hf_router_logits = hf_moe_layer.forward(x)

    mesh = jax.make_mesh((1, 1), ("dp", "tp"))
    with jax.set_mesh(mesh):
        moe_layer = Qwen3MoeSparseMoeBlock(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
        load_moe_base_weights(moe_layer, hf_moe_layer)

    final_hidden_states, router_logits = moe_layer(x.numpy(), return_router_logits=True)

    assert np.allclose(hf_router_logits, router_logits, rtol=1e-4)
    assert np.allclose(hf_final_hidden_states, final_hidden_states, rtol=1e-2, atol=1e-2)


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
    jax_module.lora_A.value = jax_module.lora_A.value.at[adapter_idx].set(jnp.array(lora_A_weights))
    jax_module.lora_B.value = jax_module.lora_B.value.at[adapter_idx].set(jnp.array(lora_B_weights))
    jax_module.lora_scaling.value = jax_module.lora_scaling.value.at[adapter_idx].set(scaling)
    jax_module.lora_ranks.value = jax_module.lora_ranks.value.at[adapter_idx].set(rank)


def test_qwen3_moe_layer_lora():
    """Test MoE LoRA by merging adapter into base weights and comparing outputs."""
    model_name = "trl-internal-testing/tiny-Qwen3MoeForCausalLM"
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", use_safetensors=True)
    base_config = PretrainedConfig.from_pretrained(model_name)
    config = Qwen3Config(base_config, max_lora_adapters=3, max_lora_rank=4, shard_attention_heads=True)

    hf_moe_layer = hf_model.model.layers[0].mlp
    x = torch.randn(3, 4, config.hidden_size)

    mesh = jax.make_mesh((1, 1), ("dp", "tp"))
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
                lora_A = rng.normal(0, 1.0, proj.lora_A.value.shape[1:])
                lora_B = rng.normal(0, 1.0, proj.lora_B.value.shape[1:])
                load_lora_weights(proj, adapter_idx, lora_A, lora_B, scaling, rank)

        # Test with different adapters per sample
        adapter_indices = jnp.array([0, 2, 1])
        output_with_lora, _ = moe_layer(x.numpy(), adapter_indices=adapter_indices, return_router_logits=True)

        # Test each sample by comparing with merged weights for its adapter
        for sample_idx in range(len(adapter_indices)):
            adapter_idx = int(adapter_indices[sample_idx])

            # Create merged model by adding LoRA weights to base weights
            moe_layer_merged = Qwen3MoeSparseMoeBlock(config, dtype=jnp.float32, rngs=nnx.Rngs(1 + adapter_idx))
            moe_layer_merged.gate.kernel[:] = moe_layer.gate.kernel[:]

            for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                proj = getattr(moe_layer.experts, proj_name)
                proj_merged = getattr(moe_layer_merged.experts, proj_name)

                # For each expert, merge: base + scaling * (lora_A @ lora_B)
                for expert_idx in range(config.num_experts):
                    lora_A = proj.lora_A.value[adapter_idx, expert_idx, :, :]
                    lora_B = proj.lora_B.value[adapter_idx, expert_idx, :, :]
                    lora_delta = scaling * (lora_A @ lora_B)

                    merged_weight = proj.weight[expert_idx, :, :] + lora_delta
                    proj_merged.weight.value = proj_merged.weight.value.at[expert_idx, :, :].set(merged_weight)

            # Run merged model on this sample
            x_sample = x[sample_idx : sample_idx + 1].numpy()
            output_merged, _ = moe_layer_merged(x_sample, return_router_logits=True)

            assert np.allclose(output_with_lora[sample_idx : sample_idx + 1], output_merged, rtol=1e-3, atol=1e-3)


def test_qwen3_lora():
    """Test multi-LoRA implementation by comparing with HuggingFace PEFT model using two different adapters."""
    base_model_name = "Qwen/Qwen3-0.6B"
    lora_adapters = ["pcmoritz/qwen3-0.6b-lora-random", "pcmoritz/qwen3-0.6b-lora-random2"]

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    # Use two different inputs to test with different adapters
    inputs = ["The capital of France is", "My name is"]
    batch = tokenizer(inputs, return_tensors="pt", padding=True)

    with tempfile.TemporaryDirectory() as base_tmp:
        base_hf_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, attn_implementation="eager", use_safetensors=True
        )
        base_hf_model.save_pretrained(base_tmp, safe_serialization=True)

        # Create HF models with different adapters
        hf_lora_models = []
        lora_configs = []
        for adapter_name in lora_adapters:
            lora_config = LoraConfig.from_pretrained(adapter_name)
            lora_config.target_modules = [
                "embed_tokens",
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
            lora_configs.append(lora_config)

            hf_model = get_peft_model(
                AutoModelForCausalLM.from_pretrained(
                    base_model_name, attn_implementation="eager", use_safetensors=True
                ),
                lora_config,
            )
            hf_model.eval()
            hf_model.load_adapter(adapter_name, adapter_name="default")
            hf_lora_models.append(hf_model)

        base_config = PretrainedConfig.from_pretrained(base_model_name)
        config = Qwen3Config(
            base_config,
            max_lora_adapters=len(lora_adapters),
            max_lora_rank=max(cfg.r for cfg in lora_configs),
            shard_attention_heads=True,
        )

        mesh = jax.make_mesh((1, 1), ("dp", "tp"))
        with jax.set_mesh(mesh):
            model = Qwen3ForCausalLM(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
            load_safetensors(base_tmp, config, model)

        # Get outputs from all HF models
        hf_outputs_list = []
        with torch.no_grad():
            for idx in range(len(lora_adapters)):
                hf_output = hf_lora_models[idx](
                    batch.input_ids[idx : idx + 1],
                    attention_mask=batch.attention_mask[idx : idx + 1],
                    output_hidden_states=True,
                    return_dict=True,
                )
                hf_outputs_list.append(hf_output)

        # Load LoRA adapter weights from all adapters
        for adapter_idx, (hf_model, lora_config) in enumerate(zip(hf_lora_models, lora_configs)):
            # Load embed_tokens LoRA weights
            hf_embed_tokens = hf_model.base_model.model.model.embed_tokens
            load_lora_weights(
                model.model.embed_tokens,
                adapter_idx=adapter_idx,
                lora_A_weights=hf_embed_tokens.lora_embedding_A["default"].detach().numpy().T,
                lora_B_weights=hf_embed_tokens.lora_embedding_B["default"].detach().numpy().T,
                scaling=lora_config.lora_alpha / lora_config.r,
                rank=lora_config.r,
            )

            # Load layer LoRA weights
            for i, layer in enumerate(model.model.layers):
                hf_layer = hf_model.base_model.model.model.layers[i]
                for module, projections in [
                    ("mlp", ["gate_proj", "up_proj", "down_proj"]),
                    ("self_attn", ["q_proj", "k_proj", "v_proj", "o_proj"]),
                ]:
                    for proj_name in projections:
                        hf_proj = getattr(getattr(hf_layer, module), proj_name)
                        load_lora_weights(
                            getattr(getattr(layer, module), proj_name),
                            adapter_idx=adapter_idx,
                            lora_A_weights=hf_proj.lora_A["default"].weight.detach().numpy().T,
                            lora_B_weights=hf_proj.lora_B["default"].weight.detach().numpy().T,
                            scaling=lora_config.lora_alpha / lora_config.r,
                            rank=lora_config.r,
                        )

        # Use different adapter indices for each input
        adapter_indices = jnp.arange(len(lora_adapters), dtype=jnp.int32)
        outputs = model(
            batch.input_ids.numpy(),
            attention_mask=batch.attention_mask.numpy(),
            output_hidden_states=True,
            adapter_indices=adapter_indices,
        )

        # Compare outputs with corresponding adapters
        for idx in range(len(lora_adapters)):
            assert np.allclose(hf_outputs_list[idx].logits[0], outputs.logits[idx], rtol=1e-3, atol=1e-3)

import os
import tempfile

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from peft import LoraConfig, get_peft_model
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from tx.layers.lora import LoRAMixin
from tx.models import Qwen3ForCausalLM
from tx.models.qwen3 import Qwen3MoeSparseMoeBlock
from tx.utils.models import load_checkpoint


@pytest.mark.parametrize("tp", [1, 2])
def test_qwen3(tp: int):
    if not jax._src.xla_bridge.backends_are_initialized():  # ty: ignore
        jax.config.update("jax_num_cpu_devices", 2)

    if tp > 1 and os.getenv("CI"):
        pytest.skip("TP > 1 currently runs out of memory in the CI")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    hf_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", attn_implementation="eager", use_safetensors=True)

    inputs = ["The capital of France is", "The most popular programming language is"]
    batch = tokenizer(inputs, return_tensors="pt", padding=True)
    with torch.no_grad():
        hf_outputs = hf_model(batch.input_ids, attention_mask=batch.attention_mask, output_hidden_states=True, return_dict=True)

    # Save the HF model checkpoint so we can load our model from it
    with tempfile.TemporaryDirectory() as tmp:
        hf_model.save_pretrained(tmp, safe_serialization=True)

        config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
        mesh = jax.make_mesh((1, tp), ("dp", "tp"))
        with jax.set_mesh(mesh):
            model = Qwen3ForCausalLM(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
        load_checkpoint(tmp, config, model)
        
        outputs = model(batch.input_ids.numpy(), attention_mask=batch.attention_mask.numpy(), output_hidden_states=True)
        assert np.allclose(hf_outputs.hidden_states[0], outputs["hidden_states"][0], rtol=1e-6)
        assert np.allclose(hf_outputs.hidden_states[1], outputs["hidden_states"][1], rtol=1e-3, atol=1e-3)
        assert np.allclose(hf_outputs.hidden_states[-1], outputs["hidden_states"][-1], rtol=1e-3, atol=1e-3)


def test_qwen3_moe_layer():
    model_name = "trl-internal-testing/tiny-Qwen3MoeForCausalLM"
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", use_safetensors=True)
    config = AutoConfig.from_pretrained(model_name)

    hf_moe_layer = hf_model.model.layers[0].mlp
    x = torch.randn(4, 2, config.hidden_size)
    with torch.no_grad():
        hf_final_hidden_states, hf_router_logits = hf_moe_layer.forward(x)

    mesh = jax.make_mesh((1, 1), ("dp", "tp"))
    with jax.set_mesh(mesh):
        moe_layer = Qwen3MoeSparseMoeBlock(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
        moe_layer.gate.kernel[:] = hf_moe_layer.gate.weight[:].detach().numpy().T
        for i, expert in enumerate(hf_moe_layer.experts):
            moe_layer.experts.gate_proj.weight.value = moe_layer.experts.gate_proj.weight.value.at[i,:,:].set(expert.gate_proj.weight.detach().numpy().T)
            moe_layer.experts.up_proj.weight.value = moe_layer.experts.up_proj.weight.value.at[i,:,:].set(expert.up_proj.weight.detach().numpy().T)
            moe_layer.experts.down_proj.weight.value = moe_layer.experts.down_proj.weight.value.at[i,:,:].set(expert.down_proj.weight.detach().numpy().T)

    final_hidden_states, router_logits = moe_layer(x.numpy(), return_router_logits=True)

    assert np.allclose(hf_router_logits, router_logits, rtol=1e-4)
    assert np.allclose(hf_final_hidden_states, final_hidden_states, rtol=1e-3, atol=1e-2)


def load_lora_weights(jax_module: LoRAMixin, hf_module: torch.nn.Module,
                      adapter_idx: int, scaling: float, rank: int, adapter_name: str = 'default') -> None:
    """Load LoRA weights from HF module to JAX module."""
    assert jax_module.lora_A is not None and jax_module.lora_B is not None and jax_module.lora_scaling is not None and jax_module.lora_ranks is not None
    jax_module.lora_A.value = jax_module.lora_A.value.at[adapter_idx].set(
        jnp.array(hf_module.lora_A[adapter_name].weight.detach().numpy().T)  # ty: ignore
    )
    jax_module.lora_B.value = jax_module.lora_B.value.at[adapter_idx].set(
        jnp.array(hf_module.lora_B[adapter_name].weight.detach().numpy().T)  # ty: ignore
    )
    jax_module.lora_scaling.value = jax_module.lora_scaling.value.at[adapter_idx].set(scaling)
    jax_module.lora_ranks.value = jax_module.lora_ranks.value.at[adapter_idx].set(rank)


def test_qwen3_lora():
    """Test multi-LoRA implementation by comparing with HuggingFace PEFT model using two different adapters."""
    base_model_name = "Qwen/Qwen3-0.6B"
    lora_adapters = ["charent/self_cognition_Alice", "charent/self_cognition_Bob"]

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    # Use two different inputs to test with different adapters
    inputs = ["The capital of France is", "My name is"]
    batch = tokenizer(inputs, return_tensors="pt", padding=True)

    with tempfile.TemporaryDirectory() as base_tmp:
        base_hf_model = AutoModelForCausalLM.from_pretrained(base_model_name, attn_implementation="eager", use_safetensors=True)
        base_hf_model.save_pretrained(base_tmp, safe_serialization=True)

        config = AutoConfig.from_pretrained(base_model_name)

        # Create HF models with different adapters
        hf_lora_models = []
        lora_configs = []
        for adapter_name in lora_adapters:
            lora_config = LoraConfig.from_pretrained(adapter_name)
            lora_config.target_modules = ['gate_proj', 'up_proj', 'down_proj']
            lora_configs.append(lora_config)

            hf_model = get_peft_model(
                AutoModelForCausalLM.from_pretrained(base_model_name, attn_implementation="eager", use_safetensors=True),
                lora_config
            )
            hf_model.eval()
            hf_model.load_adapter(adapter_name, adapter_name='default')
            hf_lora_models.append(hf_model)

        config.max_lora_adapters = len(lora_adapters)
        config.max_lora_rank = max(cfg.r for cfg in lora_configs)

        mesh = jax.make_mesh((1, 1), ("dp", "tp"))
        with jax.set_mesh(mesh):
            model = Qwen3ForCausalLM(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
            load_checkpoint(base_tmp, config, model)

        # Get outputs from all HF models
        hf_outputs_list = []
        with torch.no_grad():
            for idx in range(len(lora_adapters)):
                hf_output = hf_lora_models[idx](
                    batch.input_ids[idx:idx+1],
                    attention_mask=batch.attention_mask[idx:idx+1],
                    output_hidden_states=True,
                    return_dict=True
                )
                hf_outputs_list.append(hf_output)

        # Load LoRA adapter weights from all adapters
        for i, layer in enumerate(model.model.layers):
            if hasattr(layer.mlp, 'gate_proj') and hasattr(layer.mlp.gate_proj, 'lora_A'):
                for adapter_idx, (hf_model, lora_config) in enumerate(zip(hf_lora_models, lora_configs)):
                    hf_layer = hf_model.base_model.model.model.layers[i].mlp
                    for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                        load_lora_weights(
                            getattr(layer.mlp, proj_name), getattr(hf_layer, proj_name),
                            adapter_idx=adapter_idx, scaling=lora_config.lora_alpha / lora_config.r, rank=lora_config.r
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
            assert np.allclose(hf_outputs_list[idx].logits[0], outputs["logits"][idx], rtol=1e-3, atol=1e-3)


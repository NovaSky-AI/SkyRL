from flax import nnx
import jax
import jax.numpy as jnp
import optax
from huggingface_hub import snapshot_download
from transformers import PretrainedConfig

from skyrl.tx.models.configs import Qwen3Config
from skyrl.tx.models.qwen3 import Qwen3ForCausalLM
from skyrl.tx.utils.models import get_dtype, load_safetensors
from skyrl.tx.layers.lora import init_lora_adapter
from skyrl.tinker.types import LoraConfig
from tests.tx.models.lora_test_utils import get_adapter_params, get_out_of_rank_params, verify_params_unchanged


def test_lora_training():
    base_model = "Qwen/Qwen3-0.6B"
    base_config = PretrainedConfig.from_pretrained(base_model)
    config = Qwen3Config(base_config, max_lora_adapters=5, max_lora_rank=32, shard_attention_heads=True)

    checkpoint_path = snapshot_download(base_model, allow_patterns=["*.safetensors"])
    mesh = jax.make_mesh((1, 1), ("fsdp", "tp"), axis_types=(jax.sharding.AxisType.Auto,) * 2)
    with jax.set_mesh(mesh):
        model = Qwen3ForCausalLM(config, dtype=get_dtype(config.dtype), rngs=nnx.Rngs(0))
        load_safetensors(checkpoint_path, config, model)

        # Set different ranks for each adapter (0: rank 16, 1: rank 8)
        init_lora_adapter(model, adapter_index=0, lora_config=LoraConfig(rank=16, alpha=16, seed=0))
        init_lora_adapter(model, adapter_index=1, lora_config=LoraConfig(rank=8, alpha=8, seed=1))

        # Create optimizer that only targets LoRA A and B parameters
        optimizer = nnx.Optimizer(model, optax.adamw(1e-4), wrt=model.is_lora_param)

        batch = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]], dtype=jnp.int32)
        target_ids = batch[:, 1:]
        input_ids = batch[:, :-1]
        adapter_indices = jnp.array([0, 1], dtype=jnp.int32)
        attention_mask = jnp.ones_like(input_ids)

        def loss_fn(model, input_ids, target_ids, attention_mask):
            outputs = model(input_ids, attention_mask=attention_mask, adapter_indices=adapter_indices)
            logits = model.compute_logits(outputs.last_hidden_state, adapter_indices)
            return optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=target_ids).mean()

        # Compute gradients - we need to use nnx.split to separate parameters
        # that we want to compute gradients for
        graphdef, lora_params, non_lora_params = nnx.split(model, model.is_lora_param, ...)

        # Save initial states
        initial_adapter_2_params = get_adapter_params(lora_params, 2)
        initial_adapter_0_out_of_rank = get_out_of_rank_params(lora_params, 0, 16)
        initial_adapter_1_out_of_rank = get_out_of_rank_params(lora_params, 1, 8)

        # Training loop
        for step in range(10):

            def loss_for_lora(lora_params):
                merged_model = nnx.merge(graphdef, lora_params, non_lora_params)
                return loss_fn(merged_model, input_ids, target_ids, attention_mask)

            loss_and_grad_fn = nnx.value_and_grad(loss_for_lora)
            loss, lora_grads = loss_and_grad_fn(lora_params)

            optimizer.update(lora_params, lora_grads)

            print(f"Step {step}: loss = {float(loss):.4f}")

        # Verify adapter 2 (unused) was not modified
        final_adapter_2_params = get_adapter_params(lora_params, 2)
        verify_params_unchanged(initial_adapter_2_params, final_adapter_2_params, "Adapter 2 was modified")

        # Verify out-of-rank params were not modified
        final_adapter_0_out_of_rank = get_out_of_rank_params(lora_params, 0, 16)
        verify_params_unchanged(
            initial_adapter_0_out_of_rank, final_adapter_0_out_of_rank, "Adapter 0 out-of-rank params modified"
        )
        final_adapter_1_out_of_rank = get_out_of_rank_params(lora_params, 1, 8)
        verify_params_unchanged(
            initial_adapter_1_out_of_rank, final_adapter_1_out_of_rank, "Adapter 1 out-of-rank params modified"
        )

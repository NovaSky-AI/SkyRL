from flax import nnx
import jax
import jax.numpy as jnp
import optax
from huggingface_hub import snapshot_download
from transformers import PretrainedConfig

from tx.models.configs import DeepseekV3Config
from tx.models.deepseekv3 import DeepseekV3ForCausalLM
from tx.utils.models import get_dtype, load_safetensors
from tx.layers.lora import init_lora_adapter
from tx.tinker.types import LoraConfig


def _is_routed_expert_path(path) -> bool:
    """Disambiguate shared_experts and experts"""
    keys = []
    for p in path:
        if hasattr(p, "key"):
            keys.append(str(p.key))
        elif hasattr(p, "name"):
            keys.append(str(p.name))

    for i, key in enumerate(keys):
        if key == "experts" and i > 0 and keys[i - 1] == "mlp":
            return True
    return False


def _get_out_of_rank_params(params, adapter_idx: int, rank: int, num_experts: int):
    """Extract out-of-rank params, using effective rank for routed expert layers."""

    def slice_param(path, p):
        path_str = str(path)

        if _is_routed_expert_path(path):
            effective_rank = max(1, rank // num_experts)
        else:
            effective_rank = rank

        if "lora_A" in path_str:
            # lora_A shape: [adapters, ..., max_rank] - slice last dim
            return p[adapter_idx, ..., effective_rank:].copy()
        elif "lora_B" in path_str:
            # lora_B shape: [adapters, ..., max_rank, out] - slice second-to-last dim
            return p[adapter_idx, ..., effective_rank:, :].copy()
        return p

    return jax.tree.map_with_path(slice_param, params)


def test_lora_training_moe_rank_normalized():
    base_model = "yujiepan/deepseek-v3-tiny-random"
    base_config = PretrainedConfig.from_pretrained(base_model, trust_remote_code=True)
    config = DeepseekV3Config(base_config, max_lora_adapters=5, max_lora_rank=32, shard_attention_heads=True)

    checkpoint_path = snapshot_download(base_model, allow_patterns=["*.safetensors"])
    mesh = jax.make_mesh(
        (1, 1, 1),
        ("fsdp", "ep", "tp"),
        axis_types=(jax.sharding.AxisType.Auto,) * 3,
    )
    with jax.set_mesh(mesh):
        model = DeepseekV3ForCausalLM(config, dtype=get_dtype(config.dtype), rngs=nnx.Rngs(0))
        load_safetensors(checkpoint_path, config, model)

        # Set different ranks for each adapter (0: rank 16, 1: rank 8)
        # For routed experts with 256 experts: effective rank = max(1, rank // 256) = 1
        # For other layers: effective rank = configured rank
        init_lora_adapter(model, adapter_index=0, lora_config=LoraConfig(rank=16, alpha=16, seed=0))
        init_lora_adapter(model, adapter_index=1, lora_config=LoraConfig(rank=8, alpha=8, seed=1))

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

        graphdef, lora_params, non_lora_params = nnx.split(model, model.is_lora_param, ...)

        def get_adapter_params(params, adapter_idx):
            return jax.tree.map(lambda p: p[adapter_idx].copy(), params)

        num_experts = config.n_routed_experts

        # Save initial states
        initial_adapter_2_params = get_adapter_params(lora_params, 2)
        initial_adapter_0_out_of_rank = _get_out_of_rank_params(lora_params, 0, 16, num_experts)
        initial_adapter_1_out_of_rank = _get_out_of_rank_params(lora_params, 1, 8, num_experts)

        initial_loss = None

        # Training loop
        for step in range(10):

            def loss_for_lora(lora_params):
                merged_model = nnx.merge(graphdef, lora_params, non_lora_params)
                return loss_fn(merged_model, input_ids, target_ids, attention_mask)

            loss_and_grad_fn = nnx.value_and_grad(loss_for_lora)
            loss, lora_grads = loss_and_grad_fn(lora_params)

            if initial_loss is None:
                initial_loss = float(loss)

            optimizer.update(lora_params, lora_grads)

            print(f"Step {step}: loss = {float(loss):.4f}")

        final_loss = float(loss)

        def verify_params_unchanged(initial_params, final_params, error_msg_prefix):
            for (path, initial), (_, final) in zip(
                jax.tree.leaves_with_path(initial_params), jax.tree.leaves_with_path(final_params)
            ):
                assert jnp.allclose(initial, final), f"{error_msg_prefix} for {path}"

        assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"

        # Verify unused adapter was not modified
        final_adapter_2_params = get_adapter_params(lora_params, 2)
        verify_params_unchanged(initial_adapter_2_params, final_adapter_2_params, "Adapter 2 was modified")

        # Verify out-of-rank params were not modified
        final_adapter_0_out_of_rank = _get_out_of_rank_params(lora_params, 0, 16, num_experts)
        verify_params_unchanged(
            initial_adapter_0_out_of_rank, final_adapter_0_out_of_rank, "Adapter 0 out-of-rank params modified"
        )
        final_adapter_1_out_of_rank = _get_out_of_rank_params(lora_params, 1, 8, num_experts)
        verify_params_unchanged(
            initial_adapter_1_out_of_rank, final_adapter_1_out_of_rank, "Adapter 1 out-of-rank params modified"
        )


def test_lora_training_high_rank():
    base_model = "yujiepan/deepseek-v3-tiny-random"
    base_config = PretrainedConfig.from_pretrained(base_model, trust_remote_code=True)
    config = DeepseekV3Config(base_config, max_lora_adapters=5, max_lora_rank=32, shard_attention_heads=True)

    checkpoint_path = snapshot_download(base_model, allow_patterns=["*.safetensors"])
    mesh = jax.make_mesh(
        (1, 1, 1),
        ("fsdp", "ep", "tp"),
        axis_types=(jax.sharding.AxisType.Auto,) * 3,
    )
    with jax.set_mesh(mesh):
        model = DeepseekV3ForCausalLM(config, dtype=get_dtype(config.dtype), rngs=nnx.Rngs(0))
        load_safetensors(checkpoint_path, config, model)

        init_lora_adapter(model, adapter_index=0, lora_config=LoraConfig(rank=16, alpha=16, seed=0))
        init_lora_adapter(model, adapter_index=1, lora_config=LoraConfig(rank=8, alpha=8, seed=1))

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

        graphdef, lora_params, non_lora_params = nnx.split(model, model.is_lora_param, ...)

        def get_adapter_params(params, adapter_idx):
            return jax.tree.map(lambda p: p[adapter_idx].copy(), params)

        num_experts = config.n_routed_experts

        # Save initial states for all unused adapters
        initial_adapter_2_params = get_adapter_params(lora_params, 2)
        initial_adapter_3_params = get_adapter_params(lora_params, 3)
        initial_adapter_4_params = get_adapter_params(lora_params, 4)

        # Save out-of-rank params for adapters 0 and 1
        initial_adapter_0_out_of_rank = _get_out_of_rank_params(lora_params, 0, 16, num_experts)
        initial_adapter_1_out_of_rank = _get_out_of_rank_params(lora_params, 1, 8, num_experts)

        # Training loop
        for step in range(10):

            def loss_for_lora(lora_params):
                merged_model = nnx.merge(graphdef, lora_params, non_lora_params)
                return loss_fn(merged_model, input_ids, target_ids, attention_mask)

            loss_and_grad_fn = nnx.value_and_grad(loss_for_lora)
            loss, lora_grads = loss_and_grad_fn(lora_params)

            optimizer.update(lora_params, lora_grads)

            print(f"Step {step}: loss = {float(loss):.4f}")

        def verify_params_unchanged(initial_params, final_params, error_msg_prefix):
            for (path, initial), (_, final) in zip(
                jax.tree.leaves_with_path(initial_params), jax.tree.leaves_with_path(final_params)
            ):
                assert jnp.allclose(initial, final), f"{error_msg_prefix} for {path}"

        # Verify unused adapters (2, 3, 4) were not modified
        final_adapter_2_params = get_adapter_params(lora_params, 2)
        verify_params_unchanged(initial_adapter_2_params, final_adapter_2_params, "Adapter 2 was modified")

        final_adapter_3_params = get_adapter_params(lora_params, 3)
        verify_params_unchanged(initial_adapter_3_params, final_adapter_3_params, "Adapter 3 was modified")

        final_adapter_4_params = get_adapter_params(lora_params, 4)
        verify_params_unchanged(initial_adapter_4_params, final_adapter_4_params, "Adapter 4 was modified")

        # Verify out-of-rank params were not modified
        final_adapter_0_out_of_rank = _get_out_of_rank_params(lora_params, 0, 16, num_experts)
        verify_params_unchanged(
            initial_adapter_0_out_of_rank, final_adapter_0_out_of_rank, "Adapter 0 out-of-rank params modified"
        )
        final_adapter_1_out_of_rank = _get_out_of_rank_params(lora_params, 1, 8, num_experts)
        verify_params_unchanged(
            initial_adapter_1_out_of_rank, final_adapter_1_out_of_rank, "Adapter 1 out-of-rank params modified"
        )

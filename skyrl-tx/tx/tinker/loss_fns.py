"""Loss functions for training."""

from collections.abc import Callable
import jax
import jax.numpy as jnp
from flax import nnx


def safe_loss_mask(loss_output: jax.Array, loss_mask: jax.Array) -> jax.Array:
    "Strongly mask the loss_output to 0.0 if the loss_mask is zero."
    return jnp.where(loss_mask != 0.0, loss_mask * loss_output, jnp.zeros_like(loss_output))


def cross_entropy_loss(
    target_logprobs: jax.Array, loss_mask: jax.Array, sampling_logprobs: jax.Array, advantages: jax.Array
) -> jax.Array:
    "Standard cross-entropy loss (i.e., negative log-likelihood)."
    return -safe_loss_mask(target_logprobs, loss_mask)


def importance_sampling_loss(
    target_logprobs: jax.Array, loss_mask: jax.Array, sampling_logprobs: jax.Array, advantages: jax.Array
) -> jax.Array:
    "Importance sampling loss with target_logprobs from learner policy and sampling_logprobs from sampling policy."
    prob_ratio = jnp.exp(target_logprobs - sampling_logprobs)
    return -safe_loss_mask(prob_ratio * advantages, loss_mask)


def ppo_loss(
    target_logprobs: jax.Array, loss_mask: jax.Array, sampling_logprobs: jax.Array, advantages: jax.Array
) -> jax.Array:
    "PPO style clipped version of the importance sampling loss."
    prob_ratio = jnp.exp(target_logprobs - sampling_logprobs)
    clipped_ratio = jnp.clip(prob_ratio, 0.8, 1.2)
    unclipped = prob_ratio * advantages
    clipped = clipped_ratio * advantages
    return -safe_loss_mask(jnp.minimum(unclipped, clipped), loss_mask)


# Map from string names to loss functions
# The ordering of this map determines the indices used in jax.lax.switch
LOSS_FUNCTION_MAP = {
    "cross_entropy": cross_entropy_loss,
    "importance_sampling": importance_sampling_loss,
    "ppo": ppo_loss,
}

# Map from loss function name to index (for jax.lax.switch)
LOSS_TYPES = {name: idx for idx, name in enumerate(LOSS_FUNCTION_MAP.keys())}

# List of loss functions in order (for jax.lax.switch)
LOSS_FUNCTIONS = list(LOSS_FUNCTION_MAP.values())


def loss_and_grad_fn_lora(
    mesh: jax.sharding.Mesh,
    gradient_checkpointing: bool,
    graphdef: nnx.GraphDef,
    enforce_eager: bool,
    lora_params: nnx.State,
    non_lora_params: nnx.State,
) -> Callable:
    """Compile and cache the loss function to avoid re-jitting on every call."""

    # Wrap the model forward call to use nnx.remat for gradient checkpointing
    def _model_forward(
        model: nnx.Module, input_ids: jax.Array, attention_mask: jax.Array, adapter_indices: jax.Array
    ) -> jax.Array:
        output = model(input_ids, attention_mask=attention_mask, adapter_indices=adapter_indices)
        return output.logits

    if gradient_checkpointing:
        # policy=None corresponds full activation recomputation
        _model_forward = nnx.remat(_model_forward, policy=None)

    def loss_for_lora(
        lora_params: nnx.State,
        non_lora_params: nnx.State,
        input_ids: jax.Array,
        attention_mask: jax.Array,
        adapter_indices: jax.Array,
        target_ids: jax.Array,
        loss_mask: jax.Array,
        loss_fn_types: jax.Array,
        sampling_logprobs: jax.Array,
        advantages: jax.Array,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        model = nnx.merge(graphdef, lora_params, non_lora_params)
        logits = _model_forward(model, input_ids, attention_mask, adapter_indices)  # [B, T, V]

        logprobs = jax.nn.log_softmax(logits, axis=-1)  # [B, T, V]
        target_logprobs = jnp.take_along_axis(logprobs, target_ids[..., None], axis=-1).squeeze(-1)

        def compute_loss_per_example(loss_fn_type, target_logprobs, loss_mask, sampling_logprobs, advantages):
            return jax.lax.switch(
                loss_fn_type,
                LOSS_FUNCTIONS,
                target_logprobs,
                loss_mask,
                sampling_logprobs,
                advantages,
            )

        per_token_losses = jax.vmap(compute_loss_per_example)(
            loss_fn_types,
            target_logprobs,
            loss_mask,
            sampling_logprobs,
            advantages,
        )

        per_seq_loss = per_token_losses.sum(axis=-1) / loss_mask.sum(axis=-1)
        # Return sum of losses (we'll divide gradients by per-adapter batch size later)
        return per_seq_loss.sum(), (logits, per_token_losses)

    # Only differentiate with respect to lora_params (argnums=0)
    loss_and_grad_fn = jax.value_and_grad(loss_for_lora, argnums=0, has_aux=True)

    if enforce_eager:
        # Disable JIT compilation for debugging
        return loss_and_grad_fn
    else:
        # Retrieve the sharding of lora and non_lora params and compute the sharding of inputs and outputs
        lora_shardings = jax.tree.map(
            lambda spec: jax.NamedSharding(mesh, spec), nnx.get_partition_spec(lora_params)
        )
        non_lora_shardings = jax.tree.map(
            lambda spec: jax.NamedSharding(mesh, spec), nnx.get_partition_spec(non_lora_params)
        )
        replicated = jax.NamedSharding(mesh, jax.P(None))
        scalar = jax.NamedSharding(mesh, jax.P())
        return jax.jit(
            loss_and_grad_fn,
            # One input sharding parameter for each argument of loss_for_lora
            in_shardings=(lora_shardings, non_lora_shardings) + (replicated,) * 8,
            # One output sharding parameter for each return value of loss_for_lora
            out_shardings=((scalar, (replicated, replicated)), lora_shardings),
        )


def loss_and_grad_fn_full_finetuning(
    mesh: jax.sharding.Mesh,
    gradient_checkpointing: bool,
    graphdef: nnx.GraphDef,
    enforce_eager: bool,
    lora_params: nnx.State,
    non_lora_params: nnx.State,
) -> Callable:
    """Compile and cache the loss function to avoid re-jitting on every call."""

    # Wrap the model forward call to use nnx.remat for gradient checkpointing
    def _model_forward(
        model: nnx.Module, input_ids: jax.Array, attention_mask: jax.Array, adapter_indices: jax.Array
    ) -> jax.Array:
        output = model(input_ids, attention_mask=attention_mask, adapter_indices=adapter_indices)
        return output.logits

    if gradient_checkpointing:
        # policy=None corresponds full activation recomputation
        _model_forward = nnx.remat(_model_forward, policy=None)

    def loss_full_parameter(
        lora_params: nnx.State,
        non_lora_params: nnx.State,
        input_ids: jax.Array,
        attention_mask: jax.Array,
        adapter_indices: jax.Array,
        target_ids: jax.Array,
        loss_mask: jax.Array,
        loss_fn_types: jax.Array,
        sampling_logprobs: jax.Array,
        advantages: jax.Array,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        model = nnx.merge(graphdef, lora_params, non_lora_params)
        logits = _model_forward(model, input_ids, attention_mask, adapter_indices)  # [B, T, V]

        logprobs = jax.nn.log_softmax(logits, axis=-1)  # [B, T, V]
        target_logprobs = jnp.take_along_axis(logprobs, target_ids[..., None], axis=-1).squeeze(-1)

        def compute_loss_per_example(loss_fn_type, target_logprobs, loss_mask, sampling_logprobs, advantages):
            return jax.lax.switch(
                loss_fn_type,
                LOSS_FUNCTIONS,
                target_logprobs,
                loss_mask,
                sampling_logprobs,
                advantages,
            )

        per_token_losses = jax.vmap(compute_loss_per_example)(
            loss_fn_types,
            target_logprobs,
            loss_mask,
            sampling_logprobs,
            advantages,
        )

        per_seq_loss = per_token_losses.sum(axis=-1) / loss_mask.sum(axis=-1)
        # Return sum of losses (we'll divide gradients by per-adapter batch size later)
        return per_seq_loss.sum(), (logits, per_token_losses)

    # Differentiate with respect to base model's parameters (argnums=1)
    loss_and_grad_fn = nnx.value_and_grad(loss_full_parameter, argnums=1, has_aux=True, allow_int=True)
    

    from jax.tree_util import tree_map_with_path
    from flax.nnx.graph import Node  # Import this for cleaning up paths

    def print_integer_leaves(state: nnx.State, name: str):
        """Recursively traverses a Pytree and prints info about integer-based leaves."""
        print(f"\n🔍 Checking for integer leaves in: {name}")
        
        def check_leaf(path, value):
            # Check if the leaf is a JAX array or an nnx.Variable/Param
            if hasattr(value, 'value') and hasattr(value.value, 'dtype'):
                dtype = value.value.dtype
                if jnp.issubdtype(dtype, jnp.integer):
                    # Create a clean string path
                    path_str = ".".join(
                        p.key if isinstance(p, Node) else str(p) for p in path
                    )
                    var_type = type(value).__name__
                    shape = value.value.shape
                    
                    print(f"  🔴 INTEGER LEAF FOUND: {path_str}")
                    print(f"      Type: {var_type}, Dtype: {dtype}, Shape: {shape}")
        
            # Run the check on every leaf
            tree_map_with_path(check_leaf, state)
            print(f"✅ Finished checking: {name}\n")
            
    print_integer_leaves(non_lora_params, "non_lora_params in full finetuning loss")

    if enforce_eager:
        # Disable JIT compilation for debugging
        return loss_and_grad_fn
    else:
        # Retrieve the sharding of lora and non_lora params and compute the sharding of inputs and outputs
        lora_shardings = jax.tree.map(
            lambda spec: jax.NamedSharding(mesh, spec), nnx.get_partition_spec(lora_params)
        )
        non_lora_shardings = jax.tree.map(
            lambda spec: jax.NamedSharding(mesh, spec), nnx.get_partition_spec(non_lora_params)
        )
        replicated = jax.NamedSharding(mesh, jax.P(None))
        scalar = jax.NamedSharding(mesh, jax.P())
        
        return jax.jit(
            loss_and_grad_fn,
            # One input sharding parameter for each argument of loss_full_parameter
            in_shardings=(lora_shardings, non_lora_shardings) + (replicated,) * 8,
            # One output sharding parameter for each return value of loss_for_lora
            out_shardings=((scalar, (replicated, replicated)), non_lora_shardings),
        )
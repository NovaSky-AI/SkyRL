"""Loss functions for training (JAX implementations)."""

import jax
import jax.numpy as jnp

from tx.tinker.types import LOSS_TYPES


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
LOSS_FUNCTION_MAP = {
    "cross_entropy": cross_entropy_loss,
    "importance_sampling": importance_sampling_loss,
    "ppo": ppo_loss,
}

# Build list of functions indexed by LOSS_TYPES values (for jax.lax.switch)
# Sort by index to ensure LOSS_FUNCTIONS[idx] corresponds to the correct function
LOSS_FUNCTIONS = [LOSS_FUNCTION_MAP[name] for name, idx in sorted(LOSS_TYPES.items(), key=lambda x: x[1])]

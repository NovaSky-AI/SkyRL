"""Loss functions for training (JAX implementations)."""

import jax
import jax.numpy as jnp

from tx.tinker.types import LOSS_TYPE_NAMES


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
# The ordering must match LOSS_TYPE_NAMES from loss_types.py
LOSS_FUNCTION_MAP = {
    "cross_entropy": cross_entropy_loss,
    "importance_sampling": importance_sampling_loss,
    "ppo": ppo_loss,
}

# Validate that our implementations match the canonical list
assert list(LOSS_FUNCTION_MAP.keys()) == LOSS_TYPE_NAMES, (
    f"LOSS_FUNCTION_MAP keys {list(LOSS_FUNCTION_MAP.keys())} "
    f"don't match LOSS_TYPE_NAMES {LOSS_TYPE_NAMES}"
)

# List of loss functions in order (for jax.lax.switch)
# Order is determined by LOSS_TYPE_NAMES
LOSS_FUNCTIONS = [LOSS_FUNCTION_MAP[name] for name in LOSS_TYPE_NAMES]

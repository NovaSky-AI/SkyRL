"""Loss functions for training."""

import jax.numpy as jnp


def _cross_entropy_loss(target_logprobs, loss_mask, sampling_logprobs, advantages):
    return -target_logprobs * loss_mask


def _importance_sampling_loss(target_logprobs, loss_mask, sampling_logprobs, advantages):
    prob_ratio = jnp.exp(target_logprobs - sampling_logprobs)
    return -(prob_ratio * advantages * loss_mask)


def _ppo_loss(target_logprobs, loss_mask, sampling_logprobs, advantages):
    prob_ratio = jnp.exp(target_logprobs - sampling_logprobs)
    clipped_ratio = jnp.clip(prob_ratio, 0.8, 1.2)
    unclipped = prob_ratio * advantages
    clipped = clipped_ratio * advantages
    return -jnp.minimum(unclipped, clipped) * loss_mask


# Map from string names to loss functions
# The ordering of this map determines the indices used in jax.lax.switch
LOSS_FUNCTION_MAP = {
    "cross_entropy": _cross_entropy_loss,
    "importance_sampling": _importance_sampling_loss,
    "ppo": _ppo_loss,
}

# Map from loss function name to index (for jax.lax.switch)
LOSS_TYPES = {name: idx for idx, name in enumerate(LOSS_FUNCTION_MAP.keys())}

# List of loss functions in order (for jax.lax.switch)
LOSS_FUNCTIONS = list(LOSS_FUNCTION_MAP.values())

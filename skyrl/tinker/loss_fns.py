"""Loss functions for training (JAX implementations)."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from skyrl.tinker.types import LOSS_TYPES


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LossFnConfig:
    """Fixed loss config arrays passed through the JAX loss path."""

    clip_low_threshold: jax.Array
    clip_high_threshold: jax.Array


def safe_loss_mask(loss_output: jax.Array, loss_mask: jax.Array) -> jax.Array:
    "Strongly mask the loss_output to 0.0 if the loss_mask is zero."
    return jnp.where(loss_mask != 0.0, loss_mask * loss_output, jnp.zeros_like(loss_output))


def cross_entropy_loss(
    target_logprobs: jax.Array,
    loss_mask: jax.Array,
    _sampling_logprobs: jax.Array,
    _advantages: jax.Array,
    _loss_fn_config: LossFnConfig,
) -> jax.Array:
    "Standard cross-entropy loss (i.e., negative log-likelihood)."
    return -safe_loss_mask(target_logprobs, loss_mask)


def importance_sampling_loss(
    target_logprobs: jax.Array,
    loss_mask: jax.Array,
    sampling_logprobs: jax.Array,
    advantages: jax.Array,
    _loss_fn_config: LossFnConfig,
) -> jax.Array:
    "Importance sampling loss with target_logprobs from learner policy and sampling_logprobs from sampling policy."
    prob_ratio = jnp.exp(target_logprobs - sampling_logprobs)
    return -safe_loss_mask(prob_ratio * advantages, loss_mask)


def ppo_loss(
    target_logprobs: jax.Array,
    loss_mask: jax.Array,
    sampling_logprobs: jax.Array,
    advantages: jax.Array,
    loss_fn_config: LossFnConfig,
) -> jax.Array:
    "PPO style clipped version of the importance sampling loss."
    prob_ratio = jnp.exp(target_logprobs - sampling_logprobs)
    clip_low_threshold = loss_fn_config.clip_low_threshold
    clip_high_threshold = loss_fn_config.clip_high_threshold
    clipped_ratio = jnp.clip(prob_ratio, clip_low_threshold, clip_high_threshold)
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

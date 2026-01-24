"""Mixin for logits computation in causal language models."""

from abc import abstractmethod
from typing import Callable

import jax
import jax.numpy as jnp


# lm_head: (hidden_states, adapter_indices) -> logits
LMHead = Callable[[jax.Array, jax.Array | None], jax.Array]


class LogitsProcessorMixin:
    """Mixin providing logits/logprobs computation for causal language models."""

    @abstractmethod
    def get_lm_head(self) -> LMHead:
        """Return the lm_head callable for logits computation."""
        ...

    def compute_logits(
        self,
        hidden_states: jax.Array,
        adapter_indices: jax.Array | None = None,
    ) -> jax.Array:
        """Compute logits from hidden states. For sampling.

        Args:
            hidden_states: Hidden states from model forward [B, T, H].
            adapter_indices: Optional adapter indices for LoRA.

        Returns:
            Logits [B, T, V].
        """
        return self.get_lm_head()(hidden_states, adapter_indices)

    def compute_logprobs(
        self,
        hidden_states: jax.Array,
        target_ids: jax.Array,
        adapter_indices: jax.Array | None = None,
    ) -> jax.Array:
        """Compute logprobs from hidden states. For training and prompt logprobs.

        Args:
            hidden_states: Hidden states [B, T, H].
            target_ids: Target token IDs [B, T].
            adapter_indices: Adapter indices for LoRA on lm_head.

        Returns:
            Log probabilities for target tokens [B, T].
        """
        logits = self.compute_logits(hidden_states, adapter_indices)
        return self.logits_to_logprobs(logits, target_ids)

    @staticmethod
    def logits_to_logprobs(logits: jax.Array, target_ids: jax.Array) -> jax.Array:
        """Convert logits to logprobs. For decode logprobs when logits already computed.

        Args:
            logits: Logits [B, T, V] or [B, V].
            target_ids: Target token IDs [B, T] or [B].

        Returns:
            Log probabilities for target tokens [B, T] or [B].
        """
        log_sum_exp = jax.nn.logsumexp(logits, axis=-1, keepdims=True)
        target_logits = jnp.take_along_axis(logits, target_ids[..., None], axis=-1)
        return (target_logits - log_sum_exp).squeeze(-1)

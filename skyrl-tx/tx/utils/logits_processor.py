"""Mixin for logits computation in causal language models."""

from abc import abstractmethod
from typing import Callable

import jax
import jax.numpy as jnp

from tx.models.types import ModelForCausalLM

# lm_head: (hidden_states, adapter_indices) -> logits
LMHead = Callable[[jax.Array, jax.Array | None], jax.Array]


class LogitsProcessorMixin(ModelForCausalLM):
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
        chunk_size = self.config.loss_chunk_size
        if chunk_size > 0:
            return self._compute_chunked_logprobs(hidden_states, target_ids, chunk_size, adapter_indices)
        else:
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

    def _compute_chunked_logprobs(
        self,
        hidden_states: jax.Array,
        target_ids: jax.Array,
        chunk_size: int,
        adapter_indices: jax.Array | None,
    ) -> jax.Array:
        """Compute log probabilities using chunked lm_head computation.

        This avoids materializing the full [B*T, V] logits tensor by computing
        lm_head and log probabilities for each chunk sequentially.
        """
        B, T, H = hidden_states.shape
        total_tokens = B * T
        lm_head = self.get_lm_head()

        # Flatten batch and sequence dimensions
        flat_hidden = hidden_states.reshape(-1, H)  # [B*T, H]
        flat_target_ids = target_ids.reshape(-1)  # [B*T]

        # Expand adapter_indices from [B] to [B*T] by repeating each T times
        if adapter_indices is not None:
            flat_adapter_indices = jnp.repeat(adapter_indices, T)  # [B*T]
        else:
            flat_adapter_indices = None

        # Pad to multiple of chunk_size for clean slicing
        num_chunks = (total_tokens + chunk_size - 1) // chunk_size
        padded_size = num_chunks * chunk_size
        pad_amount = padded_size - total_tokens

        if pad_amount > 0:
            flat_hidden = jnp.pad(flat_hidden, ((0, pad_amount), (0, 0)))
            flat_target_ids = jnp.pad(flat_target_ids, (0, pad_amount))
            if flat_adapter_indices is not None:
                flat_adapter_indices = jnp.pad(flat_adapter_indices, (0, pad_amount))

        # Reshape into chunks: [num_chunks, chunk_size, H] and [num_chunks, chunk_size]
        chunked_hidden = flat_hidden.reshape(num_chunks, chunk_size, H)
        chunked_targets = flat_target_ids.reshape(num_chunks, chunk_size)
        if flat_adapter_indices is not None:
            chunked_adapter_indices = flat_adapter_indices.reshape(num_chunks, chunk_size)
        else:
            chunked_adapter_indices = None

        def compute_chunk_logprobs(args):
            """Compute lm_head and log probabilities for a chunk of tokens."""
            chunk_hidden, chunk_targets, chunk_adapters = args
            # Compute logits for this chunk: [chunk_size, H] -> [chunk_size, V]
            chunk_logits = lm_head(chunk_hidden, chunk_adapters)
            # Compute log probabilities
            log_sum_exp = jax.nn.logsumexp(chunk_logits, axis=-1, keepdims=True)
            target_logits = jnp.take_along_axis(chunk_logits, chunk_targets[..., None], axis=-1)
            return (target_logits - log_sum_exp).squeeze(-1)

        if self.config.gradient_checkpointing:
            compute_chunk_logprobs = jax.checkpoint(compute_chunk_logprobs, policy=None)

        # Process chunks sequentially using lax.map (not vmap) to reduce memory
        if chunked_adapter_indices is not None:
            all_logprobs = jax.lax.map(compute_chunk_logprobs, (chunked_hidden, chunked_targets, chunked_adapter_indices))
        else:
            # Create dummy array for lax.map (needs consistent structure)
            dummy_adapters = jnp.zeros((num_chunks, chunk_size), dtype=jnp.int32)
            def compute_chunk_logprobs_no_adapter(args):
                chunk_hidden, chunk_targets, _ = args
                chunk_logits = lm_head(chunk_hidden, None)
                log_sum_exp = jax.nn.logsumexp(chunk_logits, axis=-1, keepdims=True)
                target_logits = jnp.take_along_axis(chunk_logits, chunk_targets[..., None], axis=-1)
                return (target_logits - log_sum_exp).squeeze(-1)
            if self.config.gradient_checkpointing:
                compute_chunk_logprobs_no_adapter = jax.checkpoint(compute_chunk_logprobs_no_adapter, policy=None)
            all_logprobs = jax.lax.map(compute_chunk_logprobs_no_adapter, (chunked_hidden, chunked_targets, dummy_adapters))

        # Flatten and slice to original size, then reshape to [B, T]
        return all_logprobs.reshape(-1)[:total_tokens].reshape(B, T)

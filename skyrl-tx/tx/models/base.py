"""Base class for causal language models."""

import jax

from tx.layers.logits_processor import LogitsProcessor


class CausalLMBase:
    """Base class providing logits/logprobs computation for causal language models.

    Subclasses must set:
        - lm_head: The language model head (callable)
        - lm_head_weight: The lm_head weight matrix [H, V]
    """

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
        return LogitsProcessor.compute_logits(hidden_states, self.lm_head, adapter_indices)

    def compute_logprobs(
        self,
        hidden_states: jax.Array,
        target_ids: jax.Array,
        chunk_size: int = 0,
        gradient_checkpointing: bool = False,
    ) -> jax.Array:
        """Compute logprobs from hidden states. For training and prompt logprobs.

        Supports chunked computation to avoid materializing full [B*T, V] logits.

        Args:
            hidden_states: Hidden states [B, T, H].
            target_ids: Target token IDs [B, T].
            chunk_size: Chunk size for chunked computation (0 = non-chunked).
            gradient_checkpointing: Whether to checkpoint each chunk.

        Returns:
            Log probabilities for target tokens [B, T].
        """
        return LogitsProcessor.compute_logprobs(
            hidden_states, self.lm_head_weight, target_ids, chunk_size, gradient_checkpointing
        )

    @staticmethod
    def logits_to_logprobs(logits: jax.Array, target_ids: jax.Array) -> jax.Array:
        """Convert logits to logprobs. For decode logprobs when logits already computed.

        Args:
            logits: Logits [B, T, V] or [B, V].
            target_ids: Target token IDs [B, T] or [B].

        Returns:
            Log probabilities for target tokens [B, T] or [B].
        """
        return LogitsProcessor.logits_to_logprobs(logits, target_ids)

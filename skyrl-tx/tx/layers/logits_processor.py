"""LogitsProcessor for computing logits and logprobs from hidden states."""

import jax
import jax.numpy as jnp


class LogitsProcessor:
    """Handles logits and log probability computation from hidden states."""

    def __init__(self, config) -> None:
        self.config = config

    def __call__(
        self,
        hidden_states: jax.Array,
        lm_head,
        adapter_indices: jax.Array | None = None,
        skip_prompt_logits: bool = False,
    ) -> jax.Array:
        """Compute logits from hidden states (for sampling).

        Args:
            hidden_states: Hidden states from the model backbone.
            lm_head: Language model head (LoRALinear or embed_tokens.T).
            adapter_indices: Optional adapter indices for LoRA.
            skip_prompt_logits: If True, only compute logits for the last token (saves memory).
        """
        if skip_prompt_logits:
            hidden_states = hidden_states[:, -1:, :]
        return lm_head(hidden_states, adapter_indices)

    @staticmethod
    def compute_logprobs(
        forward_output: jax.Array,
        target_ids: jax.Array,
        lm_head_weight: jax.Array | None = None,
        chunk_size: int = 0,
        gradient_checkpointing: bool = False,
    ) -> jax.Array:
        """Compute log probabilities from model forward output.

        Supports two modes:
        - Chunked: forward_output is hidden_states [B, T, H], requires lm_head_weight
        - Non-chunked: forward_output is logits [B, T, V]

        Args:
            forward_output: Either hidden_states [B, T, H] (chunked) or logits [B, T, V].
            target_ids: Target token IDs [B, T].
            lm_head_weight: LM head weight matrix [H, V] for chunked mode (None for non-chunked).
            chunk_size: Chunk size for chunked computation (0 or negative = non-chunked).
            gradient_checkpointing: Whether to checkpoint each chunk (chunked mode only).

        Returns:
            Log probabilities for target tokens [B, T].
        """
        use_chunked = lm_head_weight is not None and chunk_size > 0

        if use_chunked:
            return LogitsProcessor._compute_chunked_logprobs(
                forward_output, lm_head_weight, target_ids, chunk_size, gradient_checkpointing
            )
        else:
            return LogitsProcessor._logits_to_logprobs(forward_output, target_ids)

    @staticmethod
    def _logits_to_logprobs(logits: jax.Array, target_ids: jax.Array) -> jax.Array:
        """Convert logits to log probabilities for target tokens."""
        log_sum_exp = jax.nn.logsumexp(logits, axis=-1, keepdims=True)
        target_logits = jnp.take_along_axis(logits, target_ids[..., None], axis=-1)
        return (target_logits - log_sum_exp).squeeze(-1)

    @staticmethod
    def _compute_chunked_logprobs(
        hidden_states: jax.Array,
        lm_head_weight: jax.Array,
        target_ids: jax.Array,
        chunk_size: int,
        gradient_checkpointing: bool,
    ) -> jax.Array:
        """Compute log probabilities using chunked lm_head computation.

        This avoids materializing the full [B*T, V] logits tensor by computing
        lm_head and log probabilities for each chunk sequentially.
        """
        B, T, H = hidden_states.shape
        total_tokens = B * T

        # Flatten batch and sequence dimensions
        flat_hidden = hidden_states.reshape(-1, H)  # [B*T, H]
        flat_target_ids = target_ids.reshape(-1)  # [B*T]

        # Pad to multiple of chunk_size for clean slicing
        num_chunks = (total_tokens + chunk_size - 1) // chunk_size
        padded_size = num_chunks * chunk_size
        pad_amount = padded_size - total_tokens

        if pad_amount > 0:
            flat_hidden = jnp.pad(flat_hidden, ((0, pad_amount), (0, 0)))
            flat_target_ids = jnp.pad(flat_target_ids, (0, pad_amount))

        # Reshape into chunks: [num_chunks, chunk_size, H] and [num_chunks, chunk_size]
        chunked_hidden = flat_hidden.reshape(num_chunks, chunk_size, H)
        chunked_targets = flat_target_ids.reshape(num_chunks, chunk_size)

        def compute_chunk_logprobs(args):
            """Compute lm_head and log probabilities for a chunk of tokens."""
            chunk_hidden, chunk_targets = args
            # Compute logits for this chunk only: [chunk_size, H] @ [H, V] = [chunk_size, V]
            chunk_logits = chunk_hidden @ lm_head_weight
            # Compute log probabilities
            log_sum_exp = jax.nn.logsumexp(chunk_logits, axis=-1, keepdims=True)
            target_logits = jnp.take_along_axis(chunk_logits, chunk_targets[..., None], axis=-1)
            return (target_logits - log_sum_exp).squeeze(-1)

        if gradient_checkpointing:
            compute_chunk_logprobs = jax.checkpoint(compute_chunk_logprobs, policy=None)

        # Process chunks sequentially using lax.map (not vmap) to reduce memory
        all_logprobs = jax.lax.map(compute_chunk_logprobs, (chunked_hidden, chunked_targets))
        # Flatten and slice to original size, then reshape to [B, T]
        return all_logprobs.reshape(-1)[:total_tokens].reshape(B, T)

"""LogitsProcessor for computing logits from hidden states."""

import jax
import jax.numpy as jnp


class LogitsProcessor:
    """Computes logits from hidden states using lm_head."""

    def __init__(self, config) -> None:
        self.config = config

    def __call__(
        self,
        hidden_states: jax.Array,
        lm_head,
        attention_mask: jax.Array,
        adapter_indices: jax.Array | None = None,
        skip_prompt_logits: bool = False,
    ) -> jax.Array:
        """Compute logits from hidden states.

        Args:
            hidden_states: Hidden states from the model backbone.
            lm_head: Language model head (LoRALinear or embed_tokens.T).
            attention_mask: Attention mask to find the last real token position.
            adapter_indices: Optional adapter indices for LoRA.
            skip_prompt_logits: If True, only compute logits for the last token (saves memory).
        """
        if skip_prompt_logits:
            last_idx = attention_mask.sum(axis=1) - 1
            hidden_states = hidden_states[jnp.arange(hidden_states.shape[0]), last_idx][:, None, :]
        return lm_head(hidden_states, adapter_indices)

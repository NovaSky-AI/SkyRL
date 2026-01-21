"""LogitsProcessor for computing logits from hidden states."""

import jax


class LogitsProcessor:
    """Computes logits from hidden states using lm_head."""

    def __init__(self, config) -> None:
        self.config = config

    def __call__(
        self,
        hidden_states: jax.Array,
        lm_head,
        adapter_indices: jax.Array | None = None,
        skip_prompt_logits: bool = False,
    ) -> jax.Array:
        """Compute logits from hidden states.

        Args:
            hidden_states: Hidden states from the model backbone.
            lm_head: Language model head (LoRALinear or embed_tokens.T).
            adapter_indices: Optional adapter indices for LoRA.
            skip_prompt_logits: If True, only compute logits for the last token (saves memory).
        """
        if skip_prompt_logits:
            hidden_states = hidden_states[:, -1:, :]
        return lm_head(hidden_states, adapter_indices)

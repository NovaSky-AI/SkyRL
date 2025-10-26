"""Model-specific output dataclasses.

This module contains model-specific output types for various model architectures.
"""

from __future__ import annotations
from dataclasses import dataclass

import jax

from tx.utils.generator import KVCache


@jax.tree_util.register_dataclass
@dataclass
class Qwen3ModelOutput:
    """Output type for Qwen3Model.

    Attributes:
        last_hidden_state: The last hidden state from the model.
        kv_cache: The updated key-value cache.
        hidden_states: All hidden states if output_hidden_states=True.
    """

    last_hidden_state: jax.Array
    kv_cache: KVCache
    hidden_states: list[jax.Array] | None = None


@jax.tree_util.register_dataclass
@dataclass
class Qwen3CausalLMOutput:
    """Output type for Qwen3ForCausalLM.

    Attributes:
        logits: The language modeling logits.
        last_hidden_state: The last hidden state from the model.
        kv_cache: The updated key-value cache.
        hidden_states: All hidden states, if output_hidden_states=True.
    """

    logits: jax.Array
    last_hidden_state: jax.Array
    kv_cache: KVCache
    hidden_states: list[jax.Array] | None = None

"""Model output dataclasses for PyTorch."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List

import torch

from tx.torch.utils.generator import KVCache


@dataclass
class ModelOutput:
    """Output type for models like Qwen3Model.

    Attributes:
        last_hidden_state: The last hidden state from the model [B, T, hidden_size]
        kv_cache: The updated key-value cache
        hidden_states: All hidden states if output_hidden_states=True
    """

    last_hidden_state: torch.Tensor
    kv_cache: KVCache
    hidden_states: List[torch.Tensor] | None = None


@dataclass
class CausalLMOutput:
    """Output type for causal language models like Qwen3ForCausalLM.

    Attributes:
        logits: The language modeling logits [B, T, vocab_size]
        last_hidden_state: The last hidden state from the model [B, T, hidden_size]
        kv_cache: The updated key-value cache
        hidden_states: All hidden states, if output_hidden_states=True
    """

    logits: torch.Tensor
    last_hidden_state: torch.Tensor
    kv_cache: KVCache
    hidden_states: List[torch.Tensor] | None = None

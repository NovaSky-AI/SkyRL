from dataclasses import dataclass

import torch


@dataclass
class ModelOutput:

    last_hidden_state: torch.Tensor


@dataclass
class CausalLMOutput:

    logits: torch.Tensor
    last_hidden_state: torch.Tensor
import random
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import distributed as dist
from typing import Optional, Dict, Any, Union, TypeVar
import torch.optim as optim
from jaxtyping import Float


DataT = TypeVar("DataT", bound=Union[Dict[str, Any], torch.Tensor])


class DistributedStrategy(ABC):
    @abstractmethod
    def setup_distributed(self):
        pass

    @abstractmethod
    def all_reduce(self, data: DataT, op="mean") -> DataT:
        """Perform all_reduce across all processes"""
        pass

    @abstractmethod
    def all_gather(self, data: DataT) -> DataT:
        """Perform all_gather across all processes"""
        pass

    @abstractmethod
    def backward(self, loss: torch.Tensor, model, optimizer: optim.Optimizer, **kwargs):
        """Perform backward pass"""
        pass

    @abstractmethod
    def optimizer_step(
        self,
        optimizer: optim.Optimizer,
        model,
        scheduler,
        name="model",
        **kwargs,
    ) -> Optional[Float[torch.Tensor, "1"]]:
        """Perform optimizer step"""
        pass

    @abstractmethod
    def save_ckpt(self, model, optimizer, scheduler, ckpt_dir, global_step, node_local_rank):
        """Save checkpoint"""
        pass

    @abstractmethod
    def load_ckpt(self, model, optimizer, scheduler, ckpt_dir, global_step, node_local_rank):
        """Load checkpoint"""
        pass

    @abstractmethod
    def save_hf_model(self, model, output_dir: str, tokenizer=None, **kwargs):
        """Save model in HuggingFace safetensors format"""
        pass

    def print(self, *msg):
        """Print only on rank 0"""
        if self.is_rank_0():
            print(*msg)

    def is_rank_0(self) -> bool:
        """Check if current process is rank 0"""
        return dist.get_rank() == 0

    def get_rank(self) -> int:
        """Get current process rank"""
        return dist.get_rank()

    @staticmethod
    def get_rng_state():
        """Get current RNG state for reproducibility"""
        rng_state = {
            "cpu": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }

        # Only save CUDA RNG state if CUDA is available and being used
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            rng_state["cuda"] = torch.cuda.get_rng_state()

        return rng_state

    @staticmethod
    def load_rng_state(rng_state):
        """Load RNG state for reproducibility"""
        torch.set_rng_state(rng_state["cpu"])
        np.random.set_state(rng_state["numpy"])
        random.setstate(rng_state["random"])

        # Only restore CUDA RNG state if it was saved and CUDA is available
        if "cuda" in rng_state and torch.cuda.is_available() and torch.cuda.device_count() > 0:
            torch.cuda.set_rng_state(rng_state["cuda"])

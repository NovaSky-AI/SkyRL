"""Minibatch reductions independent of microbatch and data-parallel partitioning."""

from dataclasses import dataclass

import torch
import torch.distributed as dist

from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch

MINIBATCH_LOSS_NORMALIZATION = "minibatch_loss_normalization"


@dataclass(frozen=True)
class MinibatchLossNormalization:
    num_sequences: float
    num_loss_tokens: float

    @classmethod
    def from_batch(
        cls,
        data: TrainingInputBatch,
        *,
        dp_group: dist.ProcessGroup | None,
        device: torch.device | str | int,
    ) -> "MinibatchLossNormalization":
        """Count once before microbatch padding, reducing only across DP replicas."""
        sample_mask = data.real_sample_mask
        token_mask = data["loss_mask"]
        counts = torch.stack((sample_mask.sum(), (token_mask * sample_mask[:, None]).sum(dtype=torch.float64))).to(
            device=device, dtype=torch.float64
        )
        if dist.is_initialized():
            dist.all_reduce(counts, group=dp_group)
        return cls(*counts.tolist())

    def token_mean_contribution(self, mean: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
        """Convert a microbatch token mean (or CP-local contribution) to a global contribution."""
        return mean * (loss_mask.sum() / max(1.0, self.num_loss_tokens))

    def sequence_mean_contribution(self, means: torch.Tensor, sample_mask: torch.Tensor) -> torch.Tensor:
        """Sum per-trajectory means; fully masked real trajectories still count in the denominator."""
        return (means * sample_mask).sum() / max(1.0, self.num_sequences)

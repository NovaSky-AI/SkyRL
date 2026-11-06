"""Optimizer-step reductions independent of microbatch, call and data-parallel partitioning."""

import math
from dataclasses import dataclass

import torch
import torch.distributed as dist

from skyrl.backends.skyrl_train.training_batch import (
    SUB_SEQ_LENGTHS,
    TrainingInputBatch,
)

MINIBATCH_LOSS_NORMALIZATION = "minibatch_loss_normalization"


@dataclass(frozen=True)
class MinibatchLossNormalization:
    """Denominators of one optimizer step's loss terms.

    Every microbatch of every ``forward_backward`` call in the step shares one instance, so the
    contributions sum to the step's loss however the step is split.
    """

    # Real trajectories, including those with no loss tokens.
    num_sequences: float
    num_loss_tokens: float
    # Real tokens of real trajectories, and the attended segments they form.
    num_model_tokens: float
    num_segments: float
    min_segment_length: float

    @classmethod
    def from_batch(
        cls,
        data: TrainingInputBatch,
        *,
        dp_group: dist.ProcessGroup | None = None,
    ) -> "MinibatchLossNormalization":
        """Count ``data`` as one optimizer step, summing over ``dp_group`` when it is one DP shard."""
        sample_mask = data.real_sample_mask.to(torch.bool)
        loss_mask = data["loss_mask"].to(torch.bool) & sample_mask[:, None]
        attention_mask = data["attention_mask"].to(torch.bool) & sample_mask[:, None]
        sub_seq_lengths = data.get(SUB_SEQ_LENGTHS)
        if sub_seq_lengths is not None:
            real_rows = sample_mask.tolist()
            segment_lengths = [int(n) for row, real in zip(sub_seq_lengths, real_rows) if real for n in row.tolist()]
        else:
            segment_lengths = attention_mask.sum(dim=-1)[sample_mask].tolist()
        counts = torch.tensor(
            [
                sample_mask.sum(),
                loss_mask.sum(),
                attention_mask.sum(),
                len(segment_lengths),
            ],
            dtype=torch.float64,
        )
        shortest = torch.tensor([min(segment_lengths, default=math.inf)], dtype=torch.float64)
        if dp_group is not None and dist.is_initialized():
            on_cuda = dist.get_backend(dp_group) == dist.Backend.NCCL
            device = torch.device("cuda", torch.cuda.current_device()) if on_cuda else torch.device("cpu")
            counts, shortest = counts.to(device), shortest.to(device)
            dist.all_reduce(counts, group=dp_group)
            dist.all_reduce(shortest, op=dist.ReduceOp.MIN, group=dp_group)
        return cls(*counts.tolist(), shortest.item())

    def token_mean_contribution(self, mean: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
        """Convert a microbatch token mean (or CP-local contribution) to a step contribution."""
        return mean * (loss_mask.sum() / max(1.0, self.num_loss_tokens))

    def sequence_mean_contribution(self, means: torch.Tensor, sample_mask: torch.Tensor) -> torch.Tensor:
        """Sum per-trajectory means; fully masked real trajectories still count in the denominator."""
        return (means * sample_mask).sum() / max(1.0, self.num_sequences)

    def num_mtp_tokens(self, depth: int) -> float:
        """Positions whose target ``depth`` tokens ahead is real and in the same segment."""
        if depth > self.min_segment_length:
            raise ValueError(
                f"MTP depth {depth} exceeds the shortest attended segment ({self.min_segment_length} tokens)"
            )
        return self.num_model_tokens - depth * self.num_segments

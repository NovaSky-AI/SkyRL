import math
import torch
import torch.distributed as dist
from skyrl_train.dataset.replay_buffer import Experience
from typing import List, Dict
from skyrl_train.training_batch import TrainingInputBatch


def reduce_metrics(metrics: Dict[str, List[float]]) -> Dict[str, float]:
    """
    Reduce metrics from a list of entries per key.
    """
    reduced_metrics = dict()
    for k, v in metrics.items():
        assert len(v) > 0, f"No metrics for key {k}"
        assert all(isinstance(x, (int, float)) for x in v), f"Metrics for key {k} are not all numbers"
        reduced_metrics[k] = sum(v) / len(v)
    return reduced_metrics


class BatchIterator:
    """A simple iterator to yield micro batches of data from the training batch."""

    def __init__(self, data: TrainingInputBatch, sample_batch_size: int, drop_last: bool = False):
        self.data = data
        self.sample_batch_size = sample_batch_size
        self.total_batch_size = data.batch_size
        self.drop_last = drop_last
        assert not drop_last, "drop_last is not supported yet"
        num_micro_batches = self.total_batch_size / self.sample_batch_size
        self.num_micro_batches = int(num_micro_batches) if drop_last else math.ceil(num_micro_batches)
        # TODO: switch to tensordict.map_iter if possible
        self._chunks = self.data.chunk(self.sample_batch_size)
        self._iter = iter(self._chunks)

    def __len__(self):
        return self.num_micro_batches

    def __iter__(self):
        return self

    def __next__(self) -> Experience:
        try:
            batch = next(self._iter)
            exp = self.batch_to_experience(batch)
            return exp
        except StopIteration:
            self._iter = iter(self._chunks)
            raise StopIteration

    @staticmethod
    def batch_to_experience(batch: TrainingInputBatch):
        # TODO (sumanthrh): other keys are not permitted right now, can go into info
        # TODO: this conversion is hidden right now, might need to be surfaced in worker explicitly.
        exp = Experience(
            sequences=batch["sequences"],
            action_log_probs=batch["action_log_probs"],
            base_action_log_probs=batch["base_action_log_probs"],
            values=batch["values"],
            returns=batch["returns"],
            advantages=batch["advantages"],
            attention_mask=batch["attention_mask"],
            loss_mask=batch["loss_mask"],
            action_mask=batch["response_mask"],
            num_actions=batch.metadata["response_length"],  # int
            rollout_logprobs=batch["rollout_logprobs"] if "rollout_logprobs" in batch else None,
            # additional info
            # can be used to log metrics etc for micro-batches in the worker
            info={},
            # propagate metadata as is
            metadata=batch.metadata,
        )
        return exp


def greedy_binpacking(token_counts: List[int], max_tokens_per_microbatch: int) -> List[List[int]]:
    """Greedily chunk a list of token counts into microbatches so that each microbatch's total token count
    does not exceed `max_tokens_per_microbatch`.

    Args:
        token_counts: List of token counts for each sample.
        max_tokens_per_microbatch: Maximum total tokens allowed per microbatch.

    Returns:
        A list of microbatches, where each microbatch is a list of indices (ints)
        referring to entries in `token_counts`.
    """
    microbatch_indices: List[List[int]] = []
    current_microbatch_indices: List[int] = []
    current_tokens = 0

    for i, tokens in enumerate(token_counts):
        # Start a new microbatch if adding this sample exceeds the max
        # TODO: Handle max(token_counts) > max_tokens_per_microbatch
        if current_tokens + tokens > max_tokens_per_microbatch:
            microbatch_indices.append(current_microbatch_indices)
            current_microbatch_indices = []
            current_tokens = 0

        current_microbatch_indices.append(i)
        current_tokens += tokens

    if current_microbatch_indices:
        microbatch_indices.append(current_microbatch_indices)

    return microbatch_indices


class BalancedBatchIterator:
    """
    An iterator that chunks batches based on token count rather than sample count.

    Packs samples into micro batches efficiently, ensuring each microbatch doesn't exceed
    max_tokens_per_microbatch. All data parallel workers will have the same number of
    micro batches (padding batches are added if needed).
    """

    def __init__(
        self,
        data: TrainingInputBatch,
        max_tokens_per_microbatch: int,
    ):
        """
        Args:
            data: The training input batch to chunk
            max_tokens_per_microbatch: Maximum number of tokens per microbatch
        """
        self._data = data
        self._max_tokens_per_microbatch = max_tokens_per_microbatch

        # Compute token counts per sample using attention_mask
        attention_mask = data["attention_mask"]
        # Count non-padding tokens per sample
        self._token_counts = attention_mask.sum(dim=1).cpu().tolist()  # [batch_size]

        # Create microbatches based on token count
        self._microbatches = greedy_binpacking(self._token_counts, self._max_tokens_per_microbatch)

        # Synchronize the number of microbatches across all DP workers
        if dist.is_initialized():
            max_num_microbatches = self._sync_num_microbatches()
        else:
            max_num_microbatches = len(self._microbatches)

        self._num_padding_microbatches = max_num_microbatches - len(self._microbatches)

    @property
    def num_microbatches(self) -> int:
        return len(self._microbatches) + self._num_padding_microbatches

    def _create_microbatch_from_indices(self, indices: List[int]) -> TrainingInputBatch:
        """Create a TrainingInputBatch from a list of sample indices."""
        # TODO: Support list indexing for TrainingInputBatch
        return TrainingInputBatch.cat([self._data[i] for i in indices])

    def _create_padding_microbatch(self) -> TrainingInputBatch:
        """Create a padding microbatch."""
        data = TrainingInputBatch(
            {
                "sequences": torch.ones((1, 1), dtype=int, device="cpu"),
                "attention_mask": torch.zeros((1, 1), dtype=int, device="cpu"),
                "action_log_probs": torch.zeros((1, 1), device="cpu"),
                "base_action_log_probs": torch.zeros((1, 1), device="cpu"),
                "values": torch.zeros((1, 1), device="cpu"),
                "returns": torch.zeros((1, 1), device="cpu"),
                "advantages": torch.zeros((1, 1), device="cpu"),
                "loss_mask": torch.zeros((1, 1), dtype=int, device="cpu"),
                "response_mask": torch.zeros((1, 1), dtype=int, device="cpu"),
            }
        )
        data.metadata = self._data.metadata
        return data

    def _sync_num_microbatches(self) -> int:
        """Ensure all DP workers have the same number of micro batches."""
        local_num_microbatches = len(self._microbatches)

        # Get the maximum number of batches across all DP workers
        # Handle case where CUDA might not be available
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
        else:
            device = torch.device("cpu")
        num_microbatches_tensor = torch.tensor(local_num_microbatches, dtype=torch.long, device=device)
        dist.all_reduce(num_microbatches_tensor, op=dist.ReduceOp.MAX)
        return num_microbatches_tensor.item()

    def __len__(self):
        return len(self._microbatches) + self._num_padding_microbatches

    def __iter__(self):
        for microbatch in self._microbatches:
            yield BatchIterator.batch_to_experience(self._create_microbatch_from_indices(microbatch))
        for _ in range(self._num_padding_microbatches):
            yield BatchIterator.batch_to_experience(self._create_padding_microbatch())

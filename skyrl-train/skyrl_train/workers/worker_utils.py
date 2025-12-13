import heapq
import math
import torch
import torch.distributed as dist
from skyrl_train.dataset.replay_buffer import Experience
from typing import List, Dict
from skyrl_train.training_batch import TrainingInputBatch, TensorBatch


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


class BaseBatchIterator:
    def __init__(self, data: TrainingInputBatch):
        self.data = data

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def reorder_microbatches(self, microbatches: List[TensorBatch]) -> TensorBatch:
        """Reorder microbatches to match the order of the original data."""
        raise NotImplementedError


class SampleBasedBatchIterator(BaseBatchIterator):
    """A simple iterator to yield micro batches of the same number of samples from the training batch."""

    def __init__(self, data: TrainingInputBatch, sample_batch_size: int, drop_last: bool = False):
        super().__init__(data)
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

    def __next__(self) -> TrainingInputBatch:
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self._chunks)
            raise StopIteration

    def reorder_microbatches(self, microbatches: List[TensorBatch]) -> TensorBatch:
        """Reorder microbatches to match the order of the original data."""
        # TODO: Define a common interface for batch iterators.
        return TensorBatch.cat(microbatches)

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


def balanced_binpacking(token_counts: List[int], max_tokens_per_microbatch: int) -> List[List[int]]:
    """Chunk a list of token counts into microbatches so that each microbatch's total token count
    does not exceed `max_tokens_per_microbatch`, and the microbatches roughly balanced.

    Balancing happens by assigning sequences to the microbatch with the least number of tokens so far.

    Args:
        token_counts: List of token counts for each sample.
        max_tokens_per_microbatch: Maximum total tokens allowed per microbatch.

    Returns:
        A list of microbatches, where each microbatch is a list of indices (ints)
        referring to entries in `token_counts`.

    >>> balanced_binpacking([10, 10, 5, 5], 15)
    [[0, 2], [1, 3]]
    >>> balanced_binpacking([10, 1, 1, 1, 1, 1], 10)
    [[0], [1, 2, 3, 4, 5]]
    >>> balanced_binpacking([8, 3, 5, 6, 2, 7], 11)
    [[0, 4], [5, 1], [3, 2]]
    """
    # TODO: Handle max(token_counts) > max_tokens_per_microbatch

    # Create list of (index, token_count) pairs and sort by token count descending
    seq_lens = [(i, seq_len) for i, seq_len in enumerate(token_counts)]
    seq_lens.sort(key=lambda x: x[1], reverse=True)

    # Track microbatch indices and their current token counts
    microbatch_indices: List[List[int]] = []

    # Heap to track the total number of tokens in each microbatch
    microbatch_tokens_heap = []  # (current_total, bin_idx)

    for idx, seq_len in seq_lens:
        placed = False

        # Look for an existing microbatch with the least number of tokens
        # that can fit the sequence without exceeding the token limit.
        if microbatch_tokens_heap:
            microbatch_len, i = microbatch_tokens_heap[0]
            new_microbatch_len = microbatch_len + seq_len
            if new_microbatch_len <= max_tokens_per_microbatch:
                microbatch_indices[i].append(idx)
                heapq.heapreplace(microbatch_tokens_heap, (new_microbatch_len, i))
                placed = True

        # If no microbatch can fit the sequence, create a new microbatch.
        if not placed:
            microbatch_indices.append([idx])
            heapq.heappush(microbatch_tokens_heap, (seq_len, len(microbatch_indices) - 1))

    return microbatch_indices


class TokenBasedBatchIterator(BaseBatchIterator):
    """An iterator that chunks microbatches based on real token count.

    Packs samples into microbatches, ensuring each microbatch doesn't exceed
    max_tokens_per_microbatch. All data parallel workers will have the same number of
    microbatches (where padding microbatches are added if needed).
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
        # TODO: Allow for different chunking strategies.
        self._microbatches = balanced_binpacking(self._token_counts, self._max_tokens_per_microbatch)

        # Synchronize the number of microbatches across all DP workers
        max_num_microbatches = self._sync_num_microbatches()

        self._num_padding_microbatches = max_num_microbatches - len(self._microbatches)

    @property
    def num_microbatches(self) -> int:
        return len(self._microbatches) + self._num_padding_microbatches

    def _create_microbatch_from_indices(self, indices: List[int]) -> TrainingInputBatch:
        """Create a TrainingInputBatch from a list of sample indices."""
        indices_tensor = torch.tensor(indices, dtype=torch.long, device="cpu")
        selected_data = {}
        for key, value in self._data.items():
            selected_data[key] = value[indices_tensor]
        microbatch = TrainingInputBatch(selected_data)
        microbatch.metadata = self._data.metadata
        return microbatch

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

        if not dist.is_initialized():
            return local_num_microbatches

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
            yield self._create_microbatch_from_indices(microbatch)
        for _ in range(self._num_padding_microbatches):
            yield self._create_padding_microbatch()

    def reorder_microbatches(self, microbatches: List[TensorBatch]) -> TensorBatch:
        """Reorder microbatch data into a single batch with the same order as the original data.

        Example: [[0, 2], [1, 3]] -> [0, 1, 2, 3]

        Args:
            microbatches: List of microbatches to reorder.

        Returns:
            A single reordered batch.
        """
        # TODO: Move this stuff to utility functions.
        non_padding_microbatches = microbatches[: len(microbatches) - self._num_padding_microbatches]

        if not non_padding_microbatches:
            # TODO: Can this happen?
            raise ValueError("Cannot reorder an empty list of microbatches.")

        # Create a reverse mapping of original idx -> (microbatch idx, sample idx)
        original_idx_to_microbatch_idx = {}

        for microbatch_idx, original_indices in enumerate(self._microbatches):
            for sample_idx, original_idx in enumerate(original_indices):
                original_idx_to_microbatch_idx[original_idx] = (microbatch_idx, sample_idx)

        # Get reference microbatch to know keys and tensor shapes
        ref_microbatch = non_padding_microbatches[0]
        reordered_data = {}

        for key, ref_value in ref_microbatch.items():
            # Get shape of a single sample (remove batch dimension)
            sample_shape = ref_value.shape[1:]
            device = ref_value.device
            dtype = ref_value.dtype

            # Pre-allocate output tensor: [batch_size, *sample_shape]
            batch_size = len(self._token_counts)
            output_tensor = torch.zeros((batch_size, *sample_shape), dtype=dtype, device=device)

            # Copy each sample directly into the correct position
            for original_idx in range(batch_size):
                microbatch_idx, sample_idx = original_idx_to_microbatch_idx[original_idx]
                source_tensor = non_padding_microbatches[microbatch_idx][key]
                output_tensor[original_idx] = source_tensor[sample_idx]

            reordered_data[key] = output_tensor

        # Create single TensorBatch with reordered data
        reordered_batch = type(ref_microbatch)(reordered_data)
        reordered_batch.metadata = ref_microbatch.metadata
        return reordered_batch

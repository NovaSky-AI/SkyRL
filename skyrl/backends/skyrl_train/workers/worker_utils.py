import math
from typing import Dict, Iterator, List

import torch
import torch.distributed as dist

from skyrl.backends.skyrl_train.distributed.strategy import DistributedStrategy
from skyrl.backends.skyrl_train.training_batch import TensorBatch, TrainingInputBatch
from skyrl.train.dataset.bin_packing import make_seq_packer
from skyrl.train.dataset.replay_buffer import Experience


def reduce_metrics(metrics: Dict[str, List[float]], sum_loss_metrics: bool = False) -> Dict[str, float]:
    """Reduce scalar metrics from a list of entries per key with the appropriate reduction.

    Default reduction is mean. Metrics ending in `_min` or `_max` use min/max respectively.

    If sum_loss_metrics is True, metrics ending in `_loss` are summed instead of averaged.
    This should be used if the scaling is already done at the advantage level.
    See `apply_loss_reduction_to_advantages_minibatch` for more details.

    Args:
        metrics: Dictionary of metrics with keys as metric names and values as lists of metric values.
            The list of values corresponds to micro-batches within a single mini-batch.
        sum_loss_metrics: If True, metrics ending in `_loss` are summed (for pre-scaled policy losses).
    """
    reduced_metrics = dict()
    for k, v in metrics.items():
        assert len(v) > 0, f"No metrics for key {k}"
        if not all(isinstance(x, (int, float)) for x in v):
            print(f"Metrics for key {k} are not all numbers: {v}")
            continue
        if k.endswith("_max"):
            reduced_metrics[k] = max(v)
        elif k.endswith("_min"):
            reduced_metrics[k] = min(v)
        elif sum_loss_metrics and k.endswith("_loss"):
            reduced_metrics[k] = sum(v)
        else:
            reduced_metrics[k] = sum(v) / len(v)
    return reduced_metrics


def all_reduce_metrics(
    metrics: Dict[str, float],
    strategy: DistributedStrategy,
    group=None,
    sum_loss_metrics: bool = False,
) -> Dict[str, float]:
    """All reduce metrics across all processes.

    Default reduction is mean. Metrics ending in `_min` or `_max` use min/max respectively.
    If sum_loss_metrics is True, metrics ending in `_loss` are summed instead of averaged.

    Args:
        metrics: Dictionary of metric name to scalar value.
        strategy: Distributed strategy for all-reduce.
        group: Process group for all-reduce.
        sum_loss_metrics: If True, metrics ending in `_loss` are summed (for pre-scaled policy losses).
    """
    min_metrics = {k: v for k, v in metrics.items() if k.endswith("_min")}
    max_metrics = {k: v for k, v in metrics.items() if k.endswith("_max")}
    sum_metrics = {k: v for k, v in metrics.items() if sum_loss_metrics and k.endswith("_loss")}
    mean_metrics = {
        k: v for k, v in metrics.items() if k not in min_metrics and k not in max_metrics and k not in sum_metrics
    }
    status_mean = strategy.all_reduce(mean_metrics, op="mean", group=group)
    status_min = strategy.all_reduce(min_metrics, op="min", group=group)
    status_max = strategy.all_reduce(max_metrics, op="max", group=group)
    status_sum = strategy.all_reduce(sum_metrics, op="sum", group=group)
    status_mean.update(status_min)
    status_mean.update(status_max)
    status_mean.update(status_sum)
    return status_mean


class BaseBatchIterator:
    """Base class for batch iterators that chunk a TrainingInputBatch into microbatches."""

    def __init__(self, data: TrainingInputBatch):
        self.data = data

    def __len__(self):
        raise NotImplementedError

    def __iter__(self) -> Iterator[TrainingInputBatch]:
        raise NotImplementedError

    def reorder_and_combine_batches(self, batches: List[TensorBatch]) -> TensorBatch:
        """Reorder and combine output batches to form a single output."""
        raise NotImplementedError

    @staticmethod
    def batch_to_experience(batch: TrainingInputBatch):
        # TODO (sumanthrh): other keys are not permitted right now, can go into info
        # TODO: this conversion is hidden right now, might need to be surfaced in worker explicitly.
        exp = Experience(
            sequences=batch["sequences"],
            action_log_probs=batch.get("action_log_probs"),
            base_action_log_probs=batch.get("base_action_log_probs"),
            values=batch.get("values"),
            returns=batch.get("returns"),
            advantages=batch.get("advantages"),
            attention_mask=batch.get("attention_mask"),
            loss_mask=batch.get("loss_mask"),
            action_mask=batch.get("response_mask"),
            num_actions=batch.metadata["response_length"],  # int
            rollout_logprobs=batch.get("rollout_logprobs"),
            rollout_expert_indices=batch.get("rollout_expert_indices"),
            # additional info
            # can be used to log metrics etc for micro-batches in the worker
            info={},
            # propagate metadata as is
            metadata=batch.metadata,
            # Multi-modal vision fields (may be absent for text-only)
            pixel_values=batch.get("pixel_values"),
            image_grid_thw=batch.get("image_grid_thw"),
            # Per-row sub-sequence lengths for sequence packing (None otherwise);
            # chunked per micro-batch by ``TensorBatch.chunk`` like any other field.
            sub_seq_lengths=batch.get("sub_seq_lengths"),
        )
        return exp


# Keep BatchIterator as an alias for backward compatibility
class BatchIterator(BaseBatchIterator):
    """A simple iterator to yield micro batches of data from the training batch.

    This is the original sample-based iterator. Kept as an alias for SampleBasedBatchIterator.
    """

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

    def __next__(self) -> Experience:
        try:
            batch = next(self._iter)
            exp = self.batch_to_experience(batch)
            return exp
        except StopIteration:
            self._iter = iter(self._chunks)
            raise StopIteration

    def reorder_and_combine_batches(self, batches: List[TensorBatch]) -> TensorBatch:
        """Concatenate output batches. No reordering needed for sample-based splitting."""
        return TensorBatch.cat(batches)


class SampleBasedBatchIterator(BaseBatchIterator):
    """Iterator that yields fixed-size sample-based microbatches from the training input.

    Yields TrainingInputBatch objects (not Experience), unlike the legacy BatchIterator.
    """

    def __init__(self, data: TrainingInputBatch, sample_batch_size: int, drop_last: bool = False):
        super().__init__(data)
        self.sample_batch_size = sample_batch_size
        self.total_batch_size = data.batch_size
        self.drop_last = drop_last
        assert not drop_last, "drop_last is not supported yet"
        num_micro_batches = self.total_batch_size / self.sample_batch_size
        self.num_micro_batches = int(num_micro_batches) if drop_last else math.ceil(num_micro_batches)
        self._chunks = self.data.chunk(self.sample_batch_size)

    def __len__(self):
        return self.num_micro_batches

    def __iter__(self) -> Iterator[TrainingInputBatch]:
        return iter(self._chunks)

    def reorder_and_combine_batches(self, batches: List[TensorBatch]) -> TensorBatch:
        """Concatenate output batches. No reordering needed for sample-based splitting."""
        return TensorBatch.cat(batches)


class TokenBasedBatchIterator(BaseBatchIterator):
    """An iterator that chunks microbatches based on real token count.

    Packs samples into microbatches using bin-packing, ensuring each microbatch
    doesn't exceed max_tokens_per_microbatch. All data parallel workers will have
    the same number of microbatches (padding microbatches are added if needed).
    """

    def __init__(
        self,
        data: TrainingInputBatch,
        max_tokens_per_microbatch: int,
    ):
        """
        Args:
            data: The training input batch to chunk.
            max_tokens_per_microbatch: Maximum number of tokens per microbatch.
        """
        super().__init__(data)
        self._max_tokens_per_microbatch = max_tokens_per_microbatch

        # Compute token counts per sample using attention_mask
        attention_mask = data["attention_mask"]
        self._token_counts = attention_mask.sum(dim=1).cpu().tolist()  # [batch_size]

        # Create microbatches based on token count. The "balanced" packer treats
        # the token budget as a soft cap: a sequence longer than the budget gets
        # its own (over-budget) microbatch rather than raising.
        packer = make_seq_packer("balanced", bin_capacity=self._max_tokens_per_microbatch)
        self._microbatches = packer.pack(self._token_counts)

        # Synchronize the number of microbatches across all DP workers
        max_num_microbatches = self._sync_num_microbatches()
        self._num_padding_microbatches = max_num_microbatches - len(self._microbatches)

    def _create_microbatch_from_indices(self, indices: List[int]) -> TrainingInputBatch:
        """Create a TrainingInputBatch from a list of sample indices."""
        indices_tensor = torch.tensor(indices, dtype=torch.long, device="cpu")
        selected_data = {}
        for key, value in self.data.items():
            if value is None:
                selected_data[key] = None
            else:
                selected_data[key] = value[indices_tensor]
        microbatch = TrainingInputBatch(selected_data)
        microbatch.metadata = self.data.metadata
        return microbatch

    def _create_padding_microbatch(self) -> TrainingInputBatch:
        """Create a padding microbatch with loss_mask=0 so it doesn't affect the loss."""
        # Match the real data's sequence length so every microbatch shares the same
        # seq_len.
        seq_len = self.data["sequences"].shape[1]
        num_actions = self.data.metadata["response_length"]
        batch_size = 1
        device = self.data["sequences"].device

        # Keep the full seq_len for shape uniformity, but mark only a single token as
        # valid in the attention mask to keep the row non-degenerate.
        attention_mask = torch.zeros((batch_size, seq_len), dtype=int, device=device)
        attention_mask[:, 0] = 1

        data = TrainingInputBatch(
            {
                "sequences": torch.randint(0, 100, (batch_size, seq_len), device=device),
                "attention_mask": attention_mask,
                "action_log_probs": 0.4 * torch.ones((batch_size, num_actions), device=device),
                "base_action_log_probs": 0.3 * torch.ones((batch_size, num_actions), device=device),
                "values": 0.5 * torch.ones((batch_size, num_actions), device=device),
                "returns": 0.5 * torch.ones((batch_size, num_actions), device=device),
                "advantages": 0.6 * torch.ones((batch_size, num_actions), device=device),
                # Loss mask is all zeros so padding samples don't contribute to the loss.
                "loss_mask": torch.zeros((batch_size, num_actions), dtype=int, device=device),
                "response_mask": torch.ones((batch_size, num_actions), dtype=int, device=device),
            }
        )
        # Add optional fields such as `rollout_logprobs` to padding batch
        if self.data.get("rollout_logprobs") is not None:
            data["rollout_logprobs"] = torch.zeros((batch_size, num_actions), device=device)
        data.metadata = self.data.metadata
        return data

    def _sync_num_microbatches(self) -> int:
        """Ensure all DP workers have the same number of micro batches."""
        local_num_microbatches = len(self._microbatches)

        if not dist.is_initialized():
            return local_num_microbatches

        # Get the maximum number of batches across all DP workers
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
        else:
            device = torch.device("cpu")
        num_microbatches_tensor = torch.tensor(local_num_microbatches, dtype=torch.long, device=device)
        dist.all_reduce(num_microbatches_tensor, op=dist.ReduceOp.MAX)
        return num_microbatches_tensor.item()

    def __len__(self):
        return len(self._microbatches) + self._num_padding_microbatches

    @property
    def num_padding_microbatches(self) -> int:
        """Number of purely-padding microbatches appended to equalize the microbatch
        count across DP ranks (each carries no real samples / loss_mask all zero)."""
        return self._num_padding_microbatches

    def __iter__(self) -> Iterator[TrainingInputBatch]:
        for microbatch_indices in self._microbatches:
            yield self._create_microbatch_from_indices(microbatch_indices)

        for _ in range(self._num_padding_microbatches):
            yield self._create_padding_microbatch()

    def reorder_and_combine_batches(self, batches: List[TensorBatch]) -> TensorBatch:
        """Reorder and combine output batches into a single batch with
        the same order as the original input data.

        Example: [[0, 2], [1, 3]] -> [0, 1, 2, 3]

        Args:
            batches: List of microbatch outputs to reorder.
        Returns:
            A single reordered batch.
        """
        non_padding_batches = batches[: len(batches) - self._num_padding_microbatches]

        if not non_padding_batches:
            raise ValueError("Cannot reorder an empty list of microbatches.")

        # Create a reverse mapping of original idx -> (microbatch idx, sample idx)
        original_idx_to_microbatch_idx = {}
        for microbatch_idx, original_indices in enumerate(self._microbatches):
            for sample_idx, original_idx in enumerate(original_indices):
                original_idx_to_microbatch_idx[original_idx] = (microbatch_idx, sample_idx)

        # Get reference microbatch to know keys and tensor shapes
        ref_microbatch = non_padding_batches[0]
        reordered_data = {}

        for key, ref_value in ref_microbatch.items():
            if ref_value is None:
                reordered_data[key] = None
                continue
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
                source_tensor = non_padding_batches[microbatch_idx][key]
                output_tensor[original_idx] = source_tensor[sample_idx]

            reordered_data[key] = output_tensor

        # Create single TensorBatch with reordered data
        reordered_batch = type(ref_microbatch)(reordered_data)
        reordered_batch.metadata = ref_microbatch.metadata
        return reordered_batch


def get_microbatch_iterator(
    data: TrainingInputBatch, micro_batch_size: int, max_tokens_per_microbatch: int
) -> BaseBatchIterator:
    """Factory function to get the appropriate microbatch iterator.

    Args:
        data: The training input batch.
        micro_batch_size: Number of samples per microbatch (used if max_tokens_per_microbatch <= 0).
        max_tokens_per_microbatch: Maximum tokens per microbatch. If > 0, uses token-based batching.

    Returns:
        A BaseBatchIterator instance.
    """
    if max_tokens_per_microbatch > 0:
        return TokenBasedBatchIterator(data, max_tokens_per_microbatch=max_tokens_per_microbatch)
    else:
        return SampleBasedBatchIterator(data, sample_batch_size=micro_batch_size, drop_last=False)

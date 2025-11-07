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
        sync_num_batches: bool = True,
    ):
        """
        Args:
            data: The training input batch to chunk
            max_tokens_per_microbatch: Maximum number of tokens per microbatch
            sync_num_batches: If True, synchronize the number of batches across DP workers
        """
        self.data = data
        self.max_tokens_per_microbatch = max_tokens_per_microbatch
        self.sync_num_batches = sync_num_batches

        # Compute token counts per sample using attention_mask
        attention_mask = data["attention_mask"]
        # Count non-padding tokens per sample
        self.token_counts = attention_mask.sum(dim=1).cpu().tolist()  # [batch_size]

        # Create chunks based on token count
        self._chunks = self._create_token_based_chunks()

        # Synchronize number of batches across DP workers if needed
        if self.sync_num_batches and dist.is_initialized():
            self._sync_num_batches()

        self._iter = iter(self._chunks)
        self.num_micro_batches = len(self._chunks)

    def _create_token_based_chunks(self) -> List[TrainingInputBatch]:
        """Create chunks by packing samples up to max_tokens_per_microbatch."""
        chunks = []
        current_chunk_indices = []
        current_token_count = 0

        for idx in range(self.data.batch_size):
            sample_token_count = self.token_counts[idx]

            # If adding this sample would exceed the limit, start a new chunk
            if current_chunk_indices and current_token_count + sample_token_count > self.max_tokens_per_microbatch:
                # Create chunk from accumulated indices
                chunk = self._create_chunk_from_indices(current_chunk_indices)
                chunks.append(chunk)
                current_chunk_indices = [idx]
                current_token_count = sample_token_count
            else:
                # Add sample to current chunk
                current_chunk_indices.append(idx)
                current_token_count += sample_token_count

        # Add final chunk if there are remaining samples
        if current_chunk_indices:
            chunk = self._create_chunk_from_indices(current_chunk_indices)
            chunks.append(chunk)

        return chunks

    def _create_chunk_from_indices(self, indices: List[int]) -> TrainingInputBatch:
        """Create a TrainingInputBatch from a list of sample indices."""
        chunk_data = {}
        for key, value in self.data.items():
            if value is not None:
                if isinstance(value, torch.Tensor):
                    chunk_data[key] = value[indices]
                else:
                    raise ValueError(f"Unsupported type {type(value)} for key {key}")
            else:
                chunk_data[key] = value

        chunk = self.data.__class__(chunk_data)
        chunk.metadata = self.data.metadata
        return chunk

    def _sync_num_batches(self):
        """Ensure all DP workers have the same number of micro batches."""
        num_batches = len(self._chunks)

        # Get the maximum number of batches across all DP workers
        # Handle case where CUDA might not be available
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
        else:
            device = torch.device("cpu")
        num_batches_tensor = torch.tensor(num_batches, dtype=torch.long, device=device)
        dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.MAX)
        max_num_batches = num_batches_tensor.item()

        # If this worker has fewer batches, pad with empty batches
        if num_batches < max_num_batches:
            num_padding_batches = max_num_batches - num_batches
            padding_batches = self._create_padding_batches(num_padding_batches)
            self._chunks.extend(padding_batches)

    def _create_padding_batches(self, num_batches: int) -> List[TrainingInputBatch]:
        """Create empty padding batches to match the number of batches on other workers."""
        padding_batches = []
        for _ in range(num_batches):
            # Create an empty batch (batch_size=0) with the same structure
            chunk_data = {}
            for key, value in self.data.items():
                if value is not None:
                    if isinstance(value, torch.Tensor):
                        # Create empty tensor with same shape except batch dimension
                        shape = list(value.shape)
                        shape[0] = 0  # batch_size = 0
                        chunk_data[key] = torch.empty(shape, dtype=value.dtype, device=value.device)
                    else:
                        raise ValueError(f"Unsupported type {type(value)} for key {key}")
                else:
                    chunk_data[key] = value

            chunk = self.data.__class__(chunk_data)
            chunk.metadata = self.data.metadata
            padding_batches.append(chunk)

        return padding_batches

    def __len__(self):
        return self.num_micro_batches

    def __iter__(self):
        return self

    def __next__(self) -> TrainingInputBatch:
        try:
            batch = next(self._iter)
            return batch
        except StopIteration:
            self._iter = iter(self._chunks)
            raise StopIteration

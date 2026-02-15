import math
from skyrl_train.dataset.replay_buffer import Experience
from typing import List, Dict
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.distributed.strategy import DistributedStrategy

import torch


def reduce_metrics(metrics: Dict[str, List[float]]) -> Dict[str, float]:
    """
    Reduce scalar metrics from a list of entries per key by averaging.
    """
    reduced_metrics = dict()
    for k, v in metrics.items():
        assert len(v) > 0, f"No metrics for key {k}"
        assert all(isinstance(x, (int, float)) for x in v), f"Metrics for key {k} are not all numbers"
        if k.endswith("_max"):
            reduced_metrics[k] = max(v)
        elif k.endswith("_min"):
            reduced_metrics[k] = min(v)
        else:
            reduced_metrics[k] = sum(v) / len(v)
    return reduced_metrics


def all_reduce_metrics(metrics: Dict[str, List[float]], strategy: DistributedStrategy) -> Dict[str, float]:
    """All reduce metrics across all processes."""
    min_metrics = {k: v for k, v in metrics.items() if k.endswith("_min")}
    max_metrics = {k: v for k, v in metrics.items() if k.endswith("_max")}
    mean_metrics = {k: v for k, v in metrics.items() if k not in min_metrics and k not in max_metrics}
    status_mean = strategy.all_reduce(mean_metrics, op="mean")
    status_min = strategy.all_reduce(min_metrics, op="min")
    status_max = strategy.all_reduce(max_metrics, op="max")
    status_mean.update(status_min)
    status_mean.update(status_max)
    return status_mean


def batch_to_experience(batch: TrainingInputBatch) -> Experience:
    """Convert a :class:`TrainingInputBatch` slice into an :class:`Experience`."""
    return Experience(
        sequences=batch["sequences"],
        action_log_probs=batch.get("action_log_probs"),
        base_action_log_probs=batch.get("base_action_log_probs"),
        values=batch.get("values"),
        returns=batch.get("returns"),
        advantages=batch.get("advantages"),
        attention_mask=batch.get("attention_mask"),
        loss_mask=batch.get("loss_mask"),
        action_mask=batch.get("response_mask"),
        num_actions=batch.metadata["response_length"],
        rollout_logprobs=batch.get("rollout_logprobs"),
        info={},
        metadata=batch.metadata,
    )


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
            exp = batch_to_experience(batch)
            return exp
        except StopIteration:
            self._iter = iter(self._chunks)
            raise StopIteration

    @staticmethod
    def batch_to_experience(batch: TrainingInputBatch):
        # TODO (sumanthrh): other keys are not permitted right now, can go into info
        # TODO: this conversion is hidden right now, might need to be surfaced in worker explicitly.
        return batch_to_experience(batch)


def _select_indices(data: TrainingInputBatch, indices: torch.Tensor) -> TrainingInputBatch:
    """Index a `TrainingInputBatch` and trim padding from both sides."""
    original_total_len = data["sequences"].shape[1]
    original_resp_len = data.metadata.get("response_length", 0)

    left_trim = 0
    right_trim = 0
    attn = data.get("attention_mask")
    if attn is not None:
        selected_attn = attn[indices]
        attn_long = selected_attn.to(torch.long)

        # left trim: min first-1 position (shared leading PADs)
        left_trim = attn_long.argmax(dim=1).min().item()

        # right trim: total_len - max(rightmost-1 position + 1)
        reversed_attn = attn_long.flip(dims=[1])
        rightmost_plus_1 = selected_attn.shape[1] - reversed_attn.argmax(dim=1)
        right_trim = original_total_len - rightmost_plus_1.max().item()

    new_resp_len = max(original_resp_len - right_trim, 0) if right_trim > 0 else original_resp_len

    selected = {}
    for key, value in data.items():
        if value is not None:
            val = value[indices]
            if val.ndim >= 2 and (right_trim > 0 or left_trim > 0):
                if val.shape[1] == original_total_len:
                    # total-width tensor (sequences, attention_mask): trim both sides
                    end = val.shape[1] - right_trim
                    val = val[:, left_trim:end]
                elif right_trim > 0:
                    # response-width tensor (loss_mask, advantages, …): right only
                    val = val[:, :-right_trim]
            selected[key] = val
        else:
            selected[key] = None

    batch = TrainingInputBatch(selected)
    # give each micro batch its own metadata
    batch.metadata = dict(data.metadata)
    if right_trim > 0 and "response_length" in batch.metadata:
        batch.metadata["response_length"] = max(new_resp_len, 1)
    return batch


class MemoryAwareBatchIterator:
    """Yield micro-batches packed by sequence length to maximise GPU utilisation.

    The idea here is to:
        1. Measure each sequence's width from the attention mask. Go to the rightmost 1 + 1
        2. Sort descending by effective width
        3. Greedily pack into micro batches
        4. Trim each micro batch (left and right padding) after grouping
    """

    def __init__(self, data: TrainingInputBatch, token_budget: int):
        self.data = data
        self.token_budget = token_budget

        attention_mask = data.get("attention_mask")
        if attention_mask is not None:
            seq_len_dim = attention_mask.shape[1]
            # For each row find the first 1 from the right
            reversed_mask = attention_mask.flip(dims=[1])
            first_one_from_right = reversed_mask.to(torch.long).argmax(dim=1)
            seq_lengths = seq_len_dim - first_one_from_right
        else:
            seq_lengths = torch.full((data.batch_size,), data["sequences"].shape[1], dtype=torch.long)

        # sort by descending sequence length
        sorted_indices = torch.argsort(seq_lengths, descending=True)
        sorted_lengths = seq_lengths[sorted_indices]

        self._micro_batches: List[TrainingInputBatch] = []
        current_indices: List[int] = []
        current_max_len = 0

        for i in range(len(sorted_indices)):
            idx = sorted_indices[i].item()
            length = sorted_lengths[i].item()

            new_max_len = max(current_max_len, length)
            new_count = len(current_indices) + 1

            # add to current batch if it fits
            if new_count * new_max_len <= token_budget:
                current_indices.append(idx)
                current_max_len = new_max_len
            else:
                if current_indices:
                    idx_tensor = torch.tensor(current_indices, dtype=torch.long)
                    self._micro_batches.append(_select_indices(data, idx_tensor))
                current_indices = [idx]
                current_max_len = length

        if current_indices:
            idx_tensor = torch.tensor(current_indices, dtype=torch.long)
            self._micro_batches.append(_select_indices(data, idx_tensor))

        self._iter = iter(self._micro_batches)

    def __len__(self):
        return len(self._micro_batches)

    def __iter__(self):
        self._iter = iter(self._micro_batches)
        return self

    def __next__(self) -> Experience:
        batch = next(self._iter)
        return batch_to_experience(batch)

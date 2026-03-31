import math
from typing import Dict, List

from skyrl.backends.skyrl_train.distributed.strategy import DistributedStrategy
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
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
        )
        return exp

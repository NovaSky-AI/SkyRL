import math
from skyrl_train.dataset.replay_buffer import Experience
from typing import List, Dict
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.distributed.strategy import DistributedStrategy


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
            # additional info
            # can be used to log metrics etc for micro-batches in the worker
            info={},
            # propagate metadata as is
            metadata=batch.metadata,
        )
        return exp

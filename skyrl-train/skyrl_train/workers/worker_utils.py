from skyrl_train.dataset.replay_buffer import Experience
from typing import List, Dict, Optional
from skyrl_train.training_batch import TrainingInputBatch
import torch
import torch.distributed as dist
from skyrl_train.utils.dynamic_batching import (
    get_seqlen_balanced_partitions,
    calculate_num_micro_batches,
    create_fixed_partitions,
)
from omegaconf import DictConfig
from loguru import logger


def reduce_metrics(metrics: Dict[str, List[float]], batch_size: int = None) -> Dict[str, float]:
    """
    Reduce metrics from a list of entries per key.
    """
    reduced_metrics = dict()
    for k, v in metrics.items():
        assert len(v) > 0, f"No metrics for key {k}"
        assert all(isinstance(x, (int, float)) for x in v), f"Metrics for key {k} are not all numbers"
        reduced_metrics[k] = sum(v) / batch_size
    return reduced_metrics


class BatchIterator:
    """A unified iterator to yield micro batches of data from the training batch.

    Supports both fixed-size and dynamic token-based batching modes.
    """

    def __init__(
        self,
        data: TrainingInputBatch,
        cfg: DictConfig,
        dp_size: int,
        drop_last: bool = False,
        worker_type: str = "policy",
        dp_group: Optional[dist.ProcessGroup] = None,
        dynamic_bsz: bool = False,
        for_inference: bool = False,
    ):
        """
        Initialize the batch iterator.

        Args:
            data: Training input batch
            cfg: Configuration object
            dp_size: Data parallel size
            drop_last: Whether to drop last incomplete batch (not supported)
            worker_type: Type of worker ("policy" or "critic")
            dp_group: Distributed process group for synchronization
            dynamic_bsz: Whether to use dynamic token-based batching
            for_inference: Whether this is for inference (affects micro-batch size)
        """
        assert not drop_last, "drop_last is not supported yet"

        self.data = data
        self.cfg = cfg
        self.dp_size = dp_size
        self.worker_type = worker_type
        self.dp_group = dp_group
        self.dynamic_bsz = dynamic_bsz
        self.for_inference = for_inference
        self.total_batch_size = data.batch_size

        logger.info(f"Total batch size: {self.total_batch_size}")

        logger.info(f"Sizes: {[len(seq) for seq in data["sequences"]]} {dp_size=}")

        self._prepare_attributes()

        self._all_micro_batches = []

        self._create_micro_batches()

        self._log_configuration()

    def _prepare_attributes(self):
        """Prepare common attributes used by both batching modes."""


        if self.worker_type == "critic":
            self.mini_batch_size_per_gpu = (
                self.cfg.trainer.critic_mini_batch_size * self.cfg.generator.n_samples_per_prompt // self.dp_size
            )
        else:
            self.mini_batch_size_per_gpu = (
                self.cfg.trainer.policy_mini_batch_size * self.cfg.generator.n_samples_per_prompt // self.dp_size
            )

        if self.mini_batch_size_per_gpu <= 0:
            raise ValueError(
                f"mini_batch_size_per_gpu must be positive, got {self.mini_batch_size_per_gpu}. "
                f"Check your configuration: worker_type={self.worker_type}, dp_size={self.dp_size}"
            )

        if not self.dynamic_bsz:

            if self.for_inference:
                self.sample_batch_size = self.cfg.trainer.micro_forward_batch_size_per_gpu
            else:
                self.sample_batch_size = self.cfg.trainer.micro_train_batch_size_per_gpu

            self.micro_batches_per_mini_batch = max(1, self.mini_batch_size_per_gpu // self.sample_batch_size)

    def _create_micro_batches(self):
        """Create micro-batches using either fixed or dynamic partitioning."""
        mini_batches = list(self.data.chunk(self.mini_batch_size_per_gpu))

        if self.dynamic_bsz:
            micro_batch_counts = self._calculate_dynamic_micro_batch_counts(mini_batches)
            synced_counts = self._synchronize_micro_batch_counts(micro_batch_counts)
        else:
            micro_batch_counts = [self.micro_batches_per_mini_batch] * len(mini_batches)
            synced_counts = micro_batch_counts

        for mini_batch, num_micro_batches in zip(mini_batches, synced_counts):
            self._partition_mini_batch(mini_batch, num_micro_batches)

    def _calculate_dynamic_micro_batch_counts(self, mini_batches: List[TrainingInputBatch]) -> List[int]:
        """Calculate the number of micro-batches needed for each mini-batch based on token counts."""
        if self.for_inference:
            self.max_token_len = getattr(
                self.cfg.trainer, "max_token_len_per_gpu_forward", self.cfg.trainer.max_token_len_per_gpu
            )
        else:
            self.max_token_len = self.cfg.trainer.max_token_len_per_gpu

        micro_batch_counts = []
        for mini_batch in mini_batches:
            token_counts = mini_batch["attention_mask"].sum(dim=1).tolist()
            num_micro_batches = calculate_num_micro_batches(token_counts, self.max_token_len)
            micro_batch_counts.append(num_micro_batches)

        return micro_batch_counts

    def _synchronize_micro_batch_counts(self, local_counts: List[int]) -> List[int]:
        """Synchronize micro-batch counts across distributed workers."""
        if self.dp_group is not None and dist.is_initialized():
            local_copy = local_counts.copy()
            counts_tensor = torch.tensor(local_counts, dtype=torch.int64, device="cuda")
            dist.all_reduce(counts_tensor, op=dist.ReduceOp.MAX, group=self.dp_group)
            synced_counts = [int(x) for x in counts_tensor.tolist()]

            logger.info(
                f"[Rank {dist.get_rank()}] BatchIterator sync - " f"Local counts: {local_copy}, Synced: {synced_counts}"
            )

            return synced_counts
        else:
            if self.dynamic_bsz:
                logger.info(f"BatchIterator - No distributed sync, using local: {local_counts}")
            return local_counts

    def _partition_mini_batch(self, mini_batch: TrainingInputBatch, num_micro_batches: int):
        """Partition a mini-batch into micro-batches."""
        batch_size = mini_batch.batch_size

        if self.dynamic_bsz:
            token_counts = mini_batch["attention_mask"].sum(dim=1).tolist()
            partitions = get_seqlen_balanced_partitions(token_counts, num_micro_batches, equal_size=False)
        else:
            partitions = create_fixed_partitions(batch_size, num_micro_batches)

        for partition in partitions:
            if not partition:
                continue

            micro_batch = mini_batch.partition(partition)

            self._all_micro_batches.append(micro_batch)

    def _log_configuration(self):
        """Log the configuration for debugging."""
        base_info = (
            f"Total batch={self.total_batch_size} | "
            f"Mini-batch/GPU={self.mini_batch_size_per_gpu} | "
            f"Micro-batches={len(self._all_micro_batches)} | "
            f"Micro-batch-size={[len(mb) for mb in self._all_micro_batches]} | "
        )

        if self.dynamic_bsz:
            logger.debug(f"BatchIterator (dynamic): {base_info} | Max tokens/GPU={self.max_token_len}")
        else:
            logger.debug(f"BatchIterator (fixed): {base_info} | Micro-batch size={self.sample_batch_size}")

    def __len__(self):
        """Return the total number of micro-batches."""
        return len(self._all_micro_batches)

    def __iter__(self):
        """Return the iterator itself."""
        for micro_batch in self._all_micro_batches:
            exp = self.batch_to_experience(micro_batch)
            if self.dynamic_bsz:
                exp.info["micro_batch_utilization"] = micro_batch["attention_mask"].sum().item() / self.max_token_len
            yield exp

    @property
    def micro_batches(self):
        """Get all micro-batches (for special cases only)."""
        return self._all_micro_batches

    @staticmethod
    def batch_to_experience(batch: TrainingInputBatch) -> Experience:
        """Convert a TrainingInputBatch to an Experience."""
        exp = Experience(
            sequences=batch["sequences"],
            rollout_logprobs=batch["rollout_log_probs"],
            returns=batch["returns"],
            advantages=batch["advantages"],
            action_log_probs=batch["action_log_probs"],
            base_action_log_probs=batch.get("base_action_log_probs", None),
            values=batch["values"],
            attention_mask=batch["attention_mask"],
            loss_mask=batch["loss_mask"],
            action_mask=batch["response_mask"],
            num_actions=batch.metadata["response_length"],
            info={},
            metadata=batch.metadata,
        )
        return exp
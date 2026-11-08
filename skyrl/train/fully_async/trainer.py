"""The fully-async trainer: inference engines on their own GPUs, generation never stops.

For details, see https://docs.skyrl.ai/docs/tutorials/fully_async.
"""

from skyrl.train.fully_async.admission import (
    AsyncStalenessManager,
    RolloutAdmissionController,
)
from skyrl.train.fully_async.base import BaseFullyAsyncTrainer


class FullyAsyncRayPPOTrainer(BaseFullyAsyncTrainer):
    """Disaggregated fully-async PPO: generation runs continuously on dedicated inference GPUs,
    bounded by :class:`AsyncStalenessManager` so no group is more than ``max_staleness_steps``
    policy versions behind training, and the weight sync pauses and resumes the engines in place."""

    def _generation_buffer_maxsize(self) -> int:
        # Bounded by capacity - consumed = B * (max_staleness_steps + 1).
        return self.mini_batch_size * (self.max_staleness_steps + 1)

    def _build_admission_controller(self) -> RolloutAdmissionController:
        return AsyncStalenessManager(
            max_concurrent_generation_groups=self.num_parallel_generation_workers,
            mini_batch_size=self.mini_batch_size,
            max_staleness_steps=self.max_staleness_steps,
        )

    def _validate_mode_config(self) -> None:
        assert (
            # otherwise wasted throughput
            self.mini_batch_size <= self.num_parallel_generation_workers
            and
            # otherwise would never use all workers due to capacity constraint
            self.num_parallel_generation_workers <= self.mini_batch_size * (self.max_staleness_steps + 1)
        ), (
            "Invalid num_parallel_generation_workers, the following must hold: "
            "mini_batch_size <= num_parallel_generation_workers <= mini_batch_size * (max_staleness_steps + 1). Got: "
            f"{self.mini_batch_size=}, {self.num_parallel_generation_workers=}, {self.max_staleness_steps=}"
        )
        # Generation and training overlap, so they need separate GPUs. A colocated variant (partial
        # rollout: cut generation at a fraction of the mini-batch, train, resume under the new
        # weights) is a sibling subclass of BaseFullyAsyncTrainer, not a mode of this one.
        assert not self.colocate_all, "colocate_all is not supported for async training yet."

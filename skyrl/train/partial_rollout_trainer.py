"""The colocated partial-rollout trainer: fully-async training when the inference engines share the GPUs.

The inference engines and the trainer share the GPUs, as in the synchronous trainer
(``trainer.placement.colocate_all=true``). The step does not wait for every trajectory. The trainer
keeps ``num_parallel_generation_workers`` groups in flight. When ``policy_mini_batch_size`` of them
have finished, the step runs in this order:

1. The inference engines abort the in-flight requests and sleep.
2. The trainer trains on the finished groups.
3. The inference engines wake with the new weights.
4. The aborted requests continue from the tokens they generated before the cut
   (see ``RemoteInferenceClient.resume_aborted_requests``).

Each step cuts the fraction ``1 - policy_mini_batch_size / num_parallel_generation_workers`` of the
in-flight groups.

Compared with the fully-async trainer, there is no GPU split and no generation during training. A
mini-batch is at most a few cuts behind the current policy. Compared with the synchronous trainer,
the step does not wait for the longest trajectory. Tokens generated before a cut carry the rollout
logprobs of the older policy. The loss must use rollout logprobs, and ``FullyAsyncRayPPOTrainer``
asserts this.
"""

from loguru import logger

from skyrl.train.fully_async_trainer import (
    FullyAsyncRayPPOTrainer,
    _AsyncStalenessManager,
)


class PartialRolloutRayPPOTrainer(FullyAsyncRayPPOTrainer):
    """The colocated partial-rollout trainer. See the module docstring."""

    # The checkpoint loads the policy back to the GPUs. It must run while the inference engines sleep.
    checkpoint_before_weight_sync = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The client holds each aborted request and resubmits it after the weight sync. The
        # generator receives one completion per trajectory, as if no cut happened.
        self.inference_engine_client.resume_aborted_requests = True

    def _generation_buffer_maxsize(self) -> int:
        # Every in-flight group can finish before the training loop drains the buffer.
        return self.num_parallel_generation_workers

    def _build_admission_controller(self) -> _AsyncStalenessManager:
        # No staleness term: the cut bounds the staleness, and a resumed group already holds its slot.
        return _AsyncStalenessManager(
            max_concurrent_generation_groups=self.num_parallel_generation_workers,
            mini_batch_size=self.mini_batch_size,
            max_staleness_steps=None,
        )

    def _validate_placement_and_capacity(self) -> None:
        assert self.colocate_all, (
            "PartialRolloutRayPPOTrainer requires trainer.placement.colocate_all=true; with separate "
            "inference GPUs use FullyAsyncRayPPOTrainer instead."
        )
        assert self.num_parallel_generation_workers >= self.mini_batch_size, (
            "trainer.fully_async.num_parallel_generation_workers must be >= policy_mini_batch_size "
            "(equal means no partial rollout: every step waits for all its groups). Got: "
            f"{self.num_parallel_generation_workers=}, {self.mini_batch_size=}"
        )
        cut_fraction = 1 - self.mini_batch_size / self.num_parallel_generation_workers
        logger.info(
            f"Partial rollout: {self.num_parallel_generation_workers} groups in flight, train on every "
            f"{self.mini_batch_size} finished groups, {cut_fraction:.0%} of in-flight groups cut per step."
        )

    async def _before_training_step(self) -> None:
        # The sleep aborts the in-flight requests, which the client holds, and frees the GPUs for training.
        await self.inference_engine_client.sleep()

    async def _sync_weights_after_training_step(self) -> None:
        # The inherited weight sync wakes the weights, broadcasts them, and wakes the key-value (KV)
        # cache. The KV cache wake resumes the inference engines, and the client resubmits the held
        # requests.
        await super()._sync_weights_after_training_step()
        # At this point the client holds every request that this step's cut aborted.
        stats = self.inference_engine_client.pop_resume_stats()
        self.all_metrics.update({f"partial_rollout/{k}": v for k, v in stats.items()})

    async def _after_training_loop(self) -> None:
        await self.inference_engine_client.sleep()

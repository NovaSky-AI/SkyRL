"""The colocated partial-rollout trainer: fully-async training when the inference engines share the GPUs.

The inference engines and the trainer share the GPUs, as in the synchronous trainer
(``trainer.placement.colocate_all=true``). The step does not wait for every trajectory. The trainer
keeps ``num_parallel_generation_workers`` groups in flight. When ``policy_mini_batch_size`` of them
have finished, the step runs in this order:

1. The inference engines pause the in-flight requests and sleep. The requests stay in the engines.
2. The trainer trains on the finished groups.
3. The inference engines wake with the new weights.
4. The paused requests continue from the tokens they generated before the cut.

``trainer.fully_async.max_staleness_steps`` selects when a finished group frees its slot. With ``0``, a
finished group holds its slot until the step that trains on it. Each step then cuts the fraction
``1 - policy_mini_batch_size / num_parallel_generation_workers`` of the in-flight groups, and the mean
staleness is about ``num_parallel_generation_workers / policy_mini_batch_size - 1`` steps. With a larger
value, a finished group frees its slot at once, as in the fully-async trainer. The inference engines
stay full, each cut hits about ``num_parallel_generation_workers`` groups, and the mean staleness is
about one step higher. See ``_PartialRolloutAdmissionController``.

Compared with the fully-async trainer, there is no GPU split and no generation during training. A
mini-batch is at most a few cuts behind the current policy. Compared with the synchronous trainer,
the step does not wait for the longest trajectory. Tokens generated before a cut carry the rollout
logprobs of the older policy. The loss must use rollout logprobs, and ``FullyAsyncRayPPOTrainer``
asserts this.

``trainer.fully_async.clear_kv_cache_on_weight_sync`` selects the handling of the key-value (KV) cache
of the paused requests. ``false`` copies the KV cache to host memory during training and restores it, so
a request resumes with no prefill and continues from KV states of the older weights. ``true`` discards
the KV cache, and the inference engine prefills the paused requests again with the new weights.

The trainer sets no limit on the number of cuts that one group survives. The metric
``partial_rollout/groups_cut`` reports the groups in flight at each cut. The metric ``async/staleness``
is the age of a trained group in steps. The age also counts the steps that a finished group waited in
the buffer.
"""

from loguru import logger

from skyrl.train.fully_async_trainer import (
    FullyAsyncRayPPOTrainer,
    _AsyncStalenessManager,
)


class _PartialRolloutAdmissionController(_AsyncStalenessManager):
    """Admit a group under two limits.

    1. At most ``max_concurrent_generation_groups`` groups generate at one time. A cut group that
       waits to resume counts as generating, so a resumed group is never blocked.
    2. At most ``max(max_concurrent_generation_groups, mini_batch_size * (max_staleness_steps + 1))``
       groups are not yet trained on. This count includes the finished groups in the buffer.

    With ``max_staleness_steps=0`` the second limit is ``max_concurrent_generation_groups``. A
    finished group then holds its slot until the step that trains on it, and each step admits
    ``mini_batch_size`` new groups. With a larger ``max_staleness_steps``, a finished group frees its
    slot at once, and the inference engines stay at ``max_concurrent_generation_groups`` groups. The
    buffer then holds more groups, and each cut hits more groups, so the staleness is higher.

    The rule of ``_AsyncStalenessManager`` has the second limit without the ``max``. It needs
    ``max_concurrent_generation_groups <= mini_batch_size * (max_staleness_steps + 1)``, which does
    not hold for partial rollout at ``max_staleness_steps=0``."""

    def _compute_capacity_unlocked(self) -> int:
        # The steps before `_current_global_step` trained on `mini_batch_size` groups each.
        trained = (self._current_global_step - 1) * self.mini_batch_size
        not_yet_trained = self._stat.accepted + self._stat.running - trained
        max_not_yet_trained = max(
            self.max_concurrent_generation_groups, self.mini_batch_size * (self.max_staleness_steps + 1)
        )
        return min(self.max_concurrent_generation_groups - self._stat.running, max_not_yet_trained - not_yet_trained)


class PartialRolloutRayPPOTrainer(FullyAsyncRayPPOTrainer):
    """The colocated partial-rollout trainer. See the module docstring."""

    def _generation_buffer_maxsize(self) -> int:
        # The admission controller allows this many groups that are not yet trained on. All of them
        # can finish before the training loop drains the buffer.
        return max(self.num_parallel_generation_workers, self.mini_batch_size * (self.max_staleness_steps + 1))

    def _build_admission_controller(self) -> _AsyncStalenessManager:
        return _PartialRolloutAdmissionController(
            max_concurrent_generation_groups=self.num_parallel_generation_workers,
            mini_batch_size=self.mini_batch_size,
            max_staleness_steps=self.max_staleness_steps,
        )

    def _holds_slot_until_trained(self) -> bool:
        return self.mini_batch_size * (self.max_staleness_steps + 1) <= self.num_parallel_generation_workers

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
        assert not self.sample_full_batch, (
            "PartialRolloutRayPPOTrainer does not support trainer.fully_async.sample_full_batch=true. When an "
            "epoch runs out of prompts mid mini-batch, the loop saves the end-of-epoch checkpoint with no "
            "training step before it, so the inference engines are awake and hold the GPU memory."
        )
        if self.cfg.trainer.fully_async.clear_kv_cache_on_weight_sync:
            assert self.cfg.generator.inference_engine.enable_prefix_caching, (
                "PartialRolloutRayPPOTrainer with trainer.fully_async.clear_kv_cache_on_weight_sync=true "
                "requires generator.inference_engine.enable_prefix_caching=true. The prefix-cache reset is "
                "the step that makes the inference engine prefill the paused requests again."
            )
        cut_fraction = 1 - self.mini_batch_size / self.num_parallel_generation_workers
        logger.info(
            f"Partial rollout: {self.num_parallel_generation_workers} groups in flight, train on every "
            f"{self.mini_batch_size} finished groups, {cut_fraction:.0%} of in-flight groups cut per step, "
            f"max_staleness_steps={self.max_staleness_steps} "
            f"({'a finished group holds its slot' if self._holds_slot_until_trained() else 'a finished group frees its slot'}), "
            f"KV cache of the paused requests: "
            f"{'cleared' if self.cfg.trainer.fully_async.clear_kv_cache_on_weight_sync else 'kept in host memory'}."
        )
        router_kwargs = self.cfg.generator.inference_engine.router_init_kwargs or {}
        if router_kwargs.get("policy") != "sticky_least_loaded":
            logger.warning(
                "Partial rollout works best with "
                "generator.inference_engine.router_init_kwargs.policy=sticky_least_loaded. The default "
                "consistent_hash policy places trajectories by session id, so long trajectories and their "
                "resumed segments pile up on one engine and can exhaust its KV cache."
            )

    async def _before_training_step(self) -> None:
        # The groups in flight at this point are the groups that this step cuts.
        self.all_metrics["partial_rollout/groups_cut"] = self._staleness_manager._stat.running
        # The inference engines keep the in-flight requests paused and free the GPUs for training.
        # The KV cache goes to host memory, unless the weight sync clears it. The inherited weight
        # sync wakes the engines and resumes the paused requests (`WorkerDispatch.save_weights_for_sampler`).
        offload_kv = not self.cfg.trainer.fully_async.clear_kv_cache_on_weight_sync
        await self.inference_engine_client.sleep_with_frozen_requests(offload_kv=offload_kv)

    async def _sync_weights_after_training_step(self, force_save: bool, is_epoch_end: bool) -> None:
        # The checkpoint loads the policy back to the GPUs. Write it first, while the inference
        # engines sleep. The loop's call after eval, when the engines are awake, then does nothing.
        await self._save_step_checkpoints(force_save, is_epoch_end)
        # The inherited weight sync wakes the engines and resumes the paused requests.
        await super()._sync_weights_after_training_step(force_save, is_epoch_end)

    async def _after_training_loop(self) -> None:
        await self.inference_engine_client.sleep()

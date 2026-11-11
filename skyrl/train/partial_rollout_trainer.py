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

``trainer.fully_async.partial_rollout.cut_mode`` selects what the cut does to the in-flight requests.
``abort`` is the sequence above. Under ``freeze``, the inference engines pause the requests instead of
aborting them, and resume them after the weight sync.
``trainer.fully_async.clear_kv_cache_on_weight_sync`` then selects the handling of the key-value (KV)
cache. ``false`` copies the KV cache to host memory during training and restores it, so a request
resumes with no prefill. ``true`` discards the KV cache, and the inference engine prefills the paused
requests again with the new weights.

The trainer sets no limit on the number of cuts that one group survives. The metrics
``partial_rollout/cuts_survived_mean`` and ``partial_rollout/cuts_survived_max`` report that number per
step over the completed requests. The metric ``async/staleness`` is the age of a trained group in steps.
The age also counts the steps that a finished group waited in the buffer.
"""

from loguru import logger

from skyrl.train.fully_async_trainer import (
    FullyAsyncRayPPOTrainer,
    _AsyncStalenessManager,
)

CUT_MODES = ("abort", "freeze")


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

    # The checkpoint loads the policy back to the GPUs. It must run while the inference engines sleep.
    checkpoint_before_weight_sync = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The client holds each aborted request and resubmits it after the weight sync. The
        # generator receives one completion per trajectory, as if no cut happened.
        self.inference_engine_client.resume_aborted_requests = True
        self._freeze_requests_at_cut = self.cfg.trainer.fully_async.partial_rollout.cut_mode == "freeze"

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
        # The base constructor calls this method before `__init__` of this class sets its attributes.
        cut_mode = self.cfg.trainer.fully_async.partial_rollout.cut_mode
        assert (
            cut_mode in CUT_MODES
        ), f"trainer.fully_async.partial_rollout.cut_mode must be one of {CUT_MODES}. Got: {cut_mode!r}"
        if cut_mode == "freeze":
            assert self.cfg.trainer.policy.model.lora.rank == 0, (
                'trainer.fully_async.partial_rollout.cut_mode="freeze" requires full-weight sync. The sleep '
                "discards the weights of the inference engines, and a LoRA sync does not send them again."
            )
            if self.cfg.trainer.fully_async.clear_kv_cache_on_weight_sync:
                assert self.cfg.generator.inference_engine.enable_prefix_caching, (
                    'trainer.fully_async.partial_rollout.cut_mode="freeze" with '
                    "trainer.fully_async.clear_kv_cache_on_weight_sync=true requires "
                    "generator.inference_engine.enable_prefix_caching=true. The prefix-cache reset is the "
                    "step that makes the inference engine prefill the frozen requests again."
                )
        cut_fraction = 1 - self.mini_batch_size / self.num_parallel_generation_workers
        logger.info(
            f"Partial rollout: {self.num_parallel_generation_workers} groups in flight, train on every "
            f"{self.mini_batch_size} finished groups, {cut_fraction:.0%} of in-flight groups cut per step, "
            f"max_staleness_steps={self.max_staleness_steps} "
            f"({'a finished group holds its slot' if self._holds_slot_until_trained() else 'a finished group frees its slot'}), "
            f"cut_mode={cut_mode}."
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
        if self._freeze_requests_at_cut:
            # The inference engines keep the in-flight requests paused and free the GPUs for training.
            # The KV cache goes to host memory, unless the weight sync clears it.
            offload_kv = not self.cfg.trainer.fully_async.clear_kv_cache_on_weight_sync
            await self.inference_engine_client.sleep_with_frozen_requests(offload_kv=offload_kv)
        else:
            # The sleep aborts the in-flight requests, which the client holds, and frees the GPUs for training.
            await self.inference_engine_client.sleep()

    async def _sync_weights_after_training_step(self) -> None:
        # The inherited weight sync wakes the weights, broadcasts them, and wakes the key-value (KV)
        # cache. Under cut_mode="abort", the KV cache wake resumes the inference engines, and the
        # client resubmits the held requests. Under cut_mode="freeze", the dispatch resumes the
        # frozen requests after the KV cache wake.
        await super()._sync_weights_after_training_step()
        # At this point the client holds every request that this step's cut aborted.
        stats = self.inference_engine_client.pop_resume_stats()
        completed = stats.pop("completed_requests")
        cuts_survived_sum = stats.pop("cuts_survived_sum")
        self.all_metrics.update({f"partial_rollout/{k}": v for k, v in stats.items()})
        self.all_metrics["partial_rollout/cuts_survived_mean"] = cuts_survived_sum / completed if completed else 0.0

    async def _after_training_loop(self) -> None:
        await self.inference_engine_client.sleep()

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

``trainer.fully_async.partial_rollout.stale_group_policy`` selects what happens to a group that reaches
``trainer.fully_async.partial_rollout.max_cuts_per_group`` cuts. ``ignore`` sets no limit: the trainer
cuts and resumes the group until it finishes. ``wait`` delays the next cut
until every such group has finished. ``drop`` does not resume such a group: the client returns its
request with stop reason ``abort``, and the trainer discards the group.

The staleness that the policy bounds is the number of cuts a request survived. The metrics
``partial_rollout/cuts_survived_mean`` and ``partial_rollout/cuts_survived_max`` report it per step over
the completed requests. The metric ``async/staleness`` is the age of a trained group in steps. The age
also counts the steps that a finished group waited in the buffer. The buffer holds at most
``num_parallel_generation_workers - 1`` groups, so the wait is short when that number is close to
``policy_mini_batch_size``.
"""

from loguru import logger

from skyrl.train.fully_async_trainer import (
    FullyAsyncRayPPOTrainer,
    GeneratedOutputGroup,
    _AsyncStalenessManager,
)

STALE_GROUP_POLICIES = ("ignore", "wait", "drop")


class _PartialRolloutAdmissionController(_AsyncStalenessManager):
    """Admit a group while fewer than ``max_concurrent_generation_groups`` groups are not yet trained on.

    A group holds its slot from submission until the step that trains on it. A finished group that
    waits in the buffer still holds its slot, and so does a cut group that waits to resume. So each
    step admits ``mini_batch_size`` new groups, and a resumed group is never blocked.

    This rule replaces the staleness capacity rule of ``_AsyncStalenessManager``. The cut bounds the
    staleness, so ``max_staleness_steps`` is not used."""

    def __init__(self, max_concurrent_generation_groups: int, mini_batch_size: int):
        # The override of `_compute_capacity_unlocked` does not read `max_staleness_steps`.
        super().__init__(max_concurrent_generation_groups, mini_batch_size, max_staleness_steps=0)

    def _compute_capacity_unlocked(self) -> int:
        # The steps before `_current_global_step` trained on `mini_batch_size` groups each.
        trained = (self._current_global_step - 1) * self.mini_batch_size
        not_yet_trained = self._stat.accepted + self._stat.running - trained
        return self.max_concurrent_generation_groups - not_yet_trained


class PartialRolloutRayPPOTrainer(FullyAsyncRayPPOTrainer):
    """The colocated partial-rollout trainer. See the module docstring."""

    # The checkpoint loads the policy back to the GPUs. It must run while the inference engines sleep.
    checkpoint_before_weight_sync = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The client holds each aborted request and resubmits it after the weight sync. The
        # generator receives one completion per trajectory, as if no cut happened.
        self.inference_engine_client.resume_aborted_requests = True
        partial_rollout_cfg = self.cfg.trainer.fully_async.partial_rollout
        self._stale_group_policy = partial_rollout_cfg.stale_group_policy
        self._max_cuts_per_group = partial_rollout_cfg.max_cuts_per_group
        self._stale_groups_dropped = 0
        if self._stale_group_policy == "drop":
            # A request that survives more than max_cuts_per_group cuts comes back with stop reason
            # "abort", and `_accept_generated_group` drops its group.
            self.inference_engine_client.max_resumes_per_request = self._max_cuts_per_group

    def _generation_buffer_maxsize(self) -> int:
        # Every in-flight group can finish before the training loop drains the buffer.
        return self.num_parallel_generation_workers

    def _build_admission_controller(self) -> _AsyncStalenessManager:
        return _PartialRolloutAdmissionController(
            max_concurrent_generation_groups=self.num_parallel_generation_workers,
            mini_batch_size=self.mini_batch_size,
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
        # The base constructor calls this method before `__init__` of this class sets its attributes.
        partial_rollout_cfg = self.cfg.trainer.fully_async.partial_rollout
        policy = partial_rollout_cfg.stale_group_policy
        max_cuts = partial_rollout_cfg.max_cuts_per_group
        assert policy in STALE_GROUP_POLICIES, (
            f"trainer.fully_async.partial_rollout.stale_group_policy must be one of {STALE_GROUP_POLICIES}. "
            f"Got: {policy!r}"
        )
        if policy != "ignore":
            assert max_cuts >= 0, (
                "trainer.fully_async.partial_rollout.max_cuts_per_group must be >= 0 under "
                f"stale_group_policy={policy!r}. Got: {max_cuts}"
            )
        cut_fraction = 1 - self.mini_batch_size / self.num_parallel_generation_workers
        logger.info(
            f"Partial rollout: {self.num_parallel_generation_workers} groups in flight, train on every "
            f"{self.mini_batch_size} finished groups, {cut_fraction:.0%} of in-flight groups cut per step, "
            f"stale_group_policy={policy}" + ("." if policy == "ignore" else f" at max_cuts_per_group={max_cuts}.")
        )
        router_kwargs = self.cfg.generator.inference_engine.router_init_kwargs or {}
        if router_kwargs.get("policy") != "sticky_least_loaded":
            logger.warning(
                "Partial rollout works best with "
                "generator.inference_engine.router_init_kwargs.policy=sticky_least_loaded. The default "
                "consistent_hash policy places trajectories by session id, so long trajectories and their "
                "resumed segments pile up on one engine and can exhaust its KV cache."
            )

    def _epoch_may_exhaust_early(self) -> bool:
        # Dropped stale groups do not reach the buffer, so an epoch can run out of prompts mid mini-batch.
        return super()._epoch_may_exhaust_early() or (
            self.cfg.trainer.fully_async.partial_rollout.stale_group_policy == "drop"
        )

    def _accept_generated_group(self, group: GeneratedOutputGroup) -> bool:
        if self._stale_group_policy != "drop":
            return True
        stop_reasons = group.generator_output.get("stop_reasons") or []
        if "abort" not in stop_reasons:
            return True
        # The client did not resume a request of this group: it reached max_cuts_per_group cuts.
        self._stale_groups_dropped += 1
        logger.info(f"Dropping group {group.uid}: it reached {self._max_cuts_per_group} cuts.")
        return False

    async def _wait_for_groups_at_cut_limit(self) -> None:
        """Delay the cut until every in-flight group at the cut limit has finished.

        A group that started at step ``s`` has survived ``global_step - s`` cuts. A group with
        ``s <= global_step - max_cuts_per_group`` would exceed the limit after one more cut. The trainer lets it finish before the inference engines sleep.
        """
        threshold = self.global_step - self._max_cuts_per_group
        failure = self._generation_failure
        waited_groups = 0
        with self._phase_gauge.timed_phase("wait_for_stale_groups", self.all_timings):
            while True:
                stale = [uid for uid, start in self._in_flight_start_steps.items() if start <= threshold]
                if not stale:
                    break
                waited_groups = max(waited_groups, len(stale))
                self._in_flight_changed.clear()
                changed = self._in_flight_changed.wait()
                await (failure.guard(changed) if failure is not None else changed)
        self.all_metrics["partial_rollout/stale_groups_waited"] = waited_groups

    async def _before_training_step(self) -> None:
        if self._stale_group_policy == "wait":
            await self._wait_for_groups_at_cut_limit()
        # The sleep aborts the in-flight requests, which the client holds, and frees the GPUs for training.
        await self.inference_engine_client.sleep()

    async def _sync_weights_after_training_step(self) -> None:
        # The inherited weight sync wakes the weights, broadcasts them, and wakes the key-value (KV)
        # cache. The KV cache wake resumes the inference engines, and the client resubmits the held
        # requests.
        await super()._sync_weights_after_training_step()
        # At this point the client holds every request that this step's cut aborted.
        stats = self.inference_engine_client.pop_resume_stats()
        completed = stats.pop("completed_requests")
        cuts_survived_sum = stats.pop("cuts_survived_sum")
        self.all_metrics.update({f"partial_rollout/{k}": v for k, v in stats.items()})
        self.all_metrics["partial_rollout/cuts_survived_mean"] = cuts_survived_sum / completed if completed else 0.0
        self.all_metrics["partial_rollout/stale_groups_dropped"] = self._stale_groups_dropped
        self._stale_groups_dropped = 0

    async def _after_training_loop(self) -> None:
        await self.inference_engine_client.sleep()

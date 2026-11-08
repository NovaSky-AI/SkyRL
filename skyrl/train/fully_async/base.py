"""Shared training loop for trainers that stream rollouts one group at a time.

For details, see https://docs.skyrl.ai/docs/tutorials/fully_async.

:class:`BaseFullyAsyncTrainer` owns everything that does not depend on where the inference
engines live: the generation workers, the completed-group buffer, mini-batch collection,
conversion to a training batch, checkpointing with consumed UIDs, metrics and the epoch
bookkeeping. A subclass supplies the policy that does depend on it through a few hooks:

- :meth:`_build_admission_controller`: when a worker may submit the next group (a staleness
  budget for the disaggregated fully-async trainer; a cut fraction of the mini-batch for a
  colocated partial-rollout trainer).
- :meth:`_generation_buffer_maxsize`: the bound on completed groups the controller can have
  buffered ahead of training.
- :meth:`_validate_mode_config`: placement and capacity assertions.
- :meth:`_before_training_step`: what happens to the engines before training (nothing when
  generation keeps running on its own GPUs; pause-and-sleep when they share the trainer's).
- :meth:`_sync_weights_after_training_step`: the weight-sync bracket (default: pause,
  broadcast, resume); a colocated subclass wakes the engines and resubmits held requests.
- :meth:`_after_training_loop`: engine teardown once training is done.

High-level notes:
- The global_step in each training loop iteration denotes the "current step being worked on", so
`global_step - 1` is the number of steps the model has finished training.
- We do not do any cross-epoch asynchrony here, so all the control logics like
  generation workers and data buffer are initialized per-epoch. The async dataloader
  and admission controller are also reset / validated at the end of each epoch.
"""

import asyncio
import enum
import inspect
import os
import sys
import time
from dataclasses import dataclass, field, replace
from typing import Any, List, Optional, Set, Tuple

import torch
from loguru import logger
from tqdm import tqdm

from skyrl.backends.skyrl_train.inference_servers.engine_utils import (
    get_sampling_params_for_backend,
)
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.utils.io import io
from skyrl.backends.skyrl_train.utils.ppo_utils import (
    LOSSES_WITH_OLD_LOGPROBS,
    PolicyLossType,
)
from skyrl.train.fully_async.admission import RolloutAdmissionController
from skyrl.train.fully_async.dataloader import AsyncDataloader
from skyrl.train.fully_async.group import GeneratedOutputGroup
from skyrl.train.generators.base import GeneratorOutput
from skyrl.train.generators.utils import (
    concatenate_generator_outputs,
    get_metrics_from_generator_output,
    prepare_generator_input,
)
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils import Timer
from skyrl.train.utils.async_utils import BackgroundFailure, cancel_background_tasks
from skyrl.train.utils.callbacks import CallbackInput
from skyrl.train.utils.metrics import ScalarGauges, TrainingPhaseGauge
from skyrl.train.utils.trainer_utils import (
    ResumeMode,
    build_dataloader,
    get_group_completion_metrics,
    get_intra_group_completion_time_std_cv,
    zero_variance_filter,
)


class _StepOutcome(enum.Enum):
    """What the step loop does after one iteration of :meth:`BaseFullyAsyncTrainer._run_step`."""

    CONTINUE = "continue"
    EPOCH_EXHAUSTED = "epoch_exhausted"
    STOP_TRAINING = "stop_training"


@dataclass
class _EpochGeneration:
    """The generation side of one epoch: the worker tasks and the buffer they fill."""

    buffer: asyncio.Queue
    failure: BackgroundFailure
    all_generators_done: asyncio.Event
    generator_tasks: List[asyncio.Task] = field(default_factory=list)
    generators_done_watcher: Optional[asyncio.Task] = None

    def background_tasks(self) -> List[asyncio.Task]:
        tasks = [*self.generator_tasks]
        if self.generators_done_watcher is not None:
            tasks.append(self.generators_done_watcher)
        return tasks

    async def stop(self) -> None:
        """Cancel the workers and the watcher, wait for them, then re-raise a recorded worker failure."""
        for t in self.generator_tasks:
            t.cancel()
        await asyncio.gather(*self.generator_tasks, return_exceptions=True)
        if self.generators_done_watcher is not None:
            self.generators_done_watcher.cancel()
            await asyncio.gather(self.generators_done_watcher, return_exceptions=True)
        self.failure.raise_if_failed()


class BaseFullyAsyncTrainer(RayPPOTrainer):
    """Streams rollouts one group at a time through ``num_parallel_generation_workers`` workers and
    trains on every ``policy_mini_batch_size`` completed groups. See the module docstring for the
    hooks a subclass implements."""

    def __init__(self, *args, **kwargs):
        # Extract cfg before base init so we can initialize async-specific knobs used by our overrides.
        cfg = kwargs.get("cfg", args[0] if len(args) > 0 else None)
        assert cfg is not None, f"cfg must be provided to {type(self).__name__}"

        # Initialize async-specific knobs
        self.num_parallel_generation_workers = cfg.trainer.fully_async.num_parallel_generation_workers
        self.mini_batch_size = cfg.trainer.policy_mini_batch_size
        self.max_staleness_steps = cfg.trainer.fully_async.max_staleness_steps
        self.sample_full_batch = cfg.trainer.fully_async.sample_full_batch

        self._phase_gauge = TrainingPhaseGauge()

        self._gen_buffer_maxsize = self._generation_buffer_maxsize()
        self._loop_gauges = ScalarGauges()
        self._loop_gauges.set(
            "skyrl_mini_batch_size", self.mini_batch_size, "Generation groups consumed per training step."
        )
        self._loop_gauges.set(
            "skyrl_gen_buffer_maxsize", self._gen_buffer_maxsize, "Staleness-bounded generation-buffer capacity."
        )

        # Initialize base trainer
        super().__init__(*args, **kwargs)

        # Some async-specific validations
        assert (
            self.cfg.trainer.fully_async.enabled
        ), "trainer.fully_async.enabled must be True when using the fully async trainer."
        assert (
            self.cfg.trainer.train_batch_size == self.cfg.trainer.policy_mini_batch_size
        ), "train_batch_size must equal policy_mini_batch_size for fully async training"
        assert (
            self.cfg.trainer.algorithm.dynamic_sampling.type is None
        ), "dynamic sampling is not supported for fully async training yet."
        if self.sample_full_batch:
            assert self.cfg.trainer.algorithm.zero_variance_filter, (
                "trainer.fully_async.sample_full_batch=True requires trainer.algorithm.zero_variance_filter=True "
                "(it is the async-native equivalent of dynamic_sampling='filter')."
            )
        assert (
            not self.cfg.generator.batched
        ), "batched is not supported for fully async training since a batched generate() call does not support pause/continue."
        # Ensure we're using a policy loss that doesn't depend on pi_old ~= pi_rollout
        loss_type = self.cfg.trainer.algorithm.policy_loss_type
        if loss_type == PolicyLossType.CISPO and self.cfg.trainer.algorithm.cispo.cispo_anchor == "rollout":
            optimizes_against_rollout = True
        else:
            optimizes_against_rollout = loss_type not in LOSSES_WITH_OLD_LOGPROBS
        assert optimizes_against_rollout, (
            f"Found trainer.algorithm.policy_loss_type={loss_type} in "
            f"{sorted([loss.value for loss in LOSSES_WITH_OLD_LOGPROBS])}. Fully async training should use "
            "rollout logprobs (i.e. rollout_is, dppo, reinforce, or cispo with cispo.cispo_anchor='rollout') instead "
            "of recomputing logprobs, since stale policies are not kept for logprob computation."
        )
        self._validate_mode_config()

        # TODO(Charlie): need to assert we are doing TIS and returning logprobs

        # Async-specific states
        self.async_train_dataloader = AsyncDataloader(self.train_dataloader, self.mini_batch_size)
        self._admission_controller = self._build_admission_controller()

    # ------------------------------------------------------------------
    # Hooks. See the module docstring for which trainer overrides what.
    # ------------------------------------------------------------------

    def _generation_buffer_maxsize(self) -> int:
        """Upper bound on completed groups buffered ahead of training. Called before the base
        trainer is initialized, so only ``self.mini_batch_size``, ``self.max_staleness_steps`` and
        ``self.num_parallel_generation_workers`` are available."""
        raise NotImplementedError

    def _build_admission_controller(self) -> RolloutAdmissionController:
        """The policy deciding when a generation worker may submit the next group."""
        raise NotImplementedError

    def _validate_mode_config(self) -> None:
        """Placement and capacity assertions specific to one trainer. Runs after the base trainer is
        initialized, with ``self.cfg`` and ``self.colocate_all`` set."""

    async def _before_training_step(self) -> None:
        """Runs once the mini-batch is collected, before the forward/backward pass. Default: nothing;
        generation keeps running on its own GPUs."""

    async def _sync_weights_after_training_step(self) -> None:
        """Push the new policy to the inference engines. The default brackets the transfer with the
        dispatch's own pause and resume of generation."""
        async with self._weight_sync_deadline():
            with self._phase_gauge.timed_phase("sync_weights", self.all_timings):
                await self.dispatch.save_weights_for_sampler()
        # `sync_weights` above is the full bracket: it also pauses and
        # resumes generation, which under vLLM DP costs seconds of
        # coordinator quiesce that is not weight-sync work. The
        # dispatch reports the transfer on its own alongside it.
        self.all_timings.update(self.dispatch.get_timing_metrics())

    async def _after_training_loop(self) -> None:
        """Runs after the last epoch, before the final checkpoint. Default: nothing."""

    # ------------------------------------------------------------------
    # Compatibility aliases.
    # ------------------------------------------------------------------

    @property
    def _staleness_manager(self) -> RolloutAdmissionController:
        """Former name of :attr:`_admission_controller`."""
        return self._admission_controller

    # ------------------------------------------------------------------
    # Shared loop.
    # ------------------------------------------------------------------

    def _build_callback_input(self, **fields) -> CallbackInput:
        """The base trainer's input, with ``steps_per_epoch`` in training steps (mini-batches).

        With ``sample_full_batch``, ``steps_per_epoch`` is an upper bound: dropped groups can end an epoch early.
        """
        return replace(super()._build_callback_input(**fields), steps_per_epoch=self.num_steps_per_epoch)

    def _build_train_dataloader_and_compute_training_steps(self):
        """
        Overrides to build dataloader for fully async training. See `AsyncDataloader` for more details.
        """
        self.train_dataloader = build_dataloader(self.cfg, self.train_dataset, is_train=True, is_fully_async=True)
        self.num_steps_per_epoch = len(self.train_dataloader) // self.mini_batch_size
        self.total_training_steps = self.num_steps_per_epoch * self.cfg.trainer.epochs
        if self.cfg.trainer.max_training_steps is not None:
            self.total_training_steps = min(self.total_training_steps, self.cfg.trainer.max_training_steps)
        logger.info(f"Length of train_dataloader: {len(self.train_dataloader)}")
        logger.info(f"Number of steps per epoch: {self.num_steps_per_epoch}")
        logger.info(f"Total training steps: {self.total_training_steps}")

    def _load_resume_state(self) -> Optional[int]:
        """Restore global_step, the consumed UIDs and the admission controller from a checkpoint.

        Returns the epoch to resume in, or None when starting fresh.
        """
        resumed_start_epoch = None
        # Load checkpoint state if resumption is enabled. Also load the data UIDs that are already trained on.
        if self.resume_mode != ResumeMode.NONE:
            with Timer("load_checkpoints"):
                (
                    self.global_step,
                    _,
                    loaded_consumed_data_uids_set,
                    loaded_filtered_data_uids_set,
                    loaded_epoch,
                ) = self.load_checkpoints()
                logger.info(f"Resumed training from global_step {self.global_step}")
                if self.global_step > 0:
                    # Set async dataloader manager and admission controller to the loaded state.
                    self.async_train_dataloader.load_state_from_checkpoint(
                        loaded_consumed_data_uids_set, loaded_filtered_data_uids_set
                    )
                    self._admission_controller.load_state_from_checkpoint(
                        self.global_step + 1
                    )  # +1 due to we haven't incremented yet
                    # Only trained UIDs map to completed steps (filtered UIDs are extra consumption),
                    # so validate the trained count is a whole number of steps.
                    num_trained_loaded = len(loaded_consumed_data_uids_set) - len(loaded_filtered_data_uids_set)
                    assert num_trained_loaded % self.mini_batch_size == 0, (
                        "Loaded trained (consumed minus filtered) data UIDs must be a multiple of "
                        f"mini_batch_size={self.mini_batch_size}. Got: {num_trained_loaded}"
                    )
                    # Use the persisted epoch; fall back to deriving it for pre-sample_full_batch
                    # checkpoints (where global_step stays aligned to epoch boundaries).
                    resumed_start_epoch = (
                        loaded_epoch if loaded_epoch is not None else self.global_step // self.num_steps_per_epoch
                    )
        return resumed_start_epoch

    async def train(self):
        """
        Main fully async training loop for PPO
        """
        self.global_step = 0
        self.epoch = 0
        resumed_start_epoch = self._load_resume_state()

        # Initialize weight sync state
        async with self._weight_sync_deadline(), Timer("init_weight_sync_state"):
            self.init_weight_sync_state()

        # sync weights to inference engines
        async with self._weight_sync_deadline(), Timer("sync_weights_to_inference_engines"):
            await self.dispatch.save_weights_for_sampler()

        # Per-step GPU utilization to the tracker. The base loop starts, flushes, and stops the
        # monitor itself. The async loop overrides train() and must wire it here.
        if self._ray_gpu_monitor is not None:
            self._ray_gpu_monitor.start()

        if self._vllm_metrics_scraper is not None:
            await self._vllm_metrics_scraper.sample()

        start_epoch = resumed_start_epoch if resumed_start_epoch is not None else 0
        self._current_epoch = start_epoch
        self._training_control.reset()
        self._fire("on_train_start")

        # Eval before training
        if self.cfg.trainer.eval_interval > 0 and self.cfg.trainer.eval_before_train:
            self._fire("on_eval_start")
            with self._phase_gauge.timed_phase("eval", self.all_timings):
                eval_metrics = await self.eval()
            self._fire("on_eval_end", metrics=eval_metrics)
            self._fire("on_log", logs=eval_metrics)
            self.tracker.log(eval_metrics, step=self.global_step, commit=True)

        # main training loop
        pbar = tqdm(total=self.total_training_steps, initial=self.global_step, desc="Training Step Progress")
        self.global_step += 1  # start training at global_step 1
        stop_training = False
        self._profiler_start()
        try:
            for epoch in range(start_epoch, self.cfg.trainer.epochs):
                stop_training = await self._run_epoch(epoch, pbar)
                if stop_training:
                    break
        finally:
            self._profiler_stop()
            if self._ray_gpu_monitor is not None:
                self._ray_gpu_monitor.stop()

        pbar.close()

        if not stop_training:
            # All epochs completed: advance past the last epoch so resuming from the final checkpoint
            # does not redo the last epoch (whose consumed UIDs were cleared at its epoch end).
            self.epoch = self.cfg.trainer.epochs

        await self._after_training_loop()

        # safety net: always save final checkpoint at end of training.
        if self.cfg.trainer.ckpt_interval > 0:
            async with self._step_deadline():
                with self._phase_gauge.timed_phase("save_checkpoints", self.all_timings):
                    ckpt_path = await asyncio.to_thread(self.save_checkpoints)
                    logger.info("Saved final checkpoint.")
            self._fire("on_save", ckpt_path=ckpt_path)
        if self.cfg.trainer.hf_save_interval > 0:
            async with self._step_deadline():
                with self._phase_gauge.timed_phase("save_hf_model", self.all_timings):
                    await asyncio.to_thread(self.save_models)
                    logger.info("Saved final model.")

        # Drain any in-flight async checkpoint write before teardown. Unconditional:
        # a save may have happened outside the periodic path. No-op when nothing is pending.
        self.dispatch.finalize_pending_saves("policy")
        if self.has_critic:
            self.dispatch.finalize_pending_saves("critic")

        await self.finalize_metrics("success")
        # The loop leaves global_step one past the last trained step; report the last trained step, as the
        # synchronous trainer does.
        self.global_step -= 1
        self._fire("on_train_end")
        self.tracker.finish()
        logger.info("Training done!")

    def _start_epoch_generation(self) -> _EpochGeneration:
        """Spawn this epoch's generation workers over a fresh buffer. Note that we do not do any
        cross-epoch asynchrony here."""
        gen = _EpochGeneration(
            # Buffer of completed generation, size bounded by capacity - consumed = B * (max_staleness_steps + 1)
            buffer=asyncio.Queue[GeneratedOutputGroup](maxsize=self._gen_buffer_maxsize),
            # First generation-worker exception; consumers waiting on the buffer raise it.
            failure=BackgroundFailure(),
            # Lets the consumer detect epoch exhaustion (all generators done + buffer empty) instead of
            # blocking forever on buffer.get() -- under sample_full_batch, drops can exhaust an epoch
            # before num_steps_per_epoch steps complete.
            all_generators_done=asyncio.Event(),
        )
        # Maintain self.num_parallel_generation_workers concurrent group-generation workers
        gen.generator_tasks = [
            asyncio.create_task(self._run_generate_for_a_group_loop(gen.buffer, gen.failure))
            for _ in range(self.num_parallel_generation_workers)
        ]
        if self.sample_full_batch:
            gen.generators_done_watcher = asyncio.create_task(
                self._watch_generators_done(gen.generator_tasks, gen.all_generators_done, gen.failure)
            )
        return gen

    async def _run_epoch(self, epoch: int, pbar: tqdm) -> bool:
        """Train every mini-batch of one epoch. Returns True when training should stop early
        (``max_training_steps`` reached)."""
        self.epoch = epoch
        self._current_epoch = epoch
        self._fire("on_epoch_start")
        # 0. Per-epoch prologue.
        gen = self._start_epoch_generation()
        try:
            # Steps trained in THIS epoch (not global_step % num_steps_per_epoch: sample_full_batch can
            # end an epoch early, drifting global_step out of epoch alignment). On resume the dataloader
            # already reflects this epoch's trained steps. The range below is just an upper bound.
            trained_steps_this_epoch = self.async_train_dataloader.num_trained() // self.mini_batch_size
            for _step_idx in range(self.global_step, (1 + epoch) * self.num_steps_per_epoch + 1):
                outcome = await self._run_step(gen, epoch, trained_steps_this_epoch, pbar)
                if outcome is _StepOutcome.EPOCH_EXHAUSTED:
                    break
                if outcome is _StepOutcome.STOP_TRAINING:
                    self._fire("on_epoch_end")
                    return True
                trained_steps_this_epoch += 1

            # 9. Per-epoch epilogue.
            if self.cfg.trainer.update_ref_every_epoch and self.ref_model is not None:
                with Timer("update_ref_with_policy", self.all_timings):
                    await asyncio.to_thread(self.update_ref_with_policy)

            # Cancel generator tasks for this epoch
            await gen.stop()

            # Per-epoch reset/validation for data loading and admission control
            assert all(
                t.done() for t in gen.generator_tasks
            ), "Generator tasks must be done before resetting the dataloader manager and validating the admission controller."
            assert (
                gen.buffer.qsize() == 0
            ), f"We expect all generation output to be consumed by the training worker at end of an epoch, got {gen.buffer.qsize()}."
            await self.async_train_dataloader.reset_at_epoch_end()
            await self._admission_controller.validate_state_at_epoch_end(self.global_step)

            self._fire("on_epoch_end")
            return False
        except BaseException:
            await cancel_background_tasks(gen.background_tasks())
            raise

    async def _run_step(self, gen: _EpochGeneration, epoch: int, trained_steps_before: int, pbar: tqdm) -> _StepOutcome:
        """One training step: collect a mini-batch, train, sync weights, eval, checkpoint, log.

        ``trained_steps_before`` is the number of steps already trained in this epoch.
        """
        async with self._step_deadline(), Timer("step", self.all_timings):
            self._loop_gauges.set(
                "skyrl_gen_buffer_qsize",
                gen.buffer.qsize(),
                "Completed generation groups buffered at step start.",
            )

            # 1. Wait until we have a full mini-batch buffered (dropping zero-variance groups if
            # sample_full_batch).
            (
                cur_generation_group_mini_batch,
                cur_dropped_groups,
                epoch_exhausted,
            ) = await self._collect_generation_mini_batch(gen.buffer, gen.all_generators_done, gen.failure)

            if epoch_exhausted:
                await self._discard_partial_epoch_end_batch(cur_generation_group_mini_batch, epoch)
                return _StepOutcome.EPOCH_EXHAUSTED

            self._fire("on_step_start")

            if self.sample_full_batch:
                # The collect loop returns exactly mini_batch_size kept groups here.
                num_dropped = len(cur_dropped_groups)
                keep_rate = self.mini_batch_size / (self.mini_batch_size + num_dropped)
                self.all_metrics["async/num_groups_dropped"] = num_dropped
                self.all_metrics["async/keep_rate"] = keep_rate
                self._loop_gauges.set(
                    "skyrl_gen_group_keep_rate",
                    keep_rate,
                    "Fraction of drained groups kept while collecting the last mini-batch.",
                )

            # 2. Post-process the generated groups, aggregating to a single GeneratorOutput, and convert to training format.
            with self._phase_gauge.timed_phase("convert_to_training_input", self.all_timings):
                training_input = await asyncio.to_thread(
                    self.convert_generation_group_mini_batch_to_training_input,
                    cur_generation_group_mini_batch,
                    cur_dropped_groups,
                )

            # 3. Run training and update consumed UIDs.
            await self._before_training_step()
            with self._phase_gauge.timed_phase("run_training", self.all_timings):
                status = await self._run_training(training_input)
                await self.async_train_dataloader.mark_consumed_uids([g.uid for g in cur_generation_group_mini_batch])
            self._fire("on_step_end", batch=training_input, metrics=status)

            # Capture callback-driven triggers, then reset.
            force_save = self._training_control.should_save
            force_eval = self._training_control.should_evaluate
            self._training_control.should_save = False
            self._training_control.should_evaluate = False

            # 4. After training: pause generation, sync weights, resume.
            await self._sync_weights_after_training_step()

        # A training step completed: count it for this epoch's bookkeeping.
        trained_steps_this_epoch = trained_steps_before + 1

        # One profiler step per async global step.
        self._profiler_step()

        # 5. Set logs for this training step.
        logger.info(status)
        self.all_metrics.update({"trainer/epoch": epoch, "trainer/global_step": self.global_step})
        pbar.update(1)

        # 6. Eval. At interval and at the last step.
        # NOTE(Charlie): eval does not overlap with training, but overlaps with generation.
        interval_eval = self.cfg.trainer.eval_interval > 0 and (
            self.global_step % self.cfg.trainer.eval_interval == 0 or self.global_step == self.total_training_steps
        )
        if force_eval or interval_eval:
            self._fire("on_eval_start")
            with self._phase_gauge.timed_phase("eval", self.all_timings):
                eval_metrics = await self.eval()
                self.all_metrics.update(eval_metrics)
            self._fire("on_eval_end", metrics=eval_metrics)

        # 7. Checkpointing. At interval, at the last step of each epoch, and when a callback asks.
        is_epoch_end = trained_steps_this_epoch == self.num_steps_per_epoch
        ckpt_interval_save = self.cfg.trainer.ckpt_interval > 0 and (
            is_epoch_end or self.global_step % self.cfg.trainer.ckpt_interval == 0
        )
        if force_save or ckpt_interval_save:
            async with self._step_deadline():
                with self._phase_gauge.timed_phase("save_checkpoints", self.all_timings):
                    ckpt_path = await asyncio.to_thread(self.save_checkpoints)
            self._fire("on_save", ckpt_path=ckpt_path)
        if self.cfg.trainer.hf_save_interval > 0:
            if is_epoch_end or self.global_step % self.cfg.trainer.hf_save_interval == 0:
                async with self._step_deadline():
                    with self._phase_gauge.timed_phase("save_hf_model", self.all_timings):
                        await asyncio.to_thread(self.save_models)

        log_payload = {**self.all_metrics, **{"timing/" + k: v for k, v in self.all_timings.items()}}
        if self._ray_gpu_monitor is not None:
            log_payload.update(self._ray_gpu_monitor.flush())
        if self._vllm_metrics_scraper is not None:
            log_payload.update(await self._vllm_metrics_scraper.sample())
        self._fire("on_log", logs=log_payload)
        self.tracker.log(log_payload, step=self.global_step, commit=True)
        self.all_metrics = {}
        self.all_timings = {}
        self.global_step += 1

        if self.cfg.trainer.max_training_steps is not None and self.global_step > self.cfg.trainer.max_training_steps:
            logger.info(f"Reached max_training_steps={self.cfg.trainer.max_training_steps}, stopping early.")
            await gen.stop()
            return _StepOutcome.STOP_TRAINING

        # 8. Notify generation workers that the capacity has increased, unblocking them.
        await self._admission_controller.notify_capacity_change(self.global_step)
        # Only trained UIDs map to completed steps; filtered/dropped UIDs are extra consumption.
        expected_trained_in_epoch = self.mini_batch_size * trained_steps_this_epoch
        actual_trained_in_epoch = self.async_train_dataloader.num_trained()
        assert actual_trained_in_epoch == expected_trained_in_epoch, (
            "Unexpected number of trained (consumed minus filtered) data UIDs. Got: "
            f"{actual_trained_in_epoch} != {expected_trained_in_epoch}"
        )
        return _StepOutcome.CONTINUE

    async def _discard_partial_epoch_end_batch(self, partial_batch: List[GeneratedOutputGroup], epoch: int) -> None:
        """Exhausted mid mini-batch under ``sample_full_batch``: discard the partial batch (marked consumed
        so it isn't regenerated on resume), and save the end-of-epoch checkpoint the normal
        is_epoch_end path would have, since the step loop breaks before reaching it."""
        if partial_batch:
            for _ in partial_batch:
                await self._admission_controller.on_rollout_filtered()
            await self.async_train_dataloader.mark_filtered_uids([g.uid for g in partial_batch])
            logger.warning(
                f"sample_full_batch: epoch {epoch} exhausted with a partial mini-batch of "
                f"{len(partial_batch)} group(s); discarding and ending the epoch."
            )
        if self.cfg.trainer.ckpt_interval > 0:
            with self._phase_gauge.timed_phase("save_checkpoints", self.all_timings):
                ckpt_path = await asyncio.to_thread(self.save_checkpoints)
            self._fire("on_save", ckpt_path=ckpt_path)
        if self.cfg.trainer.hf_save_interval > 0:
            with self._phase_gauge.timed_phase("save_hf_model", self.all_timings):
                await asyncio.to_thread(self.save_models)

    @staticmethod
    async def _watch_generators_done(
        tasks: List[asyncio.Task], all_generators_done: asyncio.Event, failure: BackgroundFailure
    ) -> None:
        """Set ``all_generators_done`` once every worker exits, unless one failed: consumers are woken by
        the failure instead, so it can't read as epoch exhaustion."""
        await asyncio.gather(*tasks, return_exceptions=True)
        if not failure.failed:
            all_generators_done.set()

    async def _drain_next_group(
        self, buffer: asyncio.Queue, all_generators_done: asyncio.Event, failure: BackgroundFailure
    ) -> Optional[GeneratedOutputGroup]:
        """Return the next generated group, or None if generation is exhausted (all workers finished
        and the buffer is empty). Raises the recorded worker failure, if any, instead of blocking.

        Only used under ``sample_full_batch``, where dropping groups can exhaust the epoch mid
        mini-batch and a plain blocking ``buffer.get()`` would hang forever.
        """
        while True:
            failure.raise_if_failed()
            if not buffer.empty():
                return buffer.get_nowait()
            if all_generators_done.is_set():
                return None
            get_task = asyncio.ensure_future(buffer.get())
            done_task = asyncio.ensure_future(all_generators_done.wait())
            try:
                done, pending = await failure.guard(
                    asyncio.wait({get_task, done_task}, return_when=asyncio.FIRST_COMPLETED)
                )
            except BaseException:
                get_task.cancel()
                done_task.cancel()
                raise
            if get_task in done:
                for t in pending:
                    t.cancel()
                return get_task.result()
            # all_generators_done fired first. Cancel the pending get and loop to re-check the buffer.
            # If the get had already pulled an item (racing the cancel), return it rather than drop it
            # (a successful get() pops from the queue). No put can actually race here since the event is
            # only set after all producers stop, but this stays correct regardless.
            get_task.cancel()
            try:
                return await get_task
            except asyncio.CancelledError:
                pass

    def _should_keep_group(self, group: GeneratedOutputGroup) -> bool:
        """Whether a group has reward variance (train on it) vs. is zero-variance (drop it).

        Token-level rewards (``List[List[float]]``) are collapsed to a per-trajectory sum, like the rest
        of the trainer. Groups with <=1 live trajectory (singletons / mostly masked) are always kept.
        """
        rewards = group.generator_output["rewards"]
        if rewards and isinstance(rewards[0], list):
            # Token-level rewards: collapse each trajectory to a scalar sequence reward.
            seq_rewards = [float(sum(r)) for r in rewards]
        else:
            seq_rewards = rewards
        kept_indices = zero_variance_filter(
            seq_rewards,
            [group.uid] * len(seq_rewards),
            loss_masks=group.generator_output["loss_masks"],
            tol=self.cfg.trainer.algorithm.zero_variance_filter_tol,
        )
        return len(kept_indices) > 0

    async def _collect_generation_mini_batch(
        self,
        generation_output_group_buffer: asyncio.Queue,
        all_generators_done: asyncio.Event,
        failure: BackgroundFailure,
    ) -> Tuple[List[GeneratedOutputGroup], List[GeneratedOutputGroup], bool]:
        """Pull a full mini-batch of generated groups from the buffer.

        Without ``sample_full_batch``, blocks until ``mini_batch_size`` groups are available. With it,
        drops zero-variance groups (freeing their capacity, marking UIDs consumed) and keeps pulling
        until ``mini_batch_size`` non-zero-variance groups are collected, which can exhaust the epoch.

        Returns ``(kept_groups, dropped_groups, epoch_exhausted)``. On exhaustion the kept groups are a
        (possibly empty) partial batch to discard; dropped groups are kept for metrics only.
        """
        kept_groups: List[GeneratedOutputGroup] = []
        dropped_groups: List[GeneratedOutputGroup] = []
        epoch_exhausted = False
        # Buffer occupancy when this step starts waiting.
        self.all_metrics["async/gen_buffer_qsize_at_wait_start"] = generation_output_group_buffer.qsize()
        with self._phase_gauge.timed_phase("wait_for_generation_buffer", self.all_timings):
            buffer_pbar = tqdm(total=self.mini_batch_size, initial=0, desc="Generation Buffer Progress", position=1)
            while len(kept_groups) < self.mini_batch_size:
                # We do finish-time FIFO here (not schedule-time FIFO).
                if not self.sample_full_batch:
                    kept_groups.append(await failure.guard(generation_output_group_buffer.get()))
                    buffer_pbar.update(1)
                    buffer_pbar.set_postfix({"buffer qsize": generation_output_group_buffer.qsize()})
                    continue

                group = await self._drain_next_group(generation_output_group_buffer, all_generators_done, failure)
                if group is None:
                    epoch_exhausted = True
                    break
                try:
                    if self._should_keep_group(group):
                        kept_groups.append(group)
                        buffer_pbar.update(1)
                        buffer_pbar.set_postfix({"buffer qsize": generation_output_group_buffer.qsize()})
                    else:
                        # Drop the zero-variance group: give its capacity back to the producers and mark
                        # its UID consumed (skipped, not regenerated, on resume).
                        dropped_groups.append(group)
                        await self._admission_controller.on_rollout_filtered()
                        await self.async_train_dataloader.mark_filtered_uids([group.uid])
                except Exception:
                    self._log_group_processing_error(group, len(kept_groups), len(dropped_groups))
                    raise
            buffer_pbar.close()
        return kept_groups, dropped_groups, epoch_exhausted

    @staticmethod
    def _log_group_processing_error(group: GeneratedOutputGroup, kept_so_far: int, dropped_so_far: int) -> None:
        """Log the offending group's reward / loss-mask shape before a drain-loop error propagates,
        flushing stderr so the log isn't lost if the process dies during teardown."""
        go = group.generator_output
        rewards = go.get("rewards") if isinstance(go, dict) else None
        loss_masks = go.get("loss_masks") if isinstance(go, dict) else None
        logger.exception(
            "sample_full_batch: error processing generated group "
            f"(uid={group.uid}, kept_so_far={kept_so_far}, dropped_so_far={dropped_so_far}). "
            f"rewards type={type(rewards).__name__} "
            f"len={len(rewards) if hasattr(rewards, '__len__') else 'n/a'} "
            f"value={rewards!r}; "
            f"loss_masks type={type(loss_masks).__name__} "
            f"len={len(loss_masks) if hasattr(loss_masks, '__len__') else 'n/a'}"
        )
        sys.stderr.flush()

    async def _run_training(self, training_input: TrainingInputBatch):
        # TODO(Charlie): share this code with the one-step-off async trainer.
        # inference and calculate values, log probs, rewards, kl divergence
        with Timer("fwd_logprobs_values_reward", self.all_timings):
            training_input = await asyncio.to_thread(self.fwd_logprobs_values_reward, training_input)

        # calculate kl divergence and create experiences
        if self.cfg.trainer.algorithm.use_kl_in_reward:
            with Timer("apply_reward_kl_penalty", self.all_timings):
                training_input = self.apply_reward_kl_penalty(training_input)

        # calculate advantages and returns / along with tensorboard logging
        with Timer("compute_advantages_and_returns", self.all_timings):
            training_input = self.compute_advantages_and_returns(training_input)
            # remove some unwanted keys
            for key in ["rewards"]:
                training_input.pop(key)
            training_input.metadata.pop("uids")

        if self.cfg.trainer.dump_data_batch:
            # dump data to file
            with Timer("dump_data_batch"):
                self.dump_data(training_input, file_name=f"global_step_{self.global_step}_training_input")

        # train policy/critic model
        with Timer("train_critic_and_policy", self.all_timings):
            status = await asyncio.to_thread(self.train_critic_and_policy, training_input)
        self.all_metrics.update(await asyncio.to_thread(self.policy_mfu_metrics, training_input))

        return status

    async def _run_generate_for_a_group_loop(
        self, generation_output_group_buffer: asyncio.Queue, failure: BackgroundFailure
    ):
        """
        Generator worker: repeatedly pulls the next prompt (possibly blocked by admission control),
        generates one single generation group, respecting a pause/resume event, and enqueues the result.
        An exception is recorded in ``failure`` (waking the consumer) and re-raised.
        """
        try:
            while True:
                # 0. Pull next batch from dataloader. If returns None, then dataloader is exhausted.
                rand_prompts = await self.async_train_dataloader.get_next_non_consumed_data()
                if rand_prompts is None:
                    return

                # 1. Prepare generator input
                assert len(rand_prompts) == 1
                generator_input, uids = prepare_generator_input(
                    rand_prompts,
                    self.cfg.generator.n_samples_per_prompt,
                    get_sampling_params_for_backend(
                        self.cfg.generator.inference_engine.backend, self.cfg.generator.sampling_params
                    ),
                    self.cfg.environment.env_class,
                    "train",
                    self.global_step,
                )
                assert all(uid == uids[0] for uid in uids), "Expect all uids to be the same"

                # 2. Acquire capacity slot.
                slot_acquired = False
                await self._admission_controller.acquire_submission_slot()
                slot_acquired = True

                # 3. Generate one rollout group
                global_step_at_start = self.global_step  # for staleness control

                group_start_time = time.monotonic()
                if "disable_tqdm" in inspect.signature(self.generator.generate).parameters:
                    # A workaround to disable tqdm for the SkyRLGymGenerator.generate method which will
                    # blast the console with each worker's progress bar.
                    cur_generator_output: GeneratorOutput = await self.generator.generate(
                        generator_input, disable_tqdm=True
                    )
                else:
                    cur_generator_output: GeneratorOutput = await self.generator.generate(generator_input)
                group_completion_time_s = time.monotonic() - group_start_time

                # 4. Enqueue the completed group and mark accepted to free capacity slot.
                try:
                    generation_output_group_buffer.put_nowait(
                        GeneratedOutputGroup(
                            generator_output=cur_generator_output,
                            uid=uids[0],
                            global_step_when_scheduled=global_step_at_start,
                            group_completion_time_s=group_completion_time_s,
                            prompts=generator_input["prompts"],
                        )
                    )
                except asyncio.QueueFull:
                    raise AssertionError("Generation buffer should never be full given admission control.")
                await self._admission_controller.on_rollout_accepted()
                slot_acquired = False
        except asyncio.CancelledError:
            # Expected on epoch end / shutdown: release any held slot so admission accounting stays
            # consistent, then exit cleanly.
            if "slot_acquired" in locals() and slot_acquired:
                try:
                    await self._admission_controller.on_rollout_rejected()
                except Exception:
                    pass
            return
        except Exception as e:
            # Logged here too: the consumer only raises it at its next buffer wait.
            logger.error(f"Generator worker errored out with exception: {e!r}")
            failure.record(e, "generation worker")
            raise

    @staticmethod
    def _reprefix_metrics(metrics: dict, suffix: str) -> dict:
        """Re-prefix metric keys into a separate view (``generate/X`` -> ``generate_<suffix>/X``),
        keeping the leading namespace so views group together in trackers."""
        out = {}
        for k, v in metrics.items():
            if "/" in k:
                prefix, rest = k.split("/", 1)
                out[f"{prefix}_{suffix}/{rest}"] = v
            else:
                out[f"{suffix}/{k}"] = v
        return out

    def convert_generation_group_mini_batch_to_training_input(
        self,
        cur_generation_group_mini_batch: List[GeneratedOutputGroup],
        dropped_groups: Optional[List[GeneratedOutputGroup]] = None,
    ) -> TrainingInputBatch:
        """Concatenate the mini-batch of generated groups and convert to a TrainingInputBatch.

        ``dropped_groups`` (zero-variance groups dropped this step under ``sample_full_batch``) are not
        trained on, but are folded into the reward/generation metrics to keep them comparable across runs.
        """
        dropped_groups = dropped_groups or []
        generator_outputs = []
        # Prompts for the kept groups, kept parallel to the concatenated `generator_output` so the
        # trajectory logger can render prompt + response (mirrors the sync trainer's use of
        # `generator_input["prompts"]`).
        prompts: List[Any] = []
        uids = []
        stalenesses = []
        staleness_violation_count = 0
        # Per-group completion times (seconds) and the intra-group spread (std and coefficient of
        # variation) of per-trajectory completion times. Helpful to understand tail latency
        group_completion_times: List[float] = []
        intra_group_stds: List[float] = []
        intra_group_cvs: List[float] = []
        for cur_generated_output_group in cur_generation_group_mini_batch:
            cur_staleness = self.global_step - cur_generated_output_group.global_step_when_scheduled
            stalenesses.append(cur_staleness)
            generator_outputs.append(cur_generated_output_group.generator_output)
            if cur_generated_output_group.prompts is not None:
                prompts.extend(cur_generated_output_group.prompts)

            # Collect per-group / per-trajectory completion-time stats for this group.
            if cur_generated_output_group.group_completion_time_s is not None:
                group_completion_times.append(cur_generated_output_group.group_completion_time_s)
            intra_group_std, intra_group_cv = get_intra_group_completion_time_std_cv(
                cur_generated_output_group.generator_output
            )
            if intra_group_std is not None:
                intra_group_stds.append(intra_group_std)
            if intra_group_cv is not None:
                intra_group_cvs.append(intra_group_cv)

            # NOTE(Charlie): for step-wise training each group can contain a variable number of entries
            # (n_samples_per_prompt * variable turns_per_trajectory), so the uid fanout is per-group.
            group_size = len(cur_generated_output_group.generator_output["response_ids"])
            uids.extend([cur_generated_output_group.uid] * group_size)

            # Check staleness violation.
            if cur_staleness > self.max_staleness_steps:
                # TODO(Charlie): should we drop, drop and resample, or just log?
                logger.warning(
                    "Staleness control violated despite using AsyncStalenessManager: "
                    f"cur_staleness={cur_staleness}, max_staleness_steps={self.max_staleness_steps}.\n"
                    "If this happens too often, consider increasing max_staleness_steps, adjusting "
                    "trainer.fully_async.num_parallel_generation_workers, or adjusting generation-training GPU allocation.\n"
                    "See https://docs.skyrl.ai/docs/tutorials/fully_async#async-staleness-manager for more details."
                )
                staleness_violation_count += 1

        generator_output = concatenate_generator_outputs(
            generator_outputs,
            step_wise=self.cfg.generator.step_wise_trajectories,
            routes_expected=self.cfg.generator.inference_engine.enable_return_routed_experts,
            support_expected=self.cfg.trainer.algorithm.enable_sample_support_replay,
            support_scores_expected=self.cfg.trainer.algorithm.enable_score_centering,
        )
        kept_rollout_metrics = generator_output["rollout_metrics"]
        assert kept_rollout_metrics is not None, "Rollout metrics should be non-null."

        # Kept + dropped view for the headline reward/generation metrics, so they stay comparable
        # across runs regardless of filtering. Dropped groups are used for metrics only, never trained.
        metrics_generator_output = None
        metrics_uids = None
        if dropped_groups:
            metrics_generator_output = concatenate_generator_outputs(
                generator_outputs + [g.generator_output for g in dropped_groups],
                step_wise=self.cfg.generator.step_wise_trajectories,
            )
            metrics_uids = []
            for g in list(cur_generation_group_mini_batch) + dropped_groups:
                metrics_uids.extend([g.uid] * len(g.generator_output["response_ids"]))

        # Headline generation (rollout) metrics: over kept + dropped when sample_full_batch dropped groups.
        rollout_metrics_source = metrics_generator_output if metrics_generator_output is not None else generator_output
        self.all_metrics.update(rollout_metrics_source["rollout_metrics"])

        group_completion_metrics = get_group_completion_metrics(
            group_completion_times, intra_group_stds, intra_group_cvs
        )

        if dropped_groups:
            # Also split generation metrics by kept vs dropped, plus reward metrics for the dropped
            # groups. (Kept reward metrics are uninformative -- pass@n ~1 -- and loss/avg_final_rewards
            # already covers the trained set.)
            dropped_generator_output = concatenate_generator_outputs(
                [g.generator_output for g in dropped_groups], step_wise=self.cfg.generator.step_wise_trajectories
            )
            self.all_metrics.update(self._reprefix_metrics(kept_rollout_metrics, "kept"))
            self.all_metrics.update(self._reprefix_metrics(dropped_generator_output["rollout_metrics"], "dropped"))
            dropped_uids = [g.uid for g in dropped_groups for _ in range(len(g.generator_output["response_ids"]))]
            dropped_reward = get_metrics_from_generator_output(dropped_generator_output, dropped_uids)

            n_samples_per_prompt = self.cfg.generator.n_samples_per_prompt
            self.all_metrics.update(
                {
                    f"reward_dropped/avg_pass_at_{n_samples_per_prompt}": dropped_reward["pass_at_n"],
                    "reward_dropped/avg_raw_reward": dropped_reward["avg_score"],
                    "reward_dropped/mean_positive_reward": dropped_reward["mean_positive_reward"],
                }
            )
            dropped_generator_output.pop("rollout_metrics", None)

            # Per-group completion times and intra-group spread for dropped groups
            dropped_group_completion_times = []
            dropped_group_intra_group_stds = []
            dropped_group_intra_group_cvs = []
            for dropped_group in dropped_groups:
                if dropped_group.group_completion_time_s is not None:
                    dropped_group_completion_times.append(dropped_group.group_completion_time_s)
                intra_group_std, intra_group_cv = get_intra_group_completion_time_std_cv(dropped_group.generator_output)
                if intra_group_std is not None:
                    dropped_group_intra_group_stds.append(intra_group_std)
                if intra_group_cv is not None:
                    dropped_group_intra_group_cvs.append(intra_group_cv)

            dropped_metrics = get_group_completion_metrics(
                dropped_group_completion_times, dropped_group_intra_group_stds, dropped_group_intra_group_cvs
            )

            self.all_metrics.update(self._reprefix_metrics(group_completion_metrics, "kept"))
            self.all_metrics.update(self._reprefix_metrics(dropped_metrics, "dropped"))

            merged_group_completion_times = group_completion_times + dropped_group_completion_times
            merged_intra_group_stds = intra_group_stds + dropped_group_intra_group_stds
            merged_intra_group_cvs = intra_group_cvs + dropped_group_intra_group_cvs
            # get merged metrics with kept + dropped groups
            group_completion_metrics = get_group_completion_metrics(
                merged_group_completion_times, merged_intra_group_stds, merged_intra_group_cvs
            )

        self.all_metrics.update(group_completion_metrics)

        generator_output.pop("rollout_metrics", None)
        if metrics_generator_output is not None:
            metrics_generator_output.pop("rollout_metrics", None)

        # Log staleness statistics for this step
        self.all_metrics.update(
            {
                "async/staleness_mean": sum(stalenesses) / len(stalenesses),
                "async/staleness_max": max(stalenesses),
                "async/staleness_min": min(stalenesses),
                "async/staleness_ratio": sum(1 for s in stalenesses if s > 0) / len(stalenesses),
                "async/staleness_violation_count": staleness_violation_count,
            }
        )

        # Per-token reward conversion (kept groups only) + reward metrics over the kept+dropped view.
        generator_output, uids = self.postprocess_generator_output(
            generator_output, uids, metrics_generator_output=metrics_generator_output, metrics_uids=metrics_uids
        )

        # print example just for debugging
        vis = self.tokenizer.decode(generator_output["response_ids"][0])
        logger.info(f"Example generated: {vis}")

        # Optionally upload up to `num_logger_train_samples` samples to tracker (mirrors the
        # synchronous trainer's train trajectory logging).
        if self.trajectory_logger is not None:
            with Timer("log_train_results"):
                self.trajectory_logger.log(
                    tracker=self.tracker,
                    num_samples=self.cfg.trainer.num_logger_train_samples,
                    prompts=prompts,
                    generator_output=generator_output,
                    tokenizer=self.tokenizer,
                    global_step=self.global_step,
                    wandb_key="trajectories/train",
                    include_idx=False,
                )

        return self.convert_to_training_input(generator_output, uids)

    def save_checkpoints(self) -> str:
        """
        Extend base checkpointing by recording consumed UIDs for fully-async training.

        Otherwise, when resuming, there is no way to know which data has been trained on.
        Returns the checkpoint folder path (forwarded from the base implementation).
        """
        consumed_uids_list = (
            self.async_train_dataloader.get_consumed_uids_list()
        )  # read first to prevent race condition
        filtered_uids_list = self.async_train_dataloader.get_filtered_uids_list()
        # The base method will save the model, dataloader path, trainer_state, and latest_ckpt_global_step.txt.
        global_step_folder = super().save_checkpoints()
        # Also save the consumed UIDs (do-not-redraw this epoch), the filtered subset (so dropped
        # prompts are skipped, not regenerated, on resume), and the epoch (not derivable from
        # global_step under sample_full_batch, where an epoch can end early).
        fully_async_state_path = os.path.join(global_step_folder, "fully_async_state.pt")
        fully_async_state = {
            "consumed_uids": consumed_uids_list,
            "filtered_uids": filtered_uids_list,
            "epoch": self.epoch,
        }
        with io.open_file(fully_async_state_path, "wb") as f:
            torch.save(fully_async_state, f)
        logger.info(f"Saved fully-async state to {fully_async_state_path}")
        return global_step_folder

    def load_checkpoints(self) -> Tuple[int, str, Optional[Set[str]], Optional[Set[str]], Optional[int]]:
        """
        Load the base checkpoint without loading the dataloader state, and load the fully-async state.

        Returns the global step to resume from, the checkpoint path, the consumed data UIDs, the
        filtered (dropped, not trained on) data UIDs, and the epoch to resume in (None if the
        checkpoint predates epoch tracking, in which case the caller derives it from global_step).
        """
        global_step, checkpoint_path = super().load_checkpoints()
        if global_step == 0:
            return 0, checkpoint_path, None, None, None
        fully_async_state_path = os.path.join(checkpoint_path, "fully_async_state.pt")
        assert io.exists(fully_async_state_path), f"Fully-async state file not found at {fully_async_state_path}"
        with io.open_file(fully_async_state_path, "rb") as f:
            fully_async_state = torch.load(f, map_location="cpu", weights_only=False)
            assert "consumed_uids" in fully_async_state, "consumed_uids key not found in fully-async state"
            consumed_data_uids_set = set(fully_async_state["consumed_uids"])
            # Backward compatible with checkpoints saved before filtered UIDs / epoch were tracked.
            filtered_data_uids_set = set(fully_async_state.get("filtered_uids", []))
            epoch = fully_async_state.get("epoch", None)
        logger.info(
            f"Loaded fully-async state with {len(consumed_data_uids_set)} consumed UIDs "
            f"({len(filtered_data_uids_set)} filtered), epoch={epoch}"
        )
        return global_step, checkpoint_path, consumed_data_uids_set, filtered_data_uids_set, epoch

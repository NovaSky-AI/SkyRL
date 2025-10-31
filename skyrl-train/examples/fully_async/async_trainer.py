import asyncio
import traceback
import sys
from loguru import logger
from skyrl_train.trainer import RayPPOTrainer
from tqdm import tqdm
from skyrl_train.utils import Timer
from skyrl_train.utils.ppo_utils import normalize_advantages_dict
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.generators.base import GeneratorOutput
from skyrl_train.utils.trainer_utils import ResumeMode
from skyrl_train.generators.utils import prepare_generator_input
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend

# Some configs
NUM_PARALLEL_GROUPS_OF_ROLLOUTS = 16  # we generate NUM_PARALLEL_GROUPS_OF_ROLLOUTS * n_samples_per_prompt individual rollouts in parallel


class FullyAsyncRayPPOTrainer(RayPPOTrainer):

    async def train(self):
        """
        Main training loop for PPO
        """
        assert not self.colocate_all, "colocate_all is not supported for async training"

        MINI_BATCH_SIZE = self.cfg.trainer.policy_mini_batch_size  # we kick off training whenever we have MINI_BATCH_SIZE groups of rollouts are in the generation buffer

        self.global_step = 0

        # Load checkpoint state if resumption is enabled
        if self.resume_mode != ResumeMode.NONE:
            with Timer("load_checkpoints"):
                self.load_checkpoints()
                logger.info(f"Resumed training from global_step {self.global_step}")

        # Initialize weight sync state
        with Timer("init_weight_sync_state"):
            self.init_weight_sync_state()

        # sync weights to inference engines
        with Timer("sync_weights_to_inference_engines"):
            await self.async_sync_policy_weights_to_inference_engines()

        # Eval before training
        if self.cfg.trainer.eval_interval > 0 and self.cfg.trainer.eval_before_train:
            with Timer("eval", self.all_timings):
                eval_metrics = await self.eval()
                self.tracker.log(eval_metrics, step=self.global_step)

        # main training loop
        pbar = tqdm(total=self.total_training_steps, initial=self.global_step, desc="Training Step Progress")
        # Start from step 1
        self.global_step += 1
        for epoch in range(self.cfg.trainer.epochs):
            
            # Shared queue of completed rollout groups
            generation_buffer_grouped = asyncio.Queue()

            # Per-epoch controls/state for generator workers
            self._dataloader_iter = iter(self.train_dataloader)
            self._dataloader_iter_lock = asyncio.Lock()

            # Maintain NUM_PARALLEL_GROUPS_OF_ROLLOUTS concurrent group-generation workers
            generator_tasks = [
                asyncio.create_task(self._run_generate_for_a_group_loop(generation_buffer_grouped))
                for _ in range(NUM_PARALLEL_GROUPS_OF_ROLLOUTS)
            ]

            # Training loop: wait until MINI_BATCH_SIZE groups are ready, then train
            for idx in range(len(self.train_dataloader)):
                with Timer("step", self.all_timings):
                    # Wait until we have enough groups buffered
                    while generation_buffer_grouped.qsize() < MINI_BATCH_SIZE:
                        logger.info(f"Current size of generation buffer: {generation_buffer_grouped.qsize()}")
                        await asyncio.sleep(1)

                    status = await self._run_training(generation_buffer_grouped)

                    # After training: pause generation, sync weights, then resume
                    await self.pause_generate()
                    async with Timer("sync_weights", self.all_timings):
                        await self.async_sync_policy_weights_to_inference_engines()
                    await self.continue_generate()

                # 5. set logs
                logger.info(status)
                # log epoch info
                self.all_metrics.update({"trainer/epoch": epoch, "trainer/global_step": self.global_step})
                self.tracker.log(self.all_metrics, step=self.global_step)
                self.all_metrics = {}
                pbar.update(1)

                if self.cfg.trainer.eval_interval > 0 and (
                    self.global_step % self.cfg.trainer.eval_interval == 0
                    or self.global_step == self.total_training_steps
                ):
                    with Timer("eval", self.all_timings):
                        eval_metrics = await self.eval()
                        self.all_metrics.update(eval_metrics)
                if self.cfg.trainer.ckpt_interval > 0 and self.global_step % self.cfg.trainer.ckpt_interval == 0:
                    with Timer("save_checkpoints", self.all_timings):
                        self.save_checkpoints()
                if self.cfg.trainer.hf_save_interval > 0 and self.global_step % self.cfg.trainer.hf_save_interval == 0:
                    with Timer("save_hf_model", self.all_timings):
                        self.save_models()
                self.tracker.log({"timing/" + k: v for k, v in self.all_timings.items()}, step=self.global_step)
                self.all_timings = {}
                self.global_step += 1

            if self.cfg.trainer.update_ref_every_epoch and self.ref_model is not None:
                with Timer("update_ref_with_policy", self.all_timings):
                    await asyncio.to_thread(self.update_ref_with_policy)

            # cancel generator tasks for this epoch
            for t in generator_tasks:
                t.cancel()
            try:
                await asyncio.gather(*generator_tasks, return_exceptions=True)
            except Exception:
                pass

        pbar.close()
        if self.cfg.trainer.ckpt_interval > 0:
            with Timer("save_checkpoints", self.all_timings):
                self.save_checkpoints()
                logger.info("Saved final checkpoint.")
        if self.cfg.trainer.hf_save_interval > 0:
            with Timer("save_hf_model", self.all_timings):
                self.save_models()
                logger.info("Saved final model.")
        logger.info("Training done!")

    async def _run_training(self, generation_buffer_grouped):
        # Get a generation future and await on the object
        generator_output, uids = await generation_buffer_grouped.get()  # GeneratorOutput, List[str]

        # print example just for debugging
        vis = self.tokenizer.decode(generator_output["response_ids"][0])
        print("example: ", vis)

        with Timer("convert_to_training_input", self.all_timings):
            training_input: TrainingInputBatch = self.convert_to_training_input(generator_output, uids)

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

            if self.cfg.trainer.algorithm.advantage_batch_normalize:
                training_input = normalize_advantages_dict(training_input)

        if self.cfg.trainer.dump_data_batch:
            # dump data to file
            with Timer("dump_data_batch"):
                self.dump_data(training_input, file_name=f"global_step_{self.global_step}_training_input")

        # train policy/critic model
        with Timer("train_critic_and_policy", self.all_timings):
            status = await asyncio.to_thread(self.train_critic_and_policy, training_input)

        return status

    async def _run_generate_for_a_group_loop(self, generation_buffer_grouped: asyncio.Queue):
        """
        Generator worker: repeatedly pulls the next prompts batch and generates one
        rollout group, respecting a pause/resume event, and enqueues the result.
        """
        try:
            while True:
                # Pull next batch from dataloader (shared among workers)
                async with self._dataloader_iter_lock:
                    try:
                        rand_prompts = next(self._dataloader_iter)
                    except StopIteration:
                        return

                # Truncate data to even shards and prepare inputs
                rand_prompts = self._remove_tail_data(rand_prompts)
                generator_input, uids = prepare_generator_input(
                    rand_prompts,
                    self.cfg.generator.n_samples_per_prompt,
                    get_sampling_params_for_backend(self.cfg.generator.backend, self.cfg.generator.sampling_params),
                    self.cfg.environment.env_class,
                    "train",
                    self.global_step,
                )

                # Generate one rollout group
                async with Timer("generate", self.all_timings):
                    generator_output: GeneratorOutput = await self.generate(generator_input)
                    generator_output = self.postprocess_generator_output(generator_output, uids)

                # Enqueue the completed group
                await generation_buffer_grouped.put((generator_output, uids))
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.error(f"Generator worker errored out with exception: {e}")
            logger.error(f"Traceback: \n{traceback.format_exc()}")
            sys.exit(1)

    async def async_sync_policy_weights_to_inference_engines(self):
        return await self.policy_model.async_run_method(
            "pass_through", "broadcast_to_inference_engines", self.inference_engine_client
        )

    async def pause_generate(self):
        await self.inference_engine_client.pause_generation()

    async def continue_generate(self):
        await self.inference_engine_client.resume_generation()

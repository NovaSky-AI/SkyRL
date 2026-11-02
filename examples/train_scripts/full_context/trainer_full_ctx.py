import random
import time
from collections import defaultdict
from typing import Dict, List

import ray
from loguru import logger

from skyrl.train.trainer import RayPPOTrainer
from skyrl.backends.skyrl_train.workers.worker_utils import reduce_metrics
from skyrl.train.utils.utils import Timer


def _cuda_peak_and_reset(worker) -> Dict[str, int]:
    import torch

    stats = {
        "max_allocated": torch.cuda.max_memory_allocated(),
        "max_reserved": torch.cuda.max_memory_reserved(),
    }
    torch.cuda.reset_peak_memory_stats()
    return stats


class FullCtxTrainer(RayPPOTrainer):
    """A dummy trainer that tests configurations with max sequence length.

    This trainer is meant to help users validate their configuration setup by:
    1. Creating max length sequences directly
    2. Running a few training steps

    This helps catch OOM issues early before running full training.

    With ``trainer.dummy_length_distribution=mixture`` it doubles as a throughput benchmark:
    prompt and response lengths are drawn per sample (a ``dummy_truncated_fraction`` of the
    responses at the max length, the rest uniform in ``[dummy_min_response_length, max]``) from a
    seed that depends only on the step, so every configuration sees the same batches. Token ids
    come from ``dummy_token_source`` -- ``dataset`` takes windows of tokenized training prompts,
    which keeps MoE routing close to real text (a single repeated token, the default, sends every
    token to the same experts).
    """

    def _dummy_lengths(self, rng: random.Random, num_samples: int) -> List[tuple]:
        cfg = self.cfg.trainer
        max_prompt = self.cfg.generator.max_input_length
        max_response = self.cfg.generator.sampling_params.max_generate_length
        if cfg.dummy_length_distribution == "max":
            return [(max_prompt, max_response)] * num_samples
        assert cfg.dummy_length_distribution == "mixture", cfg.dummy_length_distribution
        lengths = []
        for _ in range(num_samples):
            prompt = rng.randint(cfg.dummy_min_prompt_length, min(cfg.dummy_max_prompt_length, max_prompt))
            if rng.random() < cfg.dummy_truncated_fraction:
                response = max_response
            else:
                response = rng.randint(cfg.dummy_min_response_length, max_response)
            lengths.append((prompt, response))
        return lengths

    def _token_pool(self) -> List[int]:
        if getattr(self, "_dummy_token_pool", None) is None:
            pool: List[int] = []
            dataset = self.train_dataset.dataframe
            prompt_key = self.train_dataset.prompt_key
            for i in range(min(len(dataset), 4096)):
                prompt = dataset[i][prompt_key]
                text = prompt if isinstance(prompt, str) else "\n".join(m["content"] for m in prompt)
                pool.extend(self.tokenizer.encode(text, add_special_tokens=False))
            self._dummy_token_pool = pool
            logger.info(f"Dummy token pool: {len(pool)} tokens from {min(len(dataset), 4096)} prompts")
        return self._dummy_token_pool

    def _dummy_tokens(self, rng: random.Random, n: int, shared_token: int) -> List[int]:
        source = self.cfg.trainer.dummy_token_source
        if source == "repeat":
            return [shared_token] * n
        if source == "random":
            return [rng.randrange(self.tokenizer.vocab_size) for _ in range(n)]
        assert source == "dataset", source
        pool = self._token_pool()
        out: List[int] = []
        while len(out) < n:
            start = rng.randrange(len(pool))
            out.extend(pool[start : start + n - len(out)])
        return out

    def _make_dummy_batch(self, step: int):
        cfg = self.cfg
        n_prompts = cfg.trainer.train_batch_size
        n_samples = cfg.generator.n_samples_per_prompt
        rng = random.Random(cfg.trainer.dummy_seed * 1_000_003 + step)
        shared_token = rng.randint(0, self.tokenizer.vocab_size - 1)
        lengths = self._dummy_lengths(rng, n_prompts * n_samples)
        prompt_token_ids, response_ids, rewards, loss_masks, uids = [], [], [], [], []
        for p in range(n_prompts):
            # Samples of one prompt share the prompt, as in GRPO.
            prompt = self._dummy_tokens(rng, lengths[p * n_samples][0], shared_token)
            for s in range(n_samples):
                response_len = lengths[p * n_samples + s][1]
                prompt_token_ids.append(prompt)
                response_ids.append(self._dummy_tokens(rng, response_len, shared_token))
                rewards.append([0] * (response_len - 1) + [rng.randint(0, 1)])
                loss_masks.append([1] * response_len)
                uids.append(str(p))
        num_tokens = sum(len(p) + len(r) for p, r in zip(prompt_token_ids, response_ids))
        dummy_generator_output = {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": response_ids,
            "rewards": rewards,
            "loss_masks": loss_masks,
        }
        return dummy_generator_output, uids, num_tokens, sum(len(r) for r in response_ids)

    def _execute_training_step(self, model: str, data) -> Dict[str, float]:
        """Same loop as RayPPOTrainer._execute_training_step, timing forward_backward and optim_step."""
        boundaries = data.metadata[f"{model}_mini_batch_boundaries"]
        if model == "policy":
            data = self._normalize_advantages(data, boundaries, data.metadata.get("policy_prompt_boundaries"))
        all_metrics: Dict[str, List[float]] = defaultdict(list)
        all_chunk_refs = self.dispatch.stage_data(model, data, boundaries)
        fwd_bwd_s = optim_s = 0.0
        for _epoch in range(self.cfg.trainer.update_epochs_per_batch):
            for chunk_refs in all_chunk_refs:
                t0 = time.perf_counter()
                status = self.dispatch.forward_backward_from_staged(model, chunk_refs)
                t1 = time.perf_counter()
                for k, v in status.metrics.items():
                    all_metrics[k].append(v)
                grad_norm = self.dispatch.optim_step(model)
                t2 = time.perf_counter()
                fwd_bwd_s, optim_s = fwd_bwd_s + t1 - t0, optim_s + t2 - t1
                if grad_norm is not None:
                    all_metrics["grad_norm"].append(grad_norm)
        self.all_timings[f"{model}_forward_backward"] = fwd_bwd_s
        self.all_timings[f"{model}_optim_step"] = optim_s
        return reduce_metrics(all_metrics, sum_loss_metrics=False)

    def _peak_memory_gib(self) -> Dict[str, float]:
        stats = ray.get([a.__ray_call__.remote(_cuda_peak_and_reset) for a in self.policy_model._actor_handlers])
        gib = 1024**3
        return {
            "memory/policy_max_allocated_gib": max(s["max_allocated"] for s in stats) / gib,
            "memory/policy_max_reserved_gib": max(s["max_reserved"] for s in stats) / gib,
        }

    async def train(self):
        """Run a few training steps with max sequence length."""
        logger.info("Starting dummy training with max sequence length...")

        self.global_step = 0
        skip_engines = self.cfg.trainer.skip_inference_engines

        # Initialize weight sync state
        if not skip_engines:
            with Timer("init_weight_sync_state", self.all_timings):
                self.init_weight_sync_state()
        self._peak_memory_gib()  # reset the peaks left by model init

        # Run a few training steps
        self.global_step += 1  # start from 1
        self._profiler_start()
        try:
            for step in range(self.cfg.trainer.num_dummy_steps):
                logger.info(f"Running dummy training step {step + 1}/{self.cfg.trainer.num_dummy_steps}")

                # Run a single training step
                with Timer("step", self.all_timings):
                    # Create training input directly with dummy sequences
                    dummy_generator_output, uids, num_tokens, num_response_tokens = self._make_dummy_batch(step)
                    training_input = self.convert_to_training_input(dummy_generator_output, uids)

                    with Timer("fwd_logprobs_values_reward", self.all_timings):
                        training_input = self.fwd_logprobs_values_reward(training_input)

                    # 1.5 apply kl divergence penalty to rewards
                    if self.cfg.trainer.algorithm.use_kl_in_reward:
                        with Timer("apply_reward_kl_penalty", self.all_timings):
                            training_input = self.apply_reward_kl_penalty(training_input)

                    # 3. calculate advantages and returns
                    with Timer("compute_advantages_and_returns", self.all_timings):
                        training_input = self.compute_advantages_and_returns(training_input)
                        # remove some unwanted keys
                        for key in ["rewards"]:
                            training_input.pop(key)
                        training_input.metadata.pop("uids")

                    # 4. train policy/critic model
                    with Timer("train_critic_and_policy", self.all_timings):
                        status = self.train_critic_and_policy(training_input)

                    # One profiler step per global step.
                    self._profiler_step()

                timings = self.all_timings
                throughput = {
                    "throughput/num_tokens": num_tokens,
                    "throughput/num_response_tokens": num_response_tokens,
                    "throughput/fwd_tokens_per_s": num_tokens / timings["fwd_logprobs_values_reward"],
                    "throughput/train_tokens_per_s": num_tokens / timings["policy_train"],
                    "throughput/fwd_bwd_tokens_per_s": num_tokens / timings["policy_forward_backward"],
                }
                self.all_metrics.update(throughput)
                self.all_metrics.update(self._peak_memory_gib())
                summary = {k: round(v, 2) for k, v in {**throughput, **timings}.items()}
                summary.update({k: round(self.all_metrics[k], 2) for k in self.all_metrics if k.startswith("memory/")})
                logger.info(f"THROUGHPUT step {step + 1}: {summary}")
                self.tracker.log(self.all_metrics, step=self.global_step)
                self.all_metrics = {}
                self.tracker.log(
                    {"timing/" + k: v for k, v in self.all_timings.items()},
                    step=self.global_step,
                )
                self.all_timings = {}
                self.global_step += 1

                logger.info(f"Step {step + 1} completed. Status: {status}")
        finally:
            self._profiler_stop()

        self.tracker.finish()
        logger.info("Dummy training completed successfully!")

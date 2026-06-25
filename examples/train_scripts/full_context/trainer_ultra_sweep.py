"""Throughput / memory sweep trainer for Nemotron-Ultra-550B (Megatron).

Extends the dummy ``FullCtxTrainer`` to (a) build either uniform-length or
variable-length synthetic batches, (b) record per-step wall time and the peak
CUDA reserved memory across policy workers, and (c) append one JSON record per
step to ``$SWEEP_RESULTS_FILE``. It runs the *real* fwd+bwd training path
(``fwd_logprobs_values_reward`` + ``train_critic_and_policy``) so the numbers
reflect genuine training cost, but fabricates the rollout so no vLLM generation
is needed.

Driven entirely by env vars (so the same module serves every sweep config):

  SWEEP_RESULTS_FILE   path to append JSONL results to (required)
  SWEEP_TAG            label for this run (e.g. "tp8pp4ep16_mtpm128k")
  SWEEP_MODE           "uniform" (default) or "varlen"
  SWEEP_NUM_STEPS      number of measured steps (default 3; first is warmup)
  SWEEP_NUM_SEQ        total sequences per step (default = train_batch_size*n_samples)
  SWEEP_PROMPT_LEN     prompt length per sequence (default 512)
  # uniform mode:
  SWEEP_SEQ_LEN        total tokens per sequence (prompt+response), default 10240
  # varlen mode:
  SWEEP_AVG_LEN        mean total tokens/sequence (default 60000)
  SWEEP_STD_LEN        stddev of total tokens/sequence (default 30000)
  SWEEP_MIN_LEN        clamp floor for total length (default 1024)
  SWEEP_MAX_LEN        clamp ceiling for total length (default 131072)
  SWEEP_SEED           RNG seed for reproducible varlen draws (default 1234)
"""

import json
import random
import time

from loguru import logger

from skyrl.train.utils.utils import Timer

from .trainer_full_ctx import FullCtxTrainer


class UltraSweepTrainer(FullCtxTrainer):
    def _build_lengths(self, num_seq):
        """Return a list of (prompt_len, response_len) per sequence."""
        t = self.cfg.trainer
        mode = t.sweep_mode
        prompt_len = t.sweep_prompt_len
        if mode == "uniform":
            seq_len = t.sweep_seq_len
            resp = max(1, seq_len - prompt_len)
            return [(prompt_len, resp)] * num_seq, mode, seq_len, 0
        # varlen
        rng = random.Random(t.sweep_seed)
        out = []
        for _ in range(num_seq):
            total = int(round(rng.gauss(t.sweep_avg_len, t.sweep_std_len)))
            total = max(t.sweep_min_len, min(t.sweep_max_len, total))
            total = max(total, prompt_len + 1)
            out.append((prompt_len, total - prompt_len))
        return out, mode, t.sweep_avg_len, t.sweep_std_len

    def _peak_reserved_gb(self):
        """Peak CUDA high-water (reserved, allocated) and min free, GB, across policy workers.

        Uses max_reserved/max_allocated (high-water marks that survive empty_cache/offload)
        so we capture the in-step fwd/bwd peak even though this is queried after the policy
        has been offloaded back to CPU. Falls back to current reserved/allocated on older
        workers that don't return the max_* keys.
        """
        import ray

        try:
            mems = ray.get(
                self.policy_model.async_run_ray_method("pass_through", "get_cuda_memory"),
                timeout=60,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"get_cuda_memory failed: {e}")
            return None, None, None
        reserved = [m.get("max_reserved", m["reserved"]) / 1e9 for m in mems]
        allocated = [m.get("max_allocated", m["allocated"]) / 1e9 for m in mems]
        free = [m["free"] / 1e9 for m in mems]
        return max(reserved), max(allocated), min(free)

    def _record(self, rec):
        path = self.cfg.trainer.sweep_results_file
        if path:
            try:
                import os

                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"could not write results file {path}: {e}")
        logger.info("SWEEP_RESULT " + json.dumps(rec))

    async def train(self):
        tag = self.cfg.trainer.sweep_tag
        num_steps = self.cfg.trainer.num_dummy_steps
        n_samples = self.cfg.generator.n_samples_per_prompt
        default_num_seq = self.cfg.trainer.train_batch_size * n_samples
        cfg_num_seq = self.cfg.trainer.sweep_num_seq
        num_seq = cfg_num_seq if cfg_num_seq and cfg_num_seq > 0 else default_num_seq

        mc = self.cfg.trainer.policy.megatron_config
        world = self.cfg.trainer.placement.policy_num_nodes * self.cfg.trainer.placement.policy_num_gpus_per_node
        dp = world // (mc.pipeline_model_parallel_size * mc.context_parallel_size * mc.tensor_model_parallel_size)
        cfg_hdr = {
            "tag": tag,
            "tp": mc.tensor_model_parallel_size,
            "pp": mc.pipeline_model_parallel_size,
            "cp": mc.context_parallel_size,
            "ep": mc.expert_model_parallel_size,
            "etp": mc.expert_tensor_parallel_size,
            "dp": dp,
            "mtpm": self.cfg.trainer.max_tokens_per_microbatch,
            "num_seq": num_seq,
        }
        logger.info(f"[ultra-sweep] starting: {json.dumps(cfg_hdr)}")

        self.global_step = 0
        with Timer("init_weight_sync_state", self.all_timings):
            self.init_weight_sync_state()

        lengths, mode, p0, p1 = self._build_lengths(num_seq)
        total_tokens = sum(p + r for p, r in lengths)
        max_seqlen = max(p + r for p, r in lengths)
        logger.info(
            f"[ultra-sweep] mode={mode} num_seq={num_seq} total_tokens={total_tokens} "
            f"max_seqlen={max_seqlen} dp={dp} per_dp_tokens={total_tokens // dp}"
        )

        # uids group sequences into n_samples_per_prompt groups (for grpo advantage std).
        uids = []
        for i in range(num_seq):
            uids.append(str(i // n_samples))

        self.global_step += 1
        for step in range(num_steps):
            oom = False
            err = None
            t0 = time.time()
            try:
                prompt_token_ids = [[random.randint(0, self.tokenizer.vocab_size - 1)] * p for (p, r) in lengths]
                response_ids = [[random.randint(0, self.tokenizer.vocab_size - 1)] * r for (p, r) in lengths]
                rewards = [[0.0] * (r - 1) + [float(random.randint(0, 1))] for (p, r) in lengths]
                loss_masks = [[1] * r for (p, r) in lengths]
                dummy = {
                    "prompt_token_ids": prompt_token_ids,
                    "response_ids": response_ids,
                    "rewards": rewards,
                    "loss_masks": loss_masks,
                }
                training_input = self.convert_to_training_input(dummy, uids)
                with Timer("step", self.all_timings):
                    with Timer("fwd_logprobs_values_reward", self.all_timings):
                        training_input = self.fwd_logprobs_values_reward(training_input)
                    with Timer("compute_advantages_and_returns", self.all_timings):
                        training_input = self.compute_advantages_and_returns(training_input)
                        for key in ["rewards"]:
                            training_input.pop(key)
                        training_input.metadata.pop("uids")
                    with Timer("train_critic_and_policy", self.all_timings):
                        self.train_critic_and_policy(training_input)
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {str(e)[:500]}"
                if "out of memory" in str(e).lower() or "OutOfMemory" in str(e):
                    oom = True
                logger.error(f"[ultra-sweep] step {step} FAILED: {err}")

            step_time = time.time() - t0
            if err:
                # Workers can be wedged after a CUDA OOM; querying memory may hang.
                peak_res, peak_alloc, min_free = None, None, None
            else:
                peak_res, peak_alloc, min_free = self._peak_reserved_gb()
            rec = dict(cfg_hdr)
            rec.update(
                {
                    "step": step,
                    "warmup": step == 0,
                    "mode": mode,
                    "total_tokens": total_tokens,
                    "max_seqlen": max_seqlen,
                    "per_dp_tokens": total_tokens // dp,
                    "step_time_s": round(step_time, 2),
                    "tokens_per_s": round(total_tokens / step_time, 1) if step_time > 0 and not err else None,
                    "peak_reserved_gb": round(peak_res, 2) if peak_res else None,
                    "peak_alloc_gb": round(peak_alloc, 2) if peak_alloc else None,
                    "min_free_gb": round(min_free, 2) if min_free else None,
                    "oom": oom,
                    "error": err,
                }
            )
            self._record(rec)
            try:
                self.tracker.log({"timing/" + k: v for k, v in self.all_timings.items()}, step=self.global_step)
            except Exception:  # noqa: BLE001
                pass
            self.all_timings = {}
            self.all_metrics = {}
            self.global_step += 1
            if err:
                # No point continuing this config once it fails.
                break

        try:
            self.tracker.finish()
        except Exception:  # noqa: BLE001
            pass
        logger.info(f"[ultra-sweep] done: {tag}")

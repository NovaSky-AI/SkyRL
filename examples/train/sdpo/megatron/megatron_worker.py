"""Megatron TP/PP/SP SDPO worker. Vocabulary tensors stay inside each Ray worker.

Initially restricted to PP<=3, CP=EP=1 and the distribution objective. Unsupported
options fail validation rather than silently selecting a different objective.
"""

import copy
import os
from collections import defaultdict, deque
from contextlib import contextmanager
from itertools import chain
from pathlib import Path

import ray
import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from megatron.core.pipeline_parallel import get_forward_backward_func

from skyrl.backends.skyrl_train.distributed.dispatch import WorkerOutput
from skyrl.backends.skyrl_train.distributed.megatron.megatron_utils import (
    remove_left_padding,
    to_te_attention_mask,
)
from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
    MegatronPolicyWorkerBase,
)
from skyrl.backends.skyrl_train.workers.worker_utils import (
    compute_minibatch_rollout_logprob_diff_metrics,
    get_microbatch_iterator,
    reduce_metrics,
)

from .loss import (
    vocab_parallel_action_logprobs,
    vocab_parallel_entropy_weights,
    vocab_parallel_sdpo_loss,
)
from .teacher import MegatronEMATeacher
from .response_logits import response_logits_output_processor
from ..sdpo_config import validate_megatron_sdpo


class SDPOMegatronPolicyWorker(MegatronPolicyWorkerBase):
    def init_model(self, model_path, num_training_steps=1e9):
        validate_megatron_sdpo(self.cfg)
        super().init_model(model_path, num_training_steps)
        if len(self.actor_module) != 1:
            raise ValueError("SDPO Megatron does not yet support virtual pipeline stages")
        if self.is_vlm:
            if not self.cfg.distillation.vision_language:
                raise ValueError("SDPO VLM requires distillation.vision_language=true")
            self.model._assert_vlm_supported()
        text_config = getattr(self.strategy.hf_config, "text_config", self.strategy.hf_config)
        if getattr(text_config, "num_experts", 0) or getattr(text_config, "num_local_experts", 0):
            raise ValueError("Megatron SDPO currently supports dense models only")
        self._sdpo_vocab_size = text_config.vocab_size
        self.teacher = None
        if self.cfg.distillation.teacher.kind in ("ema", "frozen"):
            # A fresh provider has no student PEFT hooks or DDP/optimizer buffers.
            builder = copy.copy(self)
            builder.strategy = copy.copy(self.strategy)
            mg = self.cfg.policy.megatron_config
            builder.init_configs(
                self.cfg.distillation.teacher.path or model_path,
                mg,
                mg.model_config_kwargs,
                copy.deepcopy(mg.transformer_config_kwargs),
                bf16=self.cfg.bf16,
                flash_attn=self.cfg.flash_attn,
                language_model_only=self.cfg.policy.language_model_only,
            )
            modules = builder.make_megatron_module(wrap_with_ddp=False, bf16=self.cfg.bf16)
            self.teacher = MegatronEMATeacher(modules[0], self.cfg.distillation.teacher)
            del builder, modules
        self._sdpo_has_tokens = False
        self._advantage_abs_sum = 0.0
        self._advantage_count = 0.0

    def _response_logits(self, model, micro, response_length, teacher=False):
        prefix = "teacher_" if teacher else ""
        ids = micro[prefix + "sequences"]
        mask = micro[prefix + "attention_mask"].bool()
        positions = (mask.long().cumsum(-1) - 1).clamp_min(0)
        ids, compact_mask, positions = remove_left_padding(ids, mask, positions)
        kwargs = {}
        for key in ("pixel_values", "image_grid_thw"):
            if key == "pixel_values" and not mpu.is_pipeline_first_stage():
                continue
            values = micro.get(prefix + key)
            if values is None:
                values = micro.get(key)
            if values is not None:
                kwargs[key] = torch.cat(values.tensors, dim=0)
        if mpu.is_pipeline_last_stage():
            # Position t predicts token t+1, including the last prompt position.
            original_positions = mask.long().cumsum(-1) - 1
            slots = original_positions[:, -response_length - 1 : -1].clamp_min(0)
            kwargs.update(
                output_processor=response_logits_output_processor,
                output_processor_context={"response_positions": slots},
            )
        logits = model(
            ids,
            None if self.is_vlm else positions,
            to_te_attention_mask(compact_mask),
            **kwargs,
        )
        if not mpu.is_pipeline_last_stage():
            # Pipeline stages exchange decoder hidden states, not response logits.
            return logits
        if logits.shape[1] != response_length:
            raise RuntimeError("Megatron model did not apply response-only output projection")
        return logits

    def _score_teacher(self, model, micro, response_length):
        """Run the complete teacher pipeline; only its final stage keeps logits."""
        scored = []

        def collect_logits(logits):
            scored.append(logits.detach().cpu())
            return logits.new_zeros(()), {}

        def forward_step(iterator, stage):
            return self._response_logits(stage, next(iterator), response_length, teacher=True), collect_logits

        with torch.no_grad():
            get_forward_backward_func()(
                forward_step_func=forward_step,
                data_iterator=iter([micro]),
                model=[model],
                num_microbatches=1,
                seq_length=micro["teacher_sequences"].shape[1],
                micro_batch_size=micro.batch_size,
                forward_only=True,
            )
        return scored[0] if scored else None

    def _prepare_micro(self, micro, device):
        # The iterator owns all CPU microbatches until the loop finishes. Its
        # TensorBatch.to() mutates the container, so never call it on those
        # queued objects: that retains every processed PDF's GPU pixels.
        # A shallow container copy is sufficient: .to() replaces fields, and
        # TensorList.to() returns a new list without changing its CPU tensors.
        staged = type(micro)(dict(micro))
        staged.metadata = micro.metadata
        if not mpu.is_pipeline_first_stage():
            for key in ("pixel_values", "teacher_pixel_values"):
                if staged.get(key) is not None:
                    staged[key] = None
        return staged.to(device)

    @contextmanager
    def _teacher_phase(self):
        model = self.teacher.model if self.teacher else self.actor_module[0]
        was_training = model.training
        model.eval()
        try:
            if self.teacher:
                self.teacher.backload_to_gpu()
            yield model
        finally:
            if self.teacher and self.cfg.distillation.teacher.storage == "cpu":
                self.teacher.offload_to_cpu()
            model.train(was_training)

    def _teacher_windows(self, micros, response_length, device, metrics):
        """Cache response logits on the last PP stage; budget is per TP rank.

        Fixed sample microbatches share padded response/vocabulary dimensions.
        The first output determines capacity using its actual dtype and shape.
        One microbatch is the minimum window, even if it exceeds the budget.
        """
        capacity, offset = 1, 0
        budget = int(self.cfg.distillation.megatron_teacher_cache_gib * 1024**3)
        while offset < len(micros):
            cache = deque()
            try:
                with self._teacher_phase() as model:
                    index = offset
                    while index < min(offset + capacity, len(micros)):
                        micro = self._prepare_micro(micros[index], device)
                        try:
                            logits = self._score_teacher(model, micro, response_length)
                        finally:
                            del micro
                        if index == 0:
                            size = torch.ones((), dtype=torch.int64, device=device)
                            if mpu.is_pipeline_last_stage():
                                size.fill_(
                                    max(
                                        1,
                                        min(len(micros), budget // max(logits.nbytes, 1)),
                                    )
                                )
                            if mpu.get_pipeline_model_parallel_world_size() > 1:
                                dist.broadcast(
                                    size,
                                    src=mpu.get_pipeline_model_parallel_last_rank(),
                                    group=mpu.get_pipeline_model_parallel_group(),
                                )
                            capacity = size.item()
                        cache.append((index, logits))
                        del logits
                        index += 1
                offset = index
                cache_bytes = torch.tensor(
                    sum(x.nbytes for _, x in cache if x is not None),
                    dtype=torch.int64,
                    device=device,
                )
                if mpu.get_pipeline_model_parallel_world_size() > 1:
                    dist.broadcast(
                        cache_bytes,
                        src=mpu.get_pipeline_model_parallel_last_rank(),
                        group=mpu.get_pipeline_model_parallel_group(),
                    )
                metrics["teacher_cache_windows"] += 1
                metrics["teacher_scored_microbatches"] += len(cache)
                metrics["teacher_cache_peak_gib"] = max(metrics["teacher_cache_peak_gib"], cache_bytes.item() / 1024**3)
                yield cache
            finally:
                cache.clear()

    @staticmethod
    def _consume_teacher_cache(first, windows):
        try:
            for cache in chain((first,), windows):
                while cache:
                    yield cache.popleft()
        finally:
            first.clear()
            windows.close()

    def _advantage_signature(self):
        d = self.cfg.distillation
        return (d.loss, d.alpha, d.topk, d.add_tail, self.cfg.algorithm.temperature)

    def forward_backward(self, data, loss_fn=None, loss_fn_config=None, return_per_token_outputs=False):
        if loss_fn is not None or loss_fn_config or return_per_token_outputs:
            raise ValueError("Configure the SDPO objective through trainer.distillation; scalar metrics only")
        device = torch.device("cuda", torch.cuda.current_device())
        group = mpu.get_data_parallel_group()
        count = data["loss_mask"].sum().to(device=device, dtype=torch.float32)
        dist.all_reduce(count, group=group)
        self._sdpo_has_tokens |= count.item() > 0
        track_diff = data.get("rollout_logprobs") is not None
        config = self.cfg.distillation
        if config.rollout_is_clip is not None and not track_diff:
            raise ValueError("Rollout importance clipping requires rollout_logprobs")
        # Identical keys on all PP stages, including an entirely masked batch.
        diff_metrics = compute_minibatch_rollout_logprob_diff_metrics(
            count.new_zeros(1), count.new_zeros(1) if track_diff else None, None
        )
        micro_diff_metrics = defaultdict(list)
        if count.item() == 0:
            return WorkerOutput(
                loss_fn_outputs=[],
                metrics={
                    "sdpo_loss": 0.0,
                    "eligible_tokens": 0.0,
                    "empty_target_batch": 1.0,
                    **diff_metrics,
                },
            )
        response_length = data.metadata["response_length"]
        total = torch.zeros((), device=device)
        stats = torch.zeros(5, device=device)
        advantage_clip = None
        if config.advantage_is_clip is not None and self._advantage_count > 0:
            advantage_clip = config.advantage_is_clip * self._advantage_abs_sum / self._advantage_count
        entropy_weights = {}
        entropy_mean = count.new_ones(())
        self.model.train()
        micros = list(get_microbatch_iterator(data, self.cfg.micro_train_batch_size_per_gpu, -1))
        cache_metrics = {
            "teacher_cache_windows": 0,
            "teacher_scored_microbatches": 0,
            "teacher_cache_peak_gib": 0.0,
        }
        windows = self._teacher_windows(micros, response_length, device, cache_metrics)
        first = next(windows)
        full_cache = len(first) == len(micros)
        try:
            if config.use_entropy_weighting:
                # Preserve batch-wide normalization even when logits need windowing.
                entropy_sum = count.new_zeros(())
                rows = iter(first) if full_cache else self._consume_teacher_cache(first, windows)
                try:
                    for index, teacher_logits in rows:
                        if mpu.is_pipeline_last_stage():
                            weights = vocab_parallel_entropy_weights(
                                teacher_logits,
                                vocab_size=self._sdpo_vocab_size,
                                group=mpu.get_tensor_model_parallel_group(),
                                device=device,
                                temperature=self.cfg.algorithm.temperature,
                                beta=config.entropy_beta,
                                chunk_size=config.loss_chunk_size,
                            )
                            entropy_sum += (weights * micros[index]["loss_mask"].to(device)).sum()
                            entropy_weights[index] = weights.cpu()
                            del weights
                        del teacher_logits
                finally:
                    if not full_cache:
                        rows.close()
                    del rows
                dist.all_reduce(entropy_sum, group=group)
                if mpu.get_pipeline_model_parallel_world_size() > 1:
                    dist.broadcast(
                        entropy_sum,
                        src=mpu.get_pipeline_model_parallel_last_rank(),
                        group=mpu.get_pipeline_model_parallel_group(),
                    )
                entropy_mean = (entropy_sum / count).clamp_min(torch.finfo(torch.float32).tiny)
                if not full_cache:
                    windows = self._teacher_windows(micros, response_length, device, cache_metrics)
                    first = next(windows)
        except BaseException:
            first.clear()
            windows.close()
            raise
        cached_rows = self._consume_teacher_cache(first, windows)
        try:
            for index, teacher_logits in cached_rows:
                micro = self._prepare_micro(micros[index], device)

                def loss_callback(
                    student_logits,
                    teacher_logits=teacher_logits,
                    micro=micro,
                    index=index,
                ):
                    if track_diff:
                        action_logprobs = vocab_parallel_action_logprobs(
                            student_logits,
                            micro["sequences"][:, -response_length:],
                            vocab_size=self._sdpo_vocab_size,
                            group=mpu.get_tensor_model_parallel_group(),
                            temperature=self.cfg.algorithm.temperature,
                            chunk_size=self.cfg.distillation.loss_chunk_size,
                        )
                        for (
                            key,
                            value,
                        ) in compute_minibatch_rollout_logprob_diff_metrics(
                            action_logprobs,
                            micro["rollout_logprobs"],
                            micro["loss_mask"],
                        ).items():
                            micro_diff_metrics[key].append(value)
                    weights = micro["loss_mask"]
                    if config.use_entropy_weighting:
                        weights = weights * entropy_weights[index].to(device) / entropy_mean
                    loss, chunk_stats = vocab_parallel_sdpo_loss(
                        student_logits,
                        teacher_logits,
                        weights,
                        self.cfg.distillation,
                        vocab_size=self._sdpo_vocab_size,
                        group=mpu.get_tensor_model_parallel_group(),
                        temperature=self.cfg.algorithm.temperature,
                        response_ids=micro["sequences"][:, -response_length:],
                        rollout_logprobs=micro.get("rollout_logprobs"),
                        advantage_clip=advantage_clip,
                        return_stats=True,
                    )
                    total.add_(loss.detach())
                    stats.add_(chunk_stats)
                    # One microbatch per schedule invocation; DDP averages across DP.
                    return loss * mpu.get_data_parallel_world_size() / count, {}

                def forward_step(iterator, model, loss_callback=loss_callback):
                    batch = next(iterator)
                    return self._response_logits(model, batch, response_length), loss_callback

                try:
                    get_forward_backward_func()(
                        forward_step_func=forward_step,
                        data_iterator=iter([micro]),
                        model=self.actor_module,
                        num_microbatches=1,
                        seq_length=micro["sequences"].shape[1],
                        micro_batch_size=micro.batch_size,
                        forward_only=False,
                    )
                finally:
                    # Callbacks capture the GPU inputs and CPU teacher logits.
                    # Drop all owners before staging the next microbatch.
                    del teacher_logits, loss_callback, forward_step, micro
        finally:
            cached_rows.close()
        dist.all_reduce(total, group=group)
        dist.all_reduce(stats, group=group)
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            dist.broadcast(
                stats,
                src=mpu.get_pipeline_model_parallel_last_rank(),
                group=mpu.get_pipeline_model_parallel_group(),
            )
        if config.advantage_is_clip is not None:
            self._advantage_abs_sum += stats[0].item()
            self._advantage_count += stats[1].item()
        weighting_metrics = {}
        if config.use_entropy_weighting:
            weighting_metrics["entropy_weight_mean_raw"] = entropy_mean.item()
        if config.advantage_is_clip is not None:
            weighting_metrics.update(
                advantage_abs_mean=(stats[0] / stats[1].clamp_min(1)).item(),
                advantage_clip_threshold=advantage_clip or 0.0,
                advantage_clip_active=float(advantage_clip is not None),
                advantage_clipped_mass=(stats[2] / stats[1].clamp_min(1)).item(),
            )
        if config.rollout_is_clip is not None:
            weighting_metrics.update(
                rollout_is_mean=(stats[3] / count).item(),
                rollout_is_clipped_fraction=(stats[4] / count).item(),
            )
        if track_diff:
            if mpu.is_pipeline_last_stage():
                diff_metrics = reduce_metrics(micro_diff_metrics)
                for key, value in diff_metrics.items():
                    scalar = count.new_tensor(value)
                    op = dist.ReduceOp.SUM
                    if key.endswith("_max"):
                        op = dist.ReduceOp.MAX
                    elif key.endswith("_min"):
                        op = dist.ReduceOp.MIN
                    dist.all_reduce(scalar, op=op, group=group)
                    if op == dist.ReduceOp.SUM:
                        scalar /= mpu.get_data_parallel_world_size()
                    diff_metrics[key] = scalar.item()
            if mpu.get_pipeline_model_parallel_world_size() > 1:
                keys = sorted(diff_metrics)
                values = count.new_tensor([diff_metrics[key] for key in keys])
                dist.broadcast(
                    values,
                    src=mpu.get_pipeline_model_parallel_last_rank(),
                    group=mpu.get_pipeline_model_parallel_group(),
                )
                diff_metrics = dict(zip(keys, values.tolist(), strict=True))
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            dist.broadcast(
                total,
                src=mpu.get_pipeline_model_parallel_last_rank(),
                group=mpu.get_pipeline_model_parallel_group(),
            )
        return WorkerOutput(
            loss_fn_outputs=[],
            metrics={
                "sdpo_loss": (total / count).item(),
                "eligible_tokens": count.item(),
                "empty_target_batch": 0.0,
                **diff_metrics,
                **weighting_metrics,
                **cache_metrics,
            },
        )

    def optim_step(self):
        norm = None
        if self._sdpo_has_tokens:
            self.model.run_pending_grad_sync()
            success, norm, _ = self.optimizer.step()
            if success:
                self.scheduler.step(increment=1)
                if self.teacher:
                    self.teacher.after_update(self.actor_module[0])
        self.optimizer.zero_grad()
        for module in self.actor_module:
            module.zero_grad_buffer()
        self._sdpo_has_tokens = False
        self._micro_batches_accumulated = 0
        return norm

    def offload_to_cpu(self, offload_optimizer=True, offload_model=True):
        if offload_model and getattr(self, "teacher", None):
            self.teacher.offload_to_cpu()
        return super().offload_to_cpu(offload_optimizer, offload_model)

    def _teacher_path(self, ckpt_dir):
        return Path(ckpt_dir) / f"sdpo_megatron_teacher_rank_{dist.get_rank()}.pt"

    def _topology(self):
        return (
            dist.get_world_size(),
            mpu.get_tensor_model_parallel_world_size(),
            mpu.get_pipeline_model_parallel_world_size(),
            mpu.get_context_parallel_world_size(),
        )

    def save_checkpoint(self, ckpt_dir, tokenizer=None):
        super().save_checkpoint(ckpt_dir, tokenizer)
        self.finalize_pending_saves()
        path = self._teacher_path(ckpt_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".pt.tmp")
        torch.save(
            {
                "topology": self._topology(),
                "teacher": self.teacher.state_dict() if self.teacher else None,
                "advantage_statistics": (
                    self._advantage_abs_sum,
                    self._advantage_count,
                ),
                "advantage_signature": self._advantage_signature(),
            },
            temporary,
        )
        os.replace(temporary, path)
        dist.barrier()

    def load_checkpoint(self, ckpt_dir, load_optimizer_states=True, load_lr_scheduler_states=True):
        state = torch.load(self._teacher_path(ckpt_dir), map_location="cpu", weights_only=True)
        if state["topology"] != self._topology() or (state["teacher"] is None) != (self.teacher is None):
            raise ValueError("Megatron SDPO resume requires the same topology and teacher kind")
        result = super().load_checkpoint(ckpt_dir, load_optimizer_states, load_lr_scheduler_states)
        if self.teacher:
            self.teacher.load_state_dict(state["teacher"])
        self._advantage_abs_sum, self._advantage_count = (
            state.get("advantage_statistics", (0.0, 0.0))
            if state.get("advantage_signature") == self._advantage_signature()
            else (0.0, 0.0)
        )
        return result


PolicyWorker = ray.remote(num_gpus=1)(SDPOMegatronPolicyWorker)

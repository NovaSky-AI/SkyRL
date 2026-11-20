"""Local FSDP worker: no vocabulary tensors leave the worker."""

import math
import os
import time
from pathlib import Path

import ray
import torch
import torch.distributed as dist
from loguru import logger
from transformers import AutoConfig

from skyrl.backends.skyrl_train.distributed.dispatch import WorkerOutput
from skyrl.backends.skyrl_train.distributed.fsdp_utils import should_use_meta_init
from skyrl.backends.skyrl_train.utils.ppo_utils import ppo_policy_loss
from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import FSDPPolicyWorkerBase
from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper
from skyrl.backends.skyrl_train.workers.worker_utils import get_microbatch_iterator

from .teachers import build_teacher
from .utils import (
    chunked_action_logprobs,
    dw_sdpo_weights,
    response_entropy_sum,
    response_logits,
    sdpo_loss,
    vision_inputs,
)


class SDPOPolicyWorker(FSDPPolicyWorkerBase):
    def offload_to_cpu(self, offload_optimizer=True, offload_model=True):
        if offload_model and hasattr(self, "teacher"):
            self.teacher.offload_to_cpu()
        return super().offload_to_cpu(offload_optimizer=offload_optimizer, offload_model=offload_model)

    def backload_to_gpu(self, backload_optimizer=True, backload_model=True):
        result = super().backload_to_gpu(backload_optimizer=backload_optimizer, backload_model=backload_model)
        if backload_model and hasattr(self, "teacher"):
            self.teacher.backload_to_gpu()
        return result

    def init_model(self, model_path, num_training_steps=None):
        super().init_model(model_path, num_training_steps)
        if self.model.is_vlm and not self.cfg.distillation.vision_language:
            raise ValueError("VLM training requires distillation.vision_language=true")
        if self.model.is_vlm and not hasattr(self.model.model.config, "image_token_id"):
            raise ValueError("SDPO VLM scoring requires an image_token_id model configuration")
        for name, value in self.model.model.config.to_dict().items():
            if "dropout" in name and isinstance(value, (int, float)) and value != 0:
                raise ValueError(f"SDPO requires deterministic forwards: model config {name} must be zero")
        if any(isinstance(m, torch.nn.Dropout) and m.p != 0 for m in self.model.modules()):
            raise ValueError("SDPO requires zero model dropout")
        teacher_cfg = self.cfg.distillation.teacher
        teacher_model = None
        if teacher_cfg.kind in ("ema", "frozen"):
            teacher_model = self._init_teacher_model(teacher_cfg.path or model_path)
        self.teacher = build_teacher(self.model.model, teacher_cfg, teacher_model)
        self._sdpo_has_tokens = False
        self._advantage_abs_sum = 0.0
        self._advantage_count = 0.0

    def _init_teacher_model(self, model_path):
        """Eval FSDP model on this worker: no optimizer, same mesh as policy.

        Construction follows FSDPRefWorkerBase.init_model. Scoring stays local so
        vocabulary tensors never cross Ray. Expandable segments stay off until the
        teacher model shards exist, matching policy init / CUDA-IPC constraints.
        """
        logger.info("Initializing SDPO teacher model from {}", model_path)
        self._set_expandable_segments(False)
        try:
            model_config = AutoConfig.from_pretrained(
                model_path,
                trust_remote_code=True,
                **self.cfg.policy.model_config_kwargs,
            )
            use_meta = should_use_meta_init(
                use_meta_tensor=not model_config.tie_word_embeddings,
                mesh=self.strategy.device_mesh,
            )
            wrapped_model = HFModelWrapper(
                model_path,
                use_flash_attention_2=self.cfg.flash_attn,
                bf16=self.cfg.policy.inference_only_init,
                sequence_parallel_size=self.cfg.policy.sequence_parallel_size,
                remove_microbatch_padding=self.cfg.remove_microbatch_padding,
                model_config_kwargs=self.cfg.policy.model_config_kwargs,
                meta_init=use_meta,
                language_model_only=self.cfg.policy.language_model_only,
                logprobs_chunk_size=self.cfg.logprobs_chunk_size,
            )
            self._seq_parallel_monkey_patch(model=wrapped_model.model)
            teacher = self.strategy.prepare(wrapped_model.model)
            return teacher
        finally:
            self._set_expandable_segments(True)

    @torch.no_grad()
    def _batch_entropy_weights(self, data, response_length, device, group):
        """SRPO section 3.2: normalize over all valid SDPO tokens in this update."""
        config = self.cfg.distillation
        weights = {}
        stats = torch.zeros(2, device=device, dtype=torch.float64)
        micros = get_microbatch_iterator(
            data,
            micro_batch_size=self.cfg.micro_train_batch_size_per_gpu,
            max_tokens_per_microbatch=self.cfg.max_tokens_per_microbatch,
        )
        for index, micro in enumerate(micros, start=1):
            micro.to(device)
            inputs = {}
            try:
                mask = micro["loss_mask"]
                if config.use_sample_routing:
                    mask = mask * micro["sdpo_mask"].unsqueeze(-1).to(mask.dtype)
                need_teacher = (mask.sum() > 0).to(dtype=torch.int32)
                dist.all_reduce(need_teacher, op=dist.ReduceOp.MAX, group=group)
                if not need_teacher.item():
                    weights[index] = torch.zeros_like(mask, device="cpu", dtype=torch.float32)
                    continue
                inputs = vision_inputs(micro)
                if micro.get("teacher_pixel_values") is not None:
                    inputs.update(vision_inputs(micro, prefix="teacher_"))
                if config.limit_response_logits:
                    inputs["logits_to_keep"] = response_length + 1
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    logits = self.teacher.score(
                        self.model.model,
                        micro["teacher_sequences"],
                        micro["teacher_attention_mask"],
                        response_length,
                        **inputs,
                    ).cpu()
                raw = dw_sdpo_weights(
                    logits,
                    mask.cpu(),
                    config.entropy_beta,
                    temperature=self.cfg.algorithm.temperature,
                    chunk_size=config.loss_chunk_size,
                )
                weights[index] = raw
                stats += torch.stack(((raw * mask.cpu()).double().sum(), mask.double().sum().cpu())).to(device)
                del logits, raw
            finally:
                inputs.clear()
                micro.to("cpu")
        dist.all_reduce(stats, group=group)
        mean = (stats[0] / stats[1].clamp_min(1)).float().clamp_min(torch.finfo(torch.float32).tiny)
        return weights, mean

    def forward_backward(self, data, loss_fn=None, loss_fn_config=None, return_per_token_outputs=False):
        if loss_fn is not None or loss_fn_config:
            raise ValueError("Configure the SDPO objective through trainer.distillation")
        routing = self.cfg.distillation.use_sample_routing
        device = torch.device("cuda", torch.cuda.current_device())
        group = self.device_mesh.get_group("dp")
        count = data["loss_mask"].sum().to(device=device, dtype=torch.float32)
        dist.all_reduce(count, group=group)
        self._sdpo_has_tokens = count.item() > 0
        empty = {
            "sdpo_loss": 0.0,
            "entropy": 0.0,
            "empty_target_batch": 1.0,
            "advantage_clip_ratio": 0.0,
            "advantage_clip_active": 0.0,
        }
        if routing:
            empty.update(srpo_loss=0.0, grpo_loss=0.0)
        if not self._sdpo_has_tokens:
            return WorkerOutput(loss_fn_outputs=[], metrics=empty)

        advantage_clip = None
        if self.cfg.distillation.advantage_is_clip is not None and self._advantage_count:
            advantage_clip = self.cfg.distillation.advantage_is_clip * self._advantage_abs_sum / self._advantage_count
        advantage_stats = torch.zeros(3, device=device, dtype=torch.float64)
        advantage_min = torch.full((), float("inf"), device=device)
        advantage_max = torch.full((), float("-inf"), device=device)
        total = torch.zeros((), device=device)
        sdpo_total = torch.zeros((), device=device)
        grpo_total = torch.zeros((), device=device)
        entropy_total = torch.zeros((), device=device)
        upstream_mean = self.cfg.distillation.reduction == "upstream_microbatch_mean"
        response_length = data.metadata["response_length"]
        # Sample-based sharding gives identical microbatch counts on every rank.
        # Consume TensorBatch directly so teacher fields cannot be dropped by
        # the core batch_to_experience conversion.
        started = time.perf_counter()
        entropy_weights, entropy_mean = {}, None
        if self.cfg.distillation.use_entropy_weighting:
            entropy_weights, entropy_mean = self._batch_entropy_weights(data, response_length, device, group)
        microbatches = get_microbatch_iterator(
            data,
            micro_batch_size=self.cfg.micro_train_batch_size_per_gpu,
            max_tokens_per_microbatch=self.cfg.max_tokens_per_microbatch,
        )
        for index, micro in enumerate(microbatches, start=1):
            micro.to(device)
            student = self.model.model

            training = student.training
            student.train()
            model_inputs = vision_inputs(micro)
            if self.cfg.distillation.limit_response_logits:
                model_inputs["logits_to_keep"] = response_length + 1
            teacher_model_inputs = dict(model_inputs)
            if micro.get("teacher_pixel_values") is not None:
                teacher_model_inputs.update(vision_inputs(micro, prefix="teacher_"))
            token_mask = micro["loss_mask"]
            if routing:
                sdpo_mask = token_mask * micro["sdpo_mask"].unsqueeze(-1).to(dtype=token_mask.dtype)
                grpo_mask = token_mask * micro["grpo_mask"].unsqueeze(-1).to(dtype=token_mask.dtype)
            else:
                sdpo_mask = token_mask
                grpo_mask = None
            need_teacher = torch.ones((), device=device, dtype=torch.int32)
            if routing:
                need_teacher.zero_()
                if sdpo_mask.sum() > 0:
                    need_teacher.fill_(1)
                dist.all_reduce(need_teacher, op=dist.ReduceOp.MAX, group=group)
            teacher_logits = None
            # Score before building a student autograd graph. CPU storage moves
            # the teacher shards back off GPU when score() returns.
            if self.cfg.distillation.teacher_first and need_teacher.item():
                logger.info("SDPO rank {}: scoring teacher before student, microbatch {}", dist.get_rank(), index)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    teacher_logits = self.teacher.score(
                        student,
                        micro["teacher_sequences"],
                        micro["teacher_attention_mask"],
                        response_length,
                        **teacher_model_inputs,
                    ).cpu()
                torch.cuda.empty_cache()
                logger.info(
                    "SDPO rank {}: teacher logits on CPU; GPU allocated before student: {:.2f} GiB",
                    dist.get_rank(),
                    torch.cuda.memory_allocated() / 2**30,
                )
            try:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    student_logits = response_logits(
                        student,
                        micro["sequences"],
                        micro["attention_mask"],
                        response_length,
                        **model_inputs,
                    )
                    entropy_total += response_entropy_sum(
                        student_logits,
                        token_mask,
                        temperature=self.cfg.algorithm.temperature,
                        chunk_size=self.cfg.distillation.loss_chunk_size,
                    )
                    raw_sdpo = student_logits.new_zeros((), dtype=torch.float32)
                    raw_grpo = student_logits.new_zeros((), dtype=torch.float32)
                    if need_teacher.item():
                        if teacher_logits is None:
                            teacher_logits = self.teacher.score(
                                student,
                                micro["teacher_sequences"],
                                micro["teacher_attention_mask"],
                                response_length,
                                **teacher_model_inputs,
                            )

                        sdpo_weights = sdpo_mask
                        # Optional teacher-entropy weighting, independent of routing.
                        if self.cfg.distillation.use_entropy_weighting:
                            sdpo_weights = sdpo_mask * (entropy_weights.pop(index).to(device) / entropy_mean)
                        raw_sdpo, advantage_sum, advantage_count, micro_min, micro_max, clipped_count = sdpo_loss(
                            student_logits,
                            teacher_logits,
                            micro["sequences"][:, -response_length:],
                            sdpo_weights,  # SDPO loss mask
                            self.cfg.distillation,
                            temperature=self.cfg.algorithm.temperature,
                            rollout_logprobs=micro.get("rollout_logprobs"),
                            advantage_clip=advantage_clip,
                            return_advantage_stats=True,
                        )
                        advantage_stats += torch.stack((advantage_sum, advantage_count, clipped_count)).detach()
                        advantage_min = torch.min(advantage_min, micro_min)
                        advantage_max = torch.max(advantage_max, micro_max)
                        del teacher_logits
                    if routing and grpo_mask is not None and grpo_mask.sum() > 0:
                        old = micro.get("action_log_probs")
                        if old is None:
                            old = micro.get("rollout_logprobs")
                        if old is None:
                            raise ValueError(
                                "use_sample_routing GRPO rows need rollout_logprobs " "or a policy snapshot"
                            )
                        raw_grpo, _pg_metrics = ppo_policy_loss(
                            log_probs=chunked_action_logprobs(
                                student_logits,
                                micro["sequences"][:, -response_length:],
                                self.cfg.algorithm.temperature,
                                self.cfg.distillation.loss_chunk_size,
                            ),
                            old_log_probs=old,
                            advantages=micro["advantages"],
                            config=self.cfg.algorithm,
                            loss_mask=grpo_mask,
                            rollout_logprobs=micro.get("rollout_logprobs"),
                        )
                    raw_loss = raw_sdpo + raw_grpo
                    if not raw_loss.requires_grad:
                        # A rank with only filtered rows must still participate
                        # in FSDP backward collectives with a zero gradient.
                        raw_loss = raw_loss + student_logits[0, 0, 0] * 0.0
                    if not routing and upstream_mean:
                        scale = micro.batch_size / data.batch_size / micro["loss_mask"].sum().clamp_min(1)
                    else:
                        scale = self.mesh_rank.dp_size / count
                    self.strategy.backward(
                        raw_loss * scale,
                        self.model,
                        self.optimizer,
                    )
                total += raw_loss.detach() * scale
                sdpo_total += raw_sdpo.detach() * scale
                grpo_total += raw_grpo.detach() * scale
                del student_logits, raw_loss, raw_sdpo, raw_grpo
            finally:
                student.train(training)
            if index % 16 == 0 or index == len(microbatches):
                logger.info(
                    "SDPO rank {}: microbatch {}/{} completed, {:.1f}s elapsed",
                    dist.get_rank(),
                    index,
                    len(microbatches),
                    time.perf_counter() - started,
                )
        dist.all_reduce(advantage_stats, group=group)
        dist.all_reduce(advantage_min, op=dist.ReduceOp.MIN, group=group)
        dist.all_reduce(advantage_max, op=dist.ReduceOp.MAX, group=group)
        self._advantage_abs_sum += advantage_stats[0].item()
        self._advantage_count += advantage_stats[1].item()
        dist.all_reduce(total, group=group)
        dist.all_reduce(sdpo_total, group=group)
        dist.all_reduce(grpo_total, group=group)
        dist.all_reduce(entropy_total, group=group)
        metrics = {
            "sdpo_loss": (sdpo_total / self.mesh_rank.dp_size).item(),
            "entropy": (entropy_total / count).item(),
            "eligible_tokens": count.item(),
            "empty_target_batch": 0.0,
            "advantage_clip_ratio": (advantage_stats[2] / advantage_stats[1].clamp_min(1)).item(),
            "advantage_clip_active": float(advantage_clip is not None),
        }
        if advantage_stats[1].item() > 0:
            metrics["advantage_min"] = advantage_min.item()
            metrics["advantage_max"] = advantage_max.item()
        if self._advantage_count:
            metrics["advantage_abs_running_mean"] = self._advantage_abs_sum / self._advantage_count
        if advantage_clip is not None:
            metrics["advantage_clip"] = advantage_clip
        if routing:
            metrics["srpo_loss"] = (total / self.mesh_rank.dp_size).item()
            metrics["grpo_loss"] = (grpo_total / self.mesh_rank.dp_size).item()
        else:
            metrics["sdpo_loss"] = (total / self.mesh_rank.dp_size).item()
        return WorkerOutput(loss_fn_outputs=[], metrics=metrics)

    def optim_step(self):
        if not self._sdpo_has_tokens:
            self.optimizer.zero_grad(set_to_none=True)
            return None
        # Always detect nonfinite gradients, even if core max_norm=0. Decide
        # collectively so no rank advances its optimizer/teacher alone.
        finite = torch.ones((), device=torch.cuda.current_device(), dtype=torch.int32)
        for p in self.model.parameters():
            if p.grad is not None:
                grad = p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad
                finite.mul_(torch.isfinite(grad).all().int())
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            self.optimizer.zero_grad(set_to_none=True)
            self._sdpo_has_tokens = False
            return float("nan")
        norm = super().optim_step()
        if norm is None or math.isfinite(norm):
            self.teacher.after_update(self.model.model)
        self._sdpo_has_tokens = False
        return norm

    def _teacher_checkpoint_path(self, ckpt_dir):
        return Path(ckpt_dir) / f"sdpo_teacher_rank_{dist.get_rank()}.pt"

    def save_checkpoint(self, ckpt_dir, tokenizer=None):
        super().save_checkpoint(ckpt_dir, tokenizer)
        path = self._teacher_checkpoint_path(ckpt_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".pt.tmp")
        torch.save(
            {
                "world_size": dist.get_world_size(),
                "fsdp_size": self.cfg.policy.fsdp_config.fsdp_size,
                "teacher": self.teacher.state_dict(),
                "advantage_statistics": (self._advantage_abs_sum, self._advantage_count),
                "advantage_signature": self._advantage_signature(),
            },
            temporary,
        )
        os.replace(temporary, path)
        dist.barrier()

    def load_checkpoint(self, ckpt_dir, load_optimizer_states=True, load_lr_scheduler_states=True):
        states = super().load_checkpoint(ckpt_dir, load_optimizer_states, load_lr_scheduler_states)
        path = self._teacher_checkpoint_path(ckpt_dir)
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["world_size"] != dist.get_world_size() or state["fsdp_size"] != self.cfg.policy.fsdp_config.fsdp_size:
            raise ValueError("SDPO checkpoint resume requires the same world size and FSDP sharding topology")
        self.teacher.load_state_dict(state["teacher"], self.model.model)
        self._advantage_abs_sum, self._advantage_count = (
            state.get("advantage_statistics", (0.0, 0.0))
            if state.get("advantage_signature") == self._advantage_signature()
            else (0.0, 0.0)
        )
        return states

    def _advantage_signature(self):
        d = self.cfg.distillation
        return (d.loss, d.alpha, d.topk, d.add_tail, self.cfg.algorithm.temperature)


PolicyWorker = ray.remote(num_gpus=1)(SDPOPolicyWorker)

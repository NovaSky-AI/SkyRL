import os
from dataclasses import asdict
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import megatron.core.parallel_state as mpu
import torch
import torch.nn as nn
from megatron.core.distributed import finalize_model_grads
from megatron.core.pipeline_parallel import get_forward_backward_func
from omegaconf import OmegaConf

from skyrl.backends.skyrl_train.distributed.megatron.megatron_utils import (
    get_model_config,
    make_batch_generator,
    postprocess_packed_seqs,
    preprocess_packed_seqs,
    recover_left_padding,
    remove_left_padding,
)
from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
    from_parallel_logits_to_logprobs,
    vocab_parallel_entropy,
)
from skyrl.backends.skyrl_train.utils.ppo_utils import (
    PolicyLossRegistry,
    compute_approx_kl,
)
from skyrl.backends.skyrl_train.utils.replay_utils import (
    setup_per_microbatch_replay_backward,
    setup_per_microbatch_replay_forward,
)
from skyrl.backends.skyrl_train.mtp.adapter import project_mtp_hidden_to_logits
from skyrl.backends.skyrl_train.mtp.hidden_capture import maybe_capture_mtp_hidden
from skyrl.backends.skyrl_train.mtp.soft_ce import (
    build_teacher_logits,
    draft_hard_ce,
    draft_soft_ce,
    draft_soft_ce_topk,
    shift_mask_for_mtp,
)
from skyrl.backends.skyrl_train.utils.torch_utils import (
    build_mtp_next_token_labels,
    masked_mean,
)
from skyrl.train.config import TrainerConfig


class MegatronModelWrapper:
    def __init__(
        self,
        config: TrainerConfig,
        actor_module: List[nn.Module],
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
        policy_loss_fn: Optional[Callable] = None,
    ):
        self.cfg = config
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.policy_loss_fn = policy_loss_fn
        self.remove_microbatch_padding = self.cfg.remove_microbatch_padding

        config = get_model_config(self.actor_module[0])
        # This is set to None by default: https://github.com/NVIDIA/Megatron-LM/blob/07b22a05136a3cb08ece05f7de38cf6aeeb165fb/megatron/core/model_parallel_config.py#L95
        # use the build in finalize_model_grads function to all reduce gradients across parallelism dimensions
        config.finalize_model_grads_func = finalize_model_grads
        # Wire up the optimizer's loss scaler so Megatron's pipeline schedule can scale
        # the loss before backward (critical for fp16 dynamic loss scaling, MoE aux loss
        # scaling, and any explicit loss_scale configuration).
        if actor_optimizer is not None:
            config.grad_scale_func = actor_optimizer.scale_loss

    def train(self):
        [module.train() for module in self.actor_module]

    def eval(self):
        [module.eval() for module in self.actor_module]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        micro_batches: List[dict],
        seq_len: int,
        micro_batch_size: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Forward-only inference to compute log-probs over a full mini-batch consisting of multiple micro-batches.

        Args:
            micro_batches: List of micro-batch dicts with keys: "sequences", "attention_mask", "position_ids",
                           and "num_actions".
            seq_len: Padded sequence length per sample.
            micro_batch_size: Per-micro-batch size.
            temperature: Optional temperature scaling for logits.

        Returns:
            torch.Tensor of concatenated log-probs across micro-batches (valid on pipeline last stage only).
        """
        forward_backward_func = get_forward_backward_func()

        def collection_func(logits, data):
            sequences = data["sequences"]
            tp_grp = mpu.get_tensor_model_parallel_group()
            tp_rank = mpu.get_tensor_model_parallel_rank()

            if temperature != 1.0:
                logits.div_(temperature)

            token_logprobs = from_parallel_logits_to_logprobs(
                logits,
                sequences,
                vocab_start_index=tp_rank * logits.shape[-1],
                vocab_end_index=(tp_rank + 1) * logits.shape[-1],
                tp_group=tp_grp,
                inference_only=True,
                cp_group=None,  # we handle cp gathering in `postprocess_packed_seqs`
                chunk_size=self.cfg.logprobs_chunk_size,  # chunk seq dim to bound peak memory
            )
            return torch.tensor(0.0, device=token_logprobs.device), {"log_probs": token_logprobs}

        def forward_step(batch_iter, model):
            batch = next(batch_iter)

            rollout_expert_indices = batch.pop("rollout_expert_indices", None)
            if rollout_expert_indices is not None:
                setup_per_microbatch_replay_forward(
                    rollout_expert_indices,
                    batch["attention_mask"],
                    model_config=get_model_config(model),
                    remove_microbatch_padding=self.remove_microbatch_padding,
                )

            sequences = batch["sequences"]
            attention_mask = batch["attention_mask"].to(bool)
            position_ids = batch["position_ids"]

            if self.remove_microbatch_padding:
                new_sequences, packed_seq_params = preprocess_packed_seqs(
                    sequences,
                    attention_mask,
                    pre_process=mpu.is_pipeline_first_stage(ignore_virtual=True),
                )
                new_attention_mask = None
                new_position_ids = None
            else:
                new_sequences, new_attention_mask, new_position_ids = remove_left_padding(
                    sequences,
                    attention_mask,
                    position_ids,
                    pre_process=mpu.is_pipeline_first_stage(ignore_virtual=True),
                )
                packed_seq_params = None

            outputs = model(
                new_sequences,
                new_position_ids,
                new_attention_mask,
                packed_seq_params=packed_seq_params,
            )

            if self.remove_microbatch_padding:
                outputs = postprocess_packed_seqs(
                    outputs,
                    packed_seq_params,
                    attention_mask,
                    micro_batch_size,
                    seq_len,
                    post_process=mpu.is_pipeline_last_stage(ignore_virtual=True),
                )
            else:
                outputs = recover_left_padding(
                    outputs,
                    new_attention_mask,
                    attention_mask,
                    seq_len,
                    post_process=mpu.is_pipeline_last_stage(ignore_virtual=True),
                )

            return outputs, partial(collection_func, data=batch)

        batch_generator = make_batch_generator(micro_batches, vpp_size=len(self.actor_module))

        output = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=self.actor_module,
            num_microbatches=len(micro_batches),
            seq_length=seq_len,
            micro_batch_size=micro_batch_size,
            forward_only=True,
        )

        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            log_probs = [o["log_probs"] for o in output]
            log_probs = torch.cat(log_probs, dim=0)
            # take last num_actions tokens per micro; concatenate later
            # Assume all micros have same num_actions
            num_actions = micro_batches[0]["num_actions"]
            log_probs = log_probs[:, -num_actions:]
        else:
            # return dummy tensor for non-last pp stages
            device = micro_batches[0]["sequences"].device
            log_probs = torch.zeros(size=(1, 1), dtype=torch.bfloat16, device=device)
        return log_probs

    def forward_backward_mini_batch(
        self,
        micro_batches: List[dict],
        seq_len: int,
        micro_batch_size: int,
        temperature: float = 1.0,
        loss_fn: Optional[str] = None,
        loss_fn_config: Optional[Dict[str, Any]] = None,
        forward_only: bool = False,
    ) -> List[dict]:
        """
        Run forward-backward over a full mini-batch consisting of multiple micro-batches.

        Args:
            micro_batches: A list of micro-batch dicts. Each dict must contain keys:
                "sequences", "attention_mask", "position_ids", "num_actions",
                "old_action_log_probs", "base_action_log_probs", "advantages",
                "loss_mask", "rollout_action_logprobs".
            seq_len: Sequence length (tokens) per sample (assumed same across micros after padding).
            micro_batch_size: Micro-batch size per forward pass.
            temperature: Optional temperature for logits scaling.
            loss_fn: Optional loss function name (e.g., "cross_entropy", "ppo").
                     If provided, overrides the config's policy_loss_type.
            loss_fn_config: Optional config overrides for the loss function.
            forward_only: If True, run the forward pass without backward (no gradients).
                          Useful for evaluation / loss-only inference paths (e.g., SFT
                          ``forward(loss_fn=...)`` codepath).

        Returns:
            List[dict]: one metrics dict per micro-batch in order.
        """
        forward_backward_func = get_forward_backward_func()

        # ------------------------------------------------------------------
        # Multi-Token Prediction (MTP) — decoupled draft-head training.
        # ------------------------------------------------------------------
        # If the model was built with native MTP heads (policy.megatron_config.mtp_num_layers),
        # train them with an explicit, decoupled loss instead of Megatron's in-forward
        # process_mtp_loss / MTPLossAutoScaler path. This is model-agnostic: any model that exposes a
        # native MTP block works (GPTModel for DeepSeek/GLM/Qwen3-Next, MambaModel for Qwen3.5/
        # NemotronH). We keep the heads running inside the model's forward (so we don't have to
        # reconstruct rotary embeddings) but pass NO labels, so process_mtp_loss short-circuits and no
        # MTP gradient is coupled onto the trunk. A forward hook captures the MTP block's hidden states
        # (with its trunk input optionally detached) and we project + score them ourselves. Only
        # active during training (not forward_only / logprob-only passes).
        model_config = get_model_config(self.actor_module[0])
        mtp_enabled = (not forward_only) and bool(getattr(model_config, "mtp_num_layers", None))
        mcfg = self.cfg.policy.megatron_config
        mtp_loss_type = getattr(mcfg, "mtp_loss_type", "soft_ce")
        mtp_loss_weight = float(getattr(mcfg, "mtp_loss_weight", 0.1))
        mtp_detach_trunk = bool(getattr(mcfg, "mtp_detach_trunk", True))
        mtp_detach_shared_output = bool(getattr(mcfg, "mtp_detach_shared_output", False))
        mtp_loss_chunk_size = getattr(mcfg, "mtp_loss_chunk_size", 1024)
        mtp_loss_topk = getattr(mcfg, "mtp_loss_topk", None)

        if os.environ.get("MTP_DEBUG"):
            from megatron.core.utils import unwrap_model as _uw

            from skyrl.backends.skyrl_train.mtp.hidden_capture import _resolve_mtp_host

            _gm = _uw(self.actor_module[0])
            _host = _resolve_mtp_host(_gm)
            _lm = getattr(_gm, "language_model", None)
            print(
                f"[MTP_DEBUG] forward_only={forward_only} mtp_enabled={mtp_enabled} "
                f"model_config.mtp_num_layers={getattr(model_config, 'mtp_num_layers', 'MISSING')} "
                f"unwrapped_type={type(_gm).__name__} top_has_mtp={getattr(_gm, 'mtp', None) is not None} "
                f"has_language_model={_lm is not None} "
                f"lm_has_mtp={getattr(_lm, 'mtp', None) is not None if _lm is not None else 'NA'} "
                f"resolved_host={type(_host).__name__} host_mtp_is_none={getattr(_host, 'mtp', None) is None} "
                f"host_mtp_process={getattr(_host, 'mtp_process', 'MISSING')}",
                flush=True,
            )

        # Resolve loss function
        resolved_loss_name = loss_fn if loss_fn is not None else self.cfg.algorithm.policy_loss_type
        if loss_fn is not None:
            current_loss_fn = PolicyLossRegistry.get(loss_fn)
        else:
            current_loss_fn = self.policy_loss_fn

        # Build config for loss function, applying any overrides
        loss_config = self.cfg.algorithm
        if loss_fn_config is not None:

            new_loss_config = OmegaConf.merge(OmegaConf.create(asdict(loss_config)), OmegaConf.create(loss_fn_config))
            # NOTE: users can provide a custom loss config class, so we need to use the same class after applying overrides
            loss_config = type(loss_config).from_dict_config(new_loss_config)

        def loss_func(logits, data):
            sequences = data["sequences"]
            num_actions = data["num_actions"]
            old_action_log_probs = data["old_action_log_probs"]
            base_action_log_probs = data["base_action_log_probs"]
            advantages = data["advantages"]
            loss_mask = data["loss_mask"]
            rollout_action_logprobs = data["rollout_action_logprobs"]
            action_mask = data.get("action_mask")
            num_microbatches = data.get("num_microbatches")

            dp_size = mpu.get_data_parallel_world_size(with_context_parallel=True)
            tp_grp = mpu.get_tensor_model_parallel_group()
            tp_rank = mpu.get_tensor_model_parallel_rank()

            # temperature normalization
            if temperature != 1.0:
                logits.div_(temperature)

            token_logprobs = from_parallel_logits_to_logprobs(
                logits,
                sequences,
                vocab_start_index=tp_rank * logits.shape[-1],
                vocab_end_index=(tp_rank + 1) * logits.shape[-1],
                tp_group=tp_grp,
                inference_only=False,
                cp_group=None,  # we handle cp gathering in `postprocess_packed_seqs`
                chunk_size=self.cfg.logprobs_chunk_size,  # chunk seq dim to bound peak memory
            )

            action_log_probs = token_logprobs[:, -num_actions:]

            # policy loss should be calculated based on the selected token logprobs
            policy_loss, loss_metrics = current_loss_fn(
                action_log_probs,
                old_action_log_probs,
                advantages,
                config=loss_config,
                loss_mask=loss_mask,
                rollout_logprobs=rollout_action_logprobs,
            )

            # --- Decoupled MTP / draft loss -------------------------------------------------
            # Score the (detached-input) MTP head against the policy's own next-token
            # distribution (soft CE) or the ground-truth future tokens (hard CE), over every real
            # token. Returns a local masked-mean scalar that is folded into the loss below with the
            # same reduction treatment as the KL/entropy aux terms.
            draft_loss = None
            student_logits_list = data.get("mtp_student_logits")
            if mtp_enabled and student_logits_list:
                draft_mask = data["attention_mask"].to(logits.dtype)  # [batch, seq_len]
                vocab_size_tp = logits.shape[-1]
                # Undo the in-place temperature scaling so the teacher is the true policy
                # distribution (student logits are produced unscaled).
                teacher_src = logits if temperature == 1.0 else logits * temperature
                hard_labels = build_mtp_next_token_labels(sequences) if mtp_loss_type == "hard_ce" else None

                per_layer_losses = []
                for layer_idx, student_logits in enumerate(student_logits_list):
                    layer_mask = shift_mask_for_mtp(draft_mask, layer_idx)
                    if mtp_loss_type == "hard_ce":
                        layer_labels = torch.roll(hard_labels, shifts=-(layer_idx + 1), dims=1)
                        per_layer_losses.append(
                            draft_hard_ce(
                                student_logits,
                                layer_labels,
                                layer_mask,
                                vocab_parallel_group=tp_grp,
                                vocab_start_index=tp_rank * vocab_size_tp,
                                chunk_size=mtp_loss_chunk_size,
                            )
                        )
                    else:
                        if mtp_loss_topk:
                            # Top-k draft loss: O(seq*k) memory, no full-vocab softmax (fits + avoids
                            # fragmentation at large vocab). Reconciled across the TP group, so it
                            # scales to any tensor-parallel size incl. cross-node. Pass the *un-rolled*
                            # policy logits + roll_shift so top-k runs on the policy's own logits (no
                            # ~[S, vocab] rolled-teacher copy); only the small [B, S, k] top-k is rolled.
                            per_layer_losses.append(
                                draft_soft_ce_topk(
                                    student_logits,
                                    teacher_src,
                                    layer_mask,
                                    k=mtp_loss_topk,
                                    vocab_parallel_group=tp_grp,
                                    roll_shift=layer_idx + 1,
                                )
                            )
                        else:
                            teacher_logits = build_teacher_logits(teacher_src, layer_idx, detach=True)
                            per_layer_losses.append(
                                draft_soft_ce(
                                    student_logits,
                                    teacher_logits,
                                    layer_mask,
                                    vocab_parallel_group=tp_grp,
                                    chunk_size=mtp_loss_chunk_size,
                                )
                            )
                draft_loss = torch.stack(per_layer_losses).mean() 
                # Drop the dict's reference, this microbatch's autograd graph still holds the
                # tensor for its own backward, after which it is freed instead of lingering.
                del data["mtp_student_logits"]
                student_logits_list = None

            # SFT path: cross_entropy loss (negative log likelihood)
            if resolved_loss_name == "cross_entropy":
                loss = policy_loss
                if draft_loss is not None:
                    # Both terms are per-token means here; Megatron's /num_microbatches and the DDP
                    # DP+CP averaging reduce them consistently.
                    loss = loss + mtp_loss_weight * draft_loss

                # Compute elementwise loss for Tinker API (per-token NLL)
                with torch.no_grad():
                    elementwise_loss = -action_log_probs
                    if loss_mask is not None:
                        elementwise_loss = elementwise_loss * loss_mask

                # Build per-sequence loss_fn_outputs.
                # Compute valid_lens vectorized on GPU, then move tensors to CPU
                # exactly once before iterating in Python — avoids ~3N GPU->CPU
                # syncs per micro-batch (item()/cpu()/tolist() inside the loop).
                batch_size = action_log_probs.shape[0]
                seq_len = action_log_probs.shape[1]
                if action_mask is not None:
                    valid_lens_t = action_mask.sum(dim=-1).long()
                elif loss_mask is not None:
                    valid_lens_t = loss_mask.sum(dim=-1).long()
                else:
                    valid_lens_t = torch.full((batch_size,), seq_len, device=action_log_probs.device, dtype=torch.long)

                # Bulk GPU->CPU sync: one transfer for logprobs, elementwise_loss, and valid_lens.
                action_log_probs_cpu = action_log_probs.detach().cpu()
                elementwise_loss_cpu = elementwise_loss.detach().cpu()
                valid_lens = valid_lens_t.cpu().tolist()

                loss_fn_outputs = []
                for i in range(batch_size):
                    valid_len = valid_lens[i]
                    loss_fn_outputs.append(
                        {
                            "logprobs": (action_log_probs_cpu[i, -valid_len:].tolist() if valid_len > 0 else []),
                            "elementwise_loss": (
                                elementwise_loss_cpu[i, -valid_len:].tolist() if valid_len > 0 else []
                            ),
                        }
                    )

                metrics = {
                    "loss": loss.item(),
                    "response_length": num_actions,
                    "loss_fn_outputs": loss_fn_outputs,
                }
                if draft_loss is not None:
                    metrics["mtp_loss"] = draft_loss.detach().item()
                return loss, metrics

            # RL path: add optional KL/entropy terms
            # entropy loss
            with torch.set_grad_enabled(loss_config.use_entropy_loss):
                action_logits = logits[:, -num_actions - 1 : -1, :]
                entropy_BS = vocab_parallel_entropy(action_logits)
                entropy = masked_mean(entropy_BS, loss_mask)

            if loss_config.use_entropy_loss:
                entropy_loss_term = entropy * loss_config.entropy_loss_coef
            else:
                entropy_loss_term = torch.tensor(0.0)

            if loss_config.use_kl_loss:
                kl_loss = compute_approx_kl(
                    action_log_probs,
                    base_action_log_probs,
                    loss_mask=loss_mask,
                    kl_estimator_type=loss_config.kl_estimator_type,
                )
                kl_loss = masked_mean(kl_loss, loss_mask, dim=-1).mean()
            else:
                kl_loss = torch.tensor(0.0)
            kl_loss_term = kl_loss * loss_config.kl_loss_coef

            # Policy losses are pre-scaled to achieve the correct loss_reduction
            # when summing across the entire minibatch (see `apply_loss_reduction_to_advantages_minibatch`).
            # Megatron divides loss by num_microbatches
            # (https://github.com/NVIDIA/Megatron-LM/blob/core_v0.15.2/megatron/core/pipeline_parallel/schedules.py#L248)
            # and the data parallel all-reduce averages gradients across dp_size (including CP ranks)
            # (https://github.com/NVIDIA/Megatron-LM/blob/core_v0.15.2/megatron/core/distributed/distributed_data_parallel.py#L285)
            # so we multiply by both factors to recover the correct sum reduction.
            grad_sum_correction_factor = num_microbatches * dp_size

            # NOTE: The KL and entropy loss terms are not pre-scaled,
            # so we just average them across microbatches and DP workers.
            # Megatron's DDP averages gradients across the full DP+CP group,
            # but KL/entropy should only be averaged across DP (not CP).
            # Multiply by cp_size to counteract the unwanted CP averaging.
            cp_size = mpu.get_context_parallel_world_size()
            loss = policy_loss * grad_sum_correction_factor + (kl_loss_term - entropy_loss_term) * cp_size
            # The decoupled MTP/draft loss is a per-token mean (like KL/entropy), so fold it in with
            # the same cp_size correction. Its gradient only reaches the MTP-head parameters (and the
            # shared output/embedding unless mtp_detach_shared_output) because both the trunk hidden
            # states and the teacher distribution are detached.
            if draft_loss is not None:
                loss = loss + mtp_loss_weight * draft_loss * cp_size
            unscaled_loss = loss / grad_sum_correction_factor

            # Build per-sequence loss_fn_outputs with logprobs.
            batch_size = action_log_probs.shape[0]
            seq_len = action_log_probs.shape[1]

            if action_mask is not None:
                valid_lens = action_mask.sum(dim=1).int().tolist()
            elif loss_mask is not None:
                valid_lens = loss_mask.sum(dim=1).int().tolist()
            else:
                valid_lens = [seq_len] * batch_size

            detached_log_probs = action_log_probs.detach().cpu()
            loss_fn_outputs = []
            for i, valid_len in enumerate(valid_lens):
                loss_fn_outputs.append(
                    {
                        "logprobs": detached_log_probs[i, -valid_len:].tolist() if valid_len > 0 else [],
                    }
                )

            metrics = {
                "final_loss": unscaled_loss.detach().item(),
                "policy_loss": policy_loss.detach().item(),
                "policy_entropy": entropy.detach().item(),
                "policy_kl": kl_loss.detach().item(),
                "loss_fn_outputs": loss_fn_outputs,
            }
            if draft_loss is not None:
                metrics["mtp_loss"] = draft_loss.detach().item()
            for k, v in loss_metrics.items():
                metrics["loss_metrics/" + k] = v
            return loss, metrics

        def forward_step(batch_iter, model):
            # NOTE(Charlie): despite the name, methods like `remove_left_padding()` are padding-agnostic
            # (can be left, or right) as it uses attention_mask to locate real tokens. Same thing
            # for recover_left_padding and setup_per_microbatch_replay_forward. Especially relevant
            # after this PR https://github.com/NovaSky-AI/SkyRL/pull/1285.
            batch = next(batch_iter)

            rollout_expert_indices = batch.pop("rollout_expert_indices", None)
            if rollout_expert_indices is not None:
                setup_per_microbatch_replay_forward(
                    rollout_expert_indices,
                    batch["attention_mask"],
                    model_config=get_model_config(model),
                    remove_microbatch_padding=self.remove_microbatch_padding,
                )

            sequences = batch["sequences"]
            attention_mask = batch["attention_mask"].to(bool)
            position_ids = batch["position_ids"]

            if self.remove_microbatch_padding:
                new_sequences, packed_seq_params = preprocess_packed_seqs(
                    sequences,
                    attention_mask,
                    pre_process=mpu.is_pipeline_first_stage(ignore_virtual=True),
                )
                new_attention_mask = None
                # The trunk ignores position_ids for RoPE + THD packing (rotary comes from
                # packed_seq_params), so SkyRL normally passes None. But the native MTP block rolls
                # and re-embeds position_ids per depth, so when MTP is active we must supply them in
                # the packed layout. This is harmless to the main logits.
                if mtp_enabled:
                    new_position_ids = preprocess_packed_seqs(
                        position_ids,
                        attention_mask,
                        pre_process=mpu.is_pipeline_first_stage(ignore_virtual=True),
                    )[0]
                else:
                    new_position_ids = None
            else:
                new_sequences, new_attention_mask, new_position_ids = remove_left_padding(
                    sequences,
                    attention_mask,
                    position_ids,
                    pre_process=mpu.is_pipeline_first_stage(ignore_virtual=True),
                )
                packed_seq_params = None

            is_last_stage = mpu.is_pipeline_last_stage(ignore_virtual=True)

            def depad(tensor):
                """Recover [batch, seq_len, ...] padded layout from the internal layout,
                matching exactly how the main logits are de-padded below."""
                if self.remove_microbatch_padding:
                    return postprocess_packed_seqs(
                        tensor,
                        packed_seq_params,
                        attention_mask,
                        micro_batch_size,
                        seq_len,
                        post_process=is_last_stage,
                    )
                return recover_left_padding(
                    tensor,
                    new_attention_mask,
                    attention_mask,
                    seq_len,
                    post_process=is_last_stage,
                )

            # Run the policy forward. When MTP is active, a pre-hook records the native MTP block's
            # arguments (we pass NO labels, so the model's process_mtp_loss short-circuits and the
            # main logits stay coupled to the trunk; MTPLossAutoScaler never runs).
            student_hidden = None
            student_model = None
            with maybe_capture_mtp_hidden(model, mtp_enabled, detach_trunk=mtp_detach_trunk) as capture:
                outputs = model(
                    new_sequences,
                    new_position_ids,
                    new_attention_mask,
                    packed_seq_params=packed_seq_params,
                )
                # Replay the MTP block on *detached* trunk hidden states (decoupled draft forward)
                # while still inside the capture context (so the MTP block stays in eval mode).
                if mtp_enabled and capture is not None:
                    student_hidden = capture.compute_student_hidden_states()
                    student_model = capture.model

            outputs = depad(outputs)

            # Project the decoupled MTP hidden states through the shared output layer and de-pad into
            # the same [batch, seq_len, vocab/tp] layout as the main logits, so the draft loss can be
            # scored against the policy's own distribution (or hard labels) in loss_func.
            if student_hidden is not None:
                student_logits = project_mtp_hidden_to_logits(
                    student_hidden, student_model, detach_output_weight=mtp_detach_shared_output
                )
                batch["mtp_student_logits"] = [depad(sl) for sl in student_logits]

            if rollout_expert_indices is not None:
                setup_per_microbatch_replay_backward()

            return outputs, partial(loss_func, data=batch)

        # batch should be a list of micro-batches
        batch_generator = make_batch_generator(micro_batches, vpp_size=len(self.actor_module))

        metrics_list = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=self.actor_module,
            num_microbatches=len(micro_batches),
            seq_length=seq_len,
            micro_batch_size=micro_batch_size,
            forward_only=forward_only,
        )

        # The decoupled MTP/draft loss is computed and logged per-microbatch inside loss_func
        # (metric key "mtp_loss"); no MTPLossLoggingHelper plumbing is needed.

        # broadcast metrics to all pp ranks
        if not mpu.is_pipeline_last_stage(ignore_virtual=True):
            metrics_list = [None] * len(micro_batches)
        with torch.no_grad():
            torch.distributed.broadcast_object_list(
                metrics_list,
                src=mpu.get_pipeline_model_parallel_last_rank(),
                group=mpu.get_pipeline_model_parallel_group(),
            )

        return metrics_list

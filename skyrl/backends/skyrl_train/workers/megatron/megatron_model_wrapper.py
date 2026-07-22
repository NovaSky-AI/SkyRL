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
    model_packs_sequences_internally,
    preprocess_packed_seqs,
    recover_left_padding,
    remove_left_padding,
    to_te_attention_mask,
)
from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
    _fused_vocab_parallel_entropy_from_hidden,
    from_parallel_hidden_to_entropy_packed_sequences,
    from_parallel_hidden_to_logprobs,
    from_parallel_hidden_to_logprobs_packed_sequences,
    from_parallel_logits_to_logprobs,
    from_parallel_logits_to_logprobs_packed_sequences,
    vocab_parallel_entropy,
    vocab_parallel_entropy_packed_sequences,
)
from skyrl.backends.skyrl_train.distributed.megatron.packing_utils import is_fp8_enabled
from skyrl.backends.skyrl_train.utils.ppo_utils import (
    PolicyLossRegistry,
    compute_approx_kl,
)
from skyrl.backends.skyrl_train.utils.replay_utils import (
    setup_per_microbatch_replay_backward,
    setup_per_microbatch_replay_forward,
)
from skyrl.backends.skyrl_train.utils.torch_utils import masked_mean
from skyrl.backends.skyrl_train.workers.worker_utils import (
    compute_minibatch_rollout_logprob_diff_metrics,
)
from skyrl.train.config import TrainerConfig


def _build_packed_targets(
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    packed_seq_params,
    sub_seq_lengths: Optional[list[list[int]]] = None,
) -> torch.Tensor:
    """Pack full target token IDs without context-parallel sharding."""
    cu_padded = packed_seq_params.cu_seqlens_q_padded.to(device=sequences.device, dtype=torch.long)
    total_padded_tokens = int(cu_padded[-1].item())

    targets = torch.zeros((total_padded_tokens,), dtype=sequences.dtype, device=sequences.device)
    if sub_seq_lengths is not None:
        cu_padded_cpu = cu_padded.detach().cpu().tolist()
        seg_idx = 0
        for row_idx, row_lens in enumerate(sub_seq_lengths):
            row_offset = 0
            for seq_len in row_lens:
                seq_len = int(seq_len)
                if seg_idx + 1 >= len(cu_padded_cpu):
                    raise ValueError("sub_seq_lengths contains more sub-sequences than packed_seq_params")
                packed_start = cu_padded_cpu[seg_idx]
                targets[packed_start : packed_start + seq_len] = sequences[row_idx, row_offset : row_offset + seq_len]
                row_offset += cu_padded_cpu[seg_idx + 1] - cu_padded_cpu[seg_idx]
                seg_idx += 1
        if seg_idx != len(cu_padded_cpu) - 1:
            raise ValueError(
                f"sub_seq_lengths describes {seg_idx} sub-sequences, "
                f"but packed_seq_params describes {len(cu_padded_cpu) - 1}"
            )
        return targets.unsqueeze(0)

    attention_mask = attention_mask.to(device=sequences.device, dtype=torch.bool)
    token_offsets = attention_mask.to(torch.long).cumsum(dim=1) - 1
    packed_indices = cu_padded[:-1].unsqueeze(1) + token_offsets
    targets[packed_indices[attention_mask]] = sequences[attention_mask]
    return targets.unsqueeze(0)


def _fused_lm_head_output_processor(**kwargs):
    """GPTModel ``output_processor`` hook for the fused LM-head log-prob path.

    Skips the output-layer matmul (so the [S, B, vocab//TP] logits are never
    built), returns the decoder hidden states in [b, s, h] layout (the same
    layout the default logits path returns), and stashes the resolved
    output-layer weight into the caller-provided ``context`` dict so the fused
    log-prob / entropy can run downstream with it.
    """
    hidden_states = kwargs["hidden_states"]
    output_layer = kwargs["output_layer"]
    ctx = kwargs.get("context")
    if ctx is not None:
        output_weight = kwargs.get("output_weight")
        ctx["lm_head_weight"] = output_weight if output_weight is not None else output_layer.weight
    # With sequence parallelism the decoder hidden states are sharded along the
    # sequence dim; the ColumnParallelLinear output layer all-gathers them before
    # projecting (megatron tensor_parallel/layers.py). We skip that layer, so
    # replicate the gather here. tensor_parallel_output_grad=True makes the
    # backward reduce-scatter the hidden grad across TP ranks — exactly the sum
    # of each rank's vocab-slice grad_hidden that the fused op produces.
    if getattr(output_layer, "sequence_parallel", False):
        from megatron.core.tensor_parallel import gather_from_sequence_parallel_region

        hidden_states = gather_from_sequence_parallel_region(hidden_states, tensor_parallel_output_grad=True)
    # [s, b, h] -> [b, s, h], matching `logits.transpose(0, 1)` in the default path.
    return hidden_states.transpose(0, 1).contiguous()


class MegatronModelWrapper:
    def __init__(
        self,
        config: TrainerConfig,
        actor_module: List[nn.Module],
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
        policy_loss_fn: Optional[Callable] = None,
        is_vlm: bool = False,
    ):
        self.cfg = config
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.policy_loss_fn = policy_loss_fn
        self.remove_microbatch_padding = self.cfg.remove_microbatch_padding
        self.is_vlm = is_vlm
        # Fuse the LM-head projection into the chunked log-prob/entropy via the
        # GPTModel output_processor hook (avoids materializing the full
        # [B, S, vocab//TP] logits + its fp32 grad). See model_utils.
        self._fused_lm_head = bool(getattr(self.cfg, "fused_lm_head_logprob", False))
        self._fused_lm_head_backend = getattr(self.cfg, "fused_lm_head_logprob_backend", "torch")
        # Some models (e.g. Qwen3.5 via the VL bridge -> Qwen3VLModel) pack
        # sequences inside their own forward; SkyRL sample packing would then
        # double-pack and corrupt the GDN cu_seqlens, so refuse it. For Qwen3.5,
        # use language_model_only=True (native GPTModel GDN path) to pack.
        if self.remove_microbatch_padding and model_packs_sequences_internally(self.actor_module):
            raise ValueError(
                "remove_microbatch_padding=True (sample packing) is not supported for models that "
                "pack sequences inside their own forward (e.g. the Qwen3.5 VL Qwen3VLModel): it "
                "double-packs and corrupts the GatedDeltaNet cu_seqlens. Set "
                "trainer.policy.language_model_only=True to route Qwen3.5 to the native GPTModel GDN "
                "packing path, or set trainer.remove_microbatch_padding=False."
            )

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

    def _assert_vlm_supported(self):
        """Guard the VLM parallelism constraints carried over from the FSDP path.

        3D RoPE and multimodal token positions make sample/microbatch packing,
        context parallelism, and sequence parallelism unsafe for VLMs today.
        """
        assert not self.remove_microbatch_padding, "VLM + microbatch padding removal unsupported"
        assert mpu.get_context_parallel_world_size() == 1, "VLM + context parallelism unsupported"
        assert (
            mpu.get_tensor_model_parallel_world_size() == 1 or self.cfg.policy.sequence_parallel_size == 1
        ), "VLM + sequence parallelism unsupported"

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
        if self.is_vlm:
            self._assert_vlm_supported()
        forward_backward_func = get_forward_backward_func()

        def collection_func(logits, data):
            sequences = data["sequences"]
            packed_seq_params = data.get("packed_seq_params")
            packed_targets = data.get("packed_targets")
            tp_grp = mpu.get_tensor_model_parallel_group()
            tp_rank = mpu.get_tensor_model_parallel_rank()

            # Fused LM-head: `logits` is actually decoder hidden states [B, S, H]
            # (the output_processor skipped the projection); fold the LM-head into
            # the chunked log-prob so this forward-only pass never materializes the
            # full [B, S, vocab//TP] logits (which would OOM at long context).
            fused_lm_head = self._fused_lm_head and data.get("lm_head_weight") is not None
            lm_head_weight = data.get("lm_head_weight")
            if fused_lm_head:
                _v_local = int(lm_head_weight.shape[0])
                fused_vocab_start, fused_vocab_end = tp_rank * _v_local, (tp_rank + 1) * _v_local

            # temperature normalization (the fused path applies it inside the op)
            if temperature != 1.0 and not fused_lm_head:
                logits.div_(temperature)

            if fused_lm_head and packed_seq_params is not None and packed_targets is not None:
                token_logprobs = from_parallel_hidden_to_logprobs_packed_sequences(
                    logits,  # decoder hidden states [1, T, H]
                    lm_head_weight,
                    packed_targets,
                    packed_seq_params.cu_seqlens_q_padded,
                    sequences.shape[1],
                    vocab_start_index=fused_vocab_start,
                    vocab_end_index=fused_vocab_end,
                    group=tp_grp,
                    inference_only=True,
                    cp_group=mpu.get_context_parallel_group(),
                    chunk_size=self.cfg.logprobs_chunk_size,
                    attention_mask=data["attention_mask"],
                    sub_seq_lengths=data.get("sub_seq_lengths_list"),
                    temperature=temperature,
                    fused_backend=self._fused_lm_head_backend,
                )
            elif fused_lm_head:
                token_logprobs = from_parallel_hidden_to_logprobs(
                    logits,  # decoder hidden states [B, S, H]
                    lm_head_weight,
                    sequences,
                    vocab_start_index=fused_vocab_start,
                    vocab_end_index=fused_vocab_end,
                    tp_group=tp_grp,
                    inference_only=True,
                    cp_group=None,
                    chunk_size=self.cfg.logprobs_chunk_size,
                    temperature=temperature,
                    fused_backend=self._fused_lm_head_backend,
                )
            elif packed_seq_params is not None and packed_targets is not None:
                token_logprobs = from_parallel_logits_to_logprobs_packed_sequences(
                    logits,
                    packed_targets,
                    packed_seq_params.cu_seqlens_q_padded,
                    sequences.shape[1],
                    vocab_start_index=tp_rank * logits.shape[-1],
                    vocab_end_index=(tp_rank + 1) * logits.shape[-1],
                    group=tp_grp,
                    inference_only=True,
                    cp_group=mpu.get_context_parallel_group(),
                    chunk_size=self.cfg.logprobs_chunk_size,
                    attention_mask=data["attention_mask"],
                    sub_seq_lengths=data.get("sub_seq_lengths_list"),
                )
            else:
                token_logprobs = from_parallel_logits_to_logprobs(
                    logits,
                    sequences,
                    vocab_start_index=tp_rank * logits.shape[-1],
                    vocab_end_index=(tp_rank + 1) * logits.shape[-1],
                    tp_group=tp_grp,
                    inference_only=True,
                    cp_group=None,
                    chunk_size=self.cfg.logprobs_chunk_size,  # chunk seq dim to bound peak memory
                )
            return torch.tensor(0.0, device=token_logprobs.device), {"log_probs": token_logprobs}

        def forward_step(batch_iter, model):
            batch = next(batch_iter)

            model_config = get_model_config(model)
            fp8_enabled = is_fp8_enabled(getattr(model_config, "fp8", None))
            rollout_expert_indices = batch.pop("rollout_expert_indices", None)
            if rollout_expert_indices is not None:
                setup_per_microbatch_replay_forward(
                    rollout_expert_indices,
                    batch["attention_mask"],
                    model_config=model_config,
                    remove_microbatch_padding=self.remove_microbatch_padding,
                )

            sequences = batch["sequences"]
            attention_mask = batch["attention_mask"].to(bool)
            position_ids = batch["position_ids"]
            sub_seq_lengths_field = batch.get("sub_seq_lengths")
            sub_seq_lengths = [t.tolist() for t in sub_seq_lengths_field] if sub_seq_lengths_field is not None else None
            batch["sub_seq_lengths_list"] = sub_seq_lengths

            vlm_inputs = {}
            if batch.get("pixel_values") is not None and mpu.get_pipeline_model_parallel_rank() == 0:
                vlm_inputs["pixel_values"] = torch.cat(batch["pixel_values"].tensors, dim=0)
            if batch.get("image_grid_thw") is not None:
                vlm_inputs["image_grid_thw"] = torch.cat(batch["image_grid_thw"].tensors, dim=0)

            if self.remove_microbatch_padding:
                new_sequences, packed_seq_params = preprocess_packed_seqs(
                    sequences,
                    attention_mask,
                    pre_process=mpu.is_pipeline_first_stage(ignore_virtual=True) or self.is_vlm,
                    sub_seq_lengths=sub_seq_lengths,
                    fp8_enabled=fp8_enabled,
                )
                batch["packed_seq_params"] = packed_seq_params
                batch["packed_targets"] = _build_packed_targets(
                    sequences, attention_mask, packed_seq_params, sub_seq_lengths=sub_seq_lengths
                )
                new_attention_mask = None
                new_position_ids = None
            else:
                new_sequences, new_attention_mask, new_position_ids = remove_left_padding(
                    sequences,
                    attention_mask,
                    position_ids,
                    pre_process=mpu.is_pipeline_first_stage(ignore_virtual=True) or self.is_vlm,
                    fp8_enabled=fp8_enabled,
                )
                packed_seq_params = None
                # Qwen-style VLMs recompute 3D mRoPE positions internally from
                # image_grid_thw and ignore any position_ids passed in.
                if self.is_vlm:
                    new_position_ids = None

            if self._fused_lm_head:
                # Fused LM-head inference: the output_processor returns decoder
                # hidden states (not logits) and stashes the LM-head weight, so
                # collection_func can fold the projection into the chunked
                # log-prob op. Without this, a forward-only ref/old-logprob pass
                # (e.g. PPO reference logprobs) at long context would still
                # materialize the full [B, S, vocab//TP] logits and OOM.
                _op_ctx: dict = {}
                outputs = model(
                    new_sequences,
                    new_position_ids,
                    to_te_attention_mask(new_attention_mask),
                    packed_seq_params=packed_seq_params,
                    output_processor=_fused_lm_head_output_processor,
                    output_processor_context=_op_ctx,
                    **vlm_inputs,
                )
                batch["lm_head_weight"] = _op_ctx.get("lm_head_weight")
            else:
                outputs = model(
                    new_sequences,
                    new_position_ids,
                    to_te_attention_mask(new_attention_mask),
                    packed_seq_params=packed_seq_params,
                    **vlm_inputs,
                )

            if not self.remove_microbatch_padding:
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
        return_per_token_outputs: bool = True,
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
            return_per_token_outputs: When False, skip building per-token
                ``loss_fn_outputs`` when callers read only ``metrics``.

        Returns:
            List[dict]: one metrics dict per micro-batch in order.
        """
        if self.is_vlm:
            self._assert_vlm_supported()
        forward_backward_func = get_forward_backward_func()

        # Resolve loss function
        resolved_loss_name = loss_fn if loss_fn is not None else self.cfg.algorithm.policy_loss_type
        if loss_fn is not None:
            current_loss_fn = PolicyLossRegistry.get(loss_fn)
        else:
            current_loss_fn = self.policy_loss_fn

        # Build config for loss function, applying any overrides
        loss_config = self.cfg.algorithm
        if loss_fn_config:
            new_loss_config = OmegaConf.merge(OmegaConf.create(asdict(loss_config)), OmegaConf.create(loss_fn_config))
            # NOTE: users can provide a custom loss config class, so we need to use the same class after applying overrides
            loss_config = type(loss_config).from_dict_config(new_loss_config)

        def loss_func(logits, data):
            sequences = data["sequences"]
            packed_seq_params = data.get("packed_seq_params")
            packed_targets = data.get("packed_targets")
            num_actions = data["num_actions"]
            old_action_log_probs = data["old_action_log_probs"]
            base_action_log_probs = data["base_action_log_probs"]
            advantages = data["advantages"]
            loss_mask = data["loss_mask"]
            rollout_action_logprobs = data["rollout_action_logprobs"]
            action_mask = data.get("action_mask")
            num_microbatches = data.get("num_microbatches")
            # Number of microbatches carrying real samples (excludes fully-padding
            # microbatches added by token-based batching). Used to normalize the
            # KL/entropy terms over real microbatches only. Falls back to
            # num_microbatches when not provided (no padding microbatches).
            num_real_microbatches = data.get("num_real_microbatches", num_microbatches)

            dp_size = mpu.get_data_parallel_world_size(with_context_parallel=False)
            tp_grp = mpu.get_tensor_model_parallel_group()
            tp_rank = mpu.get_tensor_model_parallel_rank()

            # Fused LM-head: `logits` is actually decoder hidden states [B, S, H]
            # (the output_processor skipped the projection); fold the LM-head into
            # the chunked log-prob/entropy so the full logits tensor + its fp32
            # grad are never materialized.
            fused_lm_head = self._fused_lm_head and data.get("lm_head_weight") is not None
            lm_head_weight = data.get("lm_head_weight")
            if fused_lm_head and loss_config.use_entropy_loss:
                raise NotImplementedError(
                    "fused_lm_head_logprob does not support use_entropy_loss=True "
                    "(the fused entropy is a no-grad metric)."
                )
            if fused_lm_head:
                _v_local = int(lm_head_weight.shape[0])
                fused_vocab_start, fused_vocab_end = tp_rank * _v_local, (tp_rank + 1) * _v_local

            # temperature normalization (the fused path applies it inside the op)
            if temperature != 1.0 and not fused_lm_head:
                logits.div_(temperature)

            if fused_lm_head and packed_seq_params is not None and packed_targets is not None:
                token_logprobs = from_parallel_hidden_to_logprobs_packed_sequences(
                    logits,  # decoder hidden states [1, T, H]
                    lm_head_weight,
                    packed_targets,
                    packed_seq_params.cu_seqlens_q_padded,
                    sequences.shape[1],
                    vocab_start_index=fused_vocab_start,
                    vocab_end_index=fused_vocab_end,
                    group=tp_grp,
                    inference_only=False,
                    cp_group=mpu.get_context_parallel_group(),
                    chunk_size=self.cfg.logprobs_chunk_size,
                    attention_mask=data["attention_mask"],
                    sub_seq_lengths=data.get("sub_seq_lengths_list"),
                    temperature=temperature,
                    fused_backend=self._fused_lm_head_backend,
                )
            elif fused_lm_head:
                token_logprobs = from_parallel_hidden_to_logprobs(
                    logits,  # decoder hidden states [B, S, H]
                    lm_head_weight,
                    sequences,
                    vocab_start_index=fused_vocab_start,
                    vocab_end_index=fused_vocab_end,
                    tp_group=tp_grp,
                    inference_only=False,
                    cp_group=None,
                    chunk_size=self.cfg.logprobs_chunk_size,
                    temperature=temperature,
                    fused_backend=self._fused_lm_head_backend,
                )
            elif packed_seq_params is not None and packed_targets is not None:
                token_logprobs = from_parallel_logits_to_logprobs_packed_sequences(
                    logits,
                    packed_targets,
                    packed_seq_params.cu_seqlens_q_padded,
                    sequences.shape[1],
                    vocab_start_index=tp_rank * logits.shape[-1],
                    vocab_end_index=(tp_rank + 1) * logits.shape[-1],
                    group=tp_grp,
                    inference_only=False,
                    cp_group=mpu.get_context_parallel_group(),
                    chunk_size=self.cfg.logprobs_chunk_size,
                    attention_mask=data["attention_mask"],
                    sub_seq_lengths=data.get("sub_seq_lengths_list"),
                )
            else:
                token_logprobs = from_parallel_logits_to_logprobs(
                    logits,
                    sequences,
                    vocab_start_index=tp_rank * logits.shape[-1],
                    vocab_end_index=(tp_rank + 1) * logits.shape[-1],
                    tp_group=tp_grp,
                    inference_only=False,
                    cp_group=None,
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

            # SFT path: cross_entropy loss (negative log likelihood)
            if resolved_loss_name == "cross_entropy":
                loss = policy_loss

                # Only build per-token outputs for callers that consume them.
                if return_per_token_outputs:
                    # Tinker consumes per-token NLL.
                    with torch.no_grad():
                        elementwise_loss = -action_log_probs
                        if loss_mask is not None:
                            elementwise_loss = elementwise_loss * loss_mask

                    # Trim each sample with one CPU transfer per tensor.
                    batch_size = action_log_probs.shape[0]
                    seq_len = action_log_probs.shape[1]
                    if action_mask is not None:
                        valid_lens_t = action_mask.sum(dim=-1).long()
                    elif loss_mask is not None:
                        valid_lens_t = loss_mask.sum(dim=-1).long()
                    else:
                        valid_lens_t = torch.full(
                            (batch_size,), seq_len, device=action_log_probs.device, dtype=torch.long
                        )

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
                else:
                    loss_fn_outputs = [{} for _ in range(action_log_probs.shape[0])]

                metrics = {
                    "loss": loss.item(),
                    "response_length": num_actions,
                    "loss_fn_outputs": loss_fn_outputs,
                }
                return loss, metrics

            # RL path: add optional KL/entropy terms
            with torch.set_grad_enabled(loss_config.use_entropy_loss):
                if fused_lm_head and packed_seq_params is not None and packed_targets is not None:
                    entropy, entropy_for_loss = from_parallel_hidden_to_entropy_packed_sequences(
                        logits,  # decoder hidden states [1, T, H]
                        lm_head_weight,
                        packed_seq_params.cu_seqlens_q_padded,
                        sequences.shape[1],
                        num_actions,
                        data["attention_mask"],
                        loss_mask,
                        mpu.get_context_parallel_group(),
                        tp_group=tp_grp,
                        sub_seq_lengths=data.get("sub_seq_lengths_list"),
                        chunk_size=self.cfg.logprobs_chunk_size,
                        temperature=temperature,
                    )
                elif fused_lm_head:
                    action_hidden = logits[:, -num_actions - 1 : -1, :]
                    entropy_BS = _fused_vocab_parallel_entropy_from_hidden(
                        action_hidden,
                        lm_head_weight,
                        tp_grp,
                        chunk_size=self.cfg.logprobs_chunk_size,
                        temperature=temperature,
                    )
                    entropy = masked_mean(entropy_BS, loss_mask)
                    entropy_for_loss = entropy
                elif packed_seq_params is not None and packed_targets is not None:
                    entropy, entropy_for_loss = vocab_parallel_entropy_packed_sequences(
                        logits,
                        packed_seq_params.cu_seqlens_q_padded,
                        sequences.shape[1],
                        num_actions,
                        data["attention_mask"],
                        loss_mask,
                        mpu.get_context_parallel_group(),
                        sub_seq_lengths=data.get("sub_seq_lengths_list"),
                    )
                else:
                    action_logits = logits[:, -num_actions - 1 : -1, :]
                    entropy_BS = vocab_parallel_entropy(action_logits)
                    entropy = masked_mean(entropy_BS, loss_mask)
                    entropy_for_loss = entropy

            if loss_config.use_entropy_loss:
                entropy_loss_term = entropy_for_loss * loss_config.entropy_loss_coef
            else:
                entropy_loss_term = torch.tensor(0.0, device=logits.device)

            if loss_config.use_kl_loss:
                kl_loss = compute_approx_kl(
                    action_log_probs,
                    base_action_log_probs,
                    loss_mask=loss_mask,
                    kl_estimator_type=loss_config.kl_estimator_type,
                )
                kl_loss = masked_mean(kl_loss, loss_mask, dim=-1).mean()
            else:
                kl_loss = torch.tensor(0.0, device=logits.device)
            kl_loss_term = kl_loss * loss_config.kl_loss_coef

            # Policy losses are pre-scaled to achieve the correct loss_reduction
            # when summing across the entire minibatch (see `apply_loss_reduction_to_advantages_minibatch`).
            # Megatron divides loss by num_microbatches
            # (https://github.com/NVIDIA/Megatron-LM/blob/core_v0.15.2/megatron/core/pipeline_parallel/schedules.py#L248)
            # and the data parallel all-reduce averages gradients across dp_size.
            # Megatron's schedule separately multiplies loss by the CP size for two-output loss funcs,
            # so CP ranks are not included in this correction factor.
            # (https://github.com/NVIDIA/Megatron-LM/blob/core_v0.15.2/megatron/core/distributed/distributed_data_parallel.py#L285)
            # so we multiply by both factors to recover the correct sum reduction.
            grad_sum_correction_factor = num_microbatches * dp_size

            # NOTE: The KL and entropy loss terms are not pre-scaled,
            # so we just average them across microbatches and DP workers.
            # KL and entropy use Megatron's existing microbatch and CP schedule scaling.
            # Megatron divides by num_microbatches (which includes fully-padding microbatches
            # added by token-based batching). Those padding microbatches contribute 0 to
            # KL/entropy, so dividing by the full count would dilute the regularization by
            # num_real/num_total. Scale up by num_microbatches/num_real_microbatches so the
            # terms are averaged over real microbatches only (no-op when there is no padding).
            kl_entropy_microbatch_scale = num_microbatches / max(1, num_real_microbatches)
            loss = (
                policy_loss * grad_sum_correction_factor
                + (kl_loss_term - entropy_loss_term) * kl_entropy_microbatch_scale
            )
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
            for k, v in loss_metrics.items():
                metrics["loss_metrics/" + k] = v
            metrics.update(
                compute_minibatch_rollout_logprob_diff_metrics(action_log_probs, rollout_action_logprobs, loss_mask)
            )
            return loss, metrics

        def forward_step(batch_iter, model):
            # NOTE(Charlie): despite the name, methods like `remove_left_padding()` are padding-agnostic
            # (can be left, or right) as it uses attention_mask to locate real tokens. Same thing
            # for recover_left_padding and setup_per_microbatch_replay_forward. Especially relevant
            # after this PR https://github.com/NovaSky-AI/SkyRL/pull/1285.
            batch = next(batch_iter)

            model_config = get_model_config(model)
            fp8_enabled = is_fp8_enabled(getattr(model_config, "fp8", None))
            rollout_expert_indices = batch.pop("rollout_expert_indices", None)
            if rollout_expert_indices is not None:
                setup_per_microbatch_replay_forward(
                    rollout_expert_indices,
                    batch["attention_mask"],
                    model_config=model_config,
                    remove_microbatch_padding=self.remove_microbatch_padding,
                )

            sequences = batch["sequences"]
            attention_mask = batch["attention_mask"].to(bool)
            position_ids = batch["position_ids"]
            # When present, sub_seq_lengths enumerates every sub-sequence
            # inside every row of the micro-batch (controller-side mini-batch
            # packing). preprocess_packed_seqs uses it to emit cu_seqlens
            # entries covering all sub-seqs, not one per row.
            #
            # It arrives as a ``TensorList`` data field.
            # ``preprocess_packed_seqs`` and the packed-logprob scatter use
            # ``list[list[int]]``, so convert tensors -> python lists here.
            sub_seq_lengths_field = batch.get("sub_seq_lengths")
            sub_seq_lengths = [t.tolist() for t in sub_seq_lengths_field] if sub_seq_lengths_field is not None else None
            batch["sub_seq_lengths_list"] = sub_seq_lengths

            vlm_inputs = {}
            if batch.get("pixel_values") is not None and mpu.get_pipeline_model_parallel_rank() == 0:
                vlm_inputs["pixel_values"] = torch.cat(batch["pixel_values"].tensors, dim=0)
            if batch.get("image_grid_thw") is not None:
                vlm_inputs["image_grid_thw"] = torch.cat(batch["image_grid_thw"].tensors, dim=0)

            if self.remove_microbatch_padding:
                new_sequences, packed_seq_params = preprocess_packed_seqs(
                    sequences,
                    attention_mask,
                    pre_process=mpu.is_pipeline_first_stage(ignore_virtual=True) or self.is_vlm,
                    sub_seq_lengths=sub_seq_lengths,
                    fp8_enabled=fp8_enabled,
                )
                batch["packed_seq_params"] = packed_seq_params
                batch["packed_targets"] = _build_packed_targets(
                    sequences, attention_mask, packed_seq_params, sub_seq_lengths=sub_seq_lengths
                )
                new_attention_mask = None
                new_position_ids = None
            else:
                new_sequences, new_attention_mask, new_position_ids = remove_left_padding(
                    sequences,
                    attention_mask,
                    position_ids,
                    pre_process=mpu.is_pipeline_first_stage(ignore_virtual=True) or self.is_vlm,
                    fp8_enabled=fp8_enabled,
                )
                packed_seq_params = None
                # Qwen-style VLMs recompute 3D mRoPE positions internally from
                # image_grid_thw and ignore any position_ids passed in.
                if self.is_vlm:
                    new_position_ids = None

            if self._fused_lm_head:
                # output_processor returns decoder hidden states (not logits) and
                # stashes the LM-head weight; loss_func then fuses the projection.
                _op_ctx: dict = {}
                outputs = model(
                    new_sequences,
                    new_position_ids,
                    to_te_attention_mask(new_attention_mask),
                    packed_seq_params=packed_seq_params,
                    output_processor=_fused_lm_head_output_processor,
                    output_processor_context=_op_ctx,
                    **vlm_inputs,
                )
                batch["lm_head_weight"] = _op_ctx.get("lm_head_weight")
            else:
                outputs = model(
                    new_sequences,
                    new_position_ids,
                    to_te_attention_mask(new_attention_mask),
                    packed_seq_params=packed_seq_params,
                    **vlm_inputs,
                )

            if not self.remove_microbatch_padding:
                outputs = recover_left_padding(
                    outputs,
                    new_attention_mask,
                    attention_mask,
                    seq_len,
                    post_process=mpu.is_pipeline_last_stage(ignore_virtual=True),
                )

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

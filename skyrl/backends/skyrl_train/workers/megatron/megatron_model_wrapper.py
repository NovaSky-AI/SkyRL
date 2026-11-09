import math
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence

import megatron.core.parallel_state as mpu
import torch
import torch.nn as nn
from megatron.bridge.peft.utils import (
    enable_expert_parallel_grad_sync_in_finalize,
    finalize_model_grads_with_expert_adapter_sync,
)
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.utils import unwrap_model
from omegaconf import OmegaConf

from skyrl.backends.skyrl_train.distributed.megatron.fused_lm_head import (
    call_model_with_fused_lm_head,
    fused_lm_head_output_processor,
)
from skyrl.backends.skyrl_train.distributed.megatron.megatron_utils import (
    get_model_config,
    model_owns_vlm_packing,
)
from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
    from_parallel_hidden_to_logprobs,
    from_parallel_hidden_to_logprobs_packed_sequences,
    from_parallel_logits_to_logprobs,
    from_parallel_logits_to_logprobs_packed_sequences,
    vocab_parallel_entropy,
    vocab_parallel_entropy_packed_sequences,
)
from skyrl.backends.skyrl_train.distributed.megatron.quantization_utils import (
    is_fp8_enabled,
)
from skyrl.backends.skyrl_train.distributed.megatron.rank_microbatch import (
    HostMicrobatch,
    LossInputs,
    MicrobatchFeatures,
    RankMicrobatch,
    RankView,
    build_rank_microbatch,
    recover_canonical_layout,
)
from skyrl.backends.skyrl_train.mtp.adapter import project_mtp_hidden_to_logits
from skyrl.backends.skyrl_train.mtp.hidden_capture import (
    maybe_capture_mtp_hidden,
    native_mtp_disabled,
)
from skyrl.backends.skyrl_train.mtp.soft_ce import (
    build_teacher_logits,
    draft_soft_ce,
    draft_soft_ce_topk,
    shift_mask_for_mtp,
    unpadded_vocab_shard_width,
)
from skyrl.backends.skyrl_train.utils.microbatch_prefetch import MicrobatchPrefetcher
from skyrl.backends.skyrl_train.utils.ppo_utils import (
    PolicyLossRegistry,
    PolicyLossType,
    compute_approx_kl,
    score_centering_weight,
)
from skyrl.backends.skyrl_train.utils.replay_utils import (
    install_replay_forward,
    router_replay_schedule,
    setup_per_microbatch_replay_backward,
)
from skyrl.backends.skyrl_train.utils.sample_support_replay import (
    score_centering_loss_inputs,
    score_centering_metrics,
    score_prepared_sample_support,
)
from skyrl.backends.skyrl_train.utils.torch_utils import masked_mean
from skyrl.backends.skyrl_train.workers.megatron.lazy_microbatches import (
    summarize_microbatches,
)
from skyrl.backends.skyrl_train.workers.worker_utils import (
    compute_minibatch_rollout_logprob_diff_metrics,
)
from skyrl.train.config import TrainerConfig
from skyrl.train.fused_lm_head import FusedLmHeadBackend

_MTP_STUDENT_LOGITS_KEY = "mtp_student_logits"


def _upload_rank_microbatch(
    micro_batch: dict, *, view: RankView, features: MicrobatchFeatures, device: torch.device
) -> RankMicrobatch:
    """This rank's share of a host microbatch, built on the host and uploaded in one copy."""
    return build_rank_microbatch(micro_batch, view, features).to(device)


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
        # Sample packing for VLMs needs a model that computes mRoPE positions from the packed
        # stream itself; other VLMs only run unpacked (see _assert_vlm_supported).
        self.model_owns_vlm_packing = is_vlm and model_owns_vlm_packing(self.actor_module)
        # Fuse the LM-head projection into the chunked log-prob/entropy via the
        # GPTModel output_processor hook (avoids materializing the full
        # [B, S, vocab//TP] logits + its fp32 grad). See model_utils.
        self._fused_lm_head = self.cfg.fused_lm_head_logprob
        self._fused_lm_head_backend = self.cfg.fused_lm_head_logprob_backend
        self._fused_lm_head_block_sparse = self._fused_lm_head_backend == FusedLmHeadBackend.TRITON_BLOCK_SPARSE
        if self._fused_lm_head_block_sparse and self.cfg.algorithm.enable_sample_support_replay:
            raise ValueError("The triton_block_sparse fused LM-head backend does not support sample-support replay")
        # Pending grad-sync request recorded by `_defer_finalize_model_grads`, replayed
        # by `run_pending_grad_sync`. See those methods for why the sync is deferred.
        self._pending_grad_sync: Optional[dict] = None

        config = get_model_config(self.actor_module[0])
        # The fused path computes log-probs from the hidden states and the head weight, bypassing
        # the output layer's forward, so it applies the model's final logit softcap (Beam) itself.
        self._logit_softcap = getattr(config, "final_logit_softcapping", None) or None
        self.fp8_enabled = is_fp8_enabled(config.fp8)
        self.fp8_recipe = config.fp8_recipe
        # Build and upload each microbatch while the previous one computes. The result is the same
        # either way; tests turn it off to show that.
        self.prefetch_microbatches = True
        # megatron-core's mlp_chunks_for_training splits the sequence with Tensor.chunk(n), which yields fewer
        # than n pieces unless n divides the length. Each MoE call is a set of expert-parallel collectives, so
        # a per-rank piece count hangs the EP group. Pad the unpacked length (the model inputs and the replay
        # routes share one TokenLayout) so every rank gets n pieces: with sequence parallelism the MLP sees the
        # length divided by TP, so pad to a multiple of n * TP. The packed path is not covered.
        mlp_chunks = getattr(config, "mlp_chunks_for_training", 1) or 1
        sp_size = (
            mpu.get_tensor_model_parallel_world_size()
            if mlp_chunks > 1 and getattr(config, "sequence_parallel", False)
            else 1
        )
        self._seq_align = mlp_chunks * sp_size if mlp_chunks > 1 else 1
        # megatron-core's DeepSeek-V4 compressed sparse attention raises on any explicit mask. On the
        # unpacked path TokenLayout left-aligns every row, so the implicit causal mask
        # already keeps real tokens off the trailing padding; the mask is still used for logprobs.
        self._implicit_causal_mask_only = getattr(config, "experimental_attention_variant", None) == "dsv4_hybrid"
        if self._implicit_causal_mask_only:
            # Its compressed attention (and cuDNN's indexer) also needs seqlen_q <= seqlen_k * ratio, i.e. the
            # unpacked length padded to a multiple of every layer's compression ratio as well (ratio 0 is a
            # sliding-window-only layer). Trailing pad only completes the last compressed block, which only pad
            # positions can attend to under the causal mask.
            ratios = [r for r in (getattr(config, "csa_compress_ratios", None) or []) if r > 0]
            self._seq_align = math.lcm(self._seq_align, *ratios)
        # Megatron-Bridge's Qwen-VL providers force calculate_per_token_loss when CP > 1 (VLM CP).
        self._per_token_loss = bool(getattr(config, "calculate_per_token_loss", False))
        if self._per_token_loss and any(
            getattr(getattr(chunk, "ddp_config", None), "overlap_grad_reduce", False) for chunk in self.actor_module
        ):
            # _per_token_grad_scaling rescales the grad buffers between calls, which would race
            # with reductions still in flight.
            raise ValueError("calculate_per_token_loss (VLM context parallelism) requires overlap_grad_reduce=false")
        # This is set to None by default: https://github.com/NVIDIA/Megatron-LM/blob/07b22a05136a3cb08ece05f7de38cf6aeeb165fb/megatron/core/model_parallel_config.py#L95
        # use the built-in finalize_model_grads function to all reduce gradients across
        # parallelism dimensions -- but deferred to optim_step rather than run per
        # forward_backward. See `_defer_finalize_model_grads`.
        config.finalize_model_grads_func = self._defer_finalize_model_grads
        # Shared expert LoRA weights are replicated across EP; `run_pending_grad_sync`
        # sums their grads over EP once per step, so drop Bridge's per-microbatch
        # fallback hooks (which also miss grads under gradient_accumulation_fusion).
        enable_expert_parallel_grad_sync_in_finalize(self.actor_module)
        # Wire up the optimizer's loss scaler so Megatron's pipeline schedule can scale
        # the loss before backward (critical for fp16 dynamic loss scaling, MoE aux loss
        # scaling, and any explicit loss_scale configuration).
        if actor_optimizer is not None:
            config.grad_scale_func = actor_optimizer.scale_loss

    def _defer_finalize_model_grads(self, model, num_tokens=None, **kwargs) -> None:
        """Record Megatron's end-of-schedule grad sync instead of running it.

        Megatron's pipeline schedules call ``finalize_model_grads_func`` at the end of
        every ``forward_backward_func``, which is correct when one call == one optimizer
        step. SkyRL lets a logical batch span several ``forward_backward`` calls (Tinker
        splits large batches into multiple requests; callers may accumulate), and the
        sync is *not* idempotent: the DP reduce-scatter writes the reduced result into
        this rank's own shard of ``grad_data`` while leaving peer regions holding
        un-reduced local values, so reducing a second time folds already-reduced
        gradients back in. The layernorm/embedding all-reduces double-count the same way.

        So we record the request here and replay it exactly once from
        :meth:`run_pending_grad_sync`, called by the worker's ``optim_step``.

        ``num_tokens`` is only non-None under ``calculate_per_token_loss``. The schedule passes the
        call's token count, ``finalize_model_grads`` all-reduces it and divides every gradient by it,
        and Bridge's finalize wrapper passes it through:
        https://github.com/NVIDIA/Megatron-LM/blob/d476c21be396dc666d2e04fa8c199c3851a1551f/megatron/core/pipeline_parallel/schedules.py#L863-L870
        https://github.com/NVIDIA/Megatron-LM/blob/d476c21be396dc666d2e04fa8c199c3851a1551f/megatron/core/distributed/finalize_model_grads.py#L694-L713
        https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/8e7077c6826d17eb4d4d54e6eb15c5a581eda4c0/src/megatron/bridge/peft/utils.py#L1095-L1114
        We drop it and replay with ``num_tokens=None``, because one division at the end would average
        several ``forward_backward`` calls instead of adding them. Each call's gradients are divided by
        its own count instead; see :meth:`_per_token_grad_scaling` for details.
        """
        del model, num_tokens, kwargs  # replayed against self.actor_module with default process groups
        self._pending_grad_sync = {"num_tokens": None}

    @contextmanager
    def _per_token_grad_scaling(self, num_tokens_global: int):
        """Apply ``calculate_per_token_loss``'s division by the token count to this call's gradients only.

        ``loss_func`` multiplies each call's loss by that call's global token count N, so the
        buffer gets N·g from the call. Megatron would divide once at finalize, by the counts summed
        over every call before the optimizer step. With three calls of N = 100, 200, 300:

            buffer = 100·g1 + 200·g2 + 300·g3, divided by 600
                   = 1/6·g1 + 1/3·g2 + 1/2·g3        (wrong: a weighted average)

        Instead, scale the buffer by N before the call and by 1/N after it:

            call  enter (×N)      + backward   exit (÷N)   buffer
            1     0               + 100·g1     ÷100        g1
            2     200·g1          + 200·g2     ÷200        g1 + g2
            3     300·(g1+g2)     + 300·g3     ÷300        g1 + g2 + g3

        The finalize replay then runs with ``num_tokens=None`` and doesn't divide, so the result is
        g1 + g2 + g3. With a single call the enter scaling is a no-op (the buffer is zero).

        Gradients are still local here (the DP/CP reduction runs once, from
        ``run_pending_grad_sync``), and ``scale_gradients`` is the per-chunk method finalize
        uses for that division, so it covers the same buffers and needs no collective.
        """
        num_tokens_global = max(1, int(num_tokens_global))
        for chunk in self.actor_module:
            chunk.scale_gradients(float(num_tokens_global))
        try:
            yield
        finally:
            for chunk in self.actor_module:
                chunk.scale_gradients(1.0 / num_tokens_global)

    def run_pending_grad_sync(self) -> None:
        """Reduce gradients across DP/TP/PP exactly once for the accumulated window.

        Always runs the collective, even when this rank recorded nothing: a DP rank
        whose ``forward_backward`` got no microbatches never reaches the schedule's
        finalize hook, and skipping the reduce here would hang the ranks that do run it.
        """
        pending = self._pending_grad_sync
        self._pending_grad_sync = None
        finalize_model_grads_with_expert_adapter_sync(self.actor_module, pending["num_tokens"] if pending else None)

    def train(self):
        [module.train() for module in self.actor_module]

    def eval(self):
        [module.eval() for module in self.actor_module]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _assert_vlm_supported(self):
        """Guard the VLM parallelism constraints.

        With ``remove_microbatch_padding`` the model receives a [1, T] THD stream and
        Megatron-Bridge's Qwen3-VL model rebuilds 3D mRoPE positions per packed
        sub-sequence from ``packed_seq_params`` (``rope.get_rope_index``), without
        re-packing the stream (NVIDIA-NeMo/Megatron-Bridge#4532). Other VLMs only run
        unpacked (``model_owns_vlm_packing``).

        Context parallelism: for those models the full packed stream goes to every CP rank
        (``RankView.shard_for_cp=False``) and the model computes mRoPE,
        places image features and applies the CP split itself. Its rank-local logits are
        expected in the same 2*CP zigzag layout SkyRL's packed log-prob gather uses
        (test_megatron_vlm_cp_vs_no_cp checks this).
        """
        if self.remove_microbatch_padding and not self.model_owns_vlm_packing:
            model_cls = type(unwrap_model(self.actor_module[0])).__name__
            raise ValueError(
                "trainer.remove_microbatch_padding=true (sample packing) is supported for VLMs only on "
                "Megatron-Bridge's Qwen3VLModel (Qwen3-VL, Qwen3.5-VL), which rebuilds mRoPE positions per "
                f"packed sample; got {model_cls}. Set trainer.remove_microbatch_padding=false."
            )
        # CP needs packing (validate_megatron_cfg), so the check above also limits VLM CP to models that
        # apply the CP split themselves (shard_for_cp=False below relies on it). MTP with CP is refused in
        # validate_megatron_cfg too.
        assert (
            mpu.get_tensor_model_parallel_world_size() == 1 or self.cfg.policy.sequence_parallel_size == 1
        ), "VLM + sequence parallelism unsupported"

    def _microbatch_prefetchers(self, micro_batches: Sequence[dict], **feature_options: bool):
        """Megatron's data iterator: one prefetcher per model chunk, staging that chunk's microbatches.

        With virtual pipeline stages, each chunk reads its own iterator over the same microbatches.
        """
        device = torch.device("cuda", torch.cuda.current_device())
        features = self._features(**feature_options)
        prefetchers = [
            MicrobatchPrefetcher(
                micro_batches,
                partial(_upload_rank_microbatch, view=self._rank_view(chunk), features=features, device=device),
                RankMicrobatch.tensors,
                device,
                enabled=self.prefetch_microbatches,
            )
            for chunk in self.actor_module
        ]
        return prefetchers if len(prefetchers) > 1 else prefetchers[0]

    def _rank_view(self, model) -> RankView:
        """Describe this rank for one model chunk, as ``build_rank_microbatch`` consumes it."""
        return RankView.for_model(
            model,
            is_vlm=self.is_vlm,
            # Qwen3VLModel needs the full stream on every CP rank to compute mRoPE and place
            # image features, then applies the CP split itself (see RankView.shard_for_cp).
            # Other models get SkyRL's per-rank split, which GPTModel requires.
            shard_for_cp=not self.model_owns_vlm_packing,
        )

    def _features(self, *, block_sparse: bool = False, mtp: bool = False) -> MicrobatchFeatures:
        """The training features ``build_rank_microbatch`` lays a microbatch out for."""
        return MicrobatchFeatures(
            packed=self.remove_microbatch_padding,
            sample_support=self.cfg.algorithm.enable_sample_support_replay,
            block_sparse=block_sparse,
            mtp=mtp,
            fp8_enabled=self.fp8_enabled,
            fp8_recipe=self.fp8_recipe,
            seq_align=self._seq_align,
            implicit_causal_mask=self._implicit_causal_mask_only,
        )

    def forward(
        self,
        micro_batches: Sequence[HostMicrobatch],
        seq_len: int,
        micro_batch_size: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Forward-only inference to compute log-probs over a full mini-batch consisting of multiple micro-batches.

        Args:
            micro_batches: The host microbatches.
            seq_len: Padded sequence length per sample.
            micro_batch_size: Per-micro-batch size.
            temperature: Optional temperature scaling for logits.

        Returns:
            torch.Tensor of concatenated log-probs across micro-batches (valid on pipeline last stage only).
        """
        if self.is_vlm:
            self._assert_vlm_supported()
        forward_backward_func = get_forward_backward_func()

        def collection_func(logits, *, loss_inputs: LossInputs, state: dict):
            tp_grp = mpu.get_tensor_model_parallel_group()
            tp_rank = mpu.get_tensor_model_parallel_rank()

            # Fused LM-head: `logits` is actually decoder hidden states [B, S, H]
            # (the output_processor skipped the projection); fold the LM-head into
            # the chunked log-prob so this forward-only pass never materializes the
            # full [B, S, vocab//TP] logits (which would OOM at long context).
            lm_head_weight = state.get("lm_head_weight")
            fused_lm_head = self._fused_lm_head and lm_head_weight is not None
            if fused_lm_head:
                _v_local = int(lm_head_weight.shape[0])
                fused_vocab_start, fused_vocab_end = tp_rank * _v_local, (tp_rank + 1) * _v_local

            # temperature normalization (the fused path applies it inside the op)
            if temperature != 1.0 and not fused_lm_head:
                logits.div_(temperature)

            shard_vocab_size = lm_head_weight.shape[0] if fused_lm_head else logits.shape[-1]
            if self.cfg.algorithm.enable_sample_support_replay:
                token_logprobs = score_prepared_sample_support(
                    logits,
                    loss_inputs.sample_support,
                    vocab_start_index=tp_rank * shard_vocab_size,
                    vocab_end_index=(tp_rank + 1) * shard_vocab_size,
                    tp_group=tp_grp,
                    lm_head_weight=lm_head_weight if fused_lm_head else None,
                    temperature=temperature,
                    chunk_size=self.cfg.logprobs_chunk_size,
                    compute_entropy=False,
                    entropy_requires_grad=False,
                ).logprobs
            elif fused_lm_head and loss_inputs.packed_targets is not None:
                token_logprobs = from_parallel_hidden_to_logprobs_packed_sequences(
                    logits,  # decoder hidden states [1, T, H]
                    lm_head_weight,
                    loss_inputs.packed_targets,
                    loss_inputs.cu_seqlens_padded,
                    loss_inputs.seq_len,
                    vocab_start_index=fused_vocab_start,
                    vocab_end_index=fused_vocab_end,
                    group=tp_grp,
                    inference_only=True,
                    cp_group=mpu.get_context_parallel_group(),
                    chunk_size=self.cfg.logprobs_chunk_size,
                    attention_mask=loss_inputs.attention_mask,
                    sub_seq_lengths=loss_inputs.sub_seq_lengths,
                    temperature=temperature,
                    fused_backend=self._fused_lm_head_backend,
                    softcap=self._logit_softcap,
                    targets_rolled=True,
                )
            elif fused_lm_head:
                token_logprobs = from_parallel_hidden_to_logprobs(
                    logits,  # decoder hidden states [B, S, H]
                    lm_head_weight,
                    loss_inputs.sequences,
                    vocab_start_index=fused_vocab_start,
                    vocab_end_index=fused_vocab_end,
                    tp_group=tp_grp,
                    inference_only=True,
                    cp_group=None,
                    chunk_size=self.cfg.logprobs_chunk_size,
                    temperature=temperature,
                    fused_backend=self._fused_lm_head_backend,
                    softcap=self._logit_softcap,
                )
            elif loss_inputs.packed_targets is not None:
                token_logprobs = from_parallel_logits_to_logprobs_packed_sequences(
                    logits,
                    loss_inputs.packed_targets,
                    loss_inputs.cu_seqlens_padded,
                    loss_inputs.seq_len,
                    vocab_start_index=tp_rank * logits.shape[-1],
                    vocab_end_index=(tp_rank + 1) * logits.shape[-1],
                    group=tp_grp,
                    inference_only=True,
                    cp_group=mpu.get_context_parallel_group(),
                    chunk_size=self.cfg.logprobs_chunk_size,
                    attention_mask=loss_inputs.attention_mask,
                    sub_seq_lengths=loss_inputs.sub_seq_lengths,
                    targets_rolled=True,
                )
            else:
                token_logprobs = from_parallel_logits_to_logprobs(
                    logits,
                    loss_inputs.sequences,
                    vocab_start_index=tp_rank * logits.shape[-1],
                    vocab_end_index=(tp_rank + 1) * logits.shape[-1],
                    tp_group=tp_grp,
                    inference_only=True,
                    cp_group=None,
                    chunk_size=self.cfg.logprobs_chunk_size,  # chunk seq dim to bound peak memory
                )
            return torch.tensor(0.0, device=token_logprobs.device), {"log_probs": token_logprobs}

        def forward_step(batch_iter, model):
            # Microbatches are held on the host; this rank's share was derived there and uploaded
            # in one copy while the previous microbatch computed (``MicrobatchPrefetcher``).
            microbatch = next(batch_iter)
            model_inputs = microbatch.model_inputs

            if microbatch.replay_routes is not None:
                install_replay_forward(microbatch.replay_routes)

            state: dict = {}
            with native_mtp_disabled(model):
                if self._fused_lm_head:
                    # Fused LM-head inference: the output_processor returns decoder
                    # hidden states (not logits) and stashes the LM-head weight, so
                    # collection_func can fold the projection into the chunked
                    # log-prob op. Without this, a forward-only ref/old-logprob pass
                    # (e.g. PPO reference logprobs) at long context would still
                    # materialize the full [B, S, vocab//TP] logits and OOM.
                    _op_ctx: dict = {}
                    outputs = call_model_with_fused_lm_head(
                        model,
                        model_inputs.input_ids,
                        model_inputs.position_ids,
                        model_inputs.attention_mask,
                        output_processor=fused_lm_head_output_processor,
                        output_processor_context=_op_ctx,
                        **model_inputs.model_kwargs(),
                    )
                    state["lm_head_weight"] = _op_ctx.get("lm_head_weight")
                else:
                    outputs = model(
                        model_inputs.input_ids,
                        model_inputs.position_ids,
                        model_inputs.attention_mask,
                        **model_inputs.model_kwargs(),
                    )

            if microbatch.loss_inputs is not None and microbatch.loss_inputs.recover is not None:
                outputs = recover_canonical_layout(outputs, microbatch.loss_inputs)

            batch_iter.prefetch()
            return outputs, partial(collection_func, loss_inputs=microbatch.loss_inputs, state=state)

        batch_generator = self._microbatch_prefetchers(micro_batches)

        summary = summarize_microbatches(micro_batches)
        with router_replay_schedule(summary.num_without_routes < len(micro_batches)):
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
            log_probs = log_probs[:, -summary.num_actions :]
        else:
            # return dummy tensor for non-last pp stages (microbatches are held on the host)
            log_probs = torch.zeros(size=(1, 1), dtype=torch.bfloat16)
        return log_probs

    def forward_backward_mini_batch(
        self,
        micro_batches: Sequence[HostMicrobatch],
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
            micro_batches: The host microbatches, with their loss tensors.
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

        # Multi-Token Prediction (MTP): if the model was built with native MTP heads, train them with
        # an explicit decoupled loss instead of Megatron's in-forward process_mtp_loss path. The heads
        # still run inside the forward (so we reuse their rotary embeddings); the native process_mtp_loss
        # is disabled at its call sites (see mtp/native_loss_patch.py, applied at config time) so no
        # native MTP gradient couples onto the trunk. A forward hook captures the heads' hidden states
        # (with the trunk input detached) for us to score. Training only.
        model_config = get_model_config(self.actor_module[0])
        mtp_enabled = (not forward_only) and bool(getattr(model_config, "mtp_num_layers", None))
        per_token_loss = self._per_token_loss
        # Defaults live on the MegatronConfig dataclass (config.py) -- read the fields directly
        # rather than restating them in getattr fallbacks that could drift.
        mcfg = self.cfg.policy.megatron_config
        mtp_loss_weight = float(mcfg.mtp_loss_weight)
        mtp_loss_chunk_size = mcfg.mtp_loss_chunk_size
        mtp_loss_topk = mcfg.mtp_loss_topk
        summary = summarize_microbatches(micro_batches)

        if mtp_enabled:
            # The decoupled draft training records the MTP block's inputs with a forward pre-hook
            # and replays the block afterwards; module hooks do not fire inside CUDA-graph replay,
            # so the capture would silently see nothing and the draft loss would vanish.
            assert (
                not getattr(model_config, "enable_cuda_graph", False)
                and not getattr(model_config, "external_cuda_graph", False)
                and getattr(model_config, "cuda_graph_impl", "none") in (None, "none")
            ), "MTP draft training uses forward hooks and cannot be combined with Megatron CUDA graphs"

        # Resolve loss function
        resolved_loss_name = loss_fn if loss_fn is not None else self.cfg.algorithm.policy_loss_type
        if self._fused_lm_head_block_sparse and resolved_loss_name == PolicyLossType.CROSS_ENTROPY:
            if return_per_token_outputs:
                raise ValueError("The triton_block_sparse fused LM-head backend does not support per-token outputs")
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
        centering_weight = (
            score_centering_weight(resolved_loss_name, loss_config)
            if self.cfg.algorithm.enable_score_centering and not forward_only
            else None
        )

        def loss_func(logits, *, loss_inputs: LossInputs, state: dict):
            packed_targets = loss_inputs.packed_targets
            num_actions = loss_inputs.num_actions
            old_action_log_probs = loss_inputs.old_action_log_probs
            base_action_log_probs = loss_inputs.base_action_log_probs
            advantages = loss_inputs.advantages
            loss_mask = loss_inputs.loss_mask
            rollout_action_logprobs = loss_inputs.rollout_action_logprobs
            response_mask = loss_inputs.response_mask
            num_microbatches = loss_inputs.num_microbatches
            normalization = loss_inputs.loss_normalization

            sparse_active_mask = loss_inputs.active_mask

            dp_size = mpu.get_data_parallel_world_size(with_context_parallel=False)
            tp_grp = mpu.get_tensor_model_parallel_group()
            tp_rank = mpu.get_tensor_model_parallel_rank()

            # Policy, KL, entropy and MTP losses are normalized over the optimizer step, so
            # undo Megatron's gradient averaging to sum their contributions across microbatches.
            if per_token_loss:
                # calculate_per_token_loss: the schedule leaves a 3-tuple loss unscaled, DDP sums over
                # DP x CP, and this call's gradients are divided by its global token count
                # (_per_token_grad_scaling). Multiply by that count (attached by the worker) so the
                # division cancels, as in verl's Megatron engine.
                # Forward-only loss passes have no gradient sync, so the counts default there.
                num_tokens_global = loss_inputs.num_tokens_global or 1
                grad_sum_correction_factor = num_tokens_global
            else:
                # Default: Megatron divides loss by num_microbatches
                # (https://github.com/NVIDIA/Megatron-LM/blob/core_v0.15.2/megatron/core/pipeline_parallel/schedules.py#L248)
                # and the data parallel all-reduce averages gradients across dp_size.
                # Megatron's schedule separately multiplies loss by the CP size for two-output loss funcs,
                # so CP ranks are not included in this correction factor.
                # (https://github.com/NVIDIA/Megatron-LM/blob/core_v0.15.2/megatron/core/distributed/distributed_data_parallel.py#L285)
                # so we multiply by both factors to recover the correct sum reduction.
                grad_sum_correction_factor = num_microbatches * dp_size

            def loss_output(loss, metrics):
                if per_token_loss:
                    num_tokens_local = loss_inputs.num_tokens_local or 0
                    return loss, torch.tensor(num_tokens_local, dtype=torch.int, device=loss.device), metrics
                return loss, metrics

            # Fused LM-head: `logits` is actually decoder hidden states [B, S, H]
            # (the output_processor skipped the projection); fold the LM-head into
            # the chunked log-prob/entropy so the full logits tensor + its fp32
            # grad are never materialized.
            lm_head_weight = state.get("lm_head_weight")
            fused_lm_head = self._fused_lm_head and lm_head_weight is not None
            if fused_lm_head:
                _v_local = int(lm_head_weight.shape[0])
                fused_vocab_start, fused_vocab_end = tp_rank * _v_local, (tp_rank + 1) * _v_local

            # temperature normalization (the fused path applies it inside the op)
            if temperature != 1.0 and not fused_lm_head:
                logits.div_(temperature)

            shard_vocab_size = lm_head_weight.shape[0] if fused_lm_head else logits.shape[-1]
            support_entropy = None
            support_scores = None
            fused_entropy_BS = None
            compute_fused_entropy = resolved_loss_name != PolicyLossType.CROSS_ENTROPY
            if self.cfg.algorithm.enable_sample_support_replay:
                # The SFT path takes no entropy term at all, so nothing there pays for the row.
                compute_support_entropy = resolved_loss_name != PolicyLossType.CROSS_ENTROPY
                support_scores = score_prepared_sample_support(
                    logits,
                    loss_inputs.sample_support,
                    vocab_start_index=tp_rank * shard_vocab_size,
                    vocab_end_index=(tp_rank + 1) * shard_vocab_size,
                    tp_group=tp_grp,
                    lm_head_weight=lm_head_weight if fused_lm_head else None,
                    temperature=temperature,
                    chunk_size=self.cfg.logprobs_chunk_size,
                    compute_entropy=compute_support_entropy,
                    entropy_requires_grad=compute_support_entropy and loss_config.use_entropy_loss,
                    centering_weight=centering_weight,
                )
                token_logprobs = support_scores.logprobs
                support_entropy = support_scores.entropy
            elif fused_lm_head and packed_targets is not None:
                fused_result = from_parallel_hidden_to_logprobs_packed_sequences(
                    logits,  # decoder hidden states [1, T, H]
                    lm_head_weight,
                    packed_targets,
                    loss_inputs.cu_seqlens_padded,
                    loss_inputs.seq_len,
                    vocab_start_index=fused_vocab_start,
                    vocab_end_index=fused_vocab_end,
                    group=tp_grp,
                    inference_only=False,
                    cp_group=mpu.get_context_parallel_group(),
                    chunk_size=self.cfg.logprobs_chunk_size,
                    attention_mask=loss_inputs.attention_mask,
                    sub_seq_lengths=loss_inputs.sub_seq_lengths,
                    temperature=temperature,
                    fused_backend=self._fused_lm_head_backend,
                    softcap=self._logit_softcap,
                    active_mask=sparse_active_mask,
                    return_entropy=compute_fused_entropy,
                    entropy_requires_grad=compute_fused_entropy and loss_config.use_entropy_loss,
                    targets_rolled=True,
                )
                if compute_fused_entropy:
                    token_logprobs, fused_entropy_BS = fused_result
                else:
                    token_logprobs = fused_result
            elif fused_lm_head:
                fused_result = from_parallel_hidden_to_logprobs(
                    logits,  # decoder hidden states [B, S, H]
                    lm_head_weight,
                    loss_inputs.sequences,
                    vocab_start_index=fused_vocab_start,
                    vocab_end_index=fused_vocab_end,
                    tp_group=tp_grp,
                    inference_only=False,
                    cp_group=None,
                    chunk_size=self.cfg.logprobs_chunk_size,
                    temperature=temperature,
                    fused_backend=self._fused_lm_head_backend,
                    softcap=self._logit_softcap,
                    active_mask=sparse_active_mask,
                    return_entropy=compute_fused_entropy,
                    entropy_requires_grad=compute_fused_entropy and loss_config.use_entropy_loss,
                )
                if compute_fused_entropy:
                    token_logprobs, fused_entropy_BS = fused_result
                else:
                    token_logprobs = fused_result
            elif packed_targets is not None:
                token_logprobs = from_parallel_logits_to_logprobs_packed_sequences(
                    logits,
                    packed_targets,
                    loss_inputs.cu_seqlens_padded,
                    loss_inputs.seq_len,
                    vocab_start_index=tp_rank * logits.shape[-1],
                    vocab_end_index=(tp_rank + 1) * logits.shape[-1],
                    group=tp_grp,
                    inference_only=False,
                    cp_group=mpu.get_context_parallel_group(),
                    chunk_size=self.cfg.logprobs_chunk_size,
                    attention_mask=loss_inputs.attention_mask,
                    sub_seq_lengths=loss_inputs.sub_seq_lengths,
                    targets_rolled=True,
                )
            else:
                token_logprobs = from_parallel_logits_to_logprobs(
                    logits,
                    loss_inputs.sequences,
                    vocab_start_index=tp_rank * logits.shape[-1],
                    vocab_end_index=(tp_rank + 1) * logits.shape[-1],
                    tp_group=tp_grp,
                    inference_only=False,
                    cp_group=None,
                    chunk_size=self.cfg.logprobs_chunk_size,  # chunk seq dim to bound peak memory
                )

            action_log_probs = token_logprobs[:, -num_actions:]
            if self._fused_lm_head_block_sparse and resolved_loss_name != PolicyLossType.CROSS_ENTROPY:
                # Several RL objectives form ratios before applying loss_mask.
                # Give inactive operands the same finite value so masked rows
                # cannot create overflow or NaN in those intermediates.
                inactive = loss_mask <= 0
                if old_action_log_probs is not None:
                    old_action_log_probs = old_action_log_probs.masked_fill(inactive, 0.0)
                advantages = advantages.masked_fill(inactive, 0.0)
                if rollout_action_logprobs is not None:
                    rollout_action_logprobs = rollout_action_logprobs.masked_fill(inactive, 0.0)
                if base_action_log_probs is not None:
                    base_action_log_probs = base_action_log_probs.masked_fill(inactive, 0.0)

            # Megatron's canonical positions are next-token predictions, so actions are the last columns.
            action_positions = slice(-num_actions, None)
            centering_inputs = None
            loss_rollout_logprobs, score_centering_kwargs = rollout_action_logprobs, {}
            if centering_weight is not None:
                centering_inputs = score_centering_loss_inputs(
                    support_scores, action_positions, rollout_action_logprobs, advantages
                )
                loss_rollout_logprobs = centering_inputs.sampler_logprobs
                score_centering_kwargs = centering_inputs.loss_kwargs()
            # policy loss should be calculated based on the selected token logprobs
            policy_loss, loss_metrics = current_loss_fn(
                action_log_probs,
                old_action_log_probs,
                advantages,
                config=loss_config,
                loss_mask=loss_mask,
                rollout_logprobs=loss_rollout_logprobs,
                **score_centering_kwargs,
            )

            # Decoupled MTP / draft loss: soft-CE distillation of the detached-input MTP head against
            # the policy's own next-token distribution (full-vocab, or top-k when mtp_loss_topk is
            # set). Each depth's masked sum is normalized over the whole optimizer step, like entropy.
            draft_loss = None
            student_logits_list = state.get(_MTP_STUDENT_LOGITS_KEY)
            if mtp_enabled and student_logits_list:
                # Under THD sample packing the teacher (logits) and student logits are packed
                # ([1, T, vocab]); use the matching packed real-token mask + segment boundaries so the
                # MTP roll/mask respect sub-sequence boundaries (see shift_mask_for_mtp). Otherwise the
                # [batch, seq] attention mask aligns with the de-padded logits.
                packed = loss_inputs.packed
                draft_mask = (
                    loss_inputs.mtp_packed_mask
                    if packed
                    else loss_inputs.attention_mask & loss_inputs.real_sample_mask.to(torch.bool)[:, None]
                ).to(logits.dtype)
                mtp_cu_seqlens = loss_inputs.cu_seqlens_padded if packed else None
                vocab_size_tp = logits.shape[-1]
                # Undo the in-place temperature scaling so the teacher is the true policy distribution.
                # Detached: both draft-loss paths below require a detached teacher.
                teacher_src = (logits if temperature == 1.0 else logits * temperature).detach()
                # Megatron may pad the vocab to divide across TP; padded rows are never trained, so slice
                # this rank's shard to its true width to keep them out of the teacher (a view; autograd
                # zero-fills the tail).
                true_shard_width = unpadded_vocab_shard_width(
                    getattr(model_config, "vocab_size", None), vocab_size_tp, tp_rank
                )
                if true_shard_width != vocab_size_tp:
                    teacher_src = teacher_src[..., :true_shard_width]

                per_layer_losses = []
                for layer_idx, student_logits in enumerate(student_logits_list):
                    if true_shard_width != vocab_size_tp:
                        student_logits = student_logits[..., :true_shard_width]
                    layer_mask = shift_mask_for_mtp(draft_mask, layer_idx, cu_seqlens=mtp_cu_seqlens)
                    num_layer_tokens = torch.tensor(
                        normalization.num_mtp_tokens(layer_idx + 1), dtype=torch.float32, device=layer_mask.device
                    )
                    if mtp_loss_topk:
                        # Top-k draft loss: O(seq*k) memory, no full-vocab softmax. Pass the un-rolled
                        # policy logits + roll_shift so top-k runs on them directly and only the small
                        # [B, S, k] result is rolled (avoids a full rolled-teacher copy).
                        per_layer_losses.append(
                            draft_soft_ce_topk(
                                student_logits,
                                teacher_src,
                                layer_mask,
                                k=mtp_loss_topk,
                                vocab_parallel_group=tp_grp,
                                roll_shift=layer_idx + 1,
                                global_normalization_factor=num_layer_tokens,
                            )
                        )
                    else:
                        teacher_logits = build_teacher_logits(teacher_src, layer_idx)
                        per_layer_losses.append(
                            draft_soft_ce(
                                student_logits,
                                teacher_logits,
                                layer_mask,
                                vocab_parallel_group=tp_grp,
                                chunk_size=mtp_loss_chunk_size,
                                global_normalization_factor=num_layer_tokens,
                            )
                        )
                draft_loss = torch.stack(per_layer_losses).mean()
                # Drop the dict's reference so the tensor is freed after this microbatch's backward
                # (the autograd graph still holds it until then) instead of lingering.
                del state[_MTP_STUDENT_LOGITS_KEY]
                student_logits_list = None

            # SFT objectives need no RL entropy/KL pass. Anchored objectives
            # already include their token-level KL in ``policy_loss``.
            if resolved_loss_name == PolicyLossType.CROSS_ENTROPY:
                # Policy loss masks are pre-scaled to achieve the correct reduction
                # when summing across the entire minibatch (see `DefaultCollator`).
                loss = policy_loss * grad_sum_correction_factor
                if draft_loss is not None:
                    loss = loss + mtp_loss_weight * draft_loss * grad_sum_correction_factor
                unscaled_loss = policy_loss

                # Only build per-token outputs for callers that consume them.
                if return_per_token_outputs:
                    # Build per-sequence loss_fn_outputs.
                    # Compute valid_lens vectorized on GPU, then move tensors to CPU
                    # exactly once before iterating in Python — avoids ~3N GPU->CPU
                    # syncs per micro-batch (item()/cpu()/tolist() inside the loop).
                    batch_size = action_log_probs.shape[0]
                    seq_len = action_log_probs.shape[1]
                    if response_mask is not None:
                        valid_lens_t = response_mask.sum(dim=-1).long()
                    elif loss_mask is not None:
                        valid_lens_t = (loss_mask > 0).sum(dim=-1).long()
                    else:
                        valid_lens_t = torch.full(
                            (batch_size,), seq_len, device=action_log_probs.device, dtype=torch.long
                        )

                    action_log_probs_cpu = action_log_probs.detach().cpu()
                    # Per-token output remains the selected-token NLL; aggregate
                    # SFT objectives may reweight it internally.
                    with torch.no_grad():
                        elementwise_loss = -action_log_probs
                        if loss_mask is not None:
                            elementwise_loss = elementwise_loss * loss_mask
                    elementwise_loss_cpu = elementwise_loss.detach().cpu()
                    valid_lens = valid_lens_t.cpu().tolist()

                    loss_fn_outputs = []
                    for i in range(batch_size):
                        valid_len = valid_lens[i]
                        output = {"logprobs": action_log_probs_cpu[i, -valid_len:].tolist() if valid_len > 0 else []}
                        output["elementwise_loss"] = (
                            elementwise_loss_cpu[i, -valid_len:].tolist() if valid_len > 0 else []
                        )
                        loss_fn_outputs.append(output)
                else:
                    loss_fn_outputs = [{} for _ in range(action_log_probs.shape[0])]

                metrics = {
                    "loss": unscaled_loss.detach().item(),
                    "response_length": num_actions,
                    "loss_fn_outputs": loss_fn_outputs,
                }
                for key, value in loss_metrics.items():
                    metrics["loss_metrics/" + key] = value
                if draft_loss is not None:
                    metrics["mtp_loss"] = draft_loss.detach().item()
                return loss_output(loss, metrics)

            # RL path: add optional KL/entropy terms
            with torch.set_grad_enabled(loss_config.use_entropy_loss):
                if support_entropy is not None:
                    entropy = masked_mean(support_entropy[:, -num_actions:], loss_mask)
                    entropy_for_loss = entropy
                elif fused_entropy_BS is not None:
                    entropy = masked_mean(fused_entropy_BS[:, -num_actions:], loss_mask)
                    entropy_for_loss = entropy
                elif packed_targets is not None:
                    entropy, entropy_for_loss = vocab_parallel_entropy_packed_sequences(
                        logits,
                        loss_inputs.cu_seqlens_padded,
                        loss_inputs.seq_len,
                        num_actions,
                        loss_inputs.attention_mask,
                        loss_mask,
                        mpu.get_context_parallel_group(),
                        sub_seq_lengths=loss_inputs.sub_seq_lengths,
                        chunk_size=self.cfg.vocab_entropy_chunk_size,
                        chunk_memory_mb=self.cfg.vocab_entropy_chunk_memory_mb,
                    )
                else:
                    action_logits = logits[:, -num_actions - 1 : -1, :]
                    entropy_BS = vocab_parallel_entropy(
                        action_logits,
                        chunk_size=self.cfg.vocab_entropy_chunk_size,
                        chunk_memory_mb=self.cfg.vocab_entropy_chunk_memory_mb,
                    )
                    entropy = masked_mean(entropy_BS, loss_mask)
                    entropy_for_loss = entropy

            entropy = normalization.token_mean_contribution(entropy, loss_mask)
            entropy_for_loss = normalization.token_mean_contribution(entropy_for_loss, loss_mask)
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
                kl_loss = normalization.sequence_mean_contribution(
                    masked_mean(kl_loss, loss_mask, dim=-1), loss_inputs.real_sample_mask
                )
            else:
                kl_loss = torch.tensor(0.0, device=logits.device)
            kl_loss_term = kl_loss * loss_config.kl_loss_coef

            # Policy losses are pre-scaled to achieve the correct loss_reduction
            # when summing across the entire minibatch (see `apply_loss_reduction_to_advantages_minibatch`).
            loss = (policy_loss + kl_loss_term - entropy_loss_term) * grad_sum_correction_factor
            if draft_loss is not None:
                loss = loss + mtp_loss_weight * draft_loss * grad_sum_correction_factor
            unscaled_loss = loss / grad_sum_correction_factor

            # Build per-sequence loss_fn_outputs with logprobs, for callers that consume them: copying every
            # row's logprobs to host lists costs time in the token count.
            batch_size = action_log_probs.shape[0]
            if return_per_token_outputs:
                seq_len = action_log_probs.shape[1]
                if response_mask is not None:
                    valid_lens = response_mask.sum(dim=1).int().tolist()
                elif loss_mask is not None:
                    valid_lens = (loss_mask > 0).sum(dim=1).int().tolist()
                else:
                    valid_lens = [seq_len] * batch_size

                detached_log_probs = action_log_probs.detach().cpu()
                loss_fn_outputs = [
                    {"logprobs": detached_log_probs[i, -valid_len:].tolist() if valid_len > 0 else []}
                    for i, valid_len in enumerate(valid_lens)
                ]
            else:
                loss_fn_outputs = [{} for _ in range(batch_size)]

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
            if centering_inputs is not None:
                metrics.update(score_centering_metrics(support_scores, centering_inputs, action_positions, loss_mask))
            metrics.update(
                compute_minibatch_rollout_logprob_diff_metrics(action_log_probs, rollout_action_logprobs, loss_mask)
            )
            return loss_output(loss, metrics)

        def forward_step(batch_iter, model):
            # NOTE(Charlie): the token layout is padding-agnostic (left or right) as it uses
            # attention_mask to locate real tokens. Same thing for the canonical-layout recovery
            # and replay route selection. Especially relevant
            # after this PR https://github.com/NovaSky-AI/SkyRL/pull/1285.
            #
            # Microbatches are held on the host; this rank's share was derived there and uploaded
            # in one copy while the previous microbatch computed (``MicrobatchPrefetcher``).
            microbatch = next(batch_iter)
            model_inputs = microbatch.model_inputs

            # Exactly one forward setup per microbatch: each call appends this
            # microbatch's routes to the FIFO that activation-checkpoint
            # recomputation drains once during backward. A second call would
            # leave a stale entry, so backward replays the previous
            # microbatch's routes and Megatron's all-to-all split sizes stop
            # matching as soon as sequence lengths differ.
            if microbatch.replay_routes is not None:
                install_replay_forward(microbatch.replay_routes)

            # The decoupled MTP teacher/label rolls are context-parallel-unaware (a plain global
            # torch.roll, both here and in the packed path), so MTP draft training requires CP=1.
            # CP>1 would need the cross-rank boundary exchange that megatron's roll_tensor implements.
            if mtp_enabled:
                assert mpu.get_context_parallel_world_size() == 1, (
                    "MTP/draft training does not support context parallelism "
                    "(context_parallel_size > 1): the teacher/label roll is CP-unaware."
                )
                # NOTE (Alex): PP not supported yet -- the MTP block runs on the last stage, but the
                # ids it re-embeds are only laid out for the first stage.
                assert (
                    mpu.get_pipeline_model_parallel_world_size() == 1
                ), "MTP/draft training does not support pipeline parallelism (pipeline_model_parallel_size > 1)."
                # The fused LM-head path never materializes the main logits (its output_processor
                # returns decoder hidden states), but the draft loss distills against exactly those
                # logits as its teacher -- so the two cannot run together.
                assert not self._fused_lm_head, (
                    "MTP draft training is not supported with fused_lm_head_logprob=True: the fused "
                    "path skips the output-layer matmul, so there are no main logits to use as the "
                    "draft teacher. Disable one of them."
                )

            # Run the policy forward; when MTP is active a pre-hook records the MTP block's arguments.
            state: dict = {}
            student_hidden = None
            student_model = None
            with maybe_capture_mtp_hidden(model, mtp_enabled) as capture:
                if self._fused_lm_head:
                    # output_processor returns decoder hidden states (not logits) and
                    # stashes the LM-head weight; loss_func then fuses the projection.
                    _op_ctx: dict = {}
                    outputs = call_model_with_fused_lm_head(
                        model,
                        model_inputs.input_ids,
                        model_inputs.position_ids,
                        model_inputs.attention_mask,
                        output_processor=fused_lm_head_output_processor,
                        output_processor_context=_op_ctx,
                        **model_inputs.model_kwargs(),
                    )
                    state["lm_head_weight"] = _op_ctx.get("lm_head_weight")
                else:
                    outputs = model(
                        model_inputs.input_ids,
                        model_inputs.position_ids,
                        model_inputs.attention_mask,
                        **model_inputs.model_kwargs(),
                    )
                # Replay the MTP block on *detached* trunk hidden states (decoupled draft forward)
                # while still inside the capture context (so the MTP block stays in eval mode).
                if mtp_enabled and capture is not None:
                    student_hidden = capture.compute_student_hidden_states()
                    student_model = capture.model

            # Recover [batch, seq_len, ...] from Megatron's internal (left-removed) layout. Only used
            # on the non-packed path: with sample packing (remove_microbatch_padding) the logits stay
            # packed ([1, T, vocab]) and loss_func consumes the packed targets instead. MTP draft
            # training keeps the student logits in whichever layout the teacher (main logits) uses —
            # de-padded [batch, seq, vocab] without packing, packed [1, T, vocab] with it — so the two
            # always align (see the packed-aware mask in loss_func / mtp/soft_ce.py).
            recover = microbatch.loss_inputs is not None and microbatch.loss_inputs.recover is not None
            if recover:
                outputs = recover_canonical_layout(outputs, microbatch.loss_inputs)

            # Project the MTP hidden states through the shared output layer so loss_func can score the
            # draft loss, in the teacher's layout.
            if student_hidden is not None:
                student_logits = project_mtp_hidden_to_logits(student_hidden, student_model)
                if recover:
                    student_logits = [recover_canonical_layout(sl, microbatch.loss_inputs) for sl in student_logits]
                state[_MTP_STUDENT_LOGITS_KEY] = student_logits

            if microbatch.replay_routes is not None:
                setup_per_microbatch_replay_backward()

            batch_iter.prefetch()
            return outputs, partial(loss_func, loss_inputs=microbatch.loss_inputs, state=state)

        batch_generator = self._microbatch_prefetchers(
            micro_batches, block_sparse=self._fused_lm_head_block_sparse, mtp=mtp_enabled
        )

        replay_enabled = summary.num_without_routes < len(micro_batches)

        grad_scaling = (
            self._per_token_grad_scaling(summary.num_tokens_global)
            if per_token_loss and not forward_only
            else nullcontext()
        )
        with router_replay_schedule(replay_enabled), grad_scaling:
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

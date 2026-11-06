"""LoRA on FlashInfer's TRT-LLM MXFP4 MoE kernel (BF16 activations) instead of Marlin.

With LoRA enabled, vLLM 0.32 only accepts MoE expert kernels that support LoRA. For MXFP4 experts that leaves
Marlin (and the slower unfused OAI Triton path), because the TRT-LLM MXFP4 experts have no LoRA hook. On Blackwell,
Marlin's dequantize-then-mma path is ~half of a DeepSeek-V4.1-Flash decode step at 640 sequences.

vLLM already ships the LoRA scheme for TRT-LLM kernels, for BF16 weights (``trtllm_lora_moe._TrtLlmLoRAExpertsBase``,
FlashInfer PR #3153): the W13 LoRA delta is passed to the routed kernel as ``gemm1_lora_delta`` (added to FC1 before
the gated activation), the kernel returns its post-activation FC1 output, and the W2 LoRA is computed from that and
fused into the finalize. FlashInfer 0.7's ``trtllm_fp4_block_scale_routed_moe`` takes the same ``gemm1_lora_delta``
and has the same return contract, so this adds the MXFP4 (BF16 activation) member of that family: the base class
does the LoRA work, this class only issues the FP4 kernel call -- with the weights, scales and gated-activation
parameters (alpha / beta / clamp limit) exactly as vLLM's non-LoRA ``TrtLlmMxfp4ExpertsModular`` passes them.

It is registered as an extra kernel class of the ``FLASHINFER_TRTLLM_MXFP4_BF16`` backend. Only it supports LoRA,
so it is chosen only when LoRA is on (ahead of Marlin, which comes later in the backend priority list); weight
layout comes from the TRT-LLM backend's own weight processing. Activations stay BF16, as with Marlin: no change in
quantization formats, only in the FP4 GEMM kernel.

Measured on DeepSeek-V4.1-Flash (8x B200, TP8, active rank-32 adapter, 640 staggered multi-turn sessions, tuned
MoE-LoRA block sizes, mixed-step CUDA graphs): 570 -> 797 gen tok/s/GPU; pure decode 831 -> 1090. Numerics, natural
text with the adapter scaled 5x: |dlogp| vs Marlin 0.204 mean / 1.75 p99, against a Marlin-vs-Marlin run-to-run
floor of 0.187 / 1.53; mean NLL 2.201 vs Marlin 2.202 and 2.189.

DP attention + expert parallelism: the MoE layer runs on every DP rank's tokens, so this class maps global expert
ids to local LoRA slots, never takes vLLM's local no-LoRA shortcut, and splits calls above
``SKYRL_VLLM_MXFP4_TRTLLM_LORA_MOE_CHUNK`` tokens (default 32768) to bound the LoRA buffers.

Opt-in: ``SKYRL_VLLM_MXFP4_TRTLLM_LORA_MOE=1`` (applied in every vLLM worker by ``new_inference_worker_wrap``; the
DeepSeek-V4.1 recipe sets it).
Remove when vLLM ships an MXFP4 TRT-LLM LoRA experts class.
"""

import os

from loguru import logger

ENV = "SKYRL_VLLM_MXFP4_TRTLLM_LORA_MOE"
_APPLIED = False
CHUNK_ENV = "SKYRL_VLLM_MXFP4_TRTLLM_LORA_MOE_CHUNK"


def _chunk_tokens() -> int:
    """Max tokens per LoRA MoE call (default 32768: one TP rank's whole step, so TP runs are never chunked)."""
    return int(os.environ.get(CHUNK_ENV, "32768"))


def _build_class():
    import torch
    from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
    from vllm.model_executor.layers.fused_moe.experts.trtllm_lora_moe import (
        _TrtLlmLoRAExpertsBase,
    )
    from vllm.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe import (
        TrtLlmMxfp4ExpertsBase,
    )
    from vllm.model_executor.layers.fused_moe.utils import fi_moe_largest_bucket
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Static

    class TrtLlmMxfp4LoRAExperts(_TrtLlmLoRAExpertsBase):
        """MXFP4 weights, BF16 activations, TRT-LLM-Gen routed MoE + LoRA (W13 delta fused into FC1)."""

        def __init__(self, moe_config, quant_config, **kwargs):
            super().__init__(moe_config, quant_config)
            # gemm1 alpha / beta / clamp-limit tensors (and SITU mapping) exactly as the non-LoRA MXFP4 experts.
            TrtLlmMxfp4ExpertsBase.__init__(self, moe_config, quant_config)
            self.activation = moe_config.activation

        @staticmethod
        def _supports_quant_scheme(weight_key, activation_key) -> bool:
            return weight_key == kMxfp4Static and activation_key is None

        @staticmethod
        def _supports_activation(activation) -> bool:
            return TrtLlmMxfp4ExpertsBase._supports_activation(activation)

        @staticmethod
        def _supports_routing_method(routing_method, weight_key, activation_key) -> bool:
            return True  # routing is done outside the modular kernel

        def _ep_expert_map(self, global_num_experts, device):
            # Expert parallelism: the TRT-LLM kernel takes the local expert range (local_expert_offset) and no
            # expert_map, but the LoRA kernels index this rank's stacked LoRA weights by expert id, so global
            # topk_ids (0..E-1) must be mapped to local slots (other ranks' experts -> -1, skipped) or they read past
            # the local_num_experts LoRA buffers (illegal memory access). Same contiguous placement as the kernel.
            cached = getattr(self, "_ep_map", None)
            if cached is None or cached.device != device or cached.numel() != global_num_experts:
                start = self.ep_rank * self.local_num_experts
                cached = torch.full((global_num_experts,), -1, dtype=torch.int32, device=device)
                cached[start : start + self.local_num_experts] = torch.arange(
                    self.local_num_experts, dtype=torch.int32, device=device
                )
                self._ep_map = cached
            return cached

        def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation, global_num_experts,
                  expert_map=None, *args, **kwargs):
            if expert_map is None and self.local_num_experts < global_num_experts:
                expert_map = self._ep_expert_map(global_num_experts, topk_ids.device)
            # MoE layers that are never LoRA-wrapped (e.g. a speculative-decoding draft model's experts) still get
            # this class, since LoRA is on engine-wide, but no LoRA context is ever set on them: run the base MoE,
            # as the base class does for a batch with no adapter.
            if getattr(self, "_lora_context", None) is None:
                self.invoke_routed_moe(
                    hidden_states=hidden_states,
                    w1=w1,
                    w2=w2,
                    topk_ids_and_weights=(topk_ids, topk_weights),
                    gemm1_lora_delta=None,
                    global_num_experts=global_num_experts,
                    a1q_scale=None,
                    output=output,
                )
                return
            ctx = self._lora_context
            num_tokens = hidden_states.size(0)
            chunk = _chunk_tokens()
            if num_tokens <= chunk:
                return super().apply(
                    output, hidden_states, w1, w2, topk_weights, topk_ids, activation, global_num_experts,
                    expert_map, *args, **kwargs
                )
            # DP attention + EP: the naive all-gather hands every rank all DP ranks' tokens (8 x 32768 = 262144 in
            # the startup dummy run). The LoRA path allocates [T, top_k, 2I] and [T, top_k, H] bf16 buffers
            # (~15 GB each there, which also eats the KV cache budget) and the kernels index them with int32, so run
            # it over token chunks. Chunks are whole tokens: routing, LoRA mapping and finalize stay per-token.
            mapping, orig = ctx.local_token_lora_mapping, ctx.original_hidden_states
            try:
                for s in range(0, num_tokens, chunk):
                    e = min(s + chunk, num_tokens)
                    if mapping is not None:
                        ctx.local_token_lora_mapping = mapping[s:e]
                    if orig is not None:
                        ctx.original_hidden_states = orig[s:e]
                    super().apply(
                        output[s:e], hidden_states[s:e], w1, w2, topk_weights[s:e], topk_ids[s:e], activation,
                        global_num_experts, expert_map, *args, **kwargs
                    )
            finally:
                ctx.local_token_lora_mapping, ctx.original_hidden_states = mapping, orig

        @staticmethod
        def _batch_has_no_lora(lora_context) -> bool:
            # vLLM reads the local batch's no-LoRA flag, but under EP this rank's experts also serve the other DP
            # ranks' tokens (their dispatched LoRA ids are in local_token_lora_mapping): never skip LoRA there.
            if lora_context.local_token_lora_mapping is not None:
                return False
            return _TrtLlmLoRAExpertsBase._batch_has_no_lora(lora_context)

        def invoke_routed_moe(
            self,
            *,
            hidden_states,
            w1,
            w2,
            topk_ids_and_weights,
            gemm1_lora_delta,
            global_num_experts,
            a1q_scale,
            output,
        ):
            from flashinfer import trtllm_fp4_block_scale_routed_moe

            topk_ids, topk_weights = topk_ids_and_weights
            do_finalize = gemm1_lora_delta is None
            ret = trtllm_fp4_block_scale_routed_moe(
                topk_ids=(topk_ids.to(torch.int32), topk_weights),
                routing_bias=None,
                hidden_states=hidden_states,
                hidden_states_scale=None,
                gemm1_weights=w1,
                gemm1_weights_scale=self.w1_scale,
                gemm1_bias=self.w1_bias,
                gemm1_alpha=self.gemm1_alpha,
                gemm1_beta=self.gemm1_beta,
                gemm1_clamp_limit=self.gemm1_clamp_limit,
                gemm2_weights=w2,
                gemm2_weights_scale=self.w2_scale,
                gemm2_bias=self.w2_bias,
                output1_scale_scalar=None,
                output1_scale_gate_scalar=None,
                output2_scale_scalar=None,
                num_experts=global_num_experts,
                top_k=self.topk,
                n_group=None,
                topk_group=None,
                intermediate_size=self.intermediate_size_per_partition,
                local_expert_offset=self.ep_rank * self.local_num_experts,
                local_num_experts=self.local_num_experts,
                routed_scaling_factor=None,
                routing_method_type=RoutingMethodType.Renormalize,  # pre-routed, as the non-LoRA modular path
                do_finalize=do_finalize,
                enable_pdl=True,
                activation_type=TrtLlmMxfp4ExpertsBase._flashinfer_activation_type(self.activation),
                output=output if do_finalize else None,
                tune_max_num_tokens=fi_moe_largest_bucket(self.moe_config),
                gemm1_lora_delta=gemm1_lora_delta,
            )
            if not do_finalize:
                # [gemm2_output(permuted), expert_weights, expanded_idx_to_permuted_idx, gemm1_activation(permuted)]
                return list(ret)
            return [output]

    return TrtLlmMxfp4LoRAExperts


def apply_mxfp4_trtllm_lora_moe_patch() -> bool:
    """Register the class under FLASHINFER_TRTLLM_MXFP4_BF16 when the env var is set. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return True
    if os.environ.get(ENV, "0").lower() not in ("1", "true"):
        return False
    try:
        import inspect

        import flashinfer.fused_moe as fi_moe
        from vllm.model_executor.layers.fused_moe.oracle import mxfp4 as oracle
    except ImportError:
        return False
    fp4_routed = getattr(fi_moe, "trtllm_fp4_block_scale_routed_moe", None)
    if fp4_routed is None or "gemm1_lora_delta" not in inspect.signature(fp4_routed).parameters:
        logger.warning(f"{ENV}=1 but FlashInfer's FP4 routed MoE has no gemm1_lora_delta; keeping vLLM's MoE backends")
        return False
    lora_cls = _build_class()
    original = oracle.backend_to_kernel_cls

    def backend_to_kernel_cls(backend):
        classes = list(original(backend))
        if backend == oracle.Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16 and lora_cls not in classes:
            classes.append(lora_cls)  # after the non-LoRA classes: picked only when LoRA rejects them
        return classes

    oracle.backend_to_kernel_cls = backend_to_kernel_cls

    # The generic (non-gpt-oss) priority list, which DeepSeek-V4 / V4.1 use, has no BF16-activation TRT-LLM entry:
    # put it just before Marlin. Without LoRA nothing changes (the earlier MXFP8 TRT-LLM entry still wins); with LoRA
    # every earlier class rejects LoRA, so the class above is reached before Marlin.
    original_priority = oracle._get_priority_backends
    bf16_trtllm = oracle.Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16

    def _get_priority_backends():
        backends = list(original_priority())
        if bf16_trtllm not in backends and oracle.Mxfp4MoeBackend.MARLIN in backends:
            backends.insert(backends.index(oracle.Mxfp4MoeBackend.MARLIN), bf16_trtllm)
        return backends

    oracle._get_priority_backends = _get_priority_backends
    _APPLIED = True
    logger.info("MXFP4 MoE + LoRA: FlashInfer TRT-LLM experts (gemm1_lora_delta) registered ahead of Marlin")
    return True

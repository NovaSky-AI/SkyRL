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

Opt-in: ``SKYRL_VLLM_MXFP4_TRTLLM_LORA_MOE=1`` (applied in every vLLM worker by ``new_inference_worker_wrap``; the
DeepSeek-V4.1 recipe sets it).
Remove when vLLM ships an MXFP4 TRT-LLM LoRA experts class.
"""

import os

from loguru import logger

ENV = "SKYRL_VLLM_MXFP4_TRTLLM_LORA_MOE"
_APPLIED = False


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

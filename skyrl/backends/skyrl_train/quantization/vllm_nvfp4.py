"""vLLM 0.23 per-token NVFP4 activation support."""

from __future__ import annotations

import os

import torch

_PER_TOKEN_ACTIVATION_ENV = "SKYRL_VLLM_NVFP4_PER_TOKEN_ACTIVATION"
_PER_TOKEN_BASE_GLOBAL_SCALE = 1.0 / (448.0 * 6.0)
_PATCHED = False


def _materialize_expanded_parameter(layer: torch.nn.Module, name: str) -> None:
    """Replace zero-stride parameter storage with writable contiguous storage."""

    param = getattr(layer, name, None)
    if param is None or not any(size > 1 and stride == 0 for size, stride in zip(param.shape, param.stride())):
        return
    param.data = param.data.clone(memory_format=torch.contiguous_format)


def install_vllm_nvfp4_per_token_patch() -> None:
    """Enable per-token NVFP4 activations for vLLM's TRT-LLM MoE backend."""

    global _PATCHED
    if _PATCHED:
        return
    try:
        from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
            TrtLlmNvFp4ExpertsBase,
            TrtLlmNvFp4ExpertsModular,
            TrtLlmNvFp4ExpertsMonolithic,
        )
        from vllm.model_executor.layers.fused_moe.utils import (
            trtllm_moe_pack_topk_ids_weights,
        )
        from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
            activation_to_flashinfer_int,
        )
    except ImportError as exc:
        if os.getenv(_PER_TOKEN_ACTIVATION_ENV, "0") == "1":
            raise RuntimeError("W4A4 NVFP4 rollout requires vLLM's TRT-LLM NVFP4 MoE backend") from exc
        return

    original_init = TrtLlmNvFp4ExpertsBase.__init__
    original_process_weights = TrtLlmNvFp4ExpertsBase.process_weights_after_loading
    original_monolithic_support = TrtLlmNvFp4ExpertsMonolithic._supports_parallel_config
    original_monolithic_apply = TrtLlmNvFp4ExpertsMonolithic.apply
    original_workspace_shapes = TrtLlmNvFp4ExpertsModular.workspace_shapes
    original_invoke_kernel = TrtLlmNvFp4ExpertsModular._invoke_kernel
    original_apply = TrtLlmNvFp4ExpertsModular.apply

    def patched_init(self, moe_config, quant_config):
        original_init(self, moe_config, quant_config)
        self.per_token_activation = os.getenv(_PER_TOKEN_ACTIVATION_ENV, "0") == "1"

    def patched_process_weights(self, layer):
        _materialize_expanded_parameter(layer, "w13_input_scale")
        _materialize_expanded_parameter(layer, "w2_input_scale")
        return original_process_weights(self, layer)

    @staticmethod
    def patched_monolithic_support(moe_parallel_config):
        if os.getenv(_PER_TOKEN_ACTIVATION_ENV, "0") == "1":
            return False
        return original_monolithic_support(moe_parallel_config)

    @property
    def expects_unquantized_inputs(self) -> bool:
        return self.per_token_activation

    def quantize_per_token_input(self, hidden_states):
        from flashinfer import SfLayout, nvfp4_quantize

        return nvfp4_quantize(
            hidden_states,
            _PER_TOKEN_BASE_GLOBAL_SCALE,
            sfLayout=SfLayout.layout_linear,
            per_token_activation=True,
        )

    def patched_workspace_shapes(
        self,
        M,
        N,
        K,
        topk,
        global_num_experts,
        local_num_experts,
        expert_tokens_meta,
        activation,
    ):
        if not self.per_token_activation:
            return original_workspace_shapes(
                self,
                M,
                N,
                K,
                topk,
                global_num_experts,
                local_num_experts,
                expert_tokens_meta,
                activation,
            )
        return (0,), (0,), (M, self.hidden_dim)

    def patched_invoke_kernel(
        self,
        output,
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation,
        global_num_experts,
        a1q_scale,
    ):
        if not self.per_token_activation:
            return original_invoke_kernel(
                self,
                output,
                hidden_states,
                w1,
                w2,
                topk_weights,
                topk_ids,
                activation,
                global_num_experts,
                a1q_scale,
            )

        import flashinfer

        assert self.quant_config.w1_scale is not None
        assert self.quant_config.w2_scale is not None
        hidden_states, block_scale, per_token_scale = quantize_per_token_input(self, hidden_states)
        packed_tensor = trtllm_moe_pack_topk_ids_weights(topk_ids, topk_weights)
        flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe(
            topk_ids=packed_tensor,
            routing_bias=None,
            hidden_states=hidden_states,
            hidden_states_scale=block_scale.view(torch.float8_e4m3fn).reshape(*hidden_states.shape[:-1], -1),
            gemm1_weights=w1,
            gemm1_weights_scale=self.quant_config.w1_scale.view(torch.float8_e4m3fn),
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=self.gemm1_clamp_limit,
            gemm2_weights=w2,
            gemm2_weights_scale=self.quant_config.w2_scale.view(torch.float8_e4m3fn),
            gemm2_bias=None,
            output1_scale_scalar=self.g1_scale_c,
            output1_scale_gate_scalar=self.quant_config.g1_alphas,
            output2_scale_scalar=self.quant_config.g2_alphas,
            num_experts=global_num_experts,
            top_k=self.topk,
            n_group=0,
            topk_group=0,
            intermediate_size=self.intermediate_size_per_partition,
            local_expert_offset=self.ep_rank * self.local_num_experts,
            local_num_experts=self.local_num_experts,
            routed_scaling_factor=None,
            routing_method_type=1,
            do_finalize=True,
            activation_type=activation_to_flashinfer_int(activation),
            per_token_scale=per_token_scale,
            output=output,
        )

    def patched_apply(
        self,
        output,
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation,
        global_num_experts,
        expert_map,
        a1q_scale,
        a2_scale,
        workspace13,
        workspace2,
        expert_tokens_meta,
        apply_router_weight_on_input,
    ):
        if not self.per_token_activation:
            return original_apply(
                self,
                output,
                hidden_states,
                w1,
                w2,
                topk_weights,
                topk_ids,
                activation,
                global_num_experts,
                expert_map,
                a1q_scale,
                a2_scale,
                workspace13,
                workspace2,
                expert_tokens_meta,
                apply_router_weight_on_input,
            )

        assert self._supports_activation(activation)
        chunk_size = self._get_chunk_size()
        if chunk_size >= hidden_states.shape[0]:
            patched_invoke_kernel(
                self,
                output,
                hidden_states,
                w1,
                w2,
                topk_weights,
                topk_ids,
                activation,
                global_num_experts,
                None,
            )
            return

        for start in range(0, hidden_states.shape[0], chunk_size):
            end = min(start + chunk_size, hidden_states.shape[0])
            patched_invoke_kernel(
                self,
                output[start:end],
                hidden_states[start:end],
                w1,
                w2,
                topk_weights[start:end],
                topk_ids[start:end],
                activation,
                global_num_experts,
                None,
            )

    def patched_monolithic_apply(
        self,
        hidden_states,
        w1,
        w2,
        router_logits,
        activation,
        global_num_experts,
        expert_map,
        a1q_scale,
        apply_router_weight_on_input,
        num_expert_group=None,
        e_score_correction_bias=None,
        routed_scaling_factor=None,
        topk_group=None,
    ):
        if not self.per_token_activation:
            return original_monolithic_apply(
                self,
                hidden_states,
                w1,
                w2,
                router_logits,
                activation,
                global_num_experts,
                expert_map,
                a1q_scale,
                apply_router_weight_on_input,
                num_expert_group,
                e_score_correction_bias,
                routed_scaling_factor,
                topk_group,
            )

        import flashinfer

        assert self._supports_activation(activation)
        assert self.quant_config.w1_scale is not None
        assert self.quant_config.w2_scale is not None
        hidden_states, block_scale, per_token_scale = quantize_per_token_input(self, hidden_states)
        return flashinfer.fused_moe.trtllm_fp4_block_scale_moe(
            routing_logits=router_logits,
            routing_bias=e_score_correction_bias,
            hidden_states=hidden_states,
            hidden_states_scale=block_scale.view(torch.float8_e4m3fn).reshape(*hidden_states.shape[:-1], -1),
            gemm1_weights=w1,
            gemm1_weights_scale=self.quant_config.w1_scale.view(torch.float8_e4m3fn),
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=self.gemm1_clamp_limit,
            gemm2_weights=w2,
            gemm2_weights_scale=self.quant_config.w2_scale.view(torch.float8_e4m3fn),
            gemm2_bias=None,
            output1_scale_scalar=self.g1_scale_c,
            output1_scale_gate_scalar=self.quant_config.g1_alphas,
            output2_scale_scalar=self.quant_config.g2_alphas,
            num_experts=global_num_experts,
            top_k=self.topk,
            n_group=(num_expert_group or 0),
            topk_group=(topk_group or 0),
            intermediate_size=self.intermediate_size_per_partition,
            local_expert_offset=self.ep_rank * self.local_num_experts,
            local_num_experts=self.local_num_experts,
            routed_scaling_factor=routed_scaling_factor,
            routing_method_type=self.routing_method_type,
            do_finalize=True,
            activation_type=activation_to_flashinfer_int(activation),
            per_token_scale=per_token_scale,
        )[0]

    TrtLlmNvFp4ExpertsBase.__init__ = patched_init
    TrtLlmNvFp4ExpertsBase.process_weights_after_loading = patched_process_weights
    TrtLlmNvFp4ExpertsMonolithic._supports_parallel_config = patched_monolithic_support
    TrtLlmNvFp4ExpertsBase.expects_unquantized_inputs = expects_unquantized_inputs
    TrtLlmNvFp4ExpertsModular.workspace_shapes = patched_workspace_shapes
    TrtLlmNvFp4ExpertsModular._invoke_kernel = patched_invoke_kernel
    TrtLlmNvFp4ExpertsModular.apply = patched_apply
    TrtLlmNvFp4ExpertsMonolithic.apply = patched_monolithic_apply
    _PATCHED = True

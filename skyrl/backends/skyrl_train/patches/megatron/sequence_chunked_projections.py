from types import MethodType

import torch
import torch.nn.functional as F

from skyrl.backends.skyrl_train.patches.megatron.swiglu_triton import (
    install_triton_swiglu,
)


def _wrap_lora_linear_forward(module: torch.nn.Module, chunk_size: int) -> None:
    original_forward = module.forward

    def forward(
        self: torch.nn.Module,
        hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ):
        if hidden_states.shape[0] <= chunk_size or not self._adapter_enabled:
            return original_forward(hidden_states, *args, **kwargs)

        linear_output, bias, layernorm_output = self.base_linear_forward(
            hidden_states, *args, **kwargs
        )
        combined_output = torch.empty_like(linear_output)
        for start in range(0, hidden_states.shape[0], chunk_size):
            end = min(start + chunk_size, hidden_states.shape[0])
            adapter_output = self.adapter_forward(
                self.adapter,
                layernorm_output[start:end].contiguous(),
                *args,
                **kwargs,
            )
            output_slice = linear_output[start:end]
            combined_output[start:end].copy_(
                output_slice + adapter_output.reshape(output_slice.shape)
            )
        if not self._base_returns_tuple:
            return combined_output
        return combined_output, bias

    module.forward = MethodType(forward, module)
    module._skyrl_lora_sequence_chunked = True


def install_sequence_chunked_projections(
    model_chunks: list[torch.nn.Module], chunk_size: int
) -> tuple[int, int]:
    """Chunk dense SwiGLU and recurrent GDN blocks along the sequence axis."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    from megatron.bridge.peft.lora_layers import LoRALinear
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet, GatedDeltaNet2
    from megatron.core.transformer.mlp import MLP

    from skyrl.backends.skyrl_train.patches.megatron.gdn_sequence_chunking import (
        wrap_gdn_forward,
    )

    modules = [
        module for model_chunk in model_chunks for module in model_chunk.modules()
    ]
    swiglu_modules = [
        module
        for module in modules
        if isinstance(module, MLP)
        and module.config.gated_linear_unit
        and module.config.activation_func == F.silu
    ]
    for module in swiglu_modules:
        config = module.config
        if (
            config.activation_func_fp8_input_store
            or (
                config.cpu_offloading
                and getattr(config, "cpu_offloading_activations", False)
            )
            or config.activation_func_clamp_value is not None
            or getattr(config, "activation_func_tanh_clamp_scale", None) is not None
            or getattr(config, "activation_func_tanh_clamp_scale_linear", None)
            is not None
        ):
            raise ValueError(
                "Sequence-chunked Triton SwiGLU does not support activation storage or clamps"
            )
    if swiglu_modules:
        install_triton_swiglu()

    lora_count = 0
    gdn_count = 0
    for module in modules:
        if isinstance(module, LoRALinear) and not getattr(
            module, "_skyrl_lora_sequence_chunked", False
        ):
            _wrap_lora_linear_forward(module, chunk_size)
            lora_count += 1
        if isinstance(module, (GatedDeltaNet, GatedDeltaNet2)) and not getattr(
            module, "_skyrl_sequence_chunked", False
        ):
            if module.tp_size != 1 or module.cp_size != 1 or module.sp_size != 1:
                raise ValueError(
                    "Sequence-chunked GDN currently requires TP1, CP1, and sequence_parallel=False"
                )
            wrap_gdn_forward(module, chunk_size)
            gdn_count += 1

    return lora_count, gdn_count

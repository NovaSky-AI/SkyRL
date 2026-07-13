import inspect
from collections.abc import Callable
from typing import Any

from torch import Tensor


def fused_lm_head_output_processor(
    *,
    hidden_states: Tensor,
    output_layer: Any,
    output_weight: Tensor | None = None,
    context: dict[str, Any] | None = None,
    **_: Any,
) -> Tensor:
    """Return hidden states and retain the LM-head weight for fused log-probs."""
    if context is not None:
        context["lm_head_weight"] = output_weight if output_weight is not None else output_layer.weight

    if output_layer.sequence_parallel:
        from megatron.core.tensor_parallel import gather_from_sequence_parallel_region

        hidden_states = gather_from_sequence_parallel_region(
            hidden_states,
            tensor_parallel_output_grad=True,
            group=output_layer.tp_group,
        )
    elif output_layer.allreduce_dgrad:
        from megatron.core.tensor_parallel import copy_to_tensor_model_parallel_region

        hidden_states = copy_to_tensor_model_parallel_region(hidden_states, group=output_layer.tp_group)

    return hidden_states.transpose(0, 1).contiguous()


def call_model_with_fused_lm_head(
    model: Any,
    *args: Any,
    output_processor: Callable[..., Any],
    output_processor_context: dict[str, Any],
    **kwargs: Any,
) -> Any:
    """Run GPT-style or HybridModel forwards without materializing logits."""
    # Megatron is optional outside the Megatron backend, so keep these imports lazy.
    from megatron.core.fp8_utils import is_mxfp8_output_proj_active
    from megatron.core.models.hybrid.hybrid_model import HybridModel
    from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
        FineGrainedActivationOffloadingInterface as off_interface,
    )
    from megatron.core.utils import unwrap_model

    unwrapped_model = unwrap_model(model)
    if is_mxfp8_output_proj_active(unwrapped_model.config):
        raise NotImplementedError("fused_lm_head_logprob does not support MXFP8 output projection")
    if unwrapped_model.config.use_mup:
        raise NotImplementedError("fused_lm_head_logprob does not support MuP logit scaling")
    if "output_processor" in inspect.signature(unwrapped_model.forward).parameters:
        return model(
            *args,
            output_processor=output_processor,
            output_processor_context=output_processor_context,
            **kwargs,
        )
    if not isinstance(unwrapped_model, HybridModel):
        raise NotImplementedError(f"fused_lm_head_logprob is not supported for {type(unwrapped_model).__name__}")

    if unwrapped_model.config.mtp_num_layers:
        raise NotImplementedError("fused_lm_head_logprob does not support HybridModel with MTP enabled")
    if not unwrapped_model.post_process:
        return model(*args, **kwargs)

    if unwrapped_model.config.fine_grained_activation_offloading:
        for parameter in unwrapped_model.output_layer.parameters():
            off_interface.mark_not_offloadable(parameter)

    unwrapped_model.post_process = False
    try:
        hidden_states = model(*args, **kwargs)
    finally:
        unwrapped_model.post_process = True

    output_weight = (
        unwrapped_model.shared_embedding_or_output_weight()
        if unwrapped_model.share_embeddings_and_output_weights
        else None
    )
    return output_processor(
        hidden_states=hidden_states,
        output_layer=unwrapped_model.output_layer,
        output_weight=output_weight,
        context=output_processor_context,
    )

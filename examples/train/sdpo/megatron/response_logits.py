"""Response-only vocabulary projection for Megatron's output-processor hook."""

import torch


def response_logits_output_processor(
    *,
    hidden_states,
    output_layer,
    output_weight=None,
    context,
    runtime_gather_output=None,
    scale_logits=None,
    **kwargs,
):
    """Select prediction positions before the TP-sharded vocabulary projection.

    Sequence-parallel inputs are gathered, selected, then re-sharded so the
    existing output layer retains its normal gradient-reduction semantics.
    """
    slots = context["response_positions"]  # [batch, response]
    sequence_parallel = output_layer.sequence_parallel
    if sequence_parallel:
        from megatron.core.tensor_parallel import gather_from_sequence_parallel_region

        # The subsequent scatter's backward gathers the complete selected-input
        # gradient on every rank. Split it here without summing it a second time.
        hidden_states = gather_from_sequence_parallel_region(
            hidden_states, tensor_parallel_output_grad=False, group=output_layer.tp_group
        )
    selected = hidden_states.gather(0, slots.transpose(0, 1).unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1]))
    response_length = selected.shape[0]
    if sequence_parallel:
        from megatron.core.tensor_parallel import scatter_to_sequence_parallel_region

        size = torch.distributed.get_world_size(output_layer.tp_group)
        padding = (-response_length) % size
        if padding:
            selected = torch.cat((selected, selected[-1:].expand(padding, -1, -1)), dim=0)
        selected = scatter_to_sequence_parallel_region(selected.contiguous(), group=output_layer.tp_group)
    logits, _ = output_layer(selected, weight=output_weight, runtime_gather_output=runtime_gather_output)
    if scale_logits is not None:
        logits = scale_logits(logits)
    return logits[:response_length].transpose(0, 1).contiguous()

from collections.abc import Callable
from types import MethodType

import torch


class _StatefulSequenceChunkedFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        chunk_size: int,
        fn: Callable[..., tuple[torch.Tensor, ...]],
        *parameters: torch.Tensor,
    ) -> torch.Tensor:
        ctx.chunk_size = chunk_size
        ctx.fn = fn
        ctx.parameter_count = len(parameters)

        outputs = []
        boundary_states = []
        states: tuple[torch.Tensor, ...] = ()
        for start in range(0, hidden_states.shape[0], chunk_size):
            result = fn(hidden_states[start : start + chunk_size], *states)
            output, *next_states = result
            states = tuple(next_states)
            outputs.append(output)
            if start + chunk_size < hidden_states.shape[0]:
                boundary_states.extend(states)

        ctx.state_count = len(states)
        ctx.boundary_count = len(boundary_states)
        ctx.save_for_backward(hidden_states, *boundary_states, *parameters)
        return torch.cat(outputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        saved = ctx.saved_tensors
        hidden_states = saved[0]
        boundary_states = saved[1 : 1 + ctx.boundary_count]
        parameters = saved[1 + ctx.boundary_count :]
        hidden_states_grad = (
            torch.empty_like(hidden_states) if ctx.needs_input_grad[0] else None
        )
        parameter_grads: list[torch.Tensor | None] = [None] * ctx.parameter_count
        final_state_grads: tuple[torch.Tensor | None, ...] | None = None
        chunk_count = (hidden_states.shape[0] + ctx.chunk_size - 1) // ctx.chunk_size

        for chunk_index in reversed(range(chunk_count)):
            start = chunk_index * ctx.chunk_size
            end = min(start + ctx.chunk_size, hidden_states.shape[0])
            chunk = hidden_states[start:end].detach().requires_grad_(True)
            if chunk_index == 0:
                initial_states: tuple[torch.Tensor, ...] = ()
            else:
                state_start = (chunk_index - 1) * ctx.state_count
                initial_states = tuple(
                    state.detach().requires_grad_(True)
                    for state in boundary_states[
                        state_start : state_start + ctx.state_count
                    ]
                )

            with torch.enable_grad():
                result = ctx.fn(chunk, *initial_states)
            output, *next_states = result
            outputs = [output]
            output_grads = [grad_output[start:end]]
            if final_state_grads is not None:
                outputs.extend(next_states)
                output_grads.extend(
                    torch.zeros_like(state) if grad is None else grad
                    for state, grad in zip(
                        next_states, final_state_grads, strict=True
                    )
                )
            grads = torch.autograd.grad(
                outputs,
                (chunk, *initial_states, *parameters),
                output_grads,
                allow_unused=True,
            )
            if hidden_states_grad is not None:
                hidden_states_grad[start:end].copy_(grads[0])
            final_state_grads = grads[1 : 1 + len(initial_states)]
            for index, grad in enumerate(grads[1 + len(initial_states) :]):
                if grad is None:
                    continue
                if parameter_grads[index] is None:
                    parameter_grads[index] = grad.detach()
                else:
                    parameter_grads[index].add_(grad)

        return hidden_states_grad, None, None, *parameter_grads


def apply_stateful_sequence_chunked(
    fn: Callable[..., tuple[torch.Tensor, ...]],
    hidden_states: torch.Tensor,
    chunk_size: int,
    parameters: tuple[torch.Tensor, ...] | None = None,
) -> torch.Tensor:
    """Run a recurrent sequence function in chunks while carrying its states."""
    if not torch.is_grad_enabled():
        outputs = []
        states: tuple[torch.Tensor, ...] = ()
        for start in range(0, hidden_states.shape[0], chunk_size):
            output, *states = fn(hidden_states[start : start + chunk_size], *states)
            outputs.append(output)
        return torch.cat(outputs, dim=0)

    if parameters is None:
        module = getattr(fn, "__self__", None)
        if module is None:
            raise ValueError("Stateful sequence chunking requires parameters")
        parameters = tuple(
            parameter for parameter in module.parameters() if parameter.requires_grad
        )
    return _StatefulSequenceChunkedFunction.apply(
        hidden_states, chunk_size, fn, *parameters
    )


def _run_gdn_chunk(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    *states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from megatron.core.ssm.gated_delta_net.common import causal_conv1d

    conv_state, recurrent_state = states if states else (None, None)
    seq_len, batch, _ = hidden_states.shape

    qkvzba, _ = module.in_proj(hidden_states)
    qkvzba = qkvzba.transpose(0, 1)
    qkv, gate, *gate_feats = torch.split(qkvzba, module.feat_dim_split, dim=-1)
    gate = gate.reshape(batch, seq_len, -1, module.value_head_dim)

    qkv, conv_state = causal_conv1d(
        x=qkv,
        weight=module.conv1d.weight.squeeze(1),
        bias=module.conv1d.bias if module.conv_bias else None,
        activation=module.activation,
        initial_state=conv_state,
        output_final_state=True,
    )

    kernel_inputs = module._prepare_input_for_gated_delta_rule(
        qkv,
        gate,
        module.A_log,
        module.dt_bias,
        batch,
        seq_len,
        *gate_feats,
    )
    gate = kernel_inputs.pop("gate")
    core_attn_out, recurrent_state = module.gated_delta_rule(
        **kernel_inputs,
        initial_state=recurrent_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
    )

    norm_out = module._apply_gated_norm(core_attn_out, gate)
    norm_out = norm_out.reshape(batch, seq_len, -1).transpose(0, 1).contiguous()
    output, output_bias = module.out_proj(norm_out)
    if output_bias is not None and output_bias.numel() != 0:
        raise ValueError("Sequence-chunked GDN requires a bias-free output projection")
    return output, conv_state, recurrent_state


def _get_packed_sequence_ranges(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    packed_seq_params,
) -> list[tuple[int, int]]:
    if packed_seq_params is None:
        return [(0, hidden_states.shape[0])]
    if packed_seq_params.qkv_format != "thd":
        raise ValueError(
            "Sequence-chunked GDN only supports packed qkv_format='thd'"
        )
    if hidden_states.shape[1] != 1:
        raise ValueError("Packed sequence-chunked GDN requires batch size 1")
    cu_seqlens_q = module._resolve_cu_seqlens(
        packed_seq_params.cu_seqlens_q_padded,
        packed_seq_params.cu_seqlens_q,
        hidden_states.shape[0],
        "cu_seqlens_q",
    )
    cu_seqlens_kv = module._resolve_cu_seqlens(
        packed_seq_params.cu_seqlens_kv_padded,
        packed_seq_params.cu_seqlens_kv,
        hidden_states.shape[0],
        "cu_seqlens_kv",
    )
    if not torch.equal(cu_seqlens_q, cu_seqlens_kv):
        raise ValueError("Sequence-chunked GDN requires equal q and kv lengths")
    boundaries = cu_seqlens_q.tolist()
    return list(zip(boundaries[:-1], boundaries[1:], strict=True))


def wrap_gdn_forward(module: torch.nn.Module, chunk_size: int) -> None:
    """Replace one TP1/CP1 GDN forward with state-carrying Triton chunks."""
    original_forward = module.forward

    def forward(
        self: torch.nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        inference_context=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
        **kwargs,
    ):
        if (
            hidden_states.shape[0] <= chunk_size
            or inference_context is not None
            or inference_params is not None
        ):
            return original_forward(
                hidden_states,
                attention_mask,
                inference_context,
                packed_seq_params,
                sequence_len_offset,
                inference_params=inference_params,
                **kwargs,
            )

        outputs = []
        for start, end in _get_packed_sequence_ranges(
            self, hidden_states, packed_seq_params
        ):

            def run_chunk(chunk: torch.Tensor, *states: torch.Tensor):
                return _run_gdn_chunk(self, chunk, *states)

            outputs.append(
                apply_stateful_sequence_chunked(
                    run_chunk,
                    hidden_states[start:end],
                    chunk_size,
                    tuple(
                        parameter
                        for parameter in self.parameters()
                        if parameter.requires_grad
                    ),
                )
            )
        return torch.cat(outputs, dim=0), None

    module.forward = MethodType(forward, module)
    module._skyrl_sequence_chunked = True

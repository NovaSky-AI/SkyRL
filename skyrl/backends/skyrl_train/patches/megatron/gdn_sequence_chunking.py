from collections.abc import Callable
from types import MethodType

import torch
from torch.utils.checkpoint import checkpoint


def apply_stateful_sequence_chunked(
    fn: Callable[..., tuple[torch.Tensor, ...]],
    hidden_states: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Run a recurrent sequence function in chunks while carrying its states."""
    outputs = []
    states: tuple[torch.Tensor, ...] = ()
    for start in range(0, hidden_states.shape[0], chunk_size):
        chunk = hidden_states[start : start + chunk_size]
        if torch.is_grad_enabled():
            result = checkpoint(fn, chunk, *states, use_reentrant=False)
        else:
            result = fn(chunk, *states)
        output, *states = result
        outputs.append(output)
    return torch.cat(outputs, dim=0)


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
                    run_chunk, hidden_states[start:end], chunk_size
                )
            )
        return torch.cat(outputs, dim=0), None

    module.forward = MethodType(forward, module)
    module._skyrl_sequence_chunked = True

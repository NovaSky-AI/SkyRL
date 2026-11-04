"""Full activation recompute for DeepSeek-V4.1 that carries the cross-layer state (stateful layer checkpoint).

V4.1 layers share state across the stack: KV-source layers publish compressed KV (``global_kv``, in the autograd
graph: consumers backpropagate into the source's compressor) and indexer keys, index-source layers publish top-k
(``global_indices``), the candidate source publishes candidate blocks, and every hyper-connection sublayer hands its
pre-mix (``pending_pre``) to the next. megatron-core's block recompute replays layers in backward against whatever
that shared state holds by then, so SkyRL used to refuse full recompute for V4.1 (attention assert) and downgrade it
to selective for mHC models -- which leaves ~2.5 MB of activations per token on an 8-layer slice, so a 32k sequence
does not fit a GPU, let alone 256k.

Shape of NVIDIA/Megatron-LM#8018's ``CSA2HybridAdapter.checkpoint_layer`` (open): each hybrid layer is checkpointed
with the state tensors as explicit inputs and outputs, so a replay starts from the layer's own original state and
autograd links a consumer's state gradient to its producer's output. Upstream uses ``torch.utils.checkpoint``
(non-reentrant); this is the same contract as an ``autograd.Function`` (megatron-core ``CheckpointFunction`` style)
so the layer input can optionally live in pinned host memory between forward and backward
(``SKYRL_DSV41_RECOMPUTE_OFFLOAD=1``): what stays on the GPU per layer is then only the shared state, which is
small and referenced once by every layer.

R3: routes are pushed once per micro-batch per router (``set_target_indices``) and popped once by the replay in
backward (``REPLAY_BACKWARD``), the same as megatron-core's own recompute.
"""

import os
from typing import Any, Callable, Optional, Sequence, Tuple

import torch

OFFLOAD = os.environ.get("SKYRL_DSV41_RECOMPUTE_OFFLOAD", "0").lower() in ("1", "true")
# CSA2State tensor fields threaded through each layer checkpoint (``attention.CSA2State``).
STATE_FIELDS = ("global_kv", "indexer_k", "global_indices", "candidate_blocks")


def _to_host(tensor: torch.Tensor) -> torch.Tensor:
    host = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=True)
    host.copy_(tensor, non_blocking=True)  # stream-ordered: the source's storage is reused only after the copy
    return host


class StatefulLayerRecompute(torch.autograd.Function):
    """Run ``run(*inputs)`` without saving activations; replay it in backward from the saved inputs.

    ``offload_first``: keep the first input (the layer's hidden state) in pinned host memory until backward.
    Floating-point outputs are differentiable (including inputs passed through unchanged); others are not.
    """

    @staticmethod
    def forward(ctx, run: Callable, offload_first: bool, *inputs: Any):
        ctx.run = run
        ctx.meta = [(t.requires_grad, t.device) if torch.is_tensor(t) else None for t in inputs]
        ctx.offload_first = offload_first and torch.is_tensor(inputs[0]) and inputs[0].is_cuda
        ctx.inputs = [
            (_to_host(t) if i == 0 and ctx.offload_first else t.detach()) if torch.is_tensor(t) else t
            for i, t in enumerate(inputs)
        ]
        with torch.no_grad():
            outputs = run(*inputs)
        ctx.mark_non_differentiable(*[o for o in outputs if torch.is_tensor(o) and not o.is_floating_point()])
        return outputs

    @staticmethod
    def backward(ctx, *grads):
        inputs = []
        for i, (t, meta) in enumerate(zip(ctx.inputs, ctx.meta)):
            if meta is None:
                inputs.append(t)
                continue
            requires_grad, device = meta
            x = t.to(device, non_blocking=True) if (i == 0 and ctx.offload_first) else t
            inputs.append(x.detach().requires_grad_(requires_grad))
        with torch.enable_grad():
            outputs = ctx.run(*inputs)
        pairs = [(o, g) for o, g in zip(outputs, grads) if torch.is_tensor(o) and o.requires_grad and g is not None]
        if pairs:
            torch.autograd.backward([o for o, _ in pairs], [g for _, g in pairs])
        ctx.run, ctx.inputs = None, None
        return (
            None,
            None,
            *[
                x.grad if torch.is_tensor(x) and meta is not None and meta[0] else None
                for x, meta in zip(inputs, ctx.meta)
            ],
        )


def stateful_layer_forward(
    layer_forward: Callable, layer, hidden_states: torch.Tensor, args: Sequence, kwargs: dict, state_getter: Callable
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """One hybrid layer under the stateful recompute: ``layer_forward(layer, hidden, *args, **kwargs)`` with the
    forward's CSA2 / mHC state passed through as checkpoint inputs and written back from its outputs."""
    from skyrl.backends.skyrl_train.patches.megatron.deepseek_v41.attention import (
        CSA2State,
    )

    state = state_getter()
    compress_ratio = state.csa2.compress_ratio
    extras: dict = {}

    def run(hidden, global_kv, indexer_k, global_indices, candidate_blocks, pending_pre):
        local = state_getter()
        local.csa2 = CSA2State(
            global_kv=global_kv,
            compress_ratio=compress_ratio,
            indexer_k=indexer_k,
            global_indices=global_indices,
            candidate_blocks=candidate_blocks,
        )
        local.pending_pre = pending_pre
        output, context = layer_forward(layer, hidden, *args, **kwargs)
        extras["compress_ratio"] = local.csa2.compress_ratio
        extras["context"] = context
        csa2 = local.csa2
        return (output, *(getattr(csa2, name) for name in STATE_FIELDS), local.pending_pre)

    outputs = StatefulLayerRecompute.apply(
        run,
        OFFLOAD,
        hidden_states,
        *(getattr(state.csa2, name) for name in STATE_FIELDS),
        state.pending_pre,
    )
    output, *new_state, pending_pre = outputs
    state.csa2 = CSA2State(compress_ratio=extras["compress_ratio"], **dict(zip(STATE_FIELDS, new_state)))
    state.pending_pre = pending_pre
    return output, extras["context"]

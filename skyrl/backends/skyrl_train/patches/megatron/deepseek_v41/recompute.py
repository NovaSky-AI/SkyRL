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
# With offload on, the inputs of the stack's last N hybrid layers stay on the GPU: their backward runs first (no
# copy back to wait for) and the host keeps N fewer layer inputs pinned.
OFFLOAD_KEEP_LAST = int(os.environ.get("SKYRL_DSV41_RECOMPUTE_OFFLOAD_KEEP_LAST", "0"))
_BUCKET = 1 << 30  # pool capacity granularity
# CSA2State tensor fields threaded through each layer checkpoint (``attention.CSA2State``).
STATE_FIELDS = ("global_kv", "indexer_k", "global_indices", "candidate_blocks")


def _default_host_limit_bytes() -> int:
    """Pinned-host budget per process: half the node's RAM over its local ranks (unswappable pages; a pool that
    outgrew this took a 2 TB node down)."""
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        return 0
    local_ranks = int(os.environ.get("LOCAL_WORLD_SIZE", "8") or 8)
    return total // 2 // max(1, local_ranks)


class _PinnedPool:
    """Pinned host buffers for offloaded layer inputs, under a hard per-process byte limit.

    All buffers share one capacity -- the largest request so far, rounded up to 1 GiB -- so once the longest
    sequence has been seen every later request reuses a pooled buffer and pinning (seconds per ~10 GB buffer) is
    paid once. Every live buffer, pooled or in use, counts against ``limit`` (``SKYRL_DSV41_RECOMPUTE_OFFLOAD_HOST_GB``,
    default half the node's RAM over 8 local ranks); a request that would exceed it gets ``None`` and the caller
    keeps that input on the GPU. When the capacity grows, smaller buffers are released as they come back and
    PyTorch's host cache emptied, so old and new buffers never add up past the limit."""

    def __init__(self):
        self.free: list = []
        self.capacity = 0
        self.live_bytes = 0
        limit_gb = os.environ.get("SKYRL_DSV41_RECOMPUTE_OFFLOAD_HOST_GB")
        self.limit = int(float(limit_gb) * (1 << 30)) if limit_gb else _default_host_limit_bytes()

    def _release(self, buf: torch.Tensor) -> None:
        self.live_bytes -= buf.numel()
        del buf
        empty_host_cache = getattr(torch._C, "_host_emptyCache", None)
        if empty_host_cache is not None:
            empty_host_cache()

    def get(self, nbytes: int) -> Optional[torch.Tensor]:
        if nbytes > self.capacity:
            self.capacity = -(-nbytes // _BUCKET) * _BUCKET
            while self.free:
                self._release(self.free.pop())
        if self.free:
            return self.free.pop()
        if self.live_bytes + self.capacity > self.limit:
            return None
        self.live_bytes += self.capacity
        return torch.empty(self.capacity, dtype=torch.uint8, device="cpu", pin_memory=True)

    def put(self, buf: torch.Tensor) -> None:
        if buf.numel() == self.capacity:
            self.free.append(buf)
        else:
            self._release(buf)


_POOL = _PinnedPool()
_COPY_STREAM = None
# Offloaded layer inputs of the current forward, in forward order; backward walks them in reverse and prefetches
# the next one (the previous layer) while the current layer replays.
_PENDING: list = []


def _copy_stream():
    global _COPY_STREAM
    if _COPY_STREAM is None:
        _COPY_STREAM = torch.cuda.Stream()
    return _COPY_STREAM


class _Offloaded:
    """One layer input parked in pinned host memory: the device-to-host copy runs on a side stream (overlapping the
    forward's compute; the source stays allocated until it finishes), and the copy back can start early."""

    @classmethod
    def try_offload(cls, tensor: torch.Tensor):
        """The offloaded input, or the tensor itself (kept on the GPU) when the pinned budget is used up."""
        buf = _POOL.get(tensor.numel() * tensor.element_size())
        return tensor.detach() if buf is None else cls(tensor, buf)

    def __init__(self, tensor: torch.Tensor, buf: torch.Tensor):
        self.shape, self.dtype, self.device = tensor.shape, tensor.dtype, tensor.device
        nbytes = tensor.numel() * tensor.element_size()
        self.buf = buf
        host = self.buf[:nbytes].view(self.dtype).view(self.shape)
        stream = _copy_stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            host.copy_(tensor, non_blocking=True)
        tensor.record_stream(stream)  # keep the source's storage until the copy is done
        self.host = host
        self.device_copy = None
        self.ready = None
        _PENDING.append(self)

    def prefetch(self) -> None:
        if self.device_copy is not None or self.host is None:
            return
        stream = _copy_stream()
        with torch.cuda.stream(stream):
            self.device_copy = self.host.to(self.device, non_blocking=True)
            self.ready = torch.cuda.Event()
            self.ready.record(stream)

    def fetch(self) -> torch.Tensor:
        self.prefetch()
        torch.cuda.current_stream().wait_event(self.ready)
        out = self.device_copy
        out.record_stream(torch.cuda.current_stream())
        self.ready.synchronize()  # the host buffer is free for reuse once the copy back has landed
        buf, self.buf, self.host, self.device_copy = self.buf, None, None, None
        _POOL.put(buf)  # after dropping our views, so a released buffer is really unpinned
        if self in _PENDING:
            index = _PENDING.index(self)
            if index > 0:
                _PENDING[index - 1].prefetch()  # the layer whose backward runs next
            del _PENDING[index:]
        return out


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
            (_Offloaded.try_offload(t) if i == 0 and ctx.offload_first else t.detach()) if torch.is_tensor(t) else t
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
            x = t.fetch() if isinstance(t, _Offloaded) else t
            inputs.append(x.detach().requires_grad_(requires_grad))
        with torch.enable_grad():
            outputs = ctx.run(*inputs)
        pairs = [(o, g) for o, g in zip(outputs, grads) if torch.is_tensor(o) and o.requires_grad and g is not None]
        if pairs:
            torch.autograd.backward([o for o, _ in pairs], [g for _, g in pairs])
        del outputs, pairs
        clear_state = getattr(ctx.run, "clear_state", None)
        if clear_state is not None:
            # The replay wrote its outputs into the shared forward state; left there they pin this replay's graph
            # (back to its input and the input's ``.grad``) until the next forward.
            clear_state()
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

    def clear_state():
        local = state_getter()
        local.csa2 = CSA2State()
        local.pending_pre = None

    run.clear_state = clear_state

    # ``config.num_layers`` counts hybrid layers (from ``hybrid_layer_pattern``), as ``layer_number`` does.
    num_layers = getattr(getattr(layer, "config", None), "num_layers", None) or 0
    layer_number = getattr(layer, "layer_number", 0)
    keep_on_gpu = OFFLOAD_KEEP_LAST > 0 and num_layers > 0 and layer_number > num_layers - OFFLOAD_KEEP_LAST
    if not hidden_states.requires_grad:
        # The layer's parameters are not inputs of the Function, so its outputs only join the graph when an input
        # does. With a frozen embedding (LoRA) nothing upstream requires grad: without this the whole stack would be
        # cut out of backward and the adapters would get no gradient (as HF ``enable_input_require_grads``).
        hidden_states = hidden_states.detach().requires_grad_(True)
    outputs = StatefulLayerRecompute.apply(
        run,
        OFFLOAD and not keep_on_gpu,
        hidden_states,
        *(getattr(state.csa2, name) for name in STATE_FIELDS),
        state.pending_pre,
    )
    output, *new_state, pending_pre = outputs
    state.csa2 = CSA2State(compress_ratio=extras["compress_ratio"], **dict(zip(STATE_FIELDS, new_state)))
    state.pending_pre = pending_pre
    return output, extras["context"]

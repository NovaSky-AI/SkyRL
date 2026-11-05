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
from contextlib import nullcontext
from typing import Any, Callable, Optional, Sequence, Tuple

import torch

OFFLOAD = os.environ.get("SKYRL_DSV41_RECOMPUTE_OFFLOAD", "0").lower() in ("1", "true")
# With offload on, the inputs of the stack's last N hybrid layers stay on the GPU: their backward runs first (no
# copy back to wait for) and the host keeps N fewer layer inputs pinned.
OFFLOAD_KEEP_LAST = int(os.environ.get("SKYRL_DSV41_RECOMPUTE_OFFLOAD_KEEP_LAST", "0"))
# Debug: directory for a per-layer forward memory log (rank 0): allocated after the layer, the layer's own peak, and
# the bytes of the hidden state and each piece of CSA2 / mHC state saved as checkpoint inputs.
MEMLOG = os.environ.get("SKYRL_DSV41_MEMLOG", "")
# Keep only the attention sublayers' inputs: an MLP/MoE sublayer's input is the preceding attention sublayer's output,
# rebuilt in backward by re-running that attention forward (no grad) from its own saved input. Halves the stored
# layer inputs (80 hybrid sublayers -> 40 inputs) for one extra attention forward per pair.
DERIVE_MLP_INPUT = os.environ.get("SKYRL_DSV41_RECOMPUTE_DERIVE_MLP_INPUT", "0").lower() in ("1", "true")
_LAST_ATTENTION_CTX: list = [None]  # the latest attention sublayer's Function ctx (forward order)


class _SavedBreakdown:
    """MEMLOG: bytes autograd saves for backward during one layer's replay, by top-level submodule (unique storages),
    and the replay's peak over the allocation it started from."""

    def __init__(self, layer):
        self.layer = layer
        self.by_module: dict = {}
        self.seen: set = set()
        self.stack: list = []
        self.handles: list = []

    def __enter__(self):
        if self.layer is None:
            return self
        names = {id(m): n for n, m in self.layer.named_modules()}

        def pre(module, args):
            name = names.get(id(module))
            if name is not None:
                self.stack.append(".".join(name.split(".")[:3]) or "<layer>")

        def post(module, args, out):
            if id(module) in names and self.stack:
                self.stack.pop()

        self.handles = [
            torch.nn.modules.module.register_module_forward_pre_hook(pre),
            torch.nn.modules.module.register_module_forward_hook(post),
        ]

        def pack(t):
            if t.is_cuda:
                key = t.untyped_storage().data_ptr()
                if key not in self.seen:
                    self.seen.add(key)
                    where = self.stack[-1] if self.stack else "<outside>"
                    self.by_module[where] = self.by_module.get(where, 0) + t.untyped_storage().nbytes()
            return t

        self.saved = torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t)
        self.saved.__enter__()
        self.start = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *exc):
        if self.layer is not None:
            self.saved.__exit__(*exc)
            for h in self.handles:
                h.remove()
        return False

    def report(self) -> None:
        import torch.distributed as dist

        if self.layer is None or (dist.is_initialized() and dist.get_rank() != 0):
            return
        top = sorted(self.by_module.items(), key=lambda kv: -kv[1])[:8]
        parts = " ".join(f"{k}={v / 2**30:.2f}" for k, v in top)
        with open(os.path.join(MEMLOG, "memlog_rank0.txt"), "a") as f:
            f.write(
                f"replay layer {getattr(self.layer, 'layer_number', '?')}: start {self.start / 2**30:.1f} GiB, peak "
                f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB, saved {sum(self.by_module.values()) / 2**30:.2f} "
                f"GiB | {parts}\n"
            )


def _memlog(layer_number, hidden, state: dict) -> None:
    import torch.distributed as dist

    if dist.is_initialized() and dist.get_rank() != 0:
        return

    def gib(t):
        return t.numel() * t.element_size() / 2**30 if torch.is_tensor(t) else 0.0

    parts = " ".join(
        f"{k}={gib(v):.2f}{list(v.shape) if torch.is_tensor(v) else ''}" for k, v in state.items() if v is not None
    )
    with open(os.path.join(MEMLOG, "memlog_rank0.txt"), "a") as f:
        f.write(
            f"layer {layer_number}: alloc {torch.cuda.memory_allocated() / 2**30:.1f} GiB, layer peak "
            f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB, offload pool {_POOL.live_bytes / 2**30:.0f} GiB, "
            f"hidden {gib(hidden):.2f} {list(hidden.shape)} | {parts}\n"
        )
_CHUNK = 1 << 30  # pinned arena chunk size
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
    """Pinned host arena for offloaded layer inputs: fixed 1 GiB chunks under a hard per-process byte limit.

    A layer input is stored across as many chunks as it needs, so any size reuses chunks pinned for earlier
    (longer or shorter) inputs. Chunks are pinned on demand and never freed, so pinning (cudaHostAlloc, ~2 GB/s:
    2.5 s per 5 GB layer input at 128k tokens) is paid only when the step's total demand passes its previous peak,
    i.e. once, however much RL batch lengths vary step to step. (A pool of whole-input buffers re-pinned every
    buffer whenever the longest sequence grew, which in RL is most steps: ~40 s of a 48 s step.)

    Every pinned chunk counts against ``limit`` (``SKYRL_DSV41_RECOMPUTE_OFFLOAD_HOST_GB``, default half the
    node's RAM over the local ranks: pinned pages can't be swapped, and a pool that outgrew this took a 2 TB node
    down). A request that would pass it gets ``None`` and the caller keeps that input on the GPU. Chunks come back
    with the CUDA event of the copy that last read them and are reused once it has completed, without blocking.
    """

    def __init__(self):
        self.free: list = []  # chunks ready for reuse
        self.returning: list = []  # (event, chunks) whose device copy may still be reading them
        self.live_bytes = 0  # pinned bytes, free or in use
        limit_gb = os.environ.get("SKYRL_DSV41_RECOMPUTE_OFFLOAD_HOST_GB")
        self.limit = int(float(limit_gb) * (1 << 30)) if limit_gb else _default_host_limit_bytes()

    def _reclaim(self, block: bool = False) -> None:
        still = []
        for event, chunks in self.returning:
            if event is None or event.query() or block:
                if event is not None and block:
                    event.synchronize()
                self.free.extend(chunks)
            else:
                still.append((event, chunks))
        self.returning = still

    def get(self, nbytes: int) -> Optional[list]:
        """``ceil(nbytes / 1 GiB)`` pinned chunks, or ``None`` if that would pass the limit."""
        need = -(-nbytes // _CHUNK)
        self._reclaim()
        if len(self.free) < need and self.returning:
            self._reclaim(block=len(self.free) + sum(len(c) for _, c in self.returning) >= need)
        missing = need - len(self.free)
        if missing > 0 and self.live_bytes + missing * _CHUNK > self.limit:
            return None
        while len(self.free) < need:
            self.free.append(torch.empty(_CHUNK, dtype=torch.uint8, device="cpu", pin_memory=True))
            self.live_bytes += _CHUNK
        return [self.free.pop() for _ in range(need)]

    def put(self, chunks: list, event=None) -> None:
        """Return chunks; with ``event``, they are reused only after it completes."""
        self.returning.append((event, chunks))


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
    """One layer input parked in pinned host memory (a list of arena chunks): the device-to-host copy runs on a
    side stream (overlapping the forward's compute; the source stays allocated until it finishes), and the copy
    back can start early (prefetch) while the next layer's backward runs."""

    @classmethod
    def try_offload(cls, tensor: torch.Tensor):
        """The offloaded input, or the tensor itself (kept on the GPU) when the pinned budget is used up."""
        chunks = _POOL.get(tensor.numel() * tensor.element_size()) if tensor.is_contiguous() else None
        return tensor.detach() if chunks is None else cls(tensor, chunks)

    def __init__(self, tensor: torch.Tensor, chunks: list):
        self.shape, self.dtype, self.device = tensor.shape, tensor.dtype, tensor.device
        self.nbytes = tensor.numel() * tensor.element_size()
        self.chunks = chunks
        src = tensor.detach().reshape(-1).view(torch.uint8)
        stream = _copy_stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for i, chunk in enumerate(chunks):
                lo = i * _CHUNK
                n = min(_CHUNK, self.nbytes - lo)
                chunk[:n].copy_(src[lo : lo + n], non_blocking=True)
        tensor.record_stream(stream)  # keep the source's storage until the copy is done
        self.device_copy = None
        self.ready = None
        _PENDING.append(self)

    def prefetch(self) -> None:
        if self.device_copy is not None or self.chunks is None:
            return
        stream = _copy_stream()
        with torch.cuda.stream(stream):
            flat = torch.empty(self.nbytes, dtype=torch.uint8, device=self.device)
            for i, chunk in enumerate(self.chunks):
                lo = i * _CHUNK
                n = min(_CHUNK, self.nbytes - lo)
                flat[lo : lo + n].copy_(chunk[:n], non_blocking=True)
            self.device_copy = flat.view(self.dtype).view(self.shape)
            self.ready = torch.cuda.Event()
            self.ready.record(stream)

    def fetch(self) -> torch.Tensor:
        self.prefetch()
        torch.cuda.current_stream().wait_event(self.ready)
        out = self.device_copy
        out.record_stream(torch.cuda.current_stream())
        chunks, self.chunks, self.device_copy = self.chunks, None, None
        _POOL.put(chunks, self.ready)  # reusable once the copy back has landed; no host sync here
        if self in _PENDING:
            index = _PENDING.index(self)
            if index > 0:
                _PENDING[index - 1].prefetch()  # the layer whose backward runs next
            del _PENDING[index:]
        return out


class _Derived:
    """An MLP sublayer's input, not stored: rebuilt from the preceding attention sublayer (``DERIVE_MLP_INPUT``)."""

    def __init__(self, source_ctx):
        self.source = source_ctx

    def fetch(self) -> torch.Tensor:
        src, self.source = self.source, None
        inputs = []
        for i, (t, meta) in enumerate(zip(src.inputs, src.meta)):
            if meta is not None and isinstance(t, _Offloaded):
                t = t.fetch()
                src.inputs[i] = t  # keep it on the GPU for the attention's own backward, which runs right after
            inputs.append(t)
        reuse = getattr(src.run, "reuse_selection", None)
        with torch.no_grad(), (reuse(*src.selection) if reuse is not None and src.selection else nullcontext()):
            outputs = src.run(*inputs)
        clear_state = getattr(src.run, "clear_state", None)
        if clear_state is not None:
            clear_state()
        return outputs[0].detach()


class StatefulLayerRecompute(torch.autograd.Function):
    """Run ``run(*inputs)`` without saving activations; replay it in backward from the saved inputs.

    ``offload_first``: keep the first input (the layer's hidden state) in pinned host memory until backward.
    Floating-point outputs are differentiable (including inputs passed through unchanged); others are not.
    """

    @staticmethod
    def forward(ctx, run: Callable, offload_first: bool, *inputs: Any):
        ctx.run = run
        ctx.meta = [(t.requires_grad, t.device) if torch.is_tensor(t) else None for t in inputs]
        ctx.layer_number = getattr(run, "layer_number", None)
        source = _LAST_ATTENTION_CTX[0]
        derive = (
            getattr(run, "derive_input", False)
            and source is not None
            and ctx.layer_number is not None
            and source.layer_number == ctx.layer_number - 1
        )
        _LAST_ATTENTION_CTX[0] = ctx if getattr(run, "is_attention", False) else None
        ctx.offload_first = offload_first and not derive and torch.is_tensor(inputs[0]) and inputs[0].is_cuda
        ctx.inputs = [
            (_Offloaded.try_offload(t) if i == 0 and ctx.offload_first else t.detach()) if torch.is_tensor(t) else t
            for i, t in enumerate(inputs)
        ]
        if derive:
            ctx.inputs[0] = _Derived(source)
        with torch.no_grad():
            outputs = run(*inputs)
        ctx.mark_non_differentiable(*[o for o in outputs if torch.is_tensor(o) and not o.is_floating_point()])
        # Integer selections this layer produced (an index source's top-k, the candidate source's blocks): the
        # replay reuses them rather than running the indexer again (``attention.reuse_selection``).
        produced = getattr(run, "produced_selection", None)
        ctx.selection = produced(inputs, outputs) if produced is not None else None
        return outputs

    @staticmethod
    def backward(ctx, *grads):
        inputs = []
        for i, (t, meta) in enumerate(zip(ctx.inputs, ctx.meta)):
            if meta is None:
                inputs.append(t)
                continue
            requires_grad, device = meta
            x = t.fetch() if isinstance(t, (_Offloaded, _Derived)) else t
            inputs.append(x.detach().requires_grad_(requires_grad))
        reuse = getattr(ctx.run, "reuse_selection", None)
        breakdown = _SavedBreakdown(getattr(ctx.run, "layer", None)) if MEMLOG else nullcontext()
        with torch.enable_grad(), (
            reuse(*ctx.selection) if reuse is not None and ctx.selection else nullcontext()
        ), breakdown:
            outputs = ctx.run(*inputs)
        ctx.selection = None
        pairs = [(o, g) for o, g in zip(outputs, grads) if torch.is_tensor(o) and o.requires_grad and g is not None]
        if pairs:
            torch.autograd.backward([o for o, _ in pairs], [g for _, g in pairs])
        del outputs, pairs
        if MEMLOG:
            breakdown.report()
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
        reuse_selection,
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

    index_fields = [STATE_FIELDS.index("global_indices"), STATE_FIELDS.index("candidate_blocks")]

    def produced_selection(inputs, outputs):
        # inputs/outputs: (hidden, *STATE_FIELDS, pending_pre) / (output, *STATE_FIELDS, pending_pre)
        new = [outputs[1 + i] if outputs[1 + i] is not inputs[1 + i] else None for i in index_fields]
        return tuple(new) if new[0] is not None else None

    run.produced_selection = produced_selection
    run.reuse_selection = reuse_selection

    def clear_state():
        local = state_getter()
        local.csa2 = CSA2State()
        local.pending_pre = None

    run.clear_state = clear_state
    run.layer = layer
    run.layer_number = getattr(layer, "layer_number", None)
    if DERIVE_MLP_INPUT:
        is_attention = getattr(layer, "_skyrl_is_attention", None)
        if is_attention is None:
            # A real attention module has children; MoE sublayers carry an empty IdentityOp ``self_attention``.
            is_attention = any(".self_attention." in f".{n}" for n, _ in layer.named_modules())
            layer._skyrl_is_attention = is_attention
        run.is_attention = is_attention
        run.derive_input = not is_attention

    # ``config.num_layers`` counts hybrid layers (from ``hybrid_layer_pattern``), as ``layer_number`` does.
    num_layers = getattr(getattr(layer, "config", None), "num_layers", None) or 0
    layer_number = getattr(layer, "layer_number", 0)
    keep_on_gpu = OFFLOAD_KEEP_LAST > 0 and num_layers > 0 and layer_number > num_layers - OFFLOAD_KEEP_LAST
    if not hidden_states.requires_grad:
        # The layer's parameters are not inputs of the Function, so its outputs only join the graph when an input
        # does. With a frozen embedding (LoRA) nothing upstream requires grad: without this the whole stack would be
        # cut out of backward and the adapters would get no gradient (as HF ``enable_input_require_grads``).
        hidden_states = hidden_states.detach().requires_grad_(True)
    if MEMLOG:
        torch.cuda.reset_peak_memory_stats()
    state_inputs = (*(getattr(state.csa2, name) for name in STATE_FIELDS), state.pending_pre)
    outputs = StatefulLayerRecompute.apply(
        run,
        OFFLOAD and not keep_on_gpu,
        hidden_states,
        *state_inputs,
    )
    output, *new_state, pending_pre = outputs
    if MEMLOG:
        _memlog(layer_number, hidden_states, dict(zip([*STATE_FIELDS, "pending_pre"], state_inputs)))
    state.csa2 = CSA2State(compress_ratio=extras["compress_ratio"], **dict(zip(STATE_FIELDS, new_state)))
    state.pending_pre = pending_pre
    return output, extras["context"]

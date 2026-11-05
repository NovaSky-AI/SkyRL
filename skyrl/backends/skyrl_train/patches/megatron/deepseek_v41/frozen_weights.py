"""Frozen base weights kept compressed on the GPU (MXFP8), decompressed one hybrid layer at a time.

``SKYRL_DSV41_COMPRESS_FROZEN_WEIGHTS=1``. Under LoRA the base weights never change, and DeepSeek-V4.1-Flash's were
released as FP8 (attention, shared experts: e4m3 with power-of-two block scales) and MXFP4 (routed experts): the BF16
copies the trainer computes with hold at most 1-4 significant bits per value. Each frozen weight is stored as e4m3
values with one power-of-two scale per 32 consecutive elements of a row (MXFP8, ~1.03 bytes per value instead of 2),
which reproduces those BF16 values exactly: every weight is checked bit for bit after compression, and one that does
not round-trip stays in BF16. Compute is unchanged (BF16 GEMMs on the same values); only storage shrinks.

Before a hybrid layer runs (forward, logprob forward, or its replay in backward) its weights are decompressed into one
of two shared BF16 slots, and the parameters' ``.data`` point into that slot. That is only correct when one layer's
weights are needed at a time, which holds under the stateful full recompute (``deepseek_v41/recompute.py``): forward
runs each layer without saving activations, and backward replays a layer and finishes its backward before the next
one. Without it (or in eval forwards) every forward is no-grad or layer by layer, so compression is refused for a
grad-enabled forward that would keep several layers' weights alive.

Not supported with compressed weights: moving the model to CPU (colocated offload) and full-weight export/sync.
"""

import os
from typing import List, Optional

import torch
from loguru import logger

ENABLED = os.environ.get("SKYRL_DSV41_COMPRESS_FROZEN_WEIGHTS", "0").lower() in ("1", "true")
# Debug: fingerprint every compressed weight before compression and check it whenever a module is about to use it
# (forward and backward replay), raising on any mismatch. Slow; for validating the slot logic.
VERIFY = os.environ.get("SKYRL_DSV41_FROZEN_WEIGHTS_VERIFY", "0").lower() in ("1", "true")
_VERIFIED = [0]
_BLOCK = 32
_E4M3_MAX = 448.0
_MIN_NUMEL = 1 << 16

_SLOTS: List[Optional[torch.Tensor]] = [None, None]
_OWNER: List[Optional[int]] = [None, None]
_LAST_SLOT = [1]


def _quantize(w: torch.Tensor, chunk_rows: int = 4096):
    """(e4m3 values, int8 log2 scales per 32 elements) of ``w`` (..., K), K % 32 == 0; rows a chunk at a time."""
    rows = w.reshape(-1, w.shape[-1])
    q = torch.empty(rows.shape, dtype=torch.float8_e4m3fn, device=w.device)
    exps = torch.empty(rows.shape[0], rows.shape[1] // _BLOCK, dtype=torch.int8, device=w.device)
    for lo in range(0, rows.shape[0], chunk_rows):
        blocks = rows[lo : lo + chunk_rows].float().view(-1, rows.shape[1] // _BLOCK, _BLOCK)
        amax = blocks.abs().amax(dim=-1)
        exp = torch.where(amax > 0, torch.ceil(torch.log2(amax / _E4M3_MAX)), torch.zeros_like(amax)).clamp(-127, 127)
        q[lo : lo + chunk_rows] = (blocks / torch.exp2(exp).unsqueeze(-1)).to(torch.float8_e4m3fn).view(-1, rows.shape[1])
        exps[lo : lo + chunk_rows] = exp.to(torch.int8)
    return q.view(w.shape), exps.view(*w.shape[:-1], w.shape[-1] // _BLOCK)


def _dequantize_into(q: torch.Tensor, exp: torch.Tensor, out: torch.Tensor) -> None:
    blocks = q.view(*q.shape[:-1], q.shape[-1] // _BLOCK, _BLOCK)
    scale = torch.exp2(exp.float()).to(out.dtype).unsqueeze(-1)
    torch.mul(blocks.to(out.dtype), scale, out=out.view(blocks.shape))


class _LayerStore:
    def __init__(self, index: int):
        self.index = index
        self.entries = []  # (param, q, exp, offset, numel); q has the parameter's shape
        self.numel = 0


def _compressible(name: str, p: torch.nn.Parameter) -> bool:
    return (
        not p.requires_grad
        and p.is_cuda
        and p.dtype == torch.bfloat16
        and p.dim() >= 2
        and p.numel() >= _MIN_NUMEL
        and p.shape[-1] % _BLOCK == 0
        and p.is_contiguous()
        and ".engram." not in f".{name}."  # engram runs outside the layer's recompute
    )


def compress_frozen_weights(model_chunks) -> None:
    """Compress the frozen weights of every DeepSeek-V4.1 hybrid layer in ``model_chunks`` (after load + LoRA)."""
    if not ENABLED:
        return
    layers = []
    for chunk in model_chunks:
        for module in chunk.modules():
            if hasattr(module, "engram") and hasattr(module, "layer_number") and module not in layers:
                layers.append(module)
    if not layers:
        return
    if not getattr(layers[0].config, "dsv41_stateful_recompute", False):
        logger.warning(
            "SKYRL_DSV41_COMPRESS_FROZEN_WEIGHTS needs the stateful full recompute "
            "(recompute_granularity=full); weights left in BF16"
        )
        return
    before = torch.cuda.memory_allocated()
    kept, compressed, slot_numel = 0, 0, 0
    for index, layer in enumerate(layers):
        store = _LayerStore(index)
        for name, p in layer.named_parameters():
            if not _compressible(name, p):
                continue
            q, exp = _quantize(p.data)
            check = torch.empty_like(p.data)
            _dequantize_into(q, exp, check)
            if not torch.equal(check, p.data):
                kept += p.numel()
                continue
            del check
            if VERIFY:
                p._skyrl_fingerprint = _fingerprint(p.data)
            store.entries.append((p, q, exp, store.numel, p.numel()))
            store.numel += p.numel()
            compressed += p.numel()
            # Drop the BF16 copy now, not after the loop: every layer's BF16 weights plus the growing compressed
            # copies do not fit a 178 GiB B200 next to a full EP8 shard.
            p.data = torch.empty(0, dtype=p.dtype, device=p.device)
        if store.entries:
            layer._skyrl_frozen_store = store
            slot_numel = max(slot_numel, store.numel)
    for i in range(2):
        _SLOTS[i] = torch.empty(slot_numel, dtype=torch.bfloat16, device=torch.cuda.current_device())
        _OWNER[i] = None
    # Point every compressed parameter into slot 0 (its contents are only valid once its layer is materialized).
    for layer in layers:
        store = getattr(layer, "_skyrl_frozen_store", None)
        for p, q, _, offset, numel in store.entries if store is not None else ():
            p.data = _SLOTS[0][offset : offset + numel].view(q.shape)
    if VERIFY:
        for layer in layers:
            for module in layer.modules():
                if any(hasattr(p, "_skyrl_fingerprint") for p in module._parameters.values() if p is not None):
                    module.register_forward_pre_hook(_verify_hook)
    torch.cuda.empty_cache()
    after = torch.cuda.memory_allocated()
    # print as well as log: Ray drops some worker log lines from the driver's output.
    msg = (
        f"DSv4.1 frozen weights: {compressed / 1e9:.2f}B values in MXFP8 over {len(layers)} layers, "
        f"{kept / 1e9:.3f}B kept in BF16 (no exact round trip); slots 2 x {slot_numel * 2 / 2**30:.2f} GiB; "
        f"allocated {before / 2**30:.1f} -> {after / 2**30:.1f} GiB"
    )
    logger.info(msg)
    print(msg, flush=True)


def _fingerprint(t: torch.Tensor, chunk: int = 1 << 24):
    """(sum, position-weighted sum) in float64, a chunk at a time (a whole-tensor float64 copy can OOM at load)."""
    flat = t.reshape(-1)
    total = torch.zeros(2, dtype=torch.float64, device=t.device)
    for lo in range(0, flat.numel(), chunk):
        x = flat[lo : lo + chunk].double()
        weights = torch.arange(lo + 1, lo + x.numel() + 1, device=x.device, dtype=torch.float64).remainder_(1009)
        total[0] += x.sum()
        total[1] += (x * weights).sum()
    return total.cpu()


def _verify_hook(module, args):
    for name, p in module._parameters.items():
        expected = getattr(p, "_skyrl_fingerprint", None) if p is not None else None
        if expected is None:
            continue
        got = _fingerprint(p.data)
        if not torch.equal(got, expected):
            raise RuntimeError(f"DSv4.1 compressed frozen weights: stale weight {type(module).__name__}.{name}")
        _VERIFIED[0] += 1
        if _VERIFIED[0] in (1, 1000, 10000, 100000):
            logger.info(f"DSv4.1 frozen weights verify: {_VERIFIED[0]} weight uses checked, all exact")


def materialize(layer, grad_forward: bool = False) -> None:
    """Decompress ``layer``'s frozen weights into a slot and point its parameters there (no-op if resident)."""
    store = getattr(layer, "_skyrl_frozen_store", None)
    if store is None:
        return
    if grad_forward:
        raise RuntimeError(
            "DSv4.1 compressed frozen weights: a grad-enabled forward outside the stateful recompute would need "
            "every layer's weights at once"
        )
    for i in range(2):
        if _OWNER[i] == store.index:
            if _LAST_SLOT[0] != i:
                _LAST_SLOT[0] = i
            _point(store, i)
            return
    slot = 1 - _LAST_SLOT[0]
    buf = _SLOTS[slot]
    for p, q, exp, offset, numel in store.entries:
        view = buf[offset : offset + numel].view(q.shape)
        _dequantize_into(q, exp, view)
    _OWNER[slot] = store.index
    _LAST_SLOT[0] = slot
    _point(store, slot)


def _point(store: _LayerStore, slot: int) -> None:
    buf = _SLOTS[slot]
    for p, q, _, offset, numel in store.entries:
        if p.data.data_ptr() != buf[offset:].data_ptr():
            p.data = buf[offset : offset + numel].view(q.shape)

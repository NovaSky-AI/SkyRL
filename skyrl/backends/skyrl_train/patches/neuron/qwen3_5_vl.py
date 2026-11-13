"""Neuron device-safety patches for the Qwen3.5-VL model.

The Qwen3.5-VL model does several things the TorchNeuron (Trainium2) eager runtime
cannot do, or does not do safely:

1. ``Qwen3_5VisionRotaryEmbedding`` registers ``inv_freq`` as a *non-persistent*
   buffer. Under FSDP ``cpu_offload`` that buffer is left on CPU while activations
   are on the accelerator, and the vision RoPE forward multiplies
   ``position_ids * self.inv_freq`` without moving the buffer to the tensor's
   device. The patch moves the buffer onto ``position_ids.device``.
2. ``Qwen3_5TextRotaryEmbedding`` (the *language* model's MROPE) has the same
   non-persistent ``inv_freq`` buffer problem as the vision RoPE, *and* its VLM forward
   builds the 4-grid ``position_ids`` partly on CPU (the spatial grids come from the
   vision helpers, which fall back to CPU for their ``repeat_interleave``). The text
   RoPE forward then does an unsynchronized ``inv_freq @ position_ids`` matmul. The
   patch moves both onto ``x.device`` (the activation device).
3. ``get_vision_interpolation_indices_and_weights`` calls ``torch.repeat_interleave``
   with a dynamic ``repeats`` tensor, which neuronx-cc cannot lower (BIRCodeGenLoop
   crash). It falls back to CPU in eager mode; forcing it onto CPU is a sync change,
   not a correctness change.
4. ``get_vision_cu_seqlens`` builds per-frame attention boundaries with the same
   dynamic ``repeat_interleave``, which neuronx-cc cannot lower ("No active NKI
   context"). The tiny ``(num_frames+1,)`` result is computed on CPU and moved back.

Every patch inspects the tensor device at call time rather than a cached global
backend flag, so the module is a safe no-op off neuron (the two ``repeat_interleave``
wrappers return the original result unchanged unless the tensor is on ``neuron``; the
two RoPE wrappers only move a tensor when there is a real device mismatch).
"""

from loguru import logger

_APPLIED = False


def _patch_vision_rope() -> None:
    """Make the Qwen3.5 vision RoPE buffer follow ``position_ids.device``.

    transformers 5.18 changed the vision RoPE forward to ``forward(self, x,
    position_ids)`` and to return ``(cos, sin)``. Rather than reimplement that (and
    drift from upstream), keep the original forward and only move the buffer it reads,
    which is the single device-unsafe step this patch exists for.
    """
    import torch
    import transformers.models.qwen3_5.modeling_qwen3_5 as _qwen35

    cls = getattr(_qwen35, "Qwen3_5VisionRotaryEmbedding", None)
    if cls is None:
        logger.warning("Qwen3_5VisionRotaryEmbedding not found; skipping vision RoPE device patch")
        return

    _orig_forward = cls.forward

    @torch.no_grad()
    def _forward(self, x, position_ids):
        if self.inv_freq.device != position_ids.device:
            self.inv_freq = self.inv_freq.to(position_ids.device)
        return _orig_forward(self, x, position_ids)

    cls.forward = _forward
    logger.info("Patched Qwen3_5VisionRotaryEmbedding.forward (device-safe inv_freq)")


def _patch_text_rope() -> None:
    """Make the Qwen3.5 *text* RoPE device-safe under FSDP ``cpu_offload``.

    ``Qwen3_5TextRotaryEmbedding`` registers ``inv_freq`` as a *non-persistent* buffer,
    exactly like the vision RoPE, so FSDP ``cpu_offload`` leaves it on CPU while
    ``hidden_states`` live on the accelerator.  The VLM forward also builds the 4-grid
    MROPE ``position_ids`` (text + temporal + height + width) partly on CPU (the spatial
    grids come from the vision helpers, which fall back to CPU for their
    ``repeat_interleave``).  Its ``forward`` then does an unsynchronized
    ``inv_freq @ position_ids`` matmul without moving either operand.  Move both onto
    ``x.device`` (the activation device) before delegating, and leave the original
    forward otherwise untouched.
    """
    import torch
    import transformers.models.qwen3_5.modeling_qwen3_5 as _qwen35

    cls = getattr(_qwen35, "Qwen3_5TextRotaryEmbedding", None)
    if cls is None:
        logger.warning("Qwen3_5TextRotaryEmbedding not found; skipping text RoPE device patch")
        return

    _orig_forward = cls.forward

    @torch.no_grad()
    def _forward(self, x, position_ids):
        dev = x.device
        if self.inv_freq.device != dev:
            self.inv_freq = self.inv_freq.to(dev)
        if position_ids.device != dev:
            position_ids = position_ids.to(dev)
        return _orig_forward(self, x, position_ids)

    cls.forward = _forward
    logger.info("Patched Qwen3_5TextRotaryEmbedding.forward (device-safe inv_freq + position_ids)")


def _patch_vision_interp() -> None:
    """Force Qwen3.5 vision position-interpolation precompute onto CPU on neuron."""
    import transformers.models.qwen3_5.modeling_qwen3_5 as _mq

    _orig = getattr(_mq, "get_vision_interpolation_indices_and_weights", None)
    if _orig is None:
        logger.warning("get_vision_interpolation_indices_and_weights not found; skipping interp patch")
        return

    def _patched(grid_thw, *args, **kwargs):
        dev = grid_thw.device
        if dev.type == "neuron":
            indices, weights = _orig(grid_thw.to("cpu"), *args, **kwargs)
            return indices.to(dev), weights.to(dev)
        return _orig(grid_thw, *args, **kwargs)

    _mq.get_vision_interpolation_indices_and_weights = _patched
    logger.info("Patched get_vision_interpolation_indices_and_weights (repeat_interleave on CPU)")


def _patch_vision_seqlens() -> None:
    """Force the dynamic ``repeat_interleave`` in the vision seqlens helper onto CPU."""
    import transformers.vision_utils as _vu

    _orig = getattr(_vu, "get_vision_cu_seqlens", None)
    if _orig is None:
        logger.warning("get_vision_cu_seqlens not found; skipping seqlens patch")
        return

    def _patched(grid_thw, merge_temporal=False, kwargs=None):
        dev = grid_thw.device
        if dev.type == "neuron":
            return _orig(grid_thw.to("cpu"), merge_temporal=merge_temporal, kwargs=kwargs).to(dev)
        return _orig(grid_thw, merge_temporal=merge_temporal, kwargs=kwargs)

    _vu.get_vision_cu_seqlens = _patched
    logger.info("Patched get_vision_cu_seqlens (repeat_interleave on CPU)")


def apply_qwen3_5_vl_device_patch() -> None:
    """Apply all Qwen3.5-VL neuron device-safety patches (idempotent)."""
    global _APPLIED
    if _APPLIED:
        return
    _patch_vision_rope()
    _patch_text_rope()
    _patch_vision_interp()
    _patch_vision_seqlens()
    _APPLIED = True

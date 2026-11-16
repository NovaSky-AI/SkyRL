"""Neuron device-safety patches for the Qwen3-VL model.

Qwen3-VL's vision tower and its language-model MROPE do three things the TorchNeuron
(Trainium2) eager runtime cannot do, or does not do safely:

1. ``Qwen3VLVisionRotaryEmbedding`` registers ``inv_freq`` as a *non-persistent* buffer.
   Under FSDP ``cpu_offload`` that buffer stays on CPU while the position ids are on the
   accelerator, and the vision RoPE forward multiplies
   ``position_ids_expanded * self.inv_freq`` without moving the buffer to the tensor's
   device. The patch moves the buffer onto ``position_ids.device``.
2. ``Qwen3VLTextRotaryEmbedding`` (the *language* model's MROPE) has the same
   non-persistent ``inv_freq`` buffer problem, and its VLM forward builds the 4-grid
   ``position_ids`` partly on CPU (the spatial grids come from the vision helpers, which
   fall back to CPU for their ``repeat_interleave``). Its forward then does an
   unsynchronized ``inv_freq @ position_ids`` matmul. The patch moves both onto
   ``x.device``.
3. ``get_vision_interpolation_indices_and_weights`` calls ``torch.repeat_interleave``
   with a dynamic ``repeats`` tensor, which neuronx-cc cannot lower
   (``stablehlo.concatenate`` shape-inference mismatch). Qwen3-VL binds it into
   ``modeling_qwen3_vl`` via a ``from transformers.vision_utils import ...`` at module
   load, so the patch must target that bound name (not the ``vision_utils`` attribute the
   model never re-reads). Forcing it onto CPU is a sync change, not a correctness change.

Every patch inspects the tensor device at call time rather than a cached global backend
flag, so the module is a safe no-op off neuron (the two RoPE wrappers only move a tensor
when there is a real device mismatch; the interpolator wrapper returns the original result
unchanged unless the tensor is on ``neuron``).
"""

from loguru import logger

_APPLIED = False


def _patch_vision_rope() -> None:
    """Make the Qwen3-VL vision RoPE buffer follow ``position_ids.device``.

    Keep the original forward and only move the buffer it reads, which is the single
    device-unsafe step this patch exists for.
    """
    import torch
    import transformers.models.qwen3_vl.modeling_qwen3_vl as _qwen3vl

    cls = getattr(_qwen3vl, "Qwen3VLVisionRotaryEmbedding", None)
    if cls is None:
        logger.warning("Qwen3VLVisionRotaryEmbedding not found; skipping vision RoPE device patch")
        return

    _orig_forward = cls.forward

    @torch.no_grad()
    def _forward(self, x, position_ids):
        if self.inv_freq.device != position_ids.device:
            self.inv_freq = self.inv_freq.to(position_ids.device)
        return _orig_forward(self, x, position_ids)

    cls.forward = _forward
    logger.info("Patched Qwen3VLVisionRotaryEmbedding.forward (device-safe inv_freq)")


def _patch_text_rope() -> None:
    """Make the Qwen3-VL *text* RoPE device-safe under FSDP ``cpu_offload``.

    Same non-persistent ``inv_freq`` buffer problem as the vision RoPE, plus the VLM
    forward builds the 4-grid MROPE ``position_ids`` partly on CPU. Move both onto
    ``x.device`` before delegating.
    """
    import torch
    import transformers.models.qwen3_vl.modeling_qwen3_vl as _qwen3vl

    cls = getattr(_qwen3vl, "Qwen3VLTextRotaryEmbedding", None)
    if cls is None:
        logger.warning("Qwen3VLTextRotaryEmbedding not found; skipping text RoPE device patch")
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
    logger.info("Patched Qwen3VLTextRotaryEmbedding.forward (device-safe inv_freq + position_ids)")


def _patch_vision_interp() -> None:
    """Force the vision position-interpolation precompute onto CPU on neuron.

    Qwen3-VL does ``from transformers.vision_utils import
    get_vision_interpolation_indices_and_weights``, so its vision ``forward`` calls the
    *bound name* in ``modeling_qwen3_vl`` -- patching ``vision_utils`` would leave the
    model calling the original function object. Patch the model's own name instead.
    """
    import transformers.models.qwen3_vl.modeling_qwen3_vl as _mq

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
    logger.info("Patched modeling_qwen3_vl.get_vision_interpolation_indices_and_weights (CPU on neuron)")


def apply_qwen3_vl_device_patch() -> None:
    """Apply all Qwen3-VL neuron device-safety patches (idempotent)."""
    global _APPLIED
    if _APPLIED:
        return
    _patch_vision_rope()
    _patch_text_rope()
    _patch_vision_interp()
    _APPLIED = True

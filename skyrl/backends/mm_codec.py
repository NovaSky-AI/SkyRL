"""vLLM-free serialization for multimodal ``kwargs_data``.

The vLLM engine emits image features in vLLM's ``mm_serde`` base64 format; the native
(TorchNeuron) engine has no vLLM, so it needs a codec that depends only on torch and
safetensors (both already required by ``transformers``). Each native-engine blob is
tagged with ``PREFIX`` so ``decode_mm_kwargs`` can dispatch between the two formats
without probing vLLM: an untagged blob is a vLLM ``mm_serde`` item, a tagged blob is a
safetensors payload produced here.

safetensors is used instead of ``torch.save`` because the blobs travel between the
inference engine and the trainer inside the same job but are decoded by shared code that
also runs on vLLM-free nodes; safetensors keeps that path free of pickle deserialization.
"""

from __future__ import annotations

import base64

import torch

try:  # pragma: no cover - import guard mirrors the optional nature of the native engine
    import safetensors.torch
except ImportError:  # pragma: no cover
    safetensors = None  # type: ignore[assignment]


PREFIX = "skyrl:torch:"
"""Blob prefix marking a native-engine payload (vs. a vLLM ``mm_serde`` item)."""


def is_native_blob(blob: str) -> bool:
    """Return True when ``blob`` was produced by :func:`encode_mm_kwargs_item`."""
    return isinstance(blob, str) and blob.startswith(PREFIX)


def encode_mm_kwargs_item(data: dict[str, torch.Tensor]) -> str:
    """Serialize a per-image dict (``pixel_values`` / ``image_grid_thw``) to a tagged b64 string."""
    if safetensors is None:  # pragma: no cover
        raise RuntimeError("safetensors is required to encode multimodal kwargs")
    # ``safetensors.torch.save`` (no filename) returns the serialized bytes directly.
    raw = safetensors.torch.save(data)
    return PREFIX + base64.b64encode(raw).decode("ascii")


def decode_mm_kwargs_item(blob: str) -> dict[str, torch.Tensor]:
    """Deserialize a tagged b64 string back into a per-image tensor dict."""
    if safetensors is None:  # pragma: no cover
        raise RuntimeError("safetensors is required to decode multimodal kwargs")
    payload = blob[len(PREFIX) :] if blob.startswith(PREFIX) else blob
    return safetensors.torch.load(base64.b64decode(payload))

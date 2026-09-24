"""Parameter shapes read from an HF checkpoint's safetensors headers.

Sizing the packed transfer buffer needs one number: the element count of the
largest single tensor that will cross the wire. Every safetensors file carries a
JSON header naming each tensor's dtype and shape, so that number is available
from the checkpoint alone -- no GPU allocation, no collective, and no dependency
on the training model being resident.

That last property is the point. Reading it from the live model instead means a
Megatron ``export_hf_weights`` pass over the parameters, which faults under
``colocate_all`` where the policy is offloaded between steps, and an FSDP
``state_dict()`` walk. Both are avoidable: the shapes are the same either way,
because the bridge truncates vocab padding back to ``hf_config.vocab_size`` on
export (``model_bridge._truncate_vocab_padding``) and stores fused MoE expert
stacks in the checkpoint exactly as they go on the wire.

Element counts, not bytes: the wire dtype comes from the inference engine config
and is not known this early, so callers multiply by ``dtype.itemsize``.
"""

import logging
import math
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

__all__ = ["max_param_numel"]


def max_param_numel(model_path: str) -> int:
    """Element count of the largest tensor in ``model_path``'s safetensors files.

    Args:
        model_path: a local checkpoint directory or an HF repo id. A repo id is
            resolved from the local HF cache only -- the caller has already
            loaded this model, so the files are present, and this must never
            trigger a download.

    Returns:
        The largest ``prod(shape)`` across every tensor, or ``0`` if the shapes
        cannot be read (no safetensors files, a ``.bin`` checkpoint, an
        unresolvable path). ``0`` means "no floor from the checkpoint"; callers
        fall back to their configured size.
    """
    files = _safetensors_files(model_path)
    if not files:
        logger.debug("No safetensors files found for %r; packed buffer falls back to the configured size.", model_path)
        return 0

    try:
        from safetensors import safe_open
    except ImportError:
        logger.debug("safetensors is not importable; packed buffer falls back to the configured size.")
        return 0

    largest = 0
    for path in files:
        try:
            with safe_open(path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    # get_slice reads the file's header entry; it does not read
                    # or map the tensor data.
                    largest = max(largest, math.prod(handle.get_slice(key).get_shape()))
        except Exception as exc:  # noqa: BLE001 - a bad shard must not stop training
            logger.warning("Could not read tensor shapes from %s: %s", path, exc)
    return largest


def _safetensors_files(model_path: str) -> List[Path]:
    """Every ``*.safetensors`` file for a local directory or a cached repo id."""
    local = Path(model_path)
    if local.is_dir():
        return sorted(local.glob("*.safetensors"))

    resolved = _snapshot_from_cache(model_path)
    return sorted(resolved.glob("*.safetensors")) if resolved else []


def _snapshot_from_cache(repo_id: str) -> Optional[Path]:
    try:
        from huggingface_hub import snapshot_download

        return Path(
            snapshot_download(
                repo_id,
                allow_patterns=["*.safetensors", "*.safetensors.index.json"],
                local_files_only=True,
            )
        )
    except Exception as exc:  # noqa: BLE001 - not cached, offline, or not a repo id
        logger.debug("Could not resolve %r in the local HF cache: %s", repo_id, exc)
        return None

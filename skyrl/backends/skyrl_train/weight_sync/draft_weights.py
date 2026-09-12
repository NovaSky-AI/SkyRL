"""Select native MTP weights for a separate vLLM draft update session."""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional

WEIGHT_UPDATE_TARGET_MODEL = "model"
WEIGHT_UPDATE_TARGET_DRAFT = "draft"
WEIGHT_UPDATE_TARGETS = (WEIGHT_UPDATE_TARGET_MODEL, WEIGHT_UPDATE_TARGET_DRAFT)

# Only native MTP drafters share the policy checkpoint.
_POLICY_MTP_METHODS = frozenset({"mtp", "deepseek_mtp"})

_MTP_BLOCK_RE = re.compile(r"(^|\.)mtp\.")
_EMBEDDING_SUFFIXES = ("embedding.word_embeddings.weight", "output_layer.weight")


def needs_draft_weight_sync(speculative_config: Optional[Mapping[str, Any]]) -> bool:
    """Whether vLLM uses the policy checkpoint's native MTP head as its drafter."""
    if not speculative_config:
        return False
    return speculative_config.get("method") in _POLICY_MTP_METHODS


def is_megatron_mtp_param(name: str) -> bool:
    """Whether a (module-unwrapped) Megatron parameter name is in the MTP block."""
    return _MTP_BLOCK_RE.search(name) is not None


def is_megatron_draft_param(name: str) -> bool:
    """Whether a Megatron parameter belongs to the draft session: MTP block, embedding, output layer."""
    return is_megatron_mtp_param(name) or name.endswith(_EMBEDDING_SUFFIXES)


def validate_weight_update_target(target: str) -> str:
    if target not in WEIGHT_UPDATE_TARGETS:
        raise ValueError(f"Unknown weight update target {target!r}; expected one of {WEIGHT_UPDATE_TARGETS}")
    return target

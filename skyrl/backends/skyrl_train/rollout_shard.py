"""What a stored rollout carries in the object store, as opposed to the metadata the trainer reads."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Dict, List, Optional

import numpy as np

from skyrl.backends.skyrl_train.utils.bit_packing import pack_ids, unpack_ids
from skyrl.backends.skyrl_train.utils.sample_support import CompactSampleSupport


class RolloutPayloadKey(StrEnum):
    """``GeneratorOutput`` keys holding a row's token payload.

    A stored trajectory keeps these in the Ray object store; every other key is metadata the
    driver may read.
    """

    PROMPT_TOKEN_IDS = "prompt_token_ids"
    RESPONSE_IDS = "response_ids"
    LOSS_MASKS = "loss_masks"
    ROLLOUT_LOGPROBS = "rollout_logprobs"
    ROLLOUT_EXPERT_INDICES = "rollout_expert_indices"
    ROLLOUT_SAMPLE_SUPPORT = "rollout_sample_support"
    PIXEL_VALUES = "pixel_values"
    IMAGE_GRID_THW = "image_grid_thw"


PAYLOAD_KEYS = frozenset(RolloutPayloadKey)


def support_id_bits(vocab_size: int) -> int:
    """The bits that hold every token id of a ``vocab_size``-token vocabulary: ``ceil(log2(vocab_size))``.

    The vocabulary is the embedding's rows (``model_config.get_text_config().vocab_size``), so no
    sampled id or support member reaches it.
    """
    return max(1, (vocab_size - 1).bit_length())


@dataclass(frozen=True)
class StoredSampleSupport:
    """A row's ``CompactSampleSupport`` as a stored payload holds it: member ids packed ``bits`` apiece.

    Members keep their order, so the ids stay in step with the offsets and member logprobs.
    """

    packed_ids: np.ndarray
    bits: int
    offsets: np.ndarray
    top_k: int
    logprobs: Optional[np.ndarray] = None

    @classmethod
    def pack(cls, support: CompactSampleSupport, bits: int) -> "StoredSampleSupport":
        """Raises if a member id needs more than ``bits``, e.g. when ``bits`` came from another model."""
        return cls(pack_ids(support.token_ids, bits), bits, support.offsets, support.top_k, support.logprobs)

    def unpack(self) -> CompactSampleSupport:
        ids = unpack_ids(self.packed_ids, int(self.offsets[-1]), self.bits)
        columns = (ids,) if self.logprobs is None else (ids, self.logprobs)
        return CompactSampleSupport(columns, self.offsets, top_k=self.top_k)


def _token_ids(row: Any) -> np.ndarray:
    ids = np.asarray(row, dtype=np.int64)
    if ids.size and (ids.min() < 0 or ids.max() >= 2**31):
        raise ValueError(f"token ids span [{ids.min()}, {ids.max()}], outside int32's non-negative range")
    return ids.astype(np.int32)


def _float32(row: Any) -> np.ndarray:
    # The same cast collation's float32 buffers apply to these values.
    return np.asarray(row, dtype=np.float32)


def _map_rows(fn, rows: List[Any]) -> List[Any]:
    return [None if row is None else fn(row) for row in rows]


_ENCODERS = {
    RolloutPayloadKey.PROMPT_TOKEN_IDS: _token_ids,
    RolloutPayloadKey.RESPONSE_IDS: _token_ids,
    RolloutPayloadKey.LOSS_MASKS: _float32,
    RolloutPayloadKey.ROLLOUT_LOGPROBS: _float32,
}


def encode_payload(payload: Dict[str, list], *, support_id_bits: Optional[int] = None) -> Dict[str, list]:
    """``payload``'s rows as a stored trajectory keeps them.

    Token ids become int32 arrays and loss masks and logprobs float32 arrays, which pickle out of
    band, so every reader on a node maps one copy. Routes and VLM inputs stay as they are. With
    ``support_id_bits``, each row's sample-support member ids are packed at that width.
    """
    stored = {key: _map_rows(_ENCODERS[key], rows) if key in _ENCODERS else rows for key, rows in payload.items()}
    support = stored.get(RolloutPayloadKey.ROLLOUT_SAMPLE_SUPPORT)
    if support_id_bits is not None and support is not None:
        stored[RolloutPayloadKey.ROLLOUT_SAMPLE_SUPPORT] = _map_rows(
            lambda rows: StoredSampleSupport.pack(rows, support_id_bits), support
        )
    return stored


def unpack_support(rows: List[Any]) -> List[Any]:
    """Stored sample-support rows as ``CompactSampleSupport``, the form collation reads."""
    return [row.unpack() if isinstance(row, StoredSampleSupport) else row for row in rows]

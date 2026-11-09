"""Stored rollouts on their way to the trainer: what a payload carries, and one DP rank's shard of a mini-batch.

The driver never loads a payload. It plans a mini-batch from per-row metadata and sends each policy
DP rank a ``RolloutShard``: refs to that rank's stored trajectories, each row's length, and each row's
advantage with the loss reduction already applied. The rank's workers plan their microbatches from
the lengths alone and collate each one from its payloads only when they train it (``ShardRows``), so
a node holds the payloads of the microbatches in flight, never a collated copy of the shard.
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import ray
import torch
from ray import ObjectRef

from skyrl.backends.skyrl_train.fields import (
    BatchMetadataKey,
    GeneratorOutputKey,
    TrainingField,
)
from skyrl.backends.skyrl_train.training_batch import (
    TrainingInputBatch,
    padding_rows,
)
from skyrl.backends.skyrl_train.utils.bit_packing import pack_ids, unpack_ids
from skyrl.backends.skyrl_train.utils.sample_support import CompactSampleSupport
from skyrl.env_vars import SKYRL_PAYLOAD_FETCH_TIMEOUT_S
from skyrl.train.dataset.preprocess import RowCollation, rows_to_training_input
from skyrl.utils import stage_trace

# Stdlib logging: Ray pickles this module's remote functions, and a loguru handler can't be.
logger = logging.getLogger(__name__)


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
    GeneratorOutputKey.PROMPT_TOKEN_IDS: _token_ids,
    GeneratorOutputKey.RESPONSE_IDS: _token_ids,
    GeneratorOutputKey.LOSS_MASKS: _float32,
    GeneratorOutputKey.ROLLOUT_LOGPROBS: _float32,
}


def encode_payload(payload: Dict[str, list], *, support_id_bits: Optional[int] = None) -> Dict[str, list]:
    """``payload``'s rows as a stored trajectory keeps them.

    Token ids become int32 arrays and loss masks and logprobs float32 arrays, which pickle out of
    band, so every reader on a node maps one copy. Routes and VLM inputs stay as they are. With
    ``support_id_bits``, each row's sample-support member ids are packed at that width.
    """
    stored = {key: _map_rows(_ENCODERS[key], rows) if key in _ENCODERS else rows for key, rows in payload.items()}
    support = stored.get(GeneratorOutputKey.ROLLOUT_SAMPLE_SUPPORT)
    if support_id_bits is not None and support is not None:
        stored[GeneratorOutputKey.ROLLOUT_SAMPLE_SUPPORT] = _map_rows(
            lambda rows: StoredSampleSupport.pack(rows, support_id_bits), support
        )
    return stored


def unpack_support(rows: List[Any]) -> List[Any]:
    """Stored sample-support rows as ``CompactSampleSupport``, the form collation reads."""
    return [row.unpack() if isinstance(row, StoredSampleSupport) else row for row in rows]


@dataclass(frozen=True)
class RolloutShard:
    """One policy DP rank's rows of a mini-batch.

    ``payload_refs`` are stored trajectories (each a dict of ``PAYLOAD_KEYS`` -> per-row list).
    Rows are numbered across them, in order. ``row_advantages`` holds each row's advantage, already
    scaled by the loss reduction, so a row's token advantages are this value over its response tokens.
    """

    payload_refs: Tuple[ObjectRef, ...]
    # Rows of each payload; a step-wise trajectory's rows are adjacent, in step order.
    payload_rows: Tuple[int, ...]
    # Each row's prompt and response tokens, as the payload holds them.
    prompt_lengths: Tuple[int, ...]
    response_lengths: Tuple[int, ...]
    row_advantages: Tuple[float, ...]
    collation: RowCollation
    # The rank has nothing to train this step: its rows are another rank's, marked as padding (as
    # ``pad_training_input_batch`` marks them), so it still runs the step's collectives.
    padding: bool = False


def build_training_batch(shard: RolloutShard) -> TrainingInputBatch:
    """Load all of ``shard``'s payloads and collate them at once: the batch its microbatches are rows of."""
    rows = ShardRows(shard)
    return rows.collate(range(rows.num_rows))


def _support_unpacked(rows: Dict[str, list]) -> Dict[str, list]:
    """Stored rows as collation reads them: only these rows' sample-support ids are unpacked."""
    key = GeneratorOutputKey.ROLLOUT_SAMPLE_SUPPORT
    return {**rows, key: unpack_support(rows[key])} if key in rows else rows


def _with_advantages(batch: TrainingInputBatch, advantages: torch.Tensor, padding: bool) -> TrainingInputBatch:
    """``batch`` with each row's advantage over its response tokens, or marked as padding."""
    response_mask = batch[TrainingField.RESPONSE_MASK]
    batch[TrainingField.ADVANTAGES] = advantages[:, None] * response_mask.float()
    batch.metadata[BatchMetadataKey.RESPONSE_LENGTH] = response_mask.shape[1]
    if padding:
        # Rows that train nothing, as ``pad_training_input_batch`` pads.
        batch[TrainingField.ADVANTAGES] = torch.zeros_like(batch[TrainingField.ADVANTAGES])
        for key in (TrainingField.LOSS_MASK, TrainingField.ROUTER_PADDING_MASK):
            if batch.get(key) is not None:
                batch[key] = padding_rows(key, batch[key], batch.batch_size)
        batch[TrainingField.REAL_SAMPLE_MASK] = padding_rows(
            TrainingField.REAL_SAMPLE_MASK, batch.real_sample_mask, batch.batch_size
        )
    return batch


class ShardRows:
    """A ``RolloutShard``'s rows on a worker that trains them, collated a microbatch at a time.

    ``collate`` loads only the payloads holding the requested rows (zero-copy views of the node's
    copy, which every worker there shares) and pads to the whole shard's widths, so a microbatch is
    exactly those rows of ``build_training_batch(shard)``.
    """

    def __init__(self, shard: RolloutShard) -> None:
        self.shard = shard
        # Each row's (payload, row of that payload).
        self._locations = [(payload, row) for payload, rows in enumerate(shard.payload_rows) for row in range(rows)]
        self.token_counts = [p + r for p, r in zip(shard.prompt_lengths, shard.response_lengths)]
        self.padded_shape = (max(self.token_counts), max(shard.response_lengths))
        # Each row's pre-update logprobs, ``[rows, response_length]``, once a forward pass has them.
        self.old_logprobs: Optional[torch.Tensor] = None

    @property
    def num_rows(self) -> int:
        return len(self._locations)

    def payload_refs(self, rows: Sequence[int]) -> List[ObjectRef]:
        """The payloads holding ``rows``, each once."""
        payloads = dict.fromkeys(self._locations[row][0] for row in rows)
        return [self.shard.payload_refs[payload] for payload in payloads]

    def collate(self, rows: Sequence[int]) -> TrainingInputBatch:
        """``rows`` of the shard, in this order, as one padded batch with their advantages."""
        indices = list(dict.fromkeys(self._locations[row][0] for row in rows))
        # Blocks only for payloads the pull-ahead hasn't brought to this node yet.
        with stage_trace.span("mb.fetch", payloads=len(indices)):
            payloads = dict(zip(indices, fetch_payloads([self.shard.payload_refs[i] for i in indices])))
        t_collate = time.perf_counter()
        keys = payloads[indices[0]].keys()
        gathered = {key: [payloads[p][key][r] for p, r in (self._locations[row] for row in rows)] for key in keys}
        del payloads
        gathered = _support_unpacked(gathered)
        batch = rows_to_training_input(gathered, self.shard.collation, padded_shape=self.padded_shape)
        index = torch.tensor(list(rows), dtype=torch.long)
        batch = _with_advantages(
            batch, torch.tensor(self.shard.row_advantages, dtype=torch.float32)[index], self.shard.padding
        )
        if self.old_logprobs is not None:
            batch[TrainingField.ACTION_LOG_PROBS] = self.old_logprobs[index]
        stage_trace.record("mb.collate", t_collate, time.perf_counter(), rows=len(rows))
        return batch


def fetch_payloads(refs: List[ObjectRef]) -> list:
    """``ray.get`` stored payloads, failing loudly rather than stalling the step's collectives."""
    try:
        return ray.get(refs, timeout=SKYRL_PAYLOAD_FETCH_TIMEOUT_S)
    except ray.exceptions.GetTimeoutError:
        raise TimeoutError(
            f"{len(refs)} stored rollouts did not arrive within SKYRL_PAYLOAD_FETCH_TIMEOUT_S="
            f"{SKYRL_PAYLOAD_FETCH_TIMEOUT_S}s"
        ) from None


# Background pulls of the payloads of upcoming microbatches; see ``pull_payloads``.
_PULLS: Optional[ThreadPoolExecutor] = None
PULL_AHEAD_MICROBATCHES = 2


def pull_payloads(refs: List[ObjectRef]) -> None:
    """Start copying ``refs`` to this node without waiting for them.

    The copies are the node's secondary copies: nothing here holds them, so Ray can evict them under
    pressure (and ``fetch_payloads`` pulls again), and a failed pull surfaces there.
    """
    global _PULLS
    if _PULLS is None:
        _PULLS = ThreadPoolExecutor(max_workers=PULL_AHEAD_MICROBATCHES, thread_name_prefix="skyrl-payload-pull")
    _PULLS.submit(_pull, refs)


def _pull(refs: List[ObjectRef]) -> None:
    with stage_trace.span("mb.pull_ahead", payloads=len(refs)) as span:
        ready, _ = ray.wait(refs, num_returns=len(refs), timeout=SKYRL_PAYLOAD_FETCH_TIMEOUT_S, fetch_local=True)
        span["ready"] = len(ready)


# A dumped shard's advantages before z-scoring and the loss reduction: the unstreamed dump's advantages.
UNSCALED_ADVANTAGES = "unscaled_advantages"


@ray.remote(num_cpus=1)
def dump_training_batch(shard: RolloutShard, unscaled_advantages: Tuple[float, ...], path: str) -> None:
    """Save ``shard`` collated whole to ``path`` (``trainer.dump_data_batch``); nothing trains on it.

    Beside the trained (scaled) ``advantages``, the batch holds each row's ``unscaled_advantages``
    (one per row, before z-scoring and the loss reduction) over its response tokens.
    """
    batch = build_training_batch(shard)
    unscaled = torch.tensor(unscaled_advantages, dtype=torch.float32)
    batch[UNSCALED_ADVANTAGES] = unscaled[:, None] * batch[TrainingField.RESPONSE_MASK].float()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    batch.save(path)

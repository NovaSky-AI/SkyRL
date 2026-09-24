"""Server-side store for Rollout Routing Replay (R3) rollout routing.

Routing is produced on the inference servers, so it stays there: each vLLM
server actor holds a :class:`RoutedExpertsStash` populated at generation time
(``/skyrl/v1/completions`` in ``vllm_server_actor.py``), and the trainer
pulls what it needs at forward_backward time by fanning a digest query out to
all servers (``/skyrl/v1/routed_experts/fetch``). Sampling responses never
carry routing to any client, and the flow is identical in colocated and
non-colocated mode. The stash survives engine sleep (only GPU state is
offloaded) and dies with the server on teardown.

Entries are keyed by ``(model name on vLLM, blake2b digest of prompt +
response tokens)`` — exactly what the trainer reconstructs from a training
sample. Fetches never delete (multi-epoch training re-fetches after each
optim_step) but mark entries consumed. Deletion is lifecycle-driven: at a
model's weight sync (the event that starts its next rollout round), consumed
entries are deleted immediately and never-fetched ones (client-filtered
samples, eval rollouts) once they are more than ``max_staleness`` syncs old.
A FIFO cap on entries bounds memory for loops that never sync weights.

Imported by the vLLM server actor and the training backend; keep it light
(numpy + stdlib only).
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np

logger = logging.getLogger(__name__)

SKYRL_ROUTED_EXPERTS_STASH_MAX_ENTRIES = int(os.environ.get("SKYRL_ROUTED_EXPERTS_STASH_MAX_ENTRIES", 16384))


def sequence_digest(token_ids: Sequence[int]) -> bytes:
    """16-byte blake2b digest of the token ids (as int64 bytes): the R3 key."""
    buf = np.asarray(token_ids, dtype=np.int64).tobytes()
    return hashlib.blake2b(buf, digest_size=16).digest()


def decode_routed_experts_b64(payload: str) -> np.ndarray:
    """Decode vLLM's base64-encoded ``.npy`` routing payload into an array."""
    return np.load(io.BytesIO(base64.b64decode(payload)), allow_pickle=False)


def dump_arrays_npz(arrays: Dict[str, np.ndarray]) -> bytes:
    """Serialize ``{digest_hex: routing}`` as uncompressed ``.npz`` bytes."""
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def load_arrays_npz(payload: bytes) -> Dict[str, np.ndarray]:
    """Inverse of :func:`dump_arrays_npz`."""
    with np.load(io.BytesIO(payload), allow_pickle=False) as npz:
        return {name: npz[name] for name in npz.files}


@dataclass
class _StashEntry:
    routing: np.ndarray
    stored_gen: int
    consumed: bool = False


class RoutedExpertsStash:
    """In-memory routing store for one vLLM server (see module docstring)."""

    def __init__(self, max_entries: int = SKYRL_ROUTED_EXPERTS_STASH_MAX_ENTRIES):
        self.max_entries = max_entries
        self._entries: "OrderedDict[tuple[str, bytes], _StashEntry]" = OrderedDict()
        self._sync_gens: dict[str, int] = {}
        self._nbytes = 0

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def nbytes(self) -> int:
        """Total bytes of stashed routing arrays (metadata excluded)."""
        return self._nbytes

    def put(self, model: str, token_ids: Sequence[int], routing: np.ndarray) -> None:
        """Insert (or refresh) one sample's routing under the current sync generation."""
        key = (model, sequence_digest(token_ids))
        old = self._entries.pop(key, None)
        if old is not None:
            self._nbytes -= old.routing.nbytes
        self._entries[key] = _StashEntry(routing=routing, stored_gen=self._sync_gens.get(model, 0))
        self._nbytes += routing.nbytes
        while len(self._entries) > self.max_entries:
            _, evicted = self._entries.popitem(last=False)
            self._nbytes -= evicted.routing.nbytes

    def get_many(self, model: str, digest_hexes: Iterable[str]) -> Dict[str, np.ndarray]:
        """Bulk lookup of the digests present on this server.

        Marks hits consumed (deleted at the model's next weight sync) but never
        deletes directly: multi-epoch training re-fetches mid-batch.
        """
        hits: Dict[str, np.ndarray] = {}
        for digest_hex in digest_hexes:
            entry = self._entries.get((model, bytes.fromhex(digest_hex)))
            if entry is not None:
                entry.consumed = True
                hits[digest_hex] = entry.routing
        return hits

    def on_weight_sync(self, model: str, max_staleness: int) -> int:
        """Advance the model's sync generation and delete its dead entries.

        Deletes consumed entries (their training batch finished before this
        sync) and never-consumed entries stashed at generation
        ``g <= new_gen - 1 - max_staleness`` (their round is too old to still
        be trained on). Returns the number of entries deleted.
        """
        new_gen = self._sync_gens.get(model, 0) + 1
        self._sync_gens[model] = new_gen
        stale_cutoff = new_gen - 1 - max_staleness

        consumed = stale = 0
        for key in [k for k in self._entries if k[0] == model]:
            entry = self._entries[key]
            if entry.consumed:
                consumed += 1
            elif entry.stored_gen <= stale_cutoff:
                stale += 1
            else:
                continue
            self._nbytes -= entry.routing.nbytes
            del self._entries[key]
        if consumed or stale:
            logger.info(
                "R3 stash: weight sync for %s (gen %d) deleted %d trained and %d never-trained entries; "
                "%d entries (%.1f MB) remain.",
                model,
                new_gen,
                consumed,
                stale,
                len(self._entries),
                self._nbytes / 2**20,
            )
        return consumed + stale

    def evict_model(self, model: str) -> int:
        """Drop all entries (and the sync generation) for a deleted model."""
        keys: List[tuple[str, bytes]] = [k for k in self._entries if k[0] == model]
        for key in keys:
            self._nbytes -= self._entries[key].routing.nbytes
            del self._entries[key]
        self._sync_gens.pop(model, None)
        return len(keys)

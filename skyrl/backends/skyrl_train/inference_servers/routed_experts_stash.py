"""Server-side store for Rollout Routing Replay (R3) rollout routing.

With R3, training must replay the MoE expert choices vLLM made during
rollout. Routing is produced on the inference servers, so it stays there:
each vLLM server actor holds a :class:`RoutedExpertsStash` populated at
generation time (see the ``/skyrl/v1/completions`` and ``/skyrl/v1/generate``
endpoints in ``vllm_server_actor.py``), and the training backend pulls the
routing it needs at ``forward_backward`` time by fanning a digest query out
to all servers (``/skyrl/v1/routed_experts/fetch``).

This makes R3 a single system with one store and one lifecycle, identical in
colocated and non-colocated mode: sampling responses never carry routing to
any client, and no state crosses process boundaries except through these
HTTP endpoints. The stash survives engine sleep (only GPU state is offloaded;
the HTTP app keeps serving) and dies with the server on teardown, so there is
nothing to wipe.

Entries are keyed by ``(model, blake2b digest of the full token sequence)``,
where ``model`` is the name the request targeted on vLLM (a LoRA adapter
name in multi-tenant serving, the base/served model name otherwise) and the
sequence is prompt + response tokens — exactly what the trainer can
reconstruct from a training sample. Fetches are read-only so multi-epoch
training can re-fetch; eviction is lifecycle-driven instead:

- ``on_weight_sync(model, max_staleness)``: a weight sync starts the next
  rollout round for that model, so entries stashed more than
  ``max_staleness`` syncs ago can no longer be trained on and are dropped.
  ``max_staleness=1`` (the trainer's default) covers on-policy and
  one-step-async loops.
- ``evict_model(model)``: the model was deleted; its routing is dead.
- Insertion-order FIFO at ``max_entries`` (``SKYRL_ROUTED_EXPERTS_STASH_MAX_ENTRIES``,
  default 16384) bounds memory for loops that never sync weights.

This module is imported by the vLLM server actor and the training backend;
keep it light (numpy + stdlib only).
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
    """16-byte blake2b digest of the token ids, hashed as int64 bytes.

    The R3 key: fixed-size regardless of sequence length, and order- and
    value-exact. The trainer computes the same digest from a training sample's
    full sequence (prompt + response) to address the stash.
    """
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
        """Read-only bulk lookup; returns only the digests present on this server.

        Read-only so multi-epoch training can fetch the same routing again;
        entries are dropped by the lifecycle hooks below, not by consumption.
        """
        hits: Dict[str, np.ndarray] = {}
        for digest_hex in digest_hexes:
            entry = self._entries.get((model, bytes.fromhex(digest_hex)))
            if entry is not None:
                hits[digest_hex] = entry.routing
        return hits

    def on_weight_sync(self, model: str, max_staleness: int) -> int:
        """Advance the model's sync generation and drop entries past the staleness window.

        An entry stashed at generation ``g`` is dropped at the sync producing
        generation ``n`` when ``g <= n - 1 - max_staleness``: training on the
        rollout round it belongs to has finished by then (rounds are separated
        by weight syncs), or the round was abandoned. Returns the drop count.
        """
        new_gen = self._sync_gens.get(model, 0) + 1
        self._sync_gens[model] = new_gen
        stale_cutoff = new_gen - 1 - max_staleness

        removed = 0
        for key in [k for k in self._entries if k[0] == model]:
            entry = self._entries[key]
            if entry.stored_gen <= stale_cutoff:
                self._nbytes -= entry.routing.nbytes
                del self._entries[key]
                removed += 1
        if removed:
            logger.info(
                "R3 stash: weight sync for %s (gen %d) dropped %d entries; %d entries (%.1f MB) remain.",
                model,
                new_gen,
                removed,
                len(self._entries),
                self._nbytes / 2**20,
            )
        return removed

    def evict_model(self, model: str) -> int:
        """Drop all entries (and the sync generation) for a deleted model."""
        keys: List[tuple[str, bytes]] = [k for k in self._entries if k[0] == model]
        for key in keys:
            self._nbytes -= self._entries[key].routing.nbytes
            del self._entries[key]
        self._sync_gens.pop(model, None)
        return len(keys)

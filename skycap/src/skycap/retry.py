"""SDK retries: the same logical call gets the original's answer, not a second sample.

A client that times out resends the same request. Without help, capture can't
tell that from a deliberate resample, so the retry samples again and the graph
forks on a reply the harness never saw. The SDK does say which it is: OpenAI's
and Anthropic's clients send ``x-stainless-retry-count`` (0 on the first
attempt), and a client may send an ``Idempotency-Key``.

So each trajectory keeps a small cache of its recent calls, keyed by the
idempotency key or else by the request body. A request marked as a retry:

* gets the stored reply when the original has finished;
* waits for the original when it is still running (the harness gave up on
  it, but the engine didn't), then gets that reply;
* runs as a new call when the original failed, since it committed nothing.

A repeated body with no retry marker is a genuine resample and runs normally.
A text-mode stream is relayed as it arrives and isn't cached.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field

RETRY_COUNT_HEADER = "x-stainless-retry-count"
IDEMPOTENCY_KEY_HEADER = "Idempotency-Key"

TTL_SECONDS = 600.0
MAX_ENTRIES = 64


def is_retry(headers: Mapping[str, str]) -> bool:
    try:
        return int(headers.get(RETRY_COUNT_HEADER, 0)) > 0
    except ValueError:
        return False


def body_digest(raw: bytes) -> str:
    return hashlib.blake2b(raw, digest_size=16).hexdigest()


@dataclass(slots=True)
class Replay:
    body: bytes
    status: int
    headers: dict[str, str]


@dataclass(slots=True)
class Entry:
    digest: str
    future: asyncio.Future[Replay | None] = field(default_factory=lambda: asyncio.get_running_loop().create_future())
    completed_at: float | None = None

    @property
    def reply(self) -> Replay | None:
        return self.future.result() if self.future.done() and not self.future.cancelled() else None


class RetryCache:
    """One trajectory's recent calls. Bounded by age and count; never persisted."""

    def __init__(self, *, ttl: float = TTL_SECONDS, max_entries: int = MAX_ENTRIES) -> None:
        self.ttl = ttl
        self.max_entries = max_entries
        self._entries: OrderedDict[str, Entry] = OrderedDict()
        self.replayed = 0
        self.coalesced = 0

    def get(self, key: str) -> Entry | None:
        self._prune()
        return self._entries.get(key)

    def start(self, key: str, digest: str) -> Entry:
        """Record a new call under ``key``, replacing whatever was there."""
        entry = Entry(digest=digest)
        self._entries[key] = entry
        self._entries.move_to_end(key)
        self._prune()
        return entry

    def complete(self, key: str, entry: Entry, reply: Replay | None) -> None:
        """Settle ``entry``. A failed call (``None``) is forgotten so a retry runs anew."""
        if not entry.future.done():
            entry.future.set_result(reply)
        entry.completed_at = time.monotonic()
        if reply is None and self._entries.get(key) is entry:
            del self._entries[key]

    def _prune(self) -> None:
        cutoff = time.monotonic() - self.ttl
        for key in [k for k, e in self._entries.items() if e.completed_at is not None and e.completed_at < cutoff]:
            del self._entries[key]
        while len(self._entries) > self.max_entries:
            oldest = next((k for k, e in self._entries.items() if e.future.done()), None)
            if oldest is None:
                return
            del self._entries[oldest]

    def __len__(self) -> int:
        return len(self._entries)

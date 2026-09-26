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
  When several retries wait on one failed call, the first starts the new call
  and the rest wait on that.

A repeated body with no retry marker is a genuine resample and runs normally.
Only a reply whose call was committed to the graph is replayed (a backend marks
it with ``committed``); a text-mode stream is relayed as it arrives and isn't.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from aiohttp import web

from skycap.openai_chat import error_body

RETRY_COUNT_HEADER = "x-stainless-retry-count"
IDEMPOTENCY_KEY_HEADER = "Idempotency-Key"

TTL_SECONDS = 600.0
MAX_ENTRIES = 64

_COMMITTED = web.ResponseKey("skycap.committed", bool)
_UNREPLAYED_HEADERS = {"content-length", "date", "server", "transfer-encoding"}


def is_retry(headers: Mapping[str, str]) -> bool:
    try:
        return int(headers.get(RETRY_COUNT_HEADER, 0)) > 0
    except ValueError:
        return False


def body_digest(raw: bytes) -> str:
    return hashlib.blake2b(raw, digest_size=16).hexdigest()


def committed(response: web.Response) -> web.Response:
    """Mark ``response`` as the reply to a call recorded in the graph, so a retry may get it."""
    response[_COMMITTED] = True
    return response


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

    async def call(
        self, headers: Mapping[str, str], raw: bytes, run: Callable[[], asyncio.Future[web.StreamResponse]]
    ) -> web.StreamResponse:
        """Answer one request: the reply of the call it retries, or a new call started by ``run``.

        ``run`` must start the call as its own task, so a client that disconnects
        (an SDK timing out, about to retry) doesn't cancel it: it finishes and
        leaves its reply for the retry.
        """
        digest = body_digest(raw)
        explicit = headers.get(IDEMPOTENCY_KEY_HEADER)
        key = f"key:{explicit}" if explicit else f"body:{digest}"
        if explicit or is_retry(headers):
            while (previous := self.get(key)) is not None:
                if previous.digest != digest:
                    return web.Response(
                        body=error_body("Idempotency-Key was reused with a different request"),
                        status=400,
                        content_type="application/json",
                    )
                waited = not previous.future.done()
                reply = await asyncio.shield(previous.future)
                if reply is not None:
                    self.replayed += 1
                    self.coalesced += waited
                    return web.Response(body=reply.body, status=reply.status, headers=reply.headers)
                if self._entries.get(key) is previous:
                    break  # it failed and no other retry has replaced it yet: this one does
        entry = self.start(key, digest)
        work = run()
        work.add_done_callback(lambda task: self.complete(key, entry, _replayable(task)))
        return await asyncio.shield(work)

    def get(self, key: str) -> Entry | None:
        self._prune()
        entry = self._entries.get(key)
        if entry is not None:
            self._entries.move_to_end(key)
        return entry

    def start(self, key: str, digest: str) -> Entry:
        """Record a new call under ``key``, replacing whatever was there."""
        entry = Entry(digest=digest)
        self._entries[key] = entry
        self._entries.move_to_end(key)
        self._prune()
        return entry

    def complete(self, key: str, entry: Entry, reply: Replay | None) -> None:
        """Settle ``entry``. A failed call (``None``) stays, so its key still refuses a different body."""
        if not entry.future.done():
            entry.future.set_result(reply)
        entry.completed_at = time.monotonic()
        self._prune()

    def _prune(self) -> None:
        """Drop settled entries past the TTL, then the oldest settled ones past the count.

        A running call's entry is never dropped: its retries wait on it, and it
        holds no body yet.
        """
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


def _replayable(task: asyncio.Future[web.StreamResponse]) -> Replay | None:
    """The finished call's reply, if a retry may be answered with it."""
    if task.cancelled() or task.exception() is not None:
        return None
    response = task.result()
    if not isinstance(response, web.Response) or not response.get(_COMMITTED) or not isinstance(response.body, bytes):
        return None
    headers = {k: v for k, v in response.headers.items() if k.lower() not in _UNREPLAYED_HEADERS}
    return Replay(body=response.body, status=response.status, headers=headers)

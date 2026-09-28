"""Token sessions: the transient state one trajectory's next exact prompt needs.

A session is a `TokenTrace` -- the committed message graph with its exact
token deltas -- plus the lock that serialises turns on it. Prefix reuse is
inherently sequential (turn N+1's prefix is turn N's output), so two turns on
one trajectory would interleave attribution rather than add throughput.

The manager owns exactly what is transient:

* per-trajectory locks;
* a bounded LRU of traces, sized by resident tokens rather than by count,
  because trajectories differ by four orders of magnitude;
* the renderer, built once per process and shared, because building one
  loads a tokenizer and takes seconds; and
* rebuilding a trace from what has been stored after eviction or a restart.

It owns nothing else. Lifecycle, persistence, transport and the
control plane are other components' business, and this one reaches none of
them: what it needs from storage arrives through `TraceSource`, which is the
single place the refactor swaps when the store changes.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import Any, Protocol

from skyrl_capture.config import TitoUpstream
from skyrl_capture.tito.renderer import TokenRenderer, build_renderer
from skyrl_capture.tito.trace import TokenTrace


class TraceSource(Protocol):
    """Where a trace comes from when this process has no memory of it.

    The live state, which is complete the moment a turn commits: there is no
    "accepted but not yet stored" to wait for before rebuilding on it.
    """

    async def load(self, trajectory_id: str) -> TokenTrace: ...


class TokenSessionManager:

    def __init__(self, *, upstream: TitoUpstream, trace_budget: int, source: TraceSource) -> None:
        self._upstream = upstream
        self._source = source
        self._trace_budget = trace_budget
        # A trace is a cache, not the record: the graph can rebuild it. So it
        # can be evicted, and it has to be -- a replica that keeps every
        # trajectory it has ever served grows until it dies.
        self._traces: OrderedDict[str, TokenTrace] = OrderedDict()
        self._locks: dict[str, asyncio.Lock] = {}
        self._renderers: dict[str, TokenRenderer] = {}
        self._renderer_locks: dict[str, asyncio.Lock] = {}
        self.traces_evicted = 0

    # -- locks -------------------------------------------------------------
    def lock(self, trajectory_id: str) -> asyncio.Lock:
        lock = self._locks.get(trajectory_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[trajectory_id] = lock
        return lock

    # -- renderer ----------------------------------------------------------
    async def renderer(self) -> TokenRenderer:
        """The renderer for the configured upstream, built once.

        Building one loads a tokenizer and fills a pool, which takes seconds --
        so it happens in a thread. Built inline it blocked the event loop for
        about four seconds on a target's first turn, stalling every other
        request in flight at that moment, which a cold replica taking traffic
        feels as a four-second cliff rather than a slow first turn.

        The lock is what keeps that cost paid once. Without it, every turn that
        arrives during the build sees an empty cache and starts its own.
        """
        upstream = self._upstream
        key = upstream.tokenizer or ""
        renderer = self._renderers.get(key)
        if renderer is not None:
            return renderer
        async with self._renderer_lock(key):
            renderer = self._renderers.get(key)
            if renderer is None:
                renderer = await asyncio.to_thread(
                    build_renderer,
                    tokenizer=upstream.tokenizer,
                    model=upstream.model,
                    config=upstream.config,
                )
                self._renderers[key] = renderer
        return renderer

    def _renderer_lock(self, key: str) -> asyncio.Lock:
        lock = self._renderer_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._renderer_locks[key] = lock
        return lock

    # -- traces ------------------------------------------------------------
    async def trace_for(self, trajectory_id: str) -> TokenTrace:
        trace = self._traces.get(trajectory_id)
        if trace is not None:
            self._traces.move_to_end(trajectory_id)
            return trace
        trace = await self._source.load(trajectory_id)
        self._traces[trajectory_id] = trace
        self._evict_until_within_budget()
        return trace

    def attach_text_renderer(self, trace: TokenTrace, renderer: Any) -> None:
        """Give the trace the renderer that created its exact tokens.

        Once per trace, from the renderer the turn already built. A node
        decodes its own tokens as it is created, so a record carries text and
        no reader needs a tokenizer -- see `TokenNode.attach_text`.
        """
        if trace._render_text is not None:  # noqa: SLF001 - its own package's field
            return
        trace.set_text_renderer(renderer.decode, renderer.turn_start_token())

    def forget(self, trajectory_id: str) -> None:
        self._traces.pop(trajectory_id, None)
        self._locks.pop(trajectory_id, None)

    def resident_tokens(self) -> int:
        """Tokens held across every cached trace."""
        return sum(trace.stored_tokens for trace in self._traces.values())

    def _evict_until_within_budget(self) -> None:
        """Drop least-recently-used traces until the budget is met.

        Never drops the trace a turn is currently using: that one was just
        touched, so it is the most recent.
        """
        if self._trace_budget <= 0:
            return
        resident = self.resident_tokens()
        while resident > self._trace_budget and len(self._traces) > 1:
            _evicted, trace = self._traces.popitem(last=False)
            resident -= trace.stored_tokens
            self.traces_evicted += 1

    # -- what health reports ----------------------------------------------
    def stats(self) -> dict[str, Any]:
        return {
            "cached_traces": len(self._traces),
            "resident_tokens": self.resident_tokens(),
            "trace_budget_tokens": self._trace_budget,
            "traces_evicted": self.traces_evicted,
            # Prefix divergences the audit classified, by class. Only populated
            # under TOKENS_AUDIT_PREFIX, and never non-zero in a healthy run --
            # both sides of that comparison are tokens this process produced.
            "audit_failures": self.audit_failures(),
        }

    def audit_failures(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for trace in self._traces.values():
            for kind, count in trace.audit_failures.items():
                totals[kind] = totals.get(kind, 0) + count
        return totals

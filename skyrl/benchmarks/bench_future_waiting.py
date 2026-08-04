#!/usr/bin/env python3
"""Measure how request waiting loads the API server's connection pool.

No GPU, no vLLM, no HTTP server -- just the database layer the API shares between
`retrieve_future` and every other endpoint.

The API's async engine takes SQLAlchemy's defaults: 15 connections (5 + 10
overflow) with a 30s checkout timeout. Waiting per caller means every in-flight
request holds a checkout on each poll, so checkouts scale with concurrency and
unrelated endpoints queue behind them. `session_heartbeat` is the canary here --
a trivial write that should always be fast.

Shapes the load like a fully-async RL run: `groups_per_batch` x `group_size`
concurrent samples, each awaiting its own result.

Usage:
  uv run --isolated --extra dev --extra tinker python skyrl/benchmarks/bench_future_waiting.py
  uv run --isolated --extra dev --extra tinker python skyrl/benchmarks/bench_future_waiting.py --waiters 4096
"""

import argparse
import asyncio
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from sqlalchemy.exc import TimeoutError as SATimeoutError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import Session, SQLModel, create_engine, select
from sqlmodel.ext.asyncio.session import AsyncSession

from skyrl.tinker import types
from skyrl.tinker.db_models import (
    FutureDB,
    RequestStatus,
    SessionDB,
    enable_sqlite_wal,
    get_async_database_url,
)

SESSION_ID = "sess_repro"


def seed(sync_engine, count: int) -> list[int]:
    with Session(sync_engine) as session:
        session.add(SessionDB(session_id=SESSION_ID, sdk_version="repro", last_heartbeat_at=datetime.now(timezone.utc)))
        prompt = types.SampleInput(
            prompt=types.ModelInput(chunks=[types.EncodedTextChunk(tokens=list(range(180)))]),
            sampling_params=types.SamplingParams(temperature=1.0, max_tokens=512, seed=0),
            num_samples=1,
            checkpoint_id="ck0",
            prompt_logprobs=False,
        ).model_dump(mode="json")
        rows = [
            FutureDB(
                request_type=types.RequestType.EXTERNAL,
                model_id="model_a",
                request_data=prompt,
                status=RequestStatus.PENDING,
            )
            for _ in range(count)
        ]
        for row in rows:
            session.add(row)
        session.commit()
        return [row.request_id for row in rows]


async def poll_per_caller(db_engine, request_id: int, counter: list[int], errors: list[str]) -> None:
    """Waiting the way retrieve_future used to: a session and query per caller."""
    poll = 0.1
    while True:
        try:
            async with AsyncSession(db_engine) as session:
                await session.exec(select(FutureDB.status).where(FutureDB.request_id == request_id))
            counter[0] += 1
        except SATimeoutError:
            # What the old endpoint did: swallow and keep polling.
            errors.append("poll pool timeout")
        await asyncio.sleep(poll)
        poll = min(poll * 1.5, 1.0)


async def heartbeat_probe(db_engine, latencies: list[float], failures: list[str], stop: asyncio.Event) -> None:
    """The session_heartbeat endpoint: a tiny write that needs a pool connection."""
    while not stop.is_set():
        started = time.perf_counter()
        try:
            async with AsyncSession(db_engine) as session:
                row = await session.get(SessionDB, SESSION_ID)
                row.last_heartbeat_at = datetime.now(timezone.utc)
                row.heartbeat_count += 1
                await session.commit()
            latencies.append((time.perf_counter() - started) * 1000)
        except Exception as exc:
            failures.append(type(exc).__name__)
        await asyncio.sleep(2.0)


async def completion_write(db_engine, request_id: int) -> str:
    """A result write-back, to check it can still get a connection under this load."""
    async with AsyncSession(db_engine) as session:
        future = await session.get(FutureDB, request_id)
        future.result_data = {"sequences": []}
        future.status = RequestStatus.COMPLETED
        future.completed_at = datetime.now(timezone.utc)
        await session.commit()
    return "ok"


async def run_mode(mode: str, db_url: str, args) -> None:
    async_engine = create_async_engine(get_async_database_url(db_url))
    enable_sqlite_wal(async_engine.sync_engine)
    pool = async_engine.pool
    capacity = pool.size() + pool._max_overflow

    sync_engine = create_engine(db_url)
    enable_sqlite_wal(sync_engine)
    SQLModel.metadata.create_all(sync_engine)
    request_ids = seed(sync_engine, args.waiters)

    polls = [0]
    poll_errors: list[str] = []
    hb_latencies: list[float] = []
    hb_failures: list[str] = []
    stop = asyncio.Event()

    if mode == "per-caller":
        tasks = [asyncio.create_task(poll_per_caller(async_engine, rid, polls, poll_errors)) for rid in request_ids]
    else:
        from skyrl.tinker.futures import FutureWaiter

        waiter = FutureWaiter(async_engine, poll_interval_sec=0.05)
        await waiter.start()
        tasks = [asyncio.create_task(waiter.wait(rid, 3600)) for rid in request_ids]

    hb_task = asyncio.create_task(heartbeat_probe(async_engine, hb_latencies, hb_failures, stop))

    await asyncio.sleep(args.duration)

    # With the pool saturated, can a sample completion still be written?
    completion_result = "ok"
    try:
        await completion_write(async_engine, request_ids[0])
    except Exception as exc:
        completion_result = f"{type(exc).__name__}: {str(exc)[:60]}"

    stop.set()
    hb_task.cancel()
    for task in tasks:
        task.cancel()
    await asyncio.gather(hb_task, *tasks, return_exceptions=True)
    if mode == "shared":
        await waiter.stop()

    demanded = args.waiters / 1.0  # each caller polls ~1/s once backoff saturates
    achieved = polls[0] / args.duration if mode == "per-caller" else None

    print(f"\n--- mode={mode} ---")
    print(f"  pool capacity: {capacity} connections, checkout timeout {pool._timeout:.0f}s")
    print(f"  concurrent waiters: {args.waiters}")
    if achieved is not None:
        print(f"  polls demanded: ~{demanded:.0f}/s   achieved: {achieved:.0f}/s")
        print(f"  poll pool timeouts swallowed: {len(poll_errors)}")
    else:
        print("  polls: 1 batched query per 50ms tick, independent of waiter count")
    if hb_latencies:
        ordered = sorted(hb_latencies)
        print(
            f"  heartbeat: n={len(ordered)}  p50={statistics.median(ordered):.0f}ms  "
            f"max={ordered[-1]:.0f}ms  failures={len(hb_failures)}"
        )
    else:
        print(f"  heartbeat: NEVER COMPLETED ONCE in {args.duration}s  failures={len(hb_failures)}")
    verdict = "OK" if completion_result == "ok" else "FAILED -> future orphaned PENDING forever"
    print(f"  sample completion write: {verdict}")
    if completion_result != "ok":
        print(f"    {completion_result}")

    await async_engine.dispose()
    sync_engine.dispose()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--waiters", type=int, default=2048, help="Concurrent in-flight requests")
    parser.add_argument("--duration", type=float, default=25.0)
    args = parser.parse_args()

    print(f"Concurrent waiters: {args.waiters}, over {args.duration}s\n")
    for mode in ("per-caller", "shared"):
        with TemporaryDirectory(prefix=f"bench_{mode}_") as tmpdir:
            await run_mode(mode, f"sqlite:///{Path(tmpdir) / 'tinker.db'}", args)


if __name__ == "__main__":
    asyncio.run(main())

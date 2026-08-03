#!/usr/bin/env python3
"""Benchmark the Tinker server's database layer under multi-LoRA concurrency.

Isolates the DB from the GPU: no backend, no HTTP server, no inference engine.
Every phase drives the same functions the API server and engine use, so the
numbers move when those code paths change.

Phases
  submit       Concurrent request submission (``create_future`` + commit), the
               path every forward_backward/forward/sample POST takes.
  engine_scan  One iteration of the engine's scheduling queries against a
               backlog of pending rows, including rows parked behind a barrier.
  complete     Result write-back (``_complete_futures``).
  await        Concurrent ``retrieve_future`` waiters.

Alongside wall-clock timings each phase reports the number of SQL statements
and write transactions issued, which is what actually serializes on SQLite's
single writer.

Usage:
  uv run --isolated --extra dev --extra tinker python skyrl/benchmarks/bench_tinker_db.py

  # Heavier multi-LoRA shape
  uv run --isolated --extra dev --extra tinker python skyrl/benchmarks/bench_tinker_db.py \
      --num-models 16 --concurrency 128 --num-requests 512
"""

import argparse
import asyncio
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import Session, SQLModel, create_engine, select
from sqlmodel.ext.asyncio.session import AsyncSession

from skyrl.tinker import types
from skyrl.tinker.api import create_future
from skyrl.tinker.config import EngineConfig
from skyrl.tinker.db_models import (
    FutureDB,
    ModelDB,
    RequestStatus,
    SessionDB,
    enable_sqlite_wal,
    get_async_database_url,
)
from skyrl.tinker.engine import TinkerEngine

try:
    from skyrl.tinker.db_models import json_engine_kwargs
except ImportError:
    # Older checkouts have no configurable JSON codec; run against their default
    # so before/after comparisons stay possible.
    def json_engine_kwargs() -> dict:
        return {}


@dataclass
class SQLCounter:
    """Counts statements and commits issued on an engine.

    ``writes`` counts data-modifying statements; on SQLite these are what
    contend for the single write lock, and ``commits`` is how many separate
    write transactions they were spread across.
    """

    statements: int = 0
    writes: int = 0
    commits: int = 0

    def attach(self, sync_engine) -> None:
        @event.listens_for(sync_engine, "before_cursor_execute")
        def _count(conn, cursor, statement, parameters, context, executemany):
            self.statements += 1
            verb = statement.lstrip().split(" ", 1)[0].upper()
            if verb in ("INSERT", "UPDATE", "DELETE"):
                self.writes += 1

        @event.listens_for(sync_engine, "commit")
        def _count_commit(conn):
            self.commits += 1

    def reset(self) -> None:
        self.statements = self.writes = self.commits = 0

    def snapshot(self) -> dict[str, int]:
        return {"statements": self.statements, "writes": self.writes, "commits": self.commits}


@dataclass
class PhaseResult:
    name: str
    n: int
    elapsed_s: float
    sql: dict[str, int]
    latencies_ms: list[float] = field(default_factory=list)
    note: str = ""

    def report(self) -> str:
        rate = self.n / self.elapsed_s if self.elapsed_s > 0 else float("inf")
        line = f"  {self.name:<12} n={self.n:>5}  {self.elapsed_s:>7.3f}s  {rate:>8.1f} op/s"
        if self.latencies_ms:
            ordered = sorted(self.latencies_ms)
            p50 = statistics.median(ordered)
            p99 = ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))]
            line += f"  p50={p50:>7.1f}ms  p99={p99:>7.1f}ms"
        line += f"  sql={self.sql['statements']:>5} (w={self.sql['writes']:>5}, commits={self.sql['commits']:>5})"
        if self.note:
            line += f"  {self.note}"
        return line


def make_forward_backward_input(seq_len: int, seqs_per_request: int) -> types.ForwardBackwardInput:
    """Build a realistically sized forward_backward payload."""
    tokens = list(range(seq_len))
    floats = [0.1234567] * seq_len
    datum = types.Datum(
        model_input=types.ModelInput(chunks=[types.EncodedTextChunk(tokens=tokens)]),
        loss_fn_inputs=types.LossFnInputs(
            target_tokens=types.TensorData(data=tokens),
            weights=types.TensorData(data=floats),
            advantages=types.TensorData(data=floats),
            logprobs=types.TensorData(data=floats),
        ),
    )
    return types.ForwardBackwardInput(data=[datum] * seqs_per_request, loss_fn="importance_sampling")


def make_sample_input(prompt_len: int) -> types.SampleInput:
    return types.SampleInput(
        prompt=types.ModelInput(chunks=[types.EncodedTextChunk(tokens=list(range(prompt_len)))]),
        sampling_params=types.SamplingParams(temperature=1.0, max_tokens=128, seed=0),
        num_samples=1,
        checkpoint_id="",
        prompt_logprobs=False,
    )


def seed_models(sync_engine, num_models: int) -> list[str]:
    """Create a session plus ``num_models`` LoRA model rows, returning model ids."""
    session_id = f"sess_{uuid4().hex[:8]}"
    model_ids = [f"model_{i}" for i in range(num_models)]
    with Session(sync_engine) as session:
        session.add(
            SessionDB(
                session_id=session_id,
                sdk_version="bench",
                last_heartbeat_at=datetime.now(timezone.utc),
            )
        )
        for i, model_id in enumerate(model_ids):
            session.add(
                ModelDB(
                    model_id=model_id,
                    base_model="bench/base",
                    lora_config={"rank": 32, "alpha": 64.0, "seed": 0},
                    status="active",
                    request_id=i,
                    session_id=session_id,
                )
            )
        session.commit()
    return model_ids


async def phase_submit(
    db_engine,
    counter: SQLCounter,
    model_ids: list[str],
    num_requests: int,
    concurrency: int,
    payload: types.ForwardBackwardInput,
) -> PhaseResult:
    """Submit ``num_requests`` forward_backward futures with ``concurrency`` in flight.

    Mirrors the API server: one AsyncSession and one commit per request.
    """
    semaphore = asyncio.Semaphore(concurrency)
    latencies: list[float] = []

    async def submit_one(index: int) -> None:
        async with semaphore:
            started = time.perf_counter()
            async with AsyncSession(db_engine) as session:
                await create_future(
                    session=session,
                    request_type=types.RequestType.FORWARD_BACKWARD,
                    model_id=model_ids[index % len(model_ids)],
                    request_data=payload,
                )
                await session.commit()
            latencies.append((time.perf_counter() - started) * 1000)

    counter.reset()
    started = time.perf_counter()
    await asyncio.gather(*(submit_one(i) for i in range(num_requests)))
    elapsed = time.perf_counter() - started
    return PhaseResult("submit", num_requests, elapsed, counter.snapshot(), latencies)


def add_barriers(sync_engine, model_ids: list[str], fraction: float) -> int:
    """Insert a pending optim_step for the first ``fraction`` of models.

    These are the scheduling barriers that park later requests, which is the
    steady state of a multi-LoRA run: some adapters stepping while others queue.
    """
    blocked = model_ids[: max(1, int(len(model_ids) * fraction))]
    payload = types.OptimStepInput(
        adam_params=types.AdamParams(learning_rate=1e-4, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0)
    ).model_dump(mode="json")
    with Session(sync_engine) as session:
        for model_id in blocked:
            session.add(
                FutureDB(
                    request_type=types.RequestType.OPTIM_STEP,
                    model_id=model_id,
                    request_data=payload,
                    status=RequestStatus.PENDING,
                )
            )
        session.commit()
    return len(blocked)


def phase_engine_scan(
    engine: TinkerEngine, counter: SQLCounter, iterations: int, name: str
) -> tuple[PhaseResult, list[str]]:
    """Time the engine's per-iteration scheduling queries.

    Returns the phase result plus the ids of the model passes the last iteration
    would have dispatched. Barriers are intentionally excluded so the caller can
    retire the passes and leave the queue in its parked state.
    """
    counter.reset()
    latencies: list[float] = []
    started = time.perf_counter()
    passes: list[str] = []
    singles = 0
    for _ in range(iterations):
        iteration_start = time.perf_counter()
        with Session(engine.db_engine) as session:
            # Tolerate engines without the shared metadata scan so this benchmark
            # can also be run against an older checkout for comparison.
            if hasattr(engine, "scan_pending_requests"):
                pending = [engine.scan_pending_requests(session)]
            else:
                pending = []
            fb = engine.find_batchable_model_passes(session, types.RequestType.FORWARD_BACKWARD, *pending)
            fwd = engine.find_batchable_model_passes(session, types.RequestType.FORWARD, *pending)
            samples = engine.find_batchable_sample(session, *pending)
            others = engine.find_single_requests(session, *pending)
        passes = [*fb, *fwd, *samples]
        singles = len(others)
        latencies.append((time.perf_counter() - iteration_start) * 1000)
    elapsed = time.perf_counter() - started
    return (
        PhaseResult(
            name,
            iterations,
            elapsed,
            counter.snapshot(),
            latencies,
            note=f"dispatchable={len(passes)} passes + {singles} singles",
        ),
        passes,
    )


def phase_complete(engine: TinkerEngine, counter: SQLCounter, request_ids: list[str]) -> PhaseResult:
    """Time result write-back for a dispatched batch."""
    result = types.ForwardBackwardOutput(
        loss_fn_output_type="importance_sampling",
        loss_fn_outputs=[{"loss": 0.5}],
        metrics={"loss:sum": 1.0, "loss:count": 2.0},
    )
    results = {request_id: result for request_id in request_ids}

    counter.reset()
    started = time.perf_counter()
    engine._complete_futures(results)
    elapsed = time.perf_counter() - started
    return PhaseResult("complete", len(results), elapsed, counter.snapshot())


async def _legacy_await_future(db_engine, request_id: int, deadline: float) -> dict | None:
    """The pre-optimization retrieve_future wait loop, for baseline comparison."""
    poll = 0.1
    max_poll = 1.0
    while time.perf_counter() < deadline:
        async with AsyncSession(db_engine) as session:
            status = (await session.exec(select(FutureDB.status).where(FutureDB.request_id == request_id))).first()
            if status in (RequestStatus.COMPLETED, RequestStatus.FAILED):
                future = (await session.exec(select(FutureDB).where(FutureDB.request_id == request_id))).first()
                return future.result_data
        await asyncio.sleep(poll)
        poll = min(poll * 1.5, max_poll)
    return None


async def phase_await(
    db_engine,
    counter: SQLCounter,
    request_ids: list[int],
    hold_s: float,
    mode: str,
) -> PhaseResult:
    """Measure DB load from concurrent waiters while results are still pending.

    Waiters start before the results land, so this counts the polling traffic a
    real run pays while the engine is busy on the GPU.
    """
    counter.reset()
    deadline = time.perf_counter() + hold_s

    if mode == "shared":
        from skyrl.tinker.futures import FutureWaiter

        waiter = FutureWaiter(db_engine)
        await waiter.start()
        tasks = [asyncio.create_task(waiter.wait(rid, hold_s)) for rid in request_ids]
    else:
        tasks = [asyncio.create_task(_legacy_await_future(db_engine, rid, deadline)) for rid in request_ids]

    started = time.perf_counter()
    await asyncio.sleep(hold_s)
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = time.perf_counter() - started

    if mode == "shared":
        await waiter.stop()

    sql = counter.snapshot()
    return PhaseResult(
        "await",
        len(request_ids),
        elapsed,
        sql,
        note=f"mode={mode}, {sql['statements'] / max(elapsed, 1e-9):.0f} stmt/s",
    )


async def run(args) -> None:
    with TemporaryDirectory(prefix="bench_tinker_db_") as tmpdir:
        db_path = Path(tmpdir) / "tinker.db"
        db_url = f"sqlite:///{db_path}"

        sync_engine = create_engine(db_url, echo=False, **json_engine_kwargs())
        enable_sqlite_wal(sync_engine)
        SQLModel.metadata.create_all(sync_engine)

        async_engine = create_async_engine(get_async_database_url(db_url), echo=False, **json_engine_kwargs())
        enable_sqlite_wal(async_engine.sync_engine)

        counter = SQLCounter()
        counter.attach(async_engine.sync_engine)
        sync_counter = SQLCounter()
        sync_counter.attach(sync_engine)

        model_ids = seed_models(sync_engine, args.num_models)

        payload = make_forward_backward_input(args.seq_len, args.seqs_per_request)
        payload_kb = len(payload.model_dump_json()) / 1024

        engine = object.__new__(TinkerEngine)
        engine.config = EngineConfig(base_model="bench/base", backend="fsdp", database_url=db_url)
        engine.db_engine = sync_engine

        print("Tinker DB benchmark")
        print(f"  db={db_path}")
        print(f"  models={args.num_models}  concurrency={args.concurrency}  requests={args.num_requests}")
        print(f"  payload={payload_kb:.1f} KiB/request ({args.seqs_per_request} seqs x {args.seq_len} tokens)")
        print()

        results: list[PhaseResult] = []

        results.append(
            await phase_submit(async_engine, counter, model_ids, args.num_requests, args.concurrency, payload)
        )

        # Reproduce the multi-LoRA steady state: some adapters have an optim_step
        # queued (a barrier) with further passes stacked behind it. Those parked
        # requests cannot be dispatched, but the scheduler still sees them on
        # every poll.
        blocked_models = add_barriers(sync_engine, model_ids, args.barrier_fraction)
        if args.blocked_requests:
            await phase_submit(
                async_engine, counter, model_ids[:blocked_models], args.blocked_requests, args.concurrency, payload
            )

        scan, dispatched = phase_engine_scan(engine, sync_counter, args.scan_iterations, "scan_ready")
        scan.note += f", barriers={blocked_models}/{args.num_models}, parked={args.blocked_requests}"
        results.append(scan)

        results.append(phase_complete(engine, sync_counter, dispatched))

        # With the dispatchable batch retired, only requests parked behind a
        # barrier remain. This is where the engine sits between barriers, and it
        # re-runs this scan every poll interval.
        parked_scan, _ = phase_engine_scan(engine, sync_counter, args.scan_iterations, "scan_parked")
        results.append(parked_scan)

        with Session(sync_engine) as session:
            pending_ids = session.exec(
                select(FutureDB.request_id)
                .where(FutureDB.status == RequestStatus.PENDING)
                .where(FutureDB.request_type == types.RequestType.FORWARD_BACKWARD)
                .limit(args.await_waiters)
            ).all()
        results.append(await phase_await(async_engine, counter, list(pending_ids), args.await_hold_s, args.poller))

        db_bytes = db_path.stat().st_size
        wal = db_path.with_name(db_path.name + "-wal")
        wal_bytes = wal.stat().st_size if wal.exists() else 0

        for result in results:
            print(result.report())
        print()
        print(f"  db file={db_bytes / 1e6:.1f} MB   wal={wal_bytes / 1e6:.1f} MB")

        await async_engine.dispose()
        sync_engine.dispose()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-models", type=int, default=8, help="Number of concurrent LoRA adapters")
    parser.add_argument("--num-requests", type=int, default=256, help="forward_backward requests to submit")
    parser.add_argument("--concurrency", type=int, default=64, help="Submissions in flight")
    parser.add_argument("--seq-len", type=int, default=512, help="Tokens per sequence in the payload")
    parser.add_argument("--seqs-per-request", type=int, default=4, help="Sequences per forward_backward request")
    parser.add_argument("--scan-iterations", type=int, default=20, help="Engine scheduling iterations to time")
    parser.add_argument(
        "--barrier-fraction",
        type=float,
        default=0.5,
        help="Fraction of models given a pending optim_step barrier before the scan phase",
    )
    parser.add_argument(
        "--blocked-requests",
        type=int,
        default=256,
        help="Extra forward_backward requests queued behind the barriers (parked, re-seen every poll)",
    )
    parser.add_argument("--await-waiters", type=int, default=128, help="Concurrent retrieve_future waiters")
    parser.add_argument("--await-hold-s", type=float, default=3.0, help="Seconds to hold waiters on pending futures")
    parser.add_argument(
        "--poller",
        choices=["legacy", "shared"],
        default="legacy",
        help="Waiting strategy: per-request loop (legacy) or shared batched poller",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))

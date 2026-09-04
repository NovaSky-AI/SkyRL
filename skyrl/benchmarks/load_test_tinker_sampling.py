#!/usr/bin/env python3
"""Load test: is the Tinker API server the bottleneck at N concurrent sample requests?

Structure follows Chuck Tang's SQLite QueuePool repro
(https://gist.github.com/j316chuck/f44f35572ffb8584519d13b943f99ef8): a barrier
fake vLLM, the real API server, and a Tinker-SDK-shaped client, extended with
orchestration, server profiling/monitoring, payload sizing and a raw client that
can exceed the SDK's in-flight cap.

Exercises the API server layers on the non-colocated SkyRL-Train path
(``backend=megatron``, ``trainer.placement.colocate_all=false``): FastAPI +
uvicorn, the SQLite-backed session/model tables, the in-memory
``ExternalFutureStore``, and ``SkyRLTrainInferenceForwardingClient`` which
forwards each sample to the engine-managed vLLM router. Everything *below* the
server is stubbed so no GPU is needed and the numbers isolate the server:

  fake router  ``--role vllm``  serves ``/v1/completions`` in place of the
               vllm-router + vLLM. ``--vllm-mode barrier`` holds requests until
               ``--barrier-requests`` are in flight and releases them at once
               (worst-case completion burst); ``--vllm-mode latency`` admits
               ``--max-num-seqs`` at a time for ``--gen-seconds`` each (a real
               engine's queueing).
  API server   ``--role server``  the real ``skyrl.tinker.api`` app under
               uvicorn, configured exactly as the non-colocated megatron server
               is, with the engine subprocess replaced by a sleeper and the
               router URL the engine would publish to ``EngineStateDB`` seeded
               to the fake router. Pass ``--server-url`` to test a real server
               instead (engine + vLLM must be up).
  load client  ``--role load``. ``--client raw`` speaks the HTTP API directly from
               ``--workers`` processes, each bound to its own loopback source IP so
               the client is not capped by one IP's ~28k ephemeral ports. ``--client
               sdk`` uses the public Tinker SDK (``sample_async``); the SDK caps
               in-flight samples per SamplingClient at the server-advertised
               ``sample_max_concurrent_requests`` (default 2000).

Usage (everything in one command, CPU only):

  uv run --extra tinker python skyrl/benchmarks/load_test_tinker_sampling.py \\
      --num-requests 131072 --forwarding-max-connections 2048

  # Realistic engine queueing: 2048 concurrent generations of 5s each
  uv run --extra tinker python skyrl/benchmarks/load_test_tinker_sampling.py \\
      --num-requests 131072 --forwarding-max-connections 2048 \\
      --vllm-mode latency --max-num-seqs 2048 --gen-seconds 5

  # Against a server you started yourself:
  uv run --extra tinker python skyrl/benchmarks/load_test_tinker_sampling.py \\
      --role load --url http://127.0.0.1:8000 --num-requests 4096

The summary reports completed/failed counts by error class, submit and
end-to-end latency percentiles, the peak number of requests the fake router saw
in flight at once (the concurrency the server actually sustained), and the API
server's peak RSS / file descriptors and ``/healthz`` latency under load.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import aiohttp
import psutil

TINY_MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"
DEFAULT_API_PORT = 18779
DEFAULT_VLLM_PORT = 18879
# Per-IP ephemeral port budget with the Linux default range (32768-60999).
DEFAULT_REQUESTS_PER_WORKER = 16384
# The Tinker SDK gives up on each retrieve_future poll after 45s and re-polls
# the same request_id (the server would hold it for up to 300s).
SDK_RETRIEVE_POLL_TIMEOUT_SECONDS = 45.0
# The SDK gives up on a request after this many consecutive connection errors.
SDK_MAX_CONNECTION_ERROR_RETRIES = 16


# --------------------------------------------------------------------------- #
# Fake vLLM
# --------------------------------------------------------------------------- #


class _SharedCounters:
    """Request counters shared by the fake router's worker processes."""

    def __init__(self) -> None:
        import multiprocessing

        self._lock = multiprocessing.Lock()
        self._values = {
            k: multiprocessing.Value("q", 0, lock=False) for k in ("received", "completed", "in_flight", "peak")
        }

    def enter(self) -> None:
        with self._lock:
            self._values["received"].value += 1
            self._values["in_flight"].value += 1
            self._values["peak"].value = max(self._values["peak"].value, self._values["in_flight"].value)

    def exit(self, completed: bool) -> None:
        with self._lock:
            self._values["in_flight"].value -= 1
            if completed:
                self._values["completed"].value += 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            v = self._values
            return {
                "received": v["received"].value,
                "completed": v["completed"].value,
                "in_flight": v["in_flight"].value,
                "peak_in_flight": v["peak"].value,
            }


def run_fake_vllm(args: argparse.Namespace) -> None:
    """Serve the fake router; with --router-workers > 1, fork workers sharing the port (SO_REUSEPORT)."""
    import multiprocessing

    counters = _SharedCounters()
    workers = max(1, args.router_workers)
    if workers == 1:
        _serve_fake_vllm(args, counters, worker_index=0, workers=1)
        return
    procs = [
        multiprocessing.Process(target=_serve_fake_vllm, args=(args, counters, i, workers), daemon=True)
        for i in range(workers)
    ]
    for proc in procs:
        proc.start()

    def terminate_workers(signum, frame):
        # SIGTERM from the orchestrator would otherwise leave the workers
        # orphaned (daemon cleanup only runs on a normal exit).
        for proc in procs:
            proc.terminate()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate_workers)
    for proc in procs:
        proc.join()


def _serve_fake_vllm(args: argparse.Namespace, counters: _SharedCounters, worker_index: int, workers: int) -> None:
    from aiohttp import web

    barrier_requests = args.barrier_requests or args.num_requests
    release = asyncio.Event()
    # Each worker admits its share of the engine's concurrent sequences.
    admit = asyncio.Semaphore(max(1, args.max_num_seqs // workers))
    body_cache: dict[tuple[int, int], bytes] = {}

    def body_for(n: int, max_tokens: int) -> bytes:
        key = (n, max_tokens)
        if key not in body_cache:
            choice = {
                "token_ids": [1] * max_tokens,
                "logprobs": {"token_logprobs": [-0.5] * max_tokens},
                "finish_reason": "length",
            }
            body_cache[key] = json.dumps({"choices": [choice] * n}, separators=(",", ":")).encode()
        return body_cache[key]

    async def barrier_timeout_watch() -> None:
        await asyncio.sleep(args.barrier_timeout)
        if not release.is_set():
            print(
                f"[fake-vllm] barrier timeout after {args.barrier_timeout}s with "
                f"{counters.snapshot()['received']}/{barrier_requests} received -- releasing",
                flush=True,
            )
            release.set()

    async def completions(request: web.Request) -> web.Response:
        payload = await request.json()
        counters.enter()
        completed = False
        try:
            if args.vllm_mode == "barrier":
                # Barrier mode is single-process: the release is local state.
                if counters.snapshot()["received"] >= barrier_requests:
                    release.set()
                await release.wait()
            else:
                async with admit:
                    await asyncio.sleep(args.gen_seconds)
            completed = True
            return web.Response(
                body=body_for(payload.get("n", 1), payload["max_tokens"]), content_type="application/json"
            )
        finally:
            counters.exit(completed)

    async def stats(_: web.Request) -> web.Response:
        return web.json_response(counters.snapshot())

    async def serve() -> None:
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app.router.add_post("/v1/completions", completions)
        app.router.add_get("/healthz", stats)
        app.router.add_get("/stats", stats)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", args.vllm_port, backlog=65535, reuse_port=workers > 1)
        await site.start()
        if args.vllm_mode == "barrier":
            asyncio.create_task(barrier_timeout_watch())
        if worker_index == 0:
            print(
                f"[fake-vllm] mode={args.vllm_mode} workers={workers} listening on 127.0.0.1:{args.vllm_port}",
                flush=True,
            )
        await asyncio.Event().wait()

    asyncio.run(serve())


# --------------------------------------------------------------------------- #
# Raw HTTP load client (one worker process)
# --------------------------------------------------------------------------- #


def worker_source_ip(worker_index: int) -> str:
    # 127.0.0.0/8 is entirely bound to lo on Linux, so any 127.x.y.z is a free
    # source address with its own ephemeral-port budget.
    return f"127.1.{worker_index // 250}.{worker_index % 250 + 1}"


async def raw_worker(args: argparse.Namespace, worker_index: int, count: int) -> dict[str, Any]:
    base = args.url.rstrip("/") + "/api/v1"
    payload = {
        "num_samples": args.num_samples,
        "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1000] * args.prompt_tokens}]},
        "sampling_params": {"max_tokens": args.max_tokens, "temperature": 1.0, "seed": 0},
        "base_model": args.base_model,
    }
    retrieve_headers = {"Accept": "application/x-protobuf, application/json"} if args.proto else {}

    submit_latency: list[float] = []
    e2e_latency: list[float] = []
    errors: Counter[str] = Counter()
    error_samples: dict[str, str] = {}
    retries_408 = 0
    reconnects = 0
    asample_retries = 0
    poll_timeouts = 0
    completed = 0

    def record_error(kind: str, detail: str) -> None:
        errors[kind] += 1
        error_samples.setdefault(kind, detail[:300])

    local_addr = (worker_source_ip(worker_index), 0) if args.source_ips else None
    connector = aiohttp.TCPConnector(limit=0, local_addr=local_addr, force_close=False)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=60, sock_read=args.poll_timeout)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:

        # The Tinker SDK holds at most 400 asample POSTs in flight per client
        # (its sample dispatch semaphore); results are then awaited with one
        # long-poll each. Mirror that so the submit burst matches production.
        submit_gate = asyncio.Semaphore(max(1, args.submit_concurrency))
        # The SDK keeps at most sample_max_concurrent_requests samples in flight
        # per SamplingClient (submit through result) and queues the rest.
        outstanding_gate = asyncio.Semaphore(max(1, args.max_outstanding)) if args.max_outstanding else None

        async def one(i: int) -> None:
            if outstanding_gate is None:
                await _one(i)
            else:
                async with outstanding_gate:
                    await _one(i)

        async def _one(i: int) -> None:
            nonlocal retries_408, reconnects, asample_retries, poll_timeouts, completed
            t0 = time.monotonic()
            phase = "asample"
            try:
                async with submit_gate:
                    connection_failures = 0
                    while True:
                        try:
                            async with session.post(f"{base}/asample", json=payload) as resp:
                                if resp.status != 200:
                                    record_error(f"asample_http_{resp.status}", await resp.text())
                                    return
                                request_id = (await resp.json())["request_id"]
                            break
                        except (
                            aiohttp.ServerDisconnectedError,
                            aiohttp.ClientOSError,
                            aiohttp.ClientConnectorError,
                            asyncio.TimeoutError,
                        ) as e:
                            # The SDK re-sends an asample whose connection failed
                            # or timed out (the server may or may not have
                            # accepted the first copy), up to 16 times with
                            # exponential backoff.
                            connection_failures += 1
                            if connection_failures > SDK_MAX_CONNECTION_ERROR_RETRIES:
                                record_error("asample_connection_retries_exhausted", str(e))
                                return
                            asample_retries += 1
                            await asyncio.sleep(min(2 ** (connection_failures - 1), 30))
                submit_latency.append(time.monotonic() - t0)
                phase = "retrieve"
                retrieve_failures = 0
                while True:
                    if time.monotonic() - t0 > args.request_timeout:
                        record_error("request_timeout", f"no result after {args.request_timeout}s")
                        return
                    try:
                        async with session.post(
                            f"{base}/retrieve_future", json={"request_id": request_id}, headers=retrieve_headers
                        ) as resp:
                            if resp.status == 408:
                                retries_408 += 1
                                continue
                            if resp.status != 200:
                                record_error(f"retrieve_http_{resp.status}", await resp.text())
                                return
                            await resp.read()
                            break
                    except asyncio.TimeoutError:
                        # Client-side poll timeout: the SDK abandons the poll
                        # and immediately re-polls the same request_id.
                        poll_timeouts += 1
                        continue
                    except (
                        aiohttp.ServerDisconnectedError,
                        aiohttp.ClientOSError,
                        aiohttp.ClientConnectorError,
                    ) as e:
                        # The server dropped or refused the connection (e.g.
                        # uvicorn's keep-alive timer firing during an event-loop
                        # stall, or a full accept backlog). retrieve_future is
                        # idempotent; the SDK retries with backoff, so do the same.
                        reconnects += 1
                        retrieve_failures += 1
                        if retrieve_failures > SDK_MAX_CONNECTION_ERROR_RETRIES:
                            record_error("retrieve_connection_retries_exhausted", str(e))
                            return
                        await asyncio.sleep(min(2 ** (retrieve_failures - 1), 30))
                e2e_latency.append(time.monotonic() - t0)
                completed += 1
            except aiohttp.ClientConnectorError as e:
                record_error(f"{phase}_connect_error", str(e))
            except (aiohttp.ServerDisconnectedError, aiohttp.ClientOSError, aiohttp.ClientPayloadError) as e:
                record_error(f"{phase}_{type(e).__name__}", str(e))
            except asyncio.TimeoutError:
                record_error(f"{phase}_client_timeout", "")
            except Exception as e:  # noqa: BLE001 - a load test wants every failure class counted
                record_error(f"{phase}_{type(e).__name__}", str(e))

        started = time.monotonic()
        tasks = []
        interval = args.ramp_seconds / count if args.ramp_seconds > 0 and count else 0.0
        for i in range(count):
            tasks.append(asyncio.ensure_future(one(i)))
            if interval:
                await asyncio.sleep(interval)
        await asyncio.gather(*tasks)
        wall = time.monotonic() - started

    return {
        "worker": worker_index,
        "requested": count,
        "completed": completed,
        "retries_408": retries_408,
        "reconnects": reconnects,
        "asample_retries": asample_retries,
        "poll_timeouts": poll_timeouts,
        "errors": dict(errors),
        "error_samples": error_samples,
        "wall_seconds": wall,
        "submit_latency": submit_latency,
        "e2e_latency": e2e_latency,
    }


def run_raw_worker(args: argparse.Namespace, worker_index: int, count: int, out_path: Path) -> None:
    result = asyncio.run(raw_worker(args, worker_index, count))
    out_path.write_text(json.dumps(result))


# --------------------------------------------------------------------------- #
# Tinker SDK load client
# --------------------------------------------------------------------------- #


async def sdk_load(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("TINKER_API_KEY", "tml-dummy")
    import tinker
    from tinker import types as ttypes

    service_client = tinker.ServiceClient(base_url=args.url, timeout=SDK_RETRIEVE_POLL_TIMEOUT_SECONDS)
    sampling_client = await service_client.create_sampling_client_async(base_model=args.base_model)
    prompt = ttypes.ModelInput.from_ints([1000] * args.prompt_tokens)
    params = ttypes.SamplingParams(max_tokens=args.max_tokens, temperature=1.0, seed=0)

    e2e_latency: list[float] = []
    errors: Counter[str] = Counter()
    error_samples: dict[str, str] = {}

    async def one() -> None:
        t0 = time.monotonic()
        try:
            await sampling_client.sample_async(prompt=prompt, num_samples=args.num_samples, sampling_params=params)
            e2e_latency.append(time.monotonic() - t0)
        except Exception as e:  # noqa: BLE001
            kind = type(e).__name__
            errors[kind] += 1
            error_samples.setdefault(kind, str(e)[:300])

    started = time.monotonic()
    await asyncio.gather(*(one() for _ in range(args.num_requests)))
    return {
        "worker": 0,
        "requested": args.num_requests,
        "completed": len(e2e_latency),
        "retries_408": 0,
        "reconnects": 0,
        "asample_retries": 0,
        "poll_timeouts": 0,
        "errors": dict(errors),
        "error_samples": error_samples,
        "wall_seconds": time.monotonic() - started,
        "submit_latency": [],
        "e2e_latency": e2e_latency,
    }


# --------------------------------------------------------------------------- #
# Load role: fan out to worker processes and merge
# --------------------------------------------------------------------------- #


def percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def pct(p: float) -> float:
        return ordered[min(len(ordered) - 1, int(math.ceil(p / 100 * len(ordered))) - 1)]

    return {
        "p50": round(statistics.median(ordered), 3),
        "p90": round(pct(90), 3),
        "p99": round(pct(99), 3),
        "max": round(ordered[-1], 3),
    }


def merge_worker_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    errors: Counter[str] = Counter()
    error_samples: dict[str, str] = {}
    submit: list[float] = []
    e2e: list[float] = []
    for r in results:
        errors.update(r["errors"])
        for kind, sample in r["error_samples"].items():
            error_samples.setdefault(kind, sample)
        submit.extend(r["submit_latency"])
        e2e.extend(r["e2e_latency"])
    completed = sum(r["completed"] for r in results)
    wall = max(r["wall_seconds"] for r in results)
    return {
        "requested": sum(r["requested"] for r in results),
        "completed": completed,
        "failed": sum(errors.values()),
        "retries_408": sum(r["retries_408"] for r in results),
        "reconnects": sum(r.get("reconnects", 0) for r in results),
        "asample_retries": sum(r.get("asample_retries", 0) for r in results),
        "poll_timeouts": sum(r.get("poll_timeouts", 0) for r in results),
        "errors": dict(errors.most_common()),
        "error_samples": error_samples,
        "wall_seconds": round(wall, 2),
        "throughput_per_s": round(completed / wall, 1) if wall else None,
        "submit_latency_s": percentiles(submit),
        "e2e_latency_s": percentiles(e2e),
    }


def run_load(args: argparse.Namespace) -> dict[str, Any]:
    if args.client == "sdk":
        return merge_worker_results([asyncio.run(sdk_load(args))])

    workers = args.workers or max(1, math.ceil(args.num_requests / DEFAULT_REQUESTS_PER_WORKER))
    per_worker = [args.num_requests // workers + (1 if i < args.num_requests % workers else 0) for i in range(workers)]
    with tempfile.TemporaryDirectory(prefix="tinker_load_") as tmp:
        procs = []
        for i, count in enumerate(per_worker):
            out = Path(tmp) / f"worker_{i}.json"
            cmd = [
                sys.executable,
                __file__,
                "--role",
                "raw-worker",
                "--url",
                args.url,
                "--base-model",
                args.base_model,
                "--num-requests",
                str(count),
                "--max-tokens",
                str(args.max_tokens),
                "--prompt-tokens",
                str(args.prompt_tokens),
                "--num-samples",
                str(args.num_samples),
                "--ramp-seconds",
                str(args.ramp_seconds),
                "--request-timeout",
                str(args.request_timeout),
                "--max-reconnects",
                str(args.max_reconnects),
                "--max-outstanding",
                str(args.max_outstanding // workers if args.max_outstanding else 0),
                "--poll-timeout",
                str(args.poll_timeout),
                "--submit-concurrency",
                str(max(1, args.submit_concurrency // workers)),
                "--worker-index",
                str(i),
                "--worker-out",
                str(out),
            ]
            if args.proto:
                cmd.append("--proto")
            if not args.source_ips:
                cmd.append("--no-source-ips")
            procs.append((subprocess.Popen(cmd), out))
        results = []
        for proc, out in procs:
            proc.wait()
            if proc.returncode != 0 or not out.exists():
                raise RuntimeError(f"load worker {out.stem} exited with {proc.returncode}")
            results.append(json.loads(out.read_text()))
    merged = merge_worker_results(results)
    merged["workers"] = workers
    return merged


# --------------------------------------------------------------------------- #
# API server role: the real app, configured like the non-colocated megatron server
# --------------------------------------------------------------------------- #

NON_COLOCATED_MEGATRON_BACKEND_CONFIG = {
    "strategy": "megatron",
    "trainer.placement.policy_num_gpus_per_node": 2,
    "trainer.placement.policy_num_nodes": 1,
    "trainer.placement.colocate_all": False,
    "trainer.policy.megatron_config.tensor_model_parallel_size": 1,
    "trainer.policy.megatron_config.pipeline_model_parallel_size": 1,
    "trainer.policy.megatron_config.lora_config.merge_lora": False,
    "trainer.policy.model.lora.max_loras": 4,
    "trainer.policy.model.lora.max_cpu_loras": 4,
}


def run_server(args: argparse.Namespace) -> None:
    """Serve ``skyrl.tinker.api`` with the engine stubbed out and the router URL pre-published.

    The lifespan runs unmodified: with ``backend=megatron`` and ``colocate_all``
    false it installs ``SkyRLTrainInferenceForwardingClient``, which resolves
    the vLLM router URL from ``EngineStateDB``. The engine (which would need
    GPUs and a real vLLM) is the only thing replaced -- by a sleeping process --
    since it is not on the async sample path being measured.
    """
    import uvicorn
    from sqlmodel import Session, SQLModel, create_engine

    from skyrl.tinker import api as tinker_api
    from skyrl.tinker.config import EngineConfig
    from skyrl.tinker.db_models import EngineStateDB
    from skyrl.utils.log import get_uvicorn_log_config

    workdir = Path(args.server_workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    database_url = f"sqlite:///{workdir / 'tinker.db'}"
    sync_engine = create_engine(database_url)
    SQLModel.metadata.create_all(sync_engine)
    with Session(sync_engine) as session:
        session.merge(EngineStateDB(singleton_id=1, inference_proxy_url=args.vllm_url))
        session.commit()
    sync_engine.dispose()

    tinker_api._build_uv_run_cmd_engine = lambda parent_cmd, engine_config: [
        sys.executable,
        "-c",
        "import time; time.sleep(10**9)",
    ]
    # Older commits lack some of these knobs; pass only the fields this
    # EngineConfig knows so the same harness can baseline them.
    optional_fields = {
        "forwarding_inference_max_connections": args.forwarding_max_connections,
        "forwarding_inference_timeout_sec": args.forwarding_timeout,
        "external_future_retrieved_ttl_sec": args.retrieved_ttl,
    }
    tinker_api.app.state.engine_config = EngineConfig(
        base_model=args.base_model,
        backend="megatron",
        backend_config=NON_COLOCATED_MEGATRON_BACKEND_CONFIG,
        database_url=database_url,
        checkpoints_base=str(workdir / "checkpoints"),
        **{k: v for k, v in optional_fields.items() if v is not None and k in EngineConfig.model_fields},
    )
    profile_path = os.environ.get("TINKER_LOADTEST_PROFILE")
    if profile_path:
        # Profile the whole server process; the orchestrator sends SIGUSR1
        # after the load finishes and the stats are dumped from the handler.
        import cProfile

        profiler = cProfile.Profile()

        def dump_profile(signum, frame):
            profiler.disable()
            profiler.dump_stats(profile_path)

        signal.signal(signal.SIGUSR1, dump_profile)
        profiler.enable()
    uvicorn.run(
        tinker_api.app,
        host="127.0.0.1",
        port=args.api_port,
        log_config=get_uvicorn_log_config(),
        backlog=getattr(tinker_api, "SKYRL_HTTP_CONNECTION_LIMIT", 2048),
        timeout_keep_alive=getattr(tinker_api, "HTTP_KEEP_ALIVE_TIMEOUT_SECONDS", 5),
    )


# --------------------------------------------------------------------------- #
# Orchestrator: fake vLLM + real API server + monitor + load
# --------------------------------------------------------------------------- #


async def http_get_json(session: aiohttp.ClientSession, url: str) -> dict | None:
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            return await resp.json()
    except Exception:  # noqa: BLE001
        return None


def wait_for_http(url: str, timeout: float) -> None:
    async def poll() -> None:
        deadline = time.monotonic() + timeout
        async with aiohttp.ClientSession() as session:
            while time.monotonic() < deadline:
                if await http_get_json(session, url) is not None:
                    return
                await asyncio.sleep(0.5)
        raise TimeoutError(f"{url} did not come up within {timeout}s")

    asyncio.run(poll())


class ServerMonitor:
    """Samples the API server's RSS / fds / CPU and /healthz latency while load runs."""

    def __init__(self, pid: int | None, health_url: str):
        self._proc = subprocess.Popen(
            [sys.executable, __file__, "--role", "monitor", "--url", health_url, "--monitor-pid", str(pid or 0)],
            stdout=subprocess.PIPE,
            text=True,
        )

    def stop(self) -> dict[str, Any]:
        self._proc.send_signal(signal.SIGINT)
        out, _ = self._proc.communicate(timeout=30)
        return json.loads(out.strip().splitlines()[-1]) if out.strip() else {}


def run_monitor(args: argparse.Namespace) -> None:
    """Subprocess body for ServerMonitor: prints one JSON summary on SIGINT."""
    proc = psutil.Process(args.monitor_pid) if args.monitor_pid else None
    peak_rss = peak_fds = 0
    cpu: list[float] = []
    health: list[float] = []
    failures = 0

    async def loop() -> None:
        nonlocal peak_rss, peak_fds, failures
        async with aiohttp.ClientSession() as session:
            while True:
                if proc is not None:
                    try:
                        # The API process only; the engine child is idle on this path.
                        peak_rss = max(peak_rss, proc.memory_info().rss)
                        peak_fds = max(peak_fds, proc.num_fds())
                        cpu.append(proc.cpu_percent(interval=None))
                    except psutil.Error:
                        pass
                t0 = time.monotonic()
                if await http_get_json(session, args.url) is None:
                    failures += 1
                else:
                    health.append(time.monotonic() - t0)
                await asyncio.sleep(1.0)

    try:
        asyncio.run(loop())
    except KeyboardInterrupt:
        pass
    print(
        json.dumps(
            {
                "peak_rss_gb": round(peak_rss / 1e9, 2),
                "peak_fds": peak_fds,
                "cpu_percent_p50": round(statistics.median(cpu), 1) if cpu else None,
                "cpu_percent_max": round(max(cpu), 1) if cpu else None,
                "healthz_latency_s": percentiles(health),
                "healthz_failures": failures,
            }
        ),
        flush=True,
    )


def start_api_server(args: argparse.Namespace, workdir: Path, vllm_url: str) -> tuple[subprocess.Popen, Path]:
    log_path = workdir / "server.log"
    cmd = [
        sys.executable,
        __file__,
        "--role",
        "server",
        "--api-port",
        str(args.api_port),
        "--base-model",
        args.base_model,
        "--vllm-url",
        vllm_url,
        "--server-workdir",
        str(workdir),
        "--forwarding-timeout",
        str(args.forwarding_timeout),
    ]
    if args.forwarding_max_connections is not None:
        cmd += ["--forwarding-max-connections", str(args.forwarding_max_connections)]
    if args.retrieved_ttl is not None:
        cmd += ["--retrieved-ttl", str(args.retrieved_ttl)]
    print(f"[orchestrator] starting API server: {' '.join(cmd)}")
    print(f"[orchestrator] server log: {log_path}")
    proc = subprocess.Popen(cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT, start_new_session=True)
    return proc, log_path


def print_sysctl_hints(args: argparse.Namespace) -> None:
    try:
        lo, hi = map(int, Path("/proc/sys/net/ipv4/ip_local_port_range").read_text().split())
        somaxconn = int(Path("/proc/sys/net/core/somaxconn").read_text())
    except OSError:
        return
    per_ip = hi - lo + 1
    workers = args.workers or max(1, math.ceil(args.num_requests / DEFAULT_REQUESTS_PER_WORKER))
    print(
        f"[orchestrator] ephemeral ports per source IP: {per_ip}, somaxconn: {somaxconn}, nofile: {os.sysconf('SC_OPEN_MAX')}"
    )
    if args.client == "raw" and not args.source_ips and args.num_requests > per_ip * 0.9:
        print("[orchestrator] WARNING: a single source IP cannot hold this many connections; drop --no-source-ips")
    if args.client == "raw" and args.source_ips and args.num_requests / workers > per_ip * 0.9:
        print(
            f"[orchestrator] WARNING: {args.num_requests / workers:.0f} connections per worker exceeds one IP's ports; raise --workers"
        )
    if args.num_requests > per_ip * 0.9:
        print(
            "[orchestrator] NOTE: the API server forwards every in-flight sample over its own outbound connection "
            f"from one IP, so more than ~{per_ip} simultaneously forwarded requests will fail with EADDRNOTAVAIL"
        )


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    print_sysctl_hints(args)
    workdir = Path(tempfile.mkdtemp(prefix="tinker_sampling_load_"))
    vllm_url = f"http://127.0.0.1:{args.vllm_port}"
    api_url = args.server_url or f"http://127.0.0.1:{args.api_port}"
    children: list[subprocess.Popen] = []
    server_proc: subprocess.Popen | None = None
    log_path: Path | None = None
    try:
        vllm_cmd = [
            sys.executable,
            __file__,
            "--role",
            "vllm",
            "--vllm-port",
            str(args.vllm_port),
            "--vllm-mode",
            args.vllm_mode,
            "--barrier-requests",
            str(args.barrier_requests or args.num_requests),
            "--barrier-timeout",
            str(args.barrier_timeout),
            "--max-num-seqs",
            str(args.max_num_seqs),
            "--gen-seconds",
            str(args.gen_seconds),
            "--router-workers",
            str(args.router_workers),
        ]
        children.append(subprocess.Popen(vllm_cmd))
        wait_for_http(f"{vllm_url}/healthz", 30)

        if args.server_url is None:
            server_proc, log_path = start_api_server(args, workdir, vllm_url)
            children.append(server_proc)
        wait_for_http(f"{api_url}/api/v1/healthz", args.server_startup_timeout)
        api_pid = server_proc.pid if server_proc is not None else None
        print(
            f"[orchestrator] API server up (pid {api_pid}); starting load: {args.num_requests} requests via {args.client}"
        )

        monitor = ServerMonitor(api_pid, f"{api_url}/api/v1/healthz")
        args.url = api_url
        load = run_load(args)
        server_stats = monitor.stop()
        if server_proc is not None and os.environ.get("TINKER_LOADTEST_PROFILE"):
            server_proc.send_signal(signal.SIGUSR1)
            time.sleep(3)

        async def vllm_stats() -> dict | None:
            async with aiohttp.ClientSession() as session:
                return await http_get_json(session, f"{vllm_url}/stats")

        report = {
            "config": {
                "num_requests": args.num_requests,
                "client": args.client,
                "vllm_mode": args.vllm_mode,
                "max_tokens": args.max_tokens,
                "prompt_tokens": args.prompt_tokens,
                "proto": args.proto,
                "forwarding_max_connections": args.forwarding_max_connections,
                "forwarding_timeout": args.forwarding_timeout,
            },
            "load": load,
            "fake_vllm": asyncio.run(vllm_stats()),
            "api_server": server_stats,
            "server_log": str(log_path) if log_path else None,
        }
        return report
    finally:
        for proc in reversed(children):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM) if proc is server_proc else proc.terminate()
                proc.wait(timeout=20)
            except Exception:  # noqa: BLE001
                proc.kill()


def print_report(report: dict[str, Any]) -> None:
    load = report["load"]
    print("\n=== Tinker sampling load test ===")
    print(f"config: {json.dumps(report['config'])}")
    print(
        f"requested={load['requested']} completed={load['completed']} failed={load['failed']} "
        f"408_retries={load['retries_408']} reconnects={load.get('reconnects', 0)} "
        f"asample_retries={load.get('asample_retries', 0)} poll_timeouts={load.get('poll_timeouts', 0)}"
    )
    print(f"wall={load['wall_seconds']}s throughput={load['throughput_per_s']} req/s")
    print(f"submit latency (s): {load['submit_latency_s']}")
    print(f"e2e latency (s):    {load['e2e_latency_s']}")
    if load["errors"]:
        print("errors by class:")
        for kind, n in load["errors"].items():
            print(f"  {kind}: {n}   e.g. {load['error_samples'].get(kind, '')!r}")
    if report.get("fake_vllm"):
        print(
            f"fake vLLM: {json.dumps(report['fake_vllm'])}  <- peak_in_flight is the concurrency the server sustained"
        )
    if report.get("api_server"):
        print(f"API server: {json.dumps(report['api_server'])}")
    if report.get("server_log"):
        print(f"server log: {report['server_log']}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--role", choices=["all", "load", "vllm", "server", "raw-worker", "monitor"], default="all")
    p.add_argument("--num-requests", type=int, default=131072)
    p.add_argument("--client", choices=["raw", "sdk"], default="raw")
    p.add_argument("--workers", type=int, default=0, help="raw client processes (default: ceil(n / 16384))")
    p.add_argument("--no-source-ips", dest="source_ips", action="store_false", help="bind all workers to 127.0.0.1")
    p.add_argument("--ramp-seconds", type=float, default=0.0, help="spread request launch over this many seconds")
    p.add_argument("--request-timeout", type=float, default=900.0, help="give up on a request after this many seconds")
    p.add_argument(
        "--poll-timeout",
        type=float,
        default=SDK_RETRIEVE_POLL_TIMEOUT_SECONDS,
        help="client-side timeout per retrieve_future poll before re-polling (Tinker SDK: 45s)",
    )
    p.add_argument(
        "--submit-concurrency",
        type=int,
        default=400,
        help="max asample POSTs in flight across all workers (Tinker SDK default: 400 per client)",
    )
    p.add_argument(
        "--max-outstanding",
        type=int,
        default=0,
        help="cap on samples in flight across all workers, like the SDK's sample_max_concurrent_requests (0 = unlimited)",
    )
    p.add_argument(
        "--max-reconnects",
        type=float,
        default=2.0,
        help="retrieve_future reconnect budget per worker, as a multiple of its request count (SDK-style retries)",
    )
    p.add_argument("--proto", action="store_true", help="retrieve results as protobuf (Accept: application/x-protobuf)")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--prompt-tokens", type=int, default=64)
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--base-model", default=TINY_MODEL)
    p.add_argument("--url", default=f"http://127.0.0.1:{DEFAULT_API_PORT}", help="API server URL for --role load")
    p.add_argument(
        "--server-url", default=None, help="--role all: use this running API server instead of launching one"
    )
    p.add_argument(
        "--forwarding-max-connections",
        type=int,
        default=None,
        help="EngineConfig.forwarding_inference_max_connections for the launched server (default: unlimited)",
    )
    p.add_argument(
        "--forwarding-timeout", type=float, default=300.0, help="EngineConfig.forwarding_inference_timeout_sec"
    )
    p.add_argument(
        "--retrieved-ttl",
        type=float,
        default=None,
        help="EngineConfig.external_future_retrieved_ttl_sec for the launched server (memory = rate x size x ttl)",
    )
    p.add_argument("--server-startup-timeout", type=float, default=120.0)
    p.add_argument("--api-port", type=int, default=DEFAULT_API_PORT)
    p.add_argument("--vllm-port", type=int, default=DEFAULT_VLLM_PORT)
    p.add_argument("--vllm-mode", choices=["barrier", "latency"], default="barrier")
    p.add_argument("--barrier-requests", type=int, default=0, help="barrier size (default: --num-requests)")
    p.add_argument("--barrier-timeout", type=float, default=600.0, help="release the barrier anyway after this long")
    p.add_argument("--max-num-seqs", type=int, default=1024, help="latency mode: concurrent generations")
    p.add_argument("--gen-seconds", type=float, default=2.0, help="latency mode: seconds per generation")
    p.add_argument(
        "--router-workers",
        type=int,
        default=1,
        help="fake router processes sharing the port (latency mode only); raise when large results saturate one process",
    )
    p.add_argument("--json-out", type=Path, default=None)
    # internal
    p.add_argument("--vllm-url", default=f"http://127.0.0.1:{DEFAULT_VLLM_PORT}")
    p.add_argument("--server-workdir", default=None)
    p.add_argument("--worker-index", type=int, default=0)
    p.add_argument("--worker-out", type=Path, default=None)
    p.add_argument("--monitor-pid", type=int, default=0)
    return p.parse_args()


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    if args.role == "vllm":
        run_fake_vllm(args)
        return 0
    if args.role == "server":
        run_server(args)
        return 0
    if args.role == "raw-worker":
        run_raw_worker(args, args.worker_index, args.num_requests, args.worker_out)
        return 0
    if args.role == "monitor":
        run_monitor(args)
        return 0
    if args.role == "load":
        report = {"config": {"num_requests": args.num_requests, "client": args.client}, "load": run_load(args)}
    else:
        report = run_all(args)
    print_report(report)
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["load"]["failed"] == 0 and report["load"]["completed"] == report["load"]["requested"] else 1


if __name__ == "__main__":
    sys.exit(main())

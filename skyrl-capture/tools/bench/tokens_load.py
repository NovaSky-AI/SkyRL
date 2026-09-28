"""How many concurrent token-in/token-out conversations one replica takes.

The text benchmark measures independent requests. This measures the thing that
makes tokens mode different: every turn depends on the one before it, and the
replica holds the trajectory's rendered token prefix between them. So the
questions are how turn latency moves as a conversation grows, how it moves as
concurrent conversations grow, and what the replica is holding while it happens.

    python -m tools.bench.tokens_load \\
      --agents 32 --turns 12

Each agent runs one trajectory and feeds every reply back verbatim, because
anything else loses prefix reuse and would benchmark the fallback path. See
docs/design/tokens_log.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing
import statistics
import time
import traceback
import uuid
from typing import Any

import httpx

from tools.bench.harness import percentiles

REPLY_PROMPT = "Continue the migration, and explain what you changed and why."
# Measured at 4.98 characters per token on Qwen3, and close to that on any
# BPE trained on English.
FILLER = "word "
FILLER_CHARS_PER_TOKEN = 5


def _prompt_of(tokens: int, turn: int, session: str = "") -> str:
    """A user message of roughly `tokens` tokens, unique to its turn.

    Context length is the variable that matters most here and the one the
    default workload has least of: real agentic sessions run a median input
    around 88k tokens, where the engine's own tokenization is tens of
    milliseconds a turn. Padding has to be unique per turn, or the engine's
    prefix cache would absorb it and the arms would stop doing equal work.
    """
    if tokens <= 0:
        return f"{REPLY_PROMPT} (turn {turn}) [{session}]"
    # Natural-looking filler, because the ratio matters. A dense marker like
    # "t5word " tokenizes at 2.3 characters per token where ordinary prose runs
    # near 5, and the text arm is charged on an estimate while the token arm is
    # charged on real token IDs -- so a filler the estimator disagrees with
    # charges one arm ~70% more input than the other and looks like proxy
    # overhead. `MOCK_CHARS_PER_TOKEN` must match FILLER_CHARS_PER_TOKEN.
    body = (FILLER * (tokens * FILLER_CHARS_PER_TOKEN // len(FILLER) + 1))[
        : tokens * FILLER_CHARS_PER_TOKEN
    ]
    return f"{REPLY_PROMPT} (turn {turn}) [{session}] {body}"


async def _health(client: httpx.AsyncClient, endpoint: str) -> dict[str, Any]:
    try:
        response = await client.get(f"{endpoint}/healthz", timeout=5.0)
        return response.json()
    except Exception:
        return {}


def _tokens_stats(health: dict[str, Any]) -> dict[str, Any]:
    tokens = health.get("tokens") or {}
    return {
        "cached_traces": tokens.get("cached_traces"),
        "resident_tokens": tokens.get("resident_tokens"),
        "traces_evicted": tokens.get("traces_evicted"),
    }


async def _create(
    client: httpx.AsyncClient, *, endpoint: str, project: str
) -> tuple[str, str, str] | None:
    """Create one trajectory, retrying a refused connection.

    Returns ``(id, base_url)``. The base URL is taken from the
    response rather than rebuilt from the id: the data plane lives under a
    fixed prefix (``/route/``) that this generator has no business knowing,
    and guessing it is how every arm here once 404'd.

    The listen backlog is small (`kern.ipc.somaxconn` is 128 on macOS), so at
    high agent counts some of these connections are dropped by the kernel
    before the server ever accepts them. That is the load generator hitting an
    OS limit during setup, not the proxy refusing work, and setup is outside
    the measured window, so retrying it costs the benchmark nothing.
    """
    control = {"content-type": "application/json"}
    for attempt in range(6):
        try:
            created = await client.post(
                f"{endpoint}/v1/trajectories",
                json={"project": project},
                headers=control,
                timeout=30.0,
            )
        except httpx.TransportError:
            await asyncio.sleep(0.25 * (attempt + 1))
            continue
        if created.status_code != 201:
            return None
        body = created.json()
        return body["id"], body["base_url"]
    return None


def load_traces(path: str, *, limit: int, max_turns: int, max_input: int) -> list[list[tuple[int, int]]]:
    """AgentX traces as per-turn (input tokens, output tokens) plans.

    The published traces carry token counts and prefix block hashes, not text --
    they are built for KV-cache and scheduler studies, where what matters is how
    much of each prompt the engine has already seen. That is enough to replay
    the shape faithfully: `in` grows the way a conversation grows, the shared
    leading block hashes say how much of it is the previous turn's prefix, and
    `out` is what the model produced.

    What they cannot give is the text, so the replay synthesises it: each turn
    adds however many tokens it takes to reach that turn's `in`, and asks for
    exactly that turn's `out`. Input and output distributions, turn counts and
    context growth are the trace's; the words are not, which is the right way
    round, because the proxy's cost depends on token counts and prefix structure
    rather than on what the tokens say.

    `max_input` drops turns whose prompt exceeds the target's context window,
    and the trace with them -- truncating mid-conversation would silently change
    the distribution being replayed.
    """
    plans: list[list[tuple[int, int]]] = []
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            trace = json.loads(line)
            plan: list[tuple[int, int]] = []
            for request in trace["requests"][:max_turns]:
                if request["in"] > max_input:
                    plan = []
                    break
                plan.append((int(request["in"]), int(request["out"])))
            if len(plan) >= 2:
                plans.append(plan)
            if len(plans) >= limit:
                break
    return plans


async def _agent(
    client: httpx.AsyncClient,
    *,
    url: str,
    headers: dict[str, str],
    turns: int,
    max_tokens: int,
    prompt_tokens: int = 0,
    session: str = "",
    plan: list[tuple[int, int]] | None = None,
    samples: list[tuple[int, float]],
    failures: list[str],
    barrier: asyncio.Barrier,
) -> None:
    """`turns` dependent turns on an already-created trajectory.

    Trajectories are created before the clock starts and finished after it
    stops. Timing them alongside the turns measured setup as though it were
    throughput: at one agent only a fifth of the wall clock was inside a turn.
    """
    # A replayed plan sets each turn's prompt and completion size; otherwise
    # every turn is the same shape.
    steps: list[tuple[int, int]] = (
        plan[:turns] if plan else [(prompt_tokens, max_tokens)] * turns
    )
    first_in, _ = steps[0] if steps else (prompt_tokens, max_tokens)
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": _prompt_of(first_in, 0, session)}
    ]
    # What the conversation already holds, so a replayed turn adds only the
    # difference between its own `in` and what is already there.
    carried = first_in
    # Every agent starts pushing at the same moment, so the measured window is
    # the one where they are all in flight.
    await barrier.wait()
    try:
        for turn, (_want_in, want_out) in enumerate(steps):
            start = time.perf_counter()
            response = await client.post(
                url,
                json={"model": "bench", "messages": messages, "max_tokens": want_out},
                headers=headers,
                timeout=600.0,
            )
            elapsed = (time.perf_counter() - start) * 1000
            if response.status_code != 200:
                detail = response.json().get("error", {}) if response.text.startswith("{") else {}
                failures.append(f"turn {turn}: {response.status_code} {detail.get('code') or response.text[:60]}")
                return
            samples.append((turn, elapsed))
            reply = response.json()["choices"][0]["message"]
            usage = response.json().get("usage") or {}
            carried = int(usage.get("total_tokens") or 0) or carried + want_out
            messages = [
                *messages,
                {key: value for key, value in reply.items() if value is not None},
            ]
            if turn + 1 < len(steps):
                # Grow the prompt to the next turn's `in`, so the replayed
                # context follows the trace's own growth rather than a constant.
                next_in = steps[turn + 1][0]
                messages.append({
                    "role": "user",
                    "content": _prompt_of(max(16, next_in - carried), turn + 2, session),
                })
    except Exception as error:  # noqa: BLE001 - a failed agent must not stop the run
        failures.append(f"{type(error).__name__}: {error}")


async def _compare(args: argparse.Namespace, report: dict[str, Any]) -> dict[str, Any]:
    async with httpx.AsyncClient() as client:
        shapes = await _shapes(
            client,
            endpoint=args.endpoint,
            trajectory_ids=report["trajectory_ids"],
        )
    shapes = [turns for turns in shapes if turns]
    if not shapes:
        return {"error": "no exchanges were recorded, so there are no shapes to replay"}
    return await run_baseline(args, shapes)


async def run(args: argparse.Namespace) -> dict[str, Any]:
    samples: list[tuple[int, float]] = []
    failures: list[str] = []
    plans: list[list[tuple[int, int]]] = []
    if getattr(args, "traces", None):
        plans = load_traces(
            args.traces,
            limit=max(args.agents, 1),
            max_turns=args.turns,
            max_input=args.max_input_tokens,
        )
        if not plans:
            failures.append(f"no replayable traces in {args.traces}")
    limits = httpx.Limits(max_connections=args.agents * 2, max_keepalive_connections=args.agents * 2)
    control = {"content-type": "application/json"}
    async with httpx.AsyncClient(limits=limits) as client:
        # Warm the replica first. A tokenizer is loaded on its target's first
        # turn and takes seconds; measured, it lands in the tail and reads as
        # though the proxy stalled.
        warm = await _create(
            client,
            endpoint=args.endpoint,
            project=args.project,
        )
        if warm is not None:
            warm_id, warm_key, warm_url = warm
            await client.post(
                f"{warm_url}/chat/completions",
                json={"model": "bench", "messages": [{"role": "user", "content": "warm"}],
                      "max_tokens": 8},
                headers={"authorization": f"Bearer {warm_key}", "content-type": "application/json"},
                timeout=300.0,
            )
            await client.post(
                f"{args.endpoint}/v1/trajectories/{warm_id}/finish",
                json={}, headers=control, timeout=30.0,
            )

        # Setup, outside the measured window.
        # In chunks, for the same backlog reason `_create` retries for.
        created: list[tuple[str, str, str] | None] = []
        for offset in range(0, args.agents, 32):
            created.extend(await asyncio.gather(*[
                _create(
                    client,
                    endpoint=args.endpoint,
                    project=args.project,
                )
                for _ in range(min(32, args.agents - offset))
            ]))
        live = [item for item in created if item is not None]
        if len(live) != args.agents:
            failures.append(f"only {len(live)} of {args.agents} trajectories were created")

        before = _tokens_stats(await _health(client, args.endpoint))
        # With several load processes, every one of them waits here so the
        # measured window is the one where all agents everywhere are in flight.
        process_barrier = getattr(args, "process_barrier", None)
        if process_barrier is not None:
            try:
                await asyncio.to_thread(process_barrier.wait, 120)
            except Exception as error:  # noqa: BLE001 - a dead sibling must not hang us
                failures.append(f"process barrier: {type(error).__name__}: {error}")
        barrier = asyncio.Barrier(len(live))
        started = time.perf_counter()
        await asyncio.gather(*[
            _agent(
                client,
                url=f"{base_url}/chat/completions",
                headers={"content-type": "application/json"},
                turns=args.turns,
                max_tokens=args.max_tokens,
                prompt_tokens=args.prompt_tokens,
                session=trajectory_id,
                plan=plans[index % len(plans)] if plans else None,
                samples=samples,
                failures=failures,
                barrier=barrier,
            )
            for index, (trajectory_id, base_url) in enumerate(live)
        ])
        wall = time.perf_counter() - started
        after = _tokens_stats(await _health(client, args.endpoint))

        # Teardown, outside the measured window. Chunked and forgiving: at high
        # agent counts finishing every trajectory at once times out, and a
        # teardown that raises would throw away a measurement that succeeded.
        for offset in range(0, len(live), 32):
            outcomes = await asyncio.gather(*[
                client.post(
                    f"{args.endpoint}/v1/trajectories/{trajectory_id}/finish",
                    json={}, headers=control, timeout=120.0,
                )
                for trajectory_id, _url in live[offset:offset + 32]
            ], return_exceptions=True)
            for outcome in outcomes:
                if isinstance(outcome, BaseException):
                    failures.append(f"finish: {type(outcome).__name__}: {outcome}")

    latencies = [value for _turn, value in samples]
    by_turn: dict[int, list[float]] = {}
    for turn, value in samples:
        by_turn.setdefault(turn, []).append(value)

    return {
        "trajectory_ids": [trajectory_id for trajectory_id, _url in live],
        "agents": args.agents,
        "turns_per_agent": args.turns,
        "replayed_traces": len(plans),
        "turns_completed": len(samples),
        "wall_seconds": round(wall, 3),
        "turns_per_second": round(len(samples) / wall, 1) if wall else 0.0,
        "latency_ms": percentiles(latencies),
        "_latencies": latencies,
        "latency_by_turn_ms": {
            turn: round(statistics.median(values), 2) for turn, values in sorted(by_turn.items())
        },
        "replica_before": before,
        "replica_after": after,
        "failures": failures[:10],
        "failure_count": len(failures),
    }


async def run_text_direct(args: argparse.Namespace) -> dict[str, Any]:
    """The same conversations, straight at the engine's chat API.

    This is the baseline both proxy arms are measured against, and it is the
    only arm with nothing of ours on the path. It runs the identical agent loop
    against the identical workload, so the three arms differ in exactly one
    thing: what sits between the client and the engine.
    """
    samples: list[tuple[int, float]] = []
    failures: list[str] = []
    plans: list[list[tuple[int, int]]] = []
    if getattr(args, "traces", None):
        plans = load_traces(
            args.traces,
            limit=max(args.agents, 1),
            max_turns=args.turns,
            max_input=args.max_input_tokens,
        )
        if not plans:
            failures.append(f"no replayable traces in {args.traces}")
    limits = httpx.Limits(
        max_connections=args.agents * 2, max_keepalive_connections=args.agents * 2
    )
    url = f"{args.upstream}/v1/chat/completions"
    headers = {"content-type": "application/json"}
    async with httpx.AsyncClient(limits=limits) as client:
        # One warm conversation, for the same reason the proxy arms have one:
        # first-call costs belong outside the measured window.
        await _agent(
            client, url=url, headers=headers, turns=1, max_tokens=args.max_tokens,
            prompt_tokens=args.prompt_tokens, samples=[], failures=[],
            barrier=asyncio.Barrier(1),
        )
        process_barrier = getattr(args, "process_barrier", None)
        if process_barrier is not None:
            try:
                await asyncio.to_thread(process_barrier.wait, 120)
            except Exception as error:  # noqa: BLE001 - a dead sibling must not hang us
                failures.append(f"process barrier: {type(error).__name__}: {error}")
        barrier = asyncio.Barrier(args.agents)
        run_id = uuid.uuid4().hex[:8]
        started = time.perf_counter()
        await asyncio.gather(*[
            _agent(
                client,
                url=url,
                headers=headers,
                turns=args.turns,
                max_tokens=args.max_tokens,
                prompt_tokens=args.prompt_tokens,
                # A conversation identity, so an engine keying its prefix cache
                # on the conversation sees these as the separate sessions they
                # are. Without it every agent shared one cache entry and the
                # direct arm got its whole prompt for free.
                session=f"{run_id}-{index}",
                plan=plans[index % len(plans)] if plans else None,
                samples=samples,
                failures=failures,
                barrier=barrier,
            )
            for index in range(args.agents)
        ])
        wall = time.perf_counter() - started

    latencies = [value for _turn, value in samples]
    by_turn: dict[int, list[float]] = {}
    for turn, value in samples:
        by_turn.setdefault(turn, []).append(value)
    return {
        "arm": "text-direct",
        "agents": args.agents,
        "turns_per_agent": args.turns,
        "replayed_traces": len(plans),
        "turns_completed": len(samples),
        "wall_seconds": round(wall, 3),
        "turns_per_second": round(len(samples) / wall, 1) if wall else 0.0,
        "latency_ms": percentiles(latencies),
        "_latencies": latencies,
        "latency_by_turn_ms": {
            turn: round(statistics.median(values), 2) for turn, values in sorted(by_turn.items())
        },
        "replica_after": {},
        "failures": failures[:10],
        "failure_count": len(failures),
    }


async def _shapes(
    client: httpx.AsyncClient, *, endpoint: str, trajectory_ids: list[str]
) -> list[list[tuple[int, int]]]:
    """Per-turn (prompt tokens, completion tokens) for each trajectory.

    Read back from what the proxy actually sent, so the baseline arm can put
    the same shapes through the engine. Comparing two arms that gave the engine
    different work would say nothing.
    """
    control: dict[str, str] = {}
    out: list[list[tuple[int, int]]] = []
    for trajectory_id in trajectory_ids:
        response = await client.get(
            f"{endpoint}/v1/trajectories/{trajectory_id}/exchanges?limit=500",
            headers=control,
            timeout=30.0,
        )
        rows = response.json().get("data", [])
        out.append([
            (
                int((row.get("usage") or {}).get("prompt_tokens") or 0),
                int((row.get("usage") or {}).get("completion_tokens") or 0),
            )
            for row in rows
        ])
    return out


async def _baseline_agent(
    client: httpx.AsyncClient,
    *,
    upstream: str,
    session: str,
    shapes: list[tuple[int, int]],
    samples: list[tuple[int, float]],
    failures: list[str],
    barrier: asyncio.Barrier,
) -> None:
    """The same token shapes, straight at the engine, no proxy in the path.

    Each turn extends the previous prompt with what the engine returned plus
    filler, so the engine's prefix cache sees the same reuse it saw through the
    proxy.
    """
    tokens: list[int] = []
    filler = 1000
    await barrier.wait()
    try:
        for turn, (prompt_tokens, completion_tokens) in enumerate(shapes):
            while len(tokens) < prompt_tokens:
                tokens.append(filler)
                filler += 1
            tokens = tokens[:prompt_tokens]
            start = time.perf_counter()
            response = await client.post(
                f"{upstream}/generate",
                json={
                    "prompt_token_ids": [tokens],
                    "sampling_params": {"max_tokens": completion_tokens, "logprobs": True},
                    "session_id": session,
                },
                timeout=120.0,
            )
            elapsed = (time.perf_counter() - start) * 1000
            if response.status_code != 200:
                failures.append(f"baseline turn {turn}: {response.status_code}")
                return
            samples.append((turn, elapsed))
            tokens = tokens + list(response.json()["response_ids"][0])
    except Exception as error:  # noqa: BLE001
        failures.append(f"baseline {type(error).__name__}: {error}")


async def run_baseline(args: argparse.Namespace, shapes: list[list[tuple[int, int]]]) -> dict[str, Any]:
    samples: list[tuple[int, float]] = []
    failures: list[str] = []
    limits = httpx.Limits(max_connections=args.agents * 2, max_keepalive_connections=args.agents * 2)
    async with httpx.AsyncClient(limits=limits) as client:
        # Warm the pool: the listen backlog is small, and hundreds of
        # simultaneous connects get reset rather than queued.
        for start_index in range(0, len(shapes), 64):
            await asyncio.gather(*[
                client.post(
                    f"{args.upstream}/generate",
                    json={"prompt_token_ids": [[1, 2]], "sampling_params": {"max_tokens": 1},
                          "session_id": f"warm-{index}"},
                    timeout=60.0,
                )
                for index in range(start_index, min(start_index + 64, len(shapes)))
            ])
        process_barrier = getattr(args, "process_barrier", None)
        if process_barrier is not None:
            try:
                await asyncio.to_thread(process_barrier.wait, 120)
            except Exception as error:  # noqa: BLE001 - a dead sibling must not hang us
                failures.append(f"process barrier: {type(error).__name__}: {error}")
        barrier = asyncio.Barrier(len(shapes))
        started = time.perf_counter()
        await asyncio.gather(*[
            _baseline_agent(
                client,
                upstream=args.upstream,
                session=f"baseline-{index}",
                shapes=turns,
                samples=samples,
                failures=failures,
                barrier=barrier,
            )
            for index, turns in enumerate(shapes)
        ])
        wall = time.perf_counter() - started
    latencies = [value for _turn, value in samples]
    return {
        "agents": len(shapes),
        "trajectory_ids": [],
        "_latencies": latencies,
        "turns_completed": len(samples),
        "wall_seconds": round(wall, 3),
        "turns_per_second": round(len(samples) / wall, 1) if wall else 0.0,
        "latency_ms": percentiles(latencies),
        "failure_count": len(failures),
        "failures": failures[:5],
    }


def _synthetic_shapes(agents: int, turns: int, out_tokens: int, growth: int = 227) -> list[list[tuple[int, int]]]:
    """The shape an agentic conversation grows into, without needing a proxy run.

    Each turn appends the model's reply plus a short user message, so the
    prompt grows by a fixed amount and only that much is uncached. Lets the
    direct arm run at a concurrency the proxy arm is also being measured at.
    """
    shapes = []
    for _ in range(agents):
        prompt = 20
        turn_shapes = []
        for _turn in range(turns):
            turn_shapes.append((prompt, out_tokens))
            prompt += out_tokens + (growth - out_tokens if growth > out_tokens else 27)
        shapes.append(turn_shapes)
    return shapes


def _crashed(agents: int, error: BaseException) -> dict[str, Any]:
    """A report standing in for a worker that never produced one.

    Without it a crashed worker leaves its siblings waiting on the process
    barrier and the parent waiting on the queue, and the run hangs rather than
    saying what went wrong.
    """
    return {
        "agents": agents,
        "turns_completed": 0,
        "wall_seconds": 0.0,
        "turns_per_second": 0.0,
        "latency_ms": {},
        "latency_by_turn_ms": {},
        "replica_after": {},
        "failure_count": agents,
        "failures": [f"worker crashed: {type(error).__name__}: {error}"],
        "traceback": traceback.format_exc(),
        "_latencies": [],
    }


def _baseline_worker(args: argparse.Namespace, agents: int, queue: Any, barrier: Any) -> None:
    local = argparse.Namespace(**vars(args))
    local.agents = agents
    local.process_barrier = barrier
    try:
        shapes = _synthetic_shapes(agents, args.turns, args.max_tokens)
        queue.put(asyncio.run(run_baseline(local, shapes)))
    except BaseException as error:  # noqa: BLE001 - the parent must hear about it
        barrier.abort()
        queue.put(_crashed(agents, error))


def _worker(args: argparse.Namespace, agents: int, queue: Any, barrier: Any) -> None:
    """One load process: its own event loop, its own share of the agents."""
    local = argparse.Namespace(**vars(args))
    local.agents = agents
    local.process_barrier = barrier
    entry = run_text_direct if args.arm == "text-direct" else run
    try:
        queue.put(asyncio.run(entry(local)))
    except BaseException as error:  # noqa: BLE001 - the parent must hear about it
        barrier.abort()
        queue.put(_crashed(agents, error))


def _merge(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine per-process reports into one.

    Throughput adds up; latency does not, so the percentiles are recomputed
    from the pooled samples rather than averaged.
    """
    pooled: list[float] = []
    for report in reports:
        pooled.extend(report.pop("_latencies", []))
    wall = max((report["wall_seconds"] for report in reports), default=0.0)
    turns = sum(report["turns_completed"] for report in reports)
    merged = dict(reports[0]) if reports else {}
    merged.update({
        "load_processes": len(reports),
        "agents": sum(report["agents"] for report in reports),
        "turns_completed": turns,
        "wall_seconds": round(wall, 3),
        "turns_per_second": round(turns / wall, 1) if wall else 0.0,
        "latency_ms": percentiles(pooled),
        "failure_count": sum(report["failure_count"] for report in reports),
        "failures": [item for report in reports for item in report["failures"]][:10],
        "trajectory_ids": [tid for report in reports for tid in report.get("trajectory_ids", [])],
    })
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080")
    parser.add_argument("--project", default="tokens-bench")
    parser.add_argument(
        "--arm",
        choices=("proxy", "text-direct"),
        default="proxy",
        help="proxy drives the capture process at --endpoint, in whichever mode it was "
             "started; text-direct drives the engine's chat API with nothing in between",
    )
    parser.add_argument("--agents", type=int, default=16, help="Concurrent trajectories")
    parser.add_argument("--turns", type=int, default=10, help="Dependent turns per trajectory")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--traces",
        help="Replay AgentX-style traces from this JSONL instead of uniform turns. "
             "Each conversation follows one trace's per-turn input and output "
             "token counts.",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=120_000,
        help="Drop a replayed trace whose prompt would exceed this. Must stay "
             "under the target's max_model_len.",
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=0,
        help="Pad each user message to about this many tokens, so the context "
             "reaches a realistic length. Real agentic sessions run a median "
             "input near 88k tokens; the default workload is nowhere near it.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the raw report")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Also drive the engine directly with the same token shapes, and report the difference",
    )
    parser.add_argument("--upstream", default="http://127.0.0.1:9188", help="Engine base URL for --compare")
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Drive the engine directly with synthetic shapes; no proxy in the path",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Load processes. One event loop saturates near 350 requests/s, well "
             "below what the proxy does, so measuring past that needs several.",
    )
    args = parser.parse_args()

    if args.processes > 1:
        share, extra = divmod(args.agents, args.processes)
        counts = [share + (1 if index < extra else 0) for index in range(args.processes)]
        counts = [count for count in counts if count]
        context = multiprocessing.get_context("spawn")
        queue: Any = context.Queue()
        barrier = context.Barrier(len(counts))
        target = _baseline_worker if args.baseline_only else _worker
        workers = [
            context.Process(target=target, args=(args, count, queue, barrier))
            for count in counts
        ]
        for worker in workers:
            worker.start()
        deadline = time.monotonic() + 900
        reports = []
        for _ in workers:
            try:
                reports.append(queue.get(timeout=max(1.0, deadline - time.monotonic())))
            except Exception:  # noqa: BLE001 - a worker died without reporting
                break
        for worker in workers:
            worker.join(timeout=30)
            if worker.is_alive():
                worker.terminate()
        report = _merge(reports)
    elif args.arm == "text-direct":
        report = asyncio.run(run_text_direct(args))
        report.setdefault("latency_by_turn_ms", {})
        report.setdefault("replica_after", {})
    elif args.baseline_only:
        report = asyncio.run(
            run_baseline(args, _synthetic_shapes(args.agents, args.turns, args.max_tokens))
        )
        report.setdefault("latency_by_turn_ms", {})
        report.setdefault("replica_after", {})
    else:
        report = asyncio.run(run(args))
    if args.compare:
        report["baseline"] = asyncio.run(_compare(args, report))
    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"\n  {args.agents} concurrent trajectories x {args.turns} turns")
    print(f"  completed {report['turns_completed']} turns in {report['wall_seconds']}s "
          f"-> {report['turns_per_second']} turns/s")
    lat = report["latency_ms"]
    if lat:
        print(f"  turn latency ms: p50={lat['p50']}  p95={lat['p95']}  p99={lat['p99']}  max={lat['max']}")
    growth = report["latency_by_turn_ms"]
    if growth:
        first, last = min(growth), max(growth)
        print(f"  by turn: turn {first}={growth[first]}ms -> turn {last}={growth[last]}ms")
    after = report["replica_after"]
    print(f"  replica after: cached_traces={after.get('cached_traces')} "
          f"resident_tokens={after.get('resident_tokens')} evicted={after.get('traces_evicted')}")
    base = report.get("baseline")
    if base:
        blat = base["latency_ms"]
        print("\n  same token shapes, engine directly (no proxy):")
        print(f"    {base['turns_per_second']} turns/s  p50={blat['p50']}  p95={blat['p95']}  p99={blat['p99']}")
        print(f"  added by the proxy: p50 {lat['p50'] - blat['p50']:+.1f} ms  "
              f"p95 {lat['p95'] - blat['p95']:+.1f} ms  p99 {lat['p99'] - blat['p99']:+.1f} ms")
    if report["failure_count"]:
        print(f"  FAILURES ({report['failure_count']}): {report['failures'][:3]}")
    print()


if __name__ == "__main__":
    main()

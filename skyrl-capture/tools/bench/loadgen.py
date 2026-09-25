"""Load generator process.

Run as its own process, several at a time. A single Python event loop driving
many concurrent HTTP requests becomes the bottleneck well before the proxy
does -- measured on a 10-core laptop, one generator process peaks near 1.7k
requests per second and *loses* throughput above roughly 8 concurrent requests,
while the same upstream serves over 4k when driven by four processes. Measuring
proxy overhead through a saturated generator would report the generator's
queueing as the proxy's latency.

    python -m tools.bench.loadgen --url ... --requests 1000 --concurrency 8

Emits one JSON document on stdout so the harness can aggregate exact samples
rather than averaged percentiles.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

import httpx
import orjson

# Cap the returned sample count so a long run cannot produce an unbounded
# stdout document. Well above any single arm here.
MAX_SAMPLES = 200_000


def build_payload(streaming: bool, prompt_repeat: int) -> bytes:
    return orjson.dumps(
        {
            "model": "bench-model",
            "messages": [
                {"role": "system", "content": "You are a benchmark fixture."},
                {"role": "user", "content": "Return a short reply. " * prompt_repeat},
            ],
            "stream": streaming,
            "max_tokens": 64,
        }
    )


async def run(arguments: argparse.Namespace) -> dict[str, object]:
    payload = build_payload(arguments.streaming, arguments.prompt_repeat)
    headers = {"content-type": "application/json"}
    if arguments.api_key:
        headers["authorization"] = f"Bearer {arguments.api_key}"

    concurrency = arguments.concurrency
    limits = httpx.Limits(
        max_connections=concurrency * 2, max_keepalive_connections=concurrency * 2
    )
    latencies: list[float] = []
    first_bytes: list[float] = []
    errors = 0

    async with httpx.AsyncClient(limits=limits, timeout=arguments.timeout) as client:
        # Warmup establishes connections and pays first-call costs, and is
        # excluded from the samples.
        await asyncio.gather(
            *(_once(client, arguments.url, headers, payload, arguments.streaming) for _ in range(concurrency)),
            return_exceptions=True,
        )

        remaining = arguments.requests
        deadline = time.perf_counter() + arguments.duration if arguments.duration else None
        interval = (concurrency / arguments.target_rps) if arguments.target_rps else 0.0

        async def worker() -> None:
            nonlocal remaining, errors
            while True:
                if deadline is not None and time.perf_counter() >= deadline:
                    return
                if deadline is None:
                    if remaining <= 0:
                        return
                    remaining -= 1
                if interval:
                    await asyncio.sleep(interval)
                outcome = await _once(client, arguments.url, headers, payload, arguments.streaming)
                if outcome is None:
                    errors += 1
                    continue
                elapsed, first_byte = outcome
                if len(latencies) < MAX_SAMPLES:
                    latencies.append(elapsed)
                    if first_byte is not None:
                        first_bytes.append(first_byte)

        started = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        wall = time.perf_counter() - started

    return {
        "latencies_ms": latencies,
        "first_byte_ms": first_bytes,
        "errors": errors,
        "wall_seconds": wall,
    }


async def _once(
    client: httpx.AsyncClient, url: str, headers: dict[str, str], payload: bytes, streaming: bool
) -> tuple[float, float | None] | None:
    started = time.perf_counter()
    try:
        if streaming:
            first_byte: float | None = None
            async with client.stream("POST", url, content=payload, headers=headers) as response:
                if response.status_code >= 400:
                    await response.aread()
                    return None
                async for chunk in response.aiter_raw():
                    if chunk and first_byte is None:
                        first_byte = (time.perf_counter() - started) * 1000.0
            return (time.perf_counter() - started) * 1000.0, first_byte
        response = await client.post(url, content=payload, headers=headers)
        if response.status_code >= 400:
            return None
        return (time.perf_counter() - started) * 1000.0, None
    except httpx.HTTPError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--target-rps", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--prompt-repeat", type=int, default=8)
    arguments = parser.parse_args()
    json.dump(asyncio.run(run(arguments)), sys.stdout)
    sys.stdout.flush()


if __name__ == "__main__":
    main()

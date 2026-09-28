"""What the tokens proxy costs on top of the inference call.

Opt in:

    CAPTURE_BENCH=1 uv run pytest tests/test_tokens_benchmark.py -s

Reports client-observed latency for a tokens turn against the mock engine,
next to the same call made straight to that engine. The difference is the
proxy's own work: render or bridge, verify, commit, build the response.
"""

from __future__ import annotations

import os
import statistics
import time

import httpx
import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("CAPTURE_BENCH"), reason="set CAPTURE_BENCH=1")

async def _time(call, repeats: int = 10) -> tuple[float, float]:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        await call()
        samples.append((time.perf_counter() - start) * 1000)
    samples.sort()
    return statistics.median(samples), samples[int(len(samples) * 0.9)]


async def test_report_tokens_overhead(stack_builder):
    """Cost per turn of an agent that feeds the model's own replies back.

    The history has to be the model's actual output, or prefix reuse cannot
    engage: a fabricated assistant message is a client-authored node and
    anchors no inference boundary. Measuring against a made-up history
    measures the wrong path.
    """
    def upstream(url: str):
        from skyrl_capture.config import TitoUpstream

        return TitoUpstream(
            type="tokens",
            url=f"{url}/generate",
            model="bench-model",
            tokenizer="builtin",
            api_key="upstream-secret",
            max_model_len=500_000,
        )

    bench = await stack_builder(upstream_for=upstream)
    created = await bench.create_trajectory()
    upstream = bench.upstream_url

    async def ask(messages):
        response = await bench.chat(created, messages)
        assert response.status_code == 200, response.text
        return response.json()["choices"][0]["message"]

    print(f"\n  {'turn':>5} {'msgs':>5} {'direct p50':>11} {'proxy p50':>10} "
          f"{'added p50':>10} {'added p90':>10}")
    async with httpx.AsyncClient(timeout=30.0) as raw:
        messages: list[dict] = [{"role": "user", "content": "Turn 0: begin the task."}]
        for turn in range(41):
            if turn in (0, 5, 20, 40):
                async def direct(depth=len(messages)) -> None:
                    await raw.post(
                        f"{upstream}/generate",
                        json={
                            "prompt_token_ids": [list(range(depth * 12))],
                            "sampling_params": {"max_tokens": 24, "logprobs": True},
                        },
                    )

                snapshot = tuple(messages)

                async def proxied(turn_messages=snapshot) -> None:
                    await ask(list(turn_messages))

                direct_p50, _ = await _time(direct)
                proxy_p50, proxy_p90 = await _time(proxied)
                print(
                    f"  {turn:>5} {len(messages):>5} {direct_p50:>10.2f}m {proxy_p50:>9.2f}m "
                    f"{proxy_p50 - direct_p50:>9.2f}m {proxy_p90 - direct_p50:>9.2f}m"
                )

            reply = await ask(messages)
            messages = messages + [
                {key: value for key, value in reply.items() if value is not None},
                {"role": "user", "content": f"Turn {turn + 1}: keep going with more detail."},
            ]

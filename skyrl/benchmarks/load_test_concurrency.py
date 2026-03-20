#!/usr/bin/env python3
"""
Load test for concurrency limits across the inference stack.

NOTE: This is not a full fledged serving benchmark script. This is primarily 
aimed to testing bottlenecks in client/ API server code in handling
concurrent requests before they are handed off to the inference engine.

Spins up vLLM server(s) + router + RemoteInferenceClient via Ray, then sends
concurrent requests to verify that the full HTTP
pipeline handles high concurrency without dropping connections.

Three modes:
  direct  - Direct to vLLM server (bypasses router)
  router  - Through the InferenceRouter (tests httpx pool + uvicorn backlog)
  e2e     - Via RemoteInferenceClient.generate() (tests aiohttp connector)

Usage:
  # Requires at least 1 GPU
  uv run --isolated --extra dev --extra fsdp python skyrl/benchmarks/load_test_concurrency.py

  # Custom options
  uv run --isolated --extra dev --extra fsdp python skyrl/benchmarks/load_test_concurrency.py \
      --num-prompts 500 --modes direct --qps 200 --max-tokens 32
"""

import argparse
import asyncio
import logging
import math
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple

import aiohttp
import ray
from ray.util.placement_group import placement_group
from transformers import AutoTokenizer

from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    RemoteInferenceClient,
)
from skyrl.backends.skyrl_train.inference_servers.router import InferenceRouter
from skyrl.backends.skyrl_train.inference_servers.server_group import ServerGroup
from skyrl.backends.skyrl_train.inference_servers.utils import build_vllm_cli_args
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.utils import ResolvedPlacementGroup, initialize_ray

logger = logging.getLogger(__name__)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SERVED_MODEL_NAME = "load_test_model"
VALID_MODES = ["direct", "router", "e2e", "fully_async"]


def get_config() -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = MODEL
    cfg.trainer.critic.model.path = ""
    cfg.trainer.placement.colocate_all = True
    cfg.trainer.placement.policy_num_gpus_per_node = 1
    cfg.trainer.logger = "console"
    cfg.generator.async_engine = True
    cfg.generator.inference_engine.num_engines = 1
    cfg.generator.inference_engine.tensor_parallel_size = 1
    cfg.generator.run_engines_locally = True
    cfg.generator.inference_engine.served_model_name = SERVED_MODEL_NAME
    cfg.generator.sampling_params.max_generate_length = 16
    return cfg


def start_servers(cfg: SkyRLTrainConfig) -> Tuple[RemoteInferenceClient, InferenceRouter, ServerGroup]:
    """Start vLLM server group, router, and build a `RemoteInferenceClient`."""
    cli_args = build_vllm_cli_args(cfg)
    ie_cfg = cfg.generator.inference_engine

    # Placement group
    num_gpus = ie_cfg.tensor_parallel_size * ie_cfg.num_engines
    raw_pg = placement_group([{"GPU": 1, "CPU": 1}] * num_gpus, strategy="PACK")
    ray.get(raw_pg.ready())
    pg = ResolvedPlacementGroup(raw_pg)

    # Server group
    server_group = ServerGroup(
        cli_args=cli_args,
        num_servers=ie_cfg.num_engines,
        placement_group=pg,
    )
    server_infos = server_group.start()
    server_urls = [info.url for info in server_infos]

    # Router
    router = InferenceRouter(server_urls=server_urls)
    proxy_url = router.start()

    # Client
    client = RemoteInferenceClient(
        proxy_url=proxy_url,
        server_urls=server_urls,
        model_name=SERVED_MODEL_NAME,
    )

    return client, router, server_group


def shutdown_servers(client, router, server_group):
    if router is not None:
        router.shutdown()
    if server_group is not None:
        server_group.shutdown()


async def _post_chat_completion(session: aiohttp.ClientSession, url: str, payload: dict) -> dict:
    async with session.post(url, json=payload) as resp:
        return {"status": resp.status, "body": await resp.json()}


async def _rate_limited_gather(coro_fns, qps):
    """Launch coroutine factories at a steady rate of *qps* per second, then await all.

    *coro_fns* is an iterable of zero-arg callables that each return a coroutine.
    Each factory is called (and thus the coroutine created & scheduled) only when
    its turn arrives, giving precise control over request launch timing.
    """
    tasks = []
    interval = 1.0 / qps
    for fn in coro_fns:
        tasks.append(asyncio.ensure_future(fn()))
        await asyncio.sleep(interval)
    return await asyncio.gather(*tasks, return_exceptions=True)


async def fire_chat_completions(base_url: str, n: int, model_name: str, max_tokens: int, qps: float = math.inf) -> dict:
    """Send *n* concurrent /v1/chat/completions requests."""
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Say hi"}],
        "max_tokens": max_tokens,
    }
    connector = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        t0 = time.monotonic()
        if math.isinf(qps):
            tasks = [_post_chat_completion(session, url, payload) for _ in range(n)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            coro_fns = [lambda: _post_chat_completion(session, url, payload) for _ in range(n)]
            results = await _rate_limited_gather(coro_fns, qps)
        elapsed = time.monotonic() - t0

    ok = sum(1 for r in results if isinstance(r, dict) and r["status"] == 200)
    errors = [r for r in results if isinstance(r, Exception) or (isinstance(r, dict) and r["status"] != 200)]
    return {"n": n, "ok": ok, "errors": len(errors), "elapsed_s": elapsed, "first_errors": errors[:3]}


async def fire_client_generate(
    client: RemoteInferenceClient, tokenizer, n: int, max_tokens: int, qps: float = math.inf
) -> dict:
    """Send *n* concurrent prompts through RemoteInferenceClient._generate_single().

    Calls _generate_single directly (generate stage only, no detokenize) so we
    can use return_exceptions=True and get per-request error counts instead of
    one failure killing the whole batch.
    """
    token_ids = tokenizer.apply_chat_template(
        [[{"role": "user", "content": "Say hi"}]],
        add_generation_prompt=True,
        tokenize=True,
    )
    sampling_params = {"max_tokens": max_tokens}

    async def _single(idx: int) -> dict:
        """Wrap _generate_single to tag errors with the phase that failed."""
        try:
            return await client._generate_single(
                prompt_token_ids=token_ids[0],
                sampling_params=sampling_params,
                session_id=None,
            )
        except Exception as e:
            # Re-raise with context about which request and what type
            raise RuntimeError(f"request {idx}: {type(e).__name__}: {e}") from e

    t0 = time.monotonic()
    if math.isinf(qps):
        tasks = [_single(i) for i in range(n)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    else:
        coro_fns = [lambda i=i: _single(i) for i in range(n)]
        results = await _rate_limited_gather(coro_fns, qps)
    elapsed = time.monotonic() - t0

    ok = sum(1 for r in results if not isinstance(r, Exception))
    errors = [r for r in results if isinstance(r, Exception)]
    return {"n": n, "ok": ok, "errors": len(errors), "elapsed_s": elapsed, "first_errors": errors[:3]}


async def fire_fully_async_generate(
    client: RemoteInferenceClient,
    tokenizer,
    n_workers: int,
    batch_size: int,
    max_tokens: int,
    pause_resume: bool = False,
) -> dict:
    """Simulate fully-async RL: n_workers concurrent generate() calls, each with batch_size prompts.

    This mirrors the production scenario where num_parallel_generation_workers concurrent
    asyncio Tasks all call generate() on the same client instance simultaneously.

    With the buggy local semaphore: each generate() creates its own Semaphore(512), so
    n_workers × batch_size requests fire simultaneously with no cross-call throttling.
    With the fix: all calls share one Semaphore(512 × num_engines) on the client instance.

    If pause_resume=True, simulates a training-step weight-sync cycle before the burst:
      1. Warmup burst (builds the connection pool)
      2. pause(KEEP) - freezes the server
      3. Sleep until all keep-alive connections go stale on both sides
      4. resume() - unfreezes the server
      5. Immediately fire n_workers concurrent generate() calls
    This is the exact trigger for ServerDisconnectedError in production async RL.
    """
    token_ids = tokenizer.apply_chat_template(
        [[{"role": "user", "content": "Say hi"}]],
        add_generation_prompt=True,
        tokenize=True,
    )
    single_token_ids = token_ids[0]
    sampling_params = {"max_tokens": max_tokens}
    input_batch = {
        "prompt_token_ids": [single_token_ids] * batch_size,
        "sampling_params": sampling_params,
    }

    async def _worker(idx: int):
        try:
            return await client.generate(input_batch)
        except Exception as e:
            raise RuntimeError(f"worker {idx}: {type(e).__name__}: {e}") from e

    async def _run_burst(label: str) -> dict:
        t0 = time.monotonic()
        tasks = [_worker(i) for i in range(n_workers)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.monotonic() - t0
        errors = [r for r in results if isinstance(r, Exception)]
        ok = sum(1 for r in results if not isinstance(r, Exception))
        return {"n": n_workers, "ok": ok, "errors": len(errors), "elapsed_s": elapsed, "first_errors": errors[:3]}

    if pause_resume:
        # Step 1: warmup burst to fill the connection pool
        print("  [pause_resume] Step 1: warmup burst to build connection pool...")
        warmup = await _run_burst("warmup")
        print(f"  [pause_resume] Warmup: ok={warmup['ok']} errors={warmup['errors']}")

        # Step 2: pause (keep mode - like weight sync)
        print("  [pause_resume] Step 2: pausing server (KEEP mode)...")
        await client.pause()

        # Step 3: sleep until connections go stale
        # uvicorn default timeout_keep_alive=5s; aiohttp keepalive_timeout=2s.
        # Sleep 8s to ensure both sides have closed keep-alive connections.
        stale_sleep = 8
        print(f"  [pause_resume] Step 3: sleeping {stale_sleep}s to stale all keep-alive connections...")
        await asyncio.sleep(stale_sleep)

        # Step 4: resume (like after weight sync completes)
        print("  [pause_resume] Step 4: resuming server...")
        await client.resume()

        # Step 5: immediately fire the burst into stale connections
        print(f"  [pause_resume] Step 5: firing {n_workers} concurrent generate() calls into stale pool...")
        return await _run_burst("post-resume")
    else:
        return await _run_burst("burst")


def print_result(result: dict):
    elapsed = result["elapsed_s"]
    throughput = result["n"] / elapsed if elapsed > 0 else float("inf")
    status = "PASS" if result["errors"] == 0 else "FAIL"
    print(
        f"  [{status}] n={result['n']:>6}  ok={result['ok']:>6}  "
        f"errors={result['errors']:>4}  time={elapsed:.2f}s  throughput={throughput:.1f} req/s"
    )
    for e in result["first_errors"]:
        print(f"         err: {str(e)[:120]}")


def _worker_fire(base_url: str, n: int, model_name: str, max_tokens: int, qps: float = math.inf) -> dict:
    """Entry point for child processes — runs fire_chat_completions in a fresh event loop."""
    return asyncio.run(fire_chat_completions(base_url, n, model_name, max_tokens, qps))


def _merge_results(results: list[dict]) -> dict:
    """Merge results from multiple workers into a single summary."""
    total_n = sum(r["n"] for r in results)
    total_ok = sum(r["ok"] for r in results)
    total_errors = sum(r["errors"] for r in results)
    max_elapsed = max(r["elapsed_s"] for r in results)
    first_errors = []
    for r in results:
        first_errors.extend(r["first_errors"])
    return {
        "n": total_n,
        "ok": total_ok,
        "errors": total_errors,
        "elapsed_s": max_elapsed,
        "first_errors": first_errors[:3],
    }


def run_chat_completions(
    base_url: str, n: int, model_name: str, max_tokens: int, qps: float = math.inf, max_workers: int = 1
):
    """Run chat completions with optional multi-process workers."""
    if max_workers > 1:
        chunk = n // max_workers
        # Per-worker QPS so aggregate matches requested QPS
        worker_qps = qps / max_workers
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(_worker_fire, base_url, chunk, model_name, max_tokens, worker_qps)
                for _ in range(max_workers)
            ]
            worker_results = [f.result() for f in futures]
        return _merge_results(worker_results)
    else:
        return asyncio.run(fire_chat_completions(base_url, n, model_name, max_tokens, qps))


def parse_args():
    parser = argparse.ArgumentParser(description="Load test inference connection limits")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to send (default: 1000)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16,
        help="Max tokens per generation (default: 16)",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default=None,
        help=f"Comma-separated modes (default: {','.join(VALID_MODES)}). "
        "direct=vLLM server only, router=router+vLLM, e2e=RemoteInferenceClient+router+vLLM",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of worker processes for direct/router modes (default: 1)",
    )

    parser.add_argument(
        "--qps",
        type=float,
        default=math.inf,
        help="Submit requests at a steady rate (requests/sec). " "Default: inf (all requests fired at time 0).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Prompts per generate() call in fully_async mode (default: 8)",
    )
    parser.add_argument(
        "--pause-resume",
        action="store_true",
        default=False,
        help="In fully_async mode: do a pause/sleep/resume cycle before the burst to stale connections (default: False)",
    )

    args = parser.parse_args()

    modes = [x.strip() for x in args.modes.split(",")] if args.modes else VALID_MODES
    for mode in modes:
        if mode not in VALID_MODES:
            parser.error(f"Invalid mode '{mode}', expected one of {VALID_MODES}")

    if args.max_workers > 1 and "e2e" in modes and len(modes) == 1:
        parser.error("--max-workers is not supported for e2e mode")

    return args.num_prompts, modes, args.max_tokens, args.qps, args.max_workers, args.batch_size, args.pause_resume


def main():
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    num_prompts, modes, max_tokens, qps, max_workers, batch_size, pause_resume = parse_args()

    print("Load test config:")
    print(f"  model={MODEL}")
    print(f"  num_prompts={num_prompts}")
    print(f"  max_tokens={max_tokens}")
    print(f"  modes={modes}")
    print(f"  qps={qps}")
    print(f"  max_workers={max_workers}")
    print(f"  batch_size={batch_size} (fully_async mode)")
    print()
    print("Starting servers...")

    cfg = get_config()
    if not ray.is_initialized():
        initialize_ray(cfg)

    client, router, server_group = start_servers(cfg)

    try:
        proxy_url = client.proxy_url
        server_urls = client.server_urls
        tokenizer = AutoTokenizer.from_pretrained(MODEL)

        print(f"Router URL:  {proxy_url}")
        print(f"Server URLs: {server_urls}\n")

        if "direct" in modes:
            print("=" * 60)
            print("Mode: direct - vLLM server (bypass router)")
            print("=" * 60)
            result = run_chat_completions(server_urls[0], num_prompts, SERVED_MODEL_NAME, max_tokens, qps, max_workers)
            print_result(result)
            print()

        if "router" in modes:
            print("=" * 60)
            print("Mode: router - through `InferenceRouter`")
            print("=" * 60)
            result = run_chat_completions(proxy_url, num_prompts, SERVED_MODEL_NAME, max_tokens, qps, max_workers)
            print_result(result)
            print()

        if "e2e" in modes:
            print("=" * 60)
            print("Mode: e2e - `RemoteInferenceClient.generate()`")
            print("=" * 60)

            async def _run_e2e():
                result = await fire_client_generate(client, tokenizer, num_prompts, max_tokens, qps)
                print_result(result)
                await client.teardown()

            asyncio.run(_run_e2e())
            print()

        if "fully_async" in modes:
            n_workers = num_prompts // batch_size
            pr_label = " + pause/resume" if pause_resume else ""
            print("=" * 60)
            print(f"Mode: fully_async{pr_label} - {n_workers} concurrent generate() calls × batch_size={batch_size}")
            print(f"  = {n_workers * batch_size} total simultaneous requests (no cross-call throttle on baseline)")
            print("=" * 60)

            async def _run_fully_async():
                result = await fire_fully_async_generate(
                    client, tokenizer, n_workers, batch_size, max_tokens, pause_resume=pause_resume
                )
                elapsed = result["elapsed_s"]
                throughput = result["n"] / elapsed if elapsed > 0 else float("inf")
                status = "PASS" if result["errors"] == 0 else "FAIL"
                print(
                    f"  [{status}] workers={result['n']:>6}  ok={result['ok']:>6}  "
                    f"errors={result['errors']:>4}  time={elapsed:.2f}s  throughput={throughput:.1f} workers/s"
                )
                for e in result["first_errors"]:
                    print(f"         err: {str(e)[:200]}")
                await client.teardown()

            asyncio.run(_run_fully_async())
            print()

    finally:
        shutdown_servers(client, router, server_group)
        print("Servers shut down.")


if __name__ == "__main__":
    main()

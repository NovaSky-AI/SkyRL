#!/usr/bin/env python3
"""
Load test for HTTP connection limits across the inference stack.

Spins up vLLM server(s) + router + RemoteInferenceClient via Ray, then sends
concurrent requests at increasing batch sizes to verify that the full HTTP
pipeline handles high concurrency without dropping connections.

Three modes:
  direct  - Direct to vLLM server (bypasses router)
  router  - Through the InferenceRouter (tests httpx pool + uvicorn backlog)
  e2e     - Via RemoteInferenceClient.generate() (tests aiohttp connector)

Usage:
  # Requires at least 1 GPU
  uv run --isolated --extra dev --extra fsdp python skyrl/benchmarks/load_test_connections.py

  # Custom levels and modes
  uv run --isolated --extra dev --extra fsdp python skyrl/benchmarks/load_test_connections.py \\
      --levels 100,500,1000 --modes direct,e2e
"""

import argparse
import asyncio
import logging
import time

import aiohttp
import ray
from ray.util.placement_group import placement_group
from transformers import AutoTokenizer

from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import RemoteInferenceClient
from skyrl.backends.skyrl_train.inference_servers.router import InferenceRouter
from skyrl.backends.skyrl_train.inference_servers.server_group import ServerGroup
from skyrl.backends.skyrl_train.inference_servers.utils import build_vllm_cli_args
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.utils import initialize_ray

logger = logging.getLogger(__name__)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SERVED_MODEL_NAME = "load_test_model"
DEFAULT_LEVELS = [100, 500, 1000, 2000, 5000, 10000]
VALID_MODES = ["direct", "router", "e2e"]


# ---------------------------------------------------------------------------
# Infrastructure setup
# ---------------------------------------------------------------------------


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


def start_infrastructure(cfg: SkyRLTrainConfig):
    """Start vLLM server group, router, and build a RemoteInferenceClient.

    Returns (client, router, server_group) — caller is responsible for shutdown.
    """
    cli_args = build_vllm_cli_args(cfg)
    ie_cfg = cfg.generator.inference_engine

    # Placement group
    num_gpus = ie_cfg.tensor_parallel_size * ie_cfg.num_engines
    pg = placement_group([{"GPU": 1, "CPU": 1}] * num_gpus, strategy="PACK")
    ray.get(pg.ready())

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


def shutdown_infrastructure(client, router, server_group):
    if router is not None:
        router.shutdown()
    if server_group is not None:
        server_group.shutdown()


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------


async def _post_chat_completion(session: aiohttp.ClientSession, url: str, payload: dict) -> dict:
    async with session.post(url, json=payload) as resp:
        return {"status": resp.status, "body": await resp.json()}


async def fire_chat_completions(base_url: str, n: int, model_name: str) -> dict:
    """Send *n* concurrent /v1/chat/completions requests."""
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Say hi"}],
        "max_tokens": 16,
    }
    connector = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [_post_chat_completion(session, url, payload) for _ in range(n)]
        t0 = time.monotonic()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.monotonic() - t0

    ok = sum(1 for r in results if isinstance(r, dict) and r["status"] == 200)
    errors = [r for r in results if isinstance(r, Exception) or (isinstance(r, dict) and r["status"] != 200)]
    return {"n": n, "ok": ok, "errors": len(errors), "elapsed": f"{elapsed:.2f}s", "first_errors": errors[:3]}


async def fire_client_generate(client: RemoteInferenceClient, tokenizer, n: int) -> dict:
    """Send *n* concurrent prompts through RemoteInferenceClient.generate().

    Bypasses client.generate() to use return_exceptions=True so we get
    per-request error counts instead of one failure killing the whole batch.
    """
    token_ids = tokenizer.apply_chat_template(
        [[{"role": "user", "content": "Say hi"}]],
        add_generation_prompt=True,
        tokenize=True,
    )
    sampling_params = {"max_tokens": 16}

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

    tasks = [_single(i) for i in range(n)]

    t0 = time.monotonic()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = time.monotonic() - t0

    ok = sum(1 for r in results if not isinstance(r, Exception))
    errors = [r for r in results if isinstance(r, Exception)]
    return {"n": n, "ok": ok, "errors": len(errors), "elapsed": f"{elapsed:.2f}s", "first_errors": errors[:3]}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_result(result: dict):
    status = "PASS" if result["errors"] == 0 else "FAIL"
    print(
        f"  [{status}] n={result['n']:>6}  ok={result['ok']:>6}  "
        f"errors={result['errors']:>4}  time={result['elapsed']}"
    )
    for e in result["first_errors"]:
        print(f"         err: {str(e)[:120]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Load test inference connection limits")
    parser.add_argument(
        "--levels",
        type=str,
        default=None,
        help=f"Comma-separated concurrency levels (default: {','.join(map(str, DEFAULT_LEVELS))})",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default=None,
        help=f"Comma-separated modes (default: {','.join(VALID_MODES)}). "
        "direct=vLLM server only, router=router+vLLM, e2e=RemoteInferenceClient+router+vLLM",
    )
    args = parser.parse_args()

    levels = [int(x) for x in args.levels.split(",")] if args.levels else DEFAULT_LEVELS
    modes = [x.strip() for x in args.modes.split(",")] if args.modes else VALID_MODES
    for mode in modes:
        if mode not in VALID_MODES:
            parser.error(f"Invalid mode '{mode}', expected one of {VALID_MODES}")

    return levels, modes


def main():
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    levels, modes = parse_args()

    print(f"Load test: model={MODEL}, levels={levels}, modes={modes}")
    print("Starting infrastructure...")

    cfg = get_config()
    if not ray.is_initialized():
        initialize_ray(cfg)

    client, router, server_group = start_infrastructure(cfg)

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
            for n in levels:
                result = asyncio.run(fire_chat_completions(server_urls[0], n, SERVED_MODEL_NAME))
                print_result(result)
            print()

        if "router" in modes:
            print("=" * 60)
            print("Mode: router - through InferenceRouter")
            print("=" * 60)
            for n in levels:
                result = asyncio.run(fire_chat_completions(proxy_url, n, SERVED_MODEL_NAME))
                print_result(result)
            print()

        if "e2e" in modes:
            print("=" * 60)
            print("Mode: e2e - RemoteInferenceClient.generate()")
            print("=" * 60)

            for n in levels:

                async def _run_e2e(n=n):
                    result = await fire_client_generate(client, tokenizer, n)
                    print_result(result)
                    await client.teardown()

                asyncio.run(_run_e2e())
            print()

    finally:
        shutdown_infrastructure(client, router, server_group)
        print("Infrastructure shut down.")


if __name__ == "__main__":
    main()

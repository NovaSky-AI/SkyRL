"""
Performance benchmark for SkyRL inference clients.

Compares RemoteInferenceClient (new) vs InferenceEngineClient (old)
with synthetic (random) or real (ShareGPT) data.

Supports batch mode (all requests concurrent) and online mode
(Poisson/Gamma rate-controlled arrivals via vLLM's get_request).

vLLM's ShareGPTDataset flattens multi-turn conversations into single
requests (first user turn -> prompt, first assistant turn -> expected
output), so the benchmark dispatch logic is identical for both datasets.

Usage:
    # Batch mode with random data
    python skyrl/benchmarks/benchmark_new_inference.py \
        --dataset-name random --num-prompts 100 --client-type new

    # Online mode with ShareGPT
    python skyrl/benchmarks/benchmark_new_inference.py \
        --dataset-name sharegpt --dataset-path <path> \
        --request-rate 10 --num-prompts 100

    # Compare old vs new client
    python skyrl/benchmarks/benchmark_new_inference.py \
        --dataset-name random --num-prompts 100 --client-type old
"""

import argparse
import asyncio
import json
import pprint
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, Union, Dict

import numpy as np
from loguru import logger
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from skyrl.backends.skyrl_train.inference_engines.base import (
    InferenceEngineInput,
)
from skyrl.backends.skyrl_train.inference_engines.inference_engine_client import (
    InferenceEngineClient,
)
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    RemoteInferenceClient,
)
from skyrl.backends.skyrl_train.inference_servers.router import InferenceRouter
from skyrl.backends.skyrl_train.inference_servers.server_group import ServerGroup
from skyrl.backends.skyrl_train.inference_servers.utils import build_vllm_cli_args
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import (
    create_ray_wrapped_inference_engines_from_config,
    create_remote_inference_engines_from_config,
)

from vllm.benchmarks.datasets import RandomDataset, SampleRequest, ShareGPTDataset
from vllm.benchmarks.serve import get_request

# ---------------------------------------------------------------------------
# Metrics dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RequestResult:
    """Result of a single inference request."""

    success: bool
    latency: float  # E2E seconds
    prompt_len: int  # input tokens
    output_len: int  # output tokens
    error: str = ""


@dataclass
class BenchmarkResult:
    """Aggregated benchmark metrics."""

    completed: int
    failed: int
    total_input_tokens: int
    total_output_tokens: int
    duration_s: float
    request_throughput: float  # req/s
    output_throughput: float  # output tokens/s
    total_token_throughput: float  # (in + out) tokens/s
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: list  # list of (percentile, value_ms)
    server_timings: Optional[Dict[str, float]] = None  # server timings


# ---------------------------------------------------------------------------
# Translation: SampleRequest -> InferenceEngineInput
# ---------------------------------------------------------------------------


def sample_request_to_engine_input(
    request: SampleRequest,
    tokenizer: PreTrainedTokenizerBase,
) -> InferenceEngineInput:
    """Convert a vLLM SampleRequest to a SkyRL InferenceEngineInput.

    Uses the prompt_token_ids path for lower overhead (bypasses chat template).
    """
    token_ids = tokenizer.encode(request.prompt)
    return InferenceEngineInput(
        prompt_token_ids=[token_ids],
        sampling_params={"max_tokens": request.expected_output_len},
    )


# ---------------------------------------------------------------------------
# Dataset loading (delegates to vLLM dataset classes)
# ---------------------------------------------------------------------------


def _load_gsm8k_requests(
    tokenizer: PreTrainedTokenizerBase,
    num_requests: int,
    output_len: int | None,
    seed: int,
) -> list[SampleRequest]:
    """Load GSM8K test split from HuggingFace and convert to SampleRequests."""
    import random as _random
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    indices = list(range(len(ds)))
    _random.seed(seed)
    _random.shuffle(indices)

    samples: list[SampleRequest] = []
    for idx in indices:
        if len(samples) >= num_requests:
            break
        row = ds[idx]
        prompt = row["question"]
        answer = row["answer"]

        prompt_ids = tokenizer.encode(prompt)
        answer_ids = tokenizer.encode(answer)
        expected_output_len = output_len if output_len is not None else len(answer_ids)

        samples.append(
            SampleRequest(
                prompt=prompt,
                prompt_len=len(prompt_ids),
                expected_output_len=expected_output_len,
            )
        )
    return samples


def get_sample_requests(
    args: argparse.Namespace,
    tokenizer: PreTrainedTokenizerBase,
) -> list[SampleRequest]:
    """Build a list of SampleRequests using vLLM dataset infrastructure."""
    if args.dataset_name == "random":
        dataset = RandomDataset(random_seed=args.seed)
        return dataset.sample(
            tokenizer=tokenizer,
            num_requests=args.num_prompts,
            input_len=args.input_len,
            output_len=args.output_len,
        )
    elif args.dataset_name == "sharegpt":
        if args.dataset_path is None:
            raise ValueError("--dataset-path is required for sharegpt dataset")
        dataset = ShareGPTDataset(
            dataset_path=args.dataset_path,
            random_seed=args.seed,
        )
        return dataset.sample(
            tokenizer=tokenizer,
            num_requests=args.num_prompts,
            output_len=args.sharegpt_output_len,
        )
    elif args.dataset_name == "gsm8k":
        return _load_gsm8k_requests(
            tokenizer=tokenizer,
            num_requests=args.num_prompts,
            output_len=args.output_len,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")


# ---------------------------------------------------------------------------
# Benchmark core
# ---------------------------------------------------------------------------


async def run_benchmark(
    client: Union[RemoteInferenceClient, InferenceEngineClient],
    requests: list[SampleRequest],
    tokenizer: PreTrainedTokenizerBase,
    request_rate: float,
    burstiness: float = 1.0,
) -> Tuple[list[RequestResult], float]:
    """Run the benchmark and return per-request results + wall-clock duration.

    Args:
        client: SkyRL inference client (new or old).
        requests: List of SampleRequests to send.
        tokenizer: Tokenizer for prompt encoding.
        request_rate: Requests/sec. Use float('inf') for batch mode.
        burstiness: Gamma distribution shape parameter (1.0 = Poisson).

    Returns:
        (results, duration_seconds)
    """
    results: list[RequestResult] = []

    async def send_request(sample: SampleRequest) -> RequestResult:
        engine_input = sample_request_to_engine_input(sample, tokenizer)
        start = time.monotonic()
        try:
            output = await client.generate(engine_input)
            elapsed = time.monotonic() - start
            output_len = len(output["response_ids"][0])
            return RequestResult(
                success=True,
                latency=elapsed,
                prompt_len=sample.prompt_len,
                output_len=output_len,
            )
        except Exception as e:
            elapsed = time.monotonic() - start
            return RequestResult(
                success=False,
                latency=elapsed,
                prompt_len=sample.prompt_len,
                output_len=0,
                error=str(e),
            )

    benchmark_start = time.monotonic()

    if request_rate == float("inf"):
        # Batch mode: fire all requests concurrently
        results = list(await asyncio.gather(*[send_request(r) for r in requests]))
    else:
        # Online mode: rate-controlled dispatch via vLLM's get_request
        tasks: list[asyncio.Task] = []
        async for request, _ in get_request(requests, request_rate, burstiness):
            tasks.append(asyncio.create_task(send_request(request)))
        results = list(await asyncio.gather(*tasks))

    duration = time.monotonic() - benchmark_start
    return results, duration


# ---------------------------------------------------------------------------
# Results calculation and printing
# ---------------------------------------------------------------------------

PERCENTILES = [50, 75, 90, 95, 99]


def calculate_results(
    results: list[RequestResult],
    duration: float,
    percentiles: list[float] | None = None,
    server_timings: Optional[Dict[str, float]] = None,
) -> BenchmarkResult:
    """Compute aggregate metrics from per-request results."""
    if percentiles is None:
        percentiles = PERCENTILES

    successful = [r for r in results if r.success]
    if not successful:
        return BenchmarkResult(
            completed=0,
            failed=len(results),
            total_input_tokens=0,
            total_output_tokens=0,
            duration_s=duration,
            request_throughput=0.0,
            output_throughput=0.0,
            total_token_throughput=0.0,
            mean_e2el_ms=0.0,
            median_e2el_ms=0.0,
            std_e2el_ms=0.0,
            percentiles_e2el_ms=[],
            server_timings=server_timings,
        )

    e2els = [r.latency for r in successful]
    total_input = sum(r.prompt_len for r in successful)
    total_output = sum(r.output_len for r in successful)

    return BenchmarkResult(
        completed=len(successful),
        failed=len(results) - len(successful),
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        duration_s=duration,
        request_throughput=len(successful) / duration,
        output_throughput=total_output / duration,
        total_token_throughput=(total_input + total_output) / duration,
        mean_e2el_ms=float(np.mean(e2els)) * 1000,
        median_e2el_ms=float(np.median(e2els)) * 1000,
        std_e2el_ms=float(np.std(e2els)) * 1000,
        percentiles_e2el_ms=[(p, float(np.percentile(e2els, p)) * 1000) for p in percentiles],
        server_timings=server_timings,
    )


def print_results(result: BenchmarkResult) -> None:
    """Print benchmark results as a formatted table."""
    print("\n" + "=" * 60)
    print("Benchmark Results")
    print("=" * 60)
    print(f"  Completed requests:        {result.completed}")
    print(f"  Failed requests:           {result.failed}")
    print(f"  Benchmark duration (s):    {result.duration_s:.2f}")
    print(f"  Total input tokens:        {result.total_input_tokens}")
    print(f"  Total output tokens:       {result.total_output_tokens}")
    print("-" * 60)
    print(f"  Request throughput (req/s): {result.request_throughput:.2f}")
    print(f"  Output throughput (tok/s):  {result.output_throughput:.2f}")
    print(f"  Total throughput (tok/s):   {result.total_token_throughput:.2f}")
    print("-" * 60)
    print(f"  Mean E2EL (ms):            {result.mean_e2el_ms:.2f}")
    print(f"  Median E2EL (ms):          {result.median_e2el_ms:.2f}")
    print(f"  Std E2EL (ms):             {result.std_e2el_ms:.2f}")
    for p, val in result.percentiles_e2el_ms:
        print(f"  P{p:<3} E2EL (ms):           {val:.2f}")
    print("=" * 60 + "\n")

    if result.server_timings:
        pprint.pprint(result.server_timings, indent=4)


def save_results(result: BenchmarkResult, args: argparse.Namespace) -> None:
    """Save benchmark results to JSON."""
    if not args.save_result:
        return

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    filename = f"benchmark_{args.client_type}_{args.dataset_name}" f"_{args.num_prompts}prompts_{int(time.time())}.json"
    filepath = result_dir / filename

    data = asdict(result)
    data["args"] = vars(args)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Results saved to {filepath}")


# ---------------------------------------------------------------------------
# Client setup (kept from original)
# ---------------------------------------------------------------------------


def get_new_inference_client(cfg: SkyRLTrainConfig):
    ie_cfg = cfg.generator.inference_engine
    is_colocated = cfg.trainer.placement.colocate_all
    external_proxy_url = ie_cfg.external_proxy_url
    external_server_urls = ie_cfg.external_server_urls

    has_external_proxy = external_proxy_url is not None
    has_external_servers = external_server_urls is not None
    server_group = None
    inference_router = None

    if has_external_proxy and has_external_servers:
        proxy_url = external_proxy_url
        server_urls = list(external_server_urls)
        logger.info(
            f"HTTP Inference: Using fully external setup - " f"proxy_url={proxy_url}, server_urls={server_urls}"
        )
    elif has_external_proxy and not has_external_servers:
        proxy_url = external_proxy_url
        server_urls = [proxy_url]
        logger.info(f"HTTP Inference: Using external proxy for both data and " f"control plane - proxy_url={proxy_url}")
    elif has_external_servers and not has_external_proxy:
        server_urls = list(external_server_urls)
        inference_router = InferenceRouter(server_urls=server_urls)
        proxy_url = inference_router.start()
        logger.info(
            f"HTTP Inference: Created internal router over external "
            f"servers - server_urls={server_urls}, proxy_url={proxy_url}"
        )
    else:
        cli_args = build_vllm_cli_args(cfg)
        server_group = ServerGroup(
            cli_args=cli_args,
            num_servers=ie_cfg.num_engines,
            placement_group=None,
            enable_dp=ie_cfg.data_parallel_size > 1,
        )
        server_infos = server_group.start()
        server_urls = [info.url for info in server_infos]
        inference_router = InferenceRouter(server_urls=server_urls, record_timings=True)
        proxy_url = inference_router.start()
        logger.info(
            f"HTTP Inference: Built servers and router internally - "
            f"proxy_url={proxy_url}, server_urls={server_urls}, "
            f"colocated={is_colocated}"
        )

    return (
        RemoteInferenceClient(
            proxy_url=proxy_url,
            server_urls=server_urls,
            model_name=cfg.trainer.policy.model.path,
            record_timings=True,
        ),
        server_group,
        inference_router,
    )


def get_legacy_inference_client(cfg: SkyRLTrainConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.trainer.policy.model.path)
    colocate_pg = None
    if cfg.generator.inference_engine.run_engines_locally:
        inference_engines = create_ray_wrapped_inference_engines_from_config(cfg, colocate_pg, tokenizer)
    else:
        inference_engines = create_remote_inference_engines_from_config(cfg, tokenizer)
    client = InferenceEngineClient(
        inference_engines,
        tokenizer,
        cfg.trainer.policy.model.path,
        cfg.trainer.policy.model.lora,
        cfg.generator.inference_engine,
    )
    return client, None, None


def make_skyrl_config(args: argparse.Namespace) -> SkyRLTrainConfig:
    overrides = [
        f"trainer.policy.model.path={args.model_name}",
        f"generator.inference_engine.num_engines={args.num_engines}",
    ]
    cfg = SkyRLTrainConfig.from_cli_overrides(overrides)
    cfg.trainer.placement.colocate_all = False
    return cfg


def get_inference_client(
    args: argparse.Namespace,
) -> Tuple[
    Union[RemoteInferenceClient, InferenceEngineClient],
    Optional[ServerGroup],
    Optional[InferenceRouter],
]:
    cfg = make_skyrl_config(args)
    if args.client_type == "old":
        return get_legacy_inference_client(cfg)
    elif args.client_type == "new":
        return get_new_inference_client(cfg)
    else:
        raise ValueError(f"Invalid client type: {args.client_type}")


# ---------------------------------------------------------------------------
# Benchmark entry point
# ---------------------------------------------------------------------------


def run_single_engine_benchmark(args: argparse.Namespace) -> None:
    """Run single-engine benchmark with the configured dataset and traffic pattern."""
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    requests = get_sample_requests(args, tokenizer)
    client, _server_group, _router = get_inference_client(args)

    timing_metrics = None

    async def _run() -> Tuple[list[RequestResult], float]:
        nonlocal timing_metrics
        # Warmup
        if args.num_warmup > 0:
            warmup_reqs = requests[: args.num_warmup]
            logger.info(f"Warming up with {len(warmup_reqs)} requests...")
            warmup_tasks = [client.generate(sample_request_to_engine_input(r, tokenizer)) for r in warmup_reqs]
            await asyncio.gather(*warmup_tasks)
            logger.info("Warmup complete.")

        timing_metrics = client.get_timing_metrics() if isinstance(client, RemoteInferenceClient) else None

        # Benchmark
        return await run_benchmark(
            client=client,
            requests=requests,
            tokenizer=tokenizer,
            request_rate=args.request_rate,
            burstiness=args.burstiness,
        )

    results, duration = asyncio.run(_run())
    bench_result = calculate_results(results, duration, server_timings=timing_metrics)
    print_results(bench_result)
    save_results(bench_result, args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SkyRL Inference Performance Benchmark")

    # Server/client
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model path",
    )
    parser.add_argument(
        "--client-type",
        type=str,
        default="new",
        choices=["old", "new"],
        help="Inference client type",
    )
    parser.add_argument(
        "--num-engines",
        type=int,
        default=1,
        help="Number of vLLM engines",
    )

    # Dataset
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="random",
        choices=["random", "sharegpt", "gsm8k"],
        help="Dataset type",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to ShareGPT JSON file",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of requests to benchmark",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=512,
        help="Mean input token length (random dataset)",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=128,
        help="Mean output token length (random dataset)",
    )
    parser.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="Override output length for ShareGPT (None = use dataset lengths)",
    )

    # Traffic
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Requests/sec (inf = batch mode)",
    )
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="Gamma distribution shape for request arrival (1.0 = Poisson)",
    )

    # Benchmark control
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=10,
        help="Number of warmup requests",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )

    # Output
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Save results to JSON",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="benchmark_results",
        help="Directory for result files",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_single_engine_benchmark(args)

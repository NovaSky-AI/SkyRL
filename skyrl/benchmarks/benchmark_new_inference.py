"""
Performance benchmark for SkyRL inference clients.

Computes inference performance metrics supporting different datasets like synthetic (random), real (ShareGPT), or math (GSM8K) data.
Supports RemoteInferenceClient (new) and InferenceEngineClient (old)

Two benchmark modes:
  - Single-turn: batch or rate-controlled dispatch of independent requests.
  - Multi-turn: agent_loop-style iterative conversation where each user turn
    is sent, the model generates a response, and the next user turn is
    appended to the growing chat history (using apply_chat_template).

Usage:
    # Single-turn batch mode with random data
    python skyrl/benchmarks/benchmark_new_inference.py \
        --dataset-name random --num-prompts 100 --client-type new

    # Single-turn with GSM8K
    python skyrl/benchmarks/benchmark_new_inference.py \
        --dataset-name gsm8k --num-prompts 100

    # Multi-turn with ShareGPT conversations
    python skyrl/benchmarks/benchmark_new_inference.py \
        --test multi-turn --dataset-name sharegpt \
        --dataset-path ShareGPT_V3.json --num-prompts 50

    # Compare old vs new client
    python skyrl/benchmarks/benchmark_new_inference.py \
        --dataset-name random --num-prompts 100 --client-type old
"""

import argparse
import asyncio
import json
import random as _random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from loguru import logger
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from vllm.benchmarks.datasets import RandomDataset, SampleRequest, ShareGPTDataset
from vllm.benchmarks.serve import get_request

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

# Default latency percentiles to compute
PERCENTILES = [50, 75, 90, 95, 99]


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


ShareGPTConversation = list[dict[str, str]]
"""Type alias for ShareGPT conversations: list of {"from": "human"/"gpt", "value": str}"""

OpenAIConversation = list[dict[str, str]]
"""Type alias for openai conversation format: List of {"role": "user", "content": "message"}"""


@dataclass
class ConversationResult:
    """Result of a single multi-turn conversation."""

    success: bool
    num_turns: int  # number of assistant turns completed
    total_latency: float  # total E2E seconds for all turns
    per_turn_latencies: list[float]  # seconds per turn
    total_input_tokens: int  # sum of prompt tokens across turns
    total_output_tokens: int  # sum of generated tokens across turns
    error: str = ""


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
# Multi-turn conversation loading
# ---------------------------------------------------------------------------


def load_sharegpt_conversations(
    dataset_path: str,
    num_conversations: int,
    min_turns: int = 2,
    seed: int = 0,
) -> list[ShareGPTConversation]:
    """Load raw multi-turn conversations from a ShareGPT JSON file.

    Unlike vLLM's ShareGPTDataset.sample() which flattens to single-turn,
    this returns full conversations for multi-turn benchmarking.

    Each conversation is a list of turns: [{"from": "human", "value": "..."},
    {"from": "gpt", "value": "..."}, ...]. Only conversations with at least
    ``min_turns`` total messages (user + assistant) are kept.

    Returns:
        List of conversations, each a list of turn dicts.
    """
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    # Filter conversations with enough turns and at least one user turn.
    # ShareGPT uses both "human" and "user" as role names for user turns.
    valid = []
    for entry in data:
        convos = entry.get("conversations", [])
        has_user = any(t.get("from") in ("human", "user") for t in convos)
        if len(convos) >= min_turns and has_user:
            valid.append(convos)

    _random.seed(seed)
    _random.shuffle(valid)
    return valid[:num_conversations]


def _conversation_to_chat_messages(
    turns: ShareGPTConversation,
) -> OpenAIConversation:
    """Convert ShareGPT turn format to chat message format.

    Maps "human" -> "user" and "gpt" -> "assistant".
    """
    role_map = {"human": "user", "gpt": "assistant"}
    return [{"role": role_map.get(t["from"], t["from"]), "content": t["value"]} for t in turns]


# ---------------------------------------------------------------------------
# Benchmark core — single-turn
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
# Benchmark core — multi-turn
# ---------------------------------------------------------------------------


async def _run_conversation(
    client: Union[RemoteInferenceClient, InferenceEngineClient],
    conversation: ShareGPTConversation,
    tokenizer: PreTrainedTokenizerBase,
    max_tokens_per_turn: int,
    mean_env_delay: float = 0.0,
) -> ConversationResult:
    """Run a single multi-turn conversation, agent_loop style.

    Iterates over (user, assistant) turn pairs from the dataset.  For each:
      1. Simulate environment interaction delay (Poisson-distributed).
      2. Append the user message to the chat history.
      3. Build the full prompt via apply_chat_template and generate.
      4. Append the **ground-truth** assistant response (from the dataset)
         to the history — not the model's output.  This keeps the
         conversation coherent and prompt lengths deterministic across
         runs, while still measuring actual generation performance.
         The KV cache for the ground-truth text won't match what was
         generated, but it still exercises prefix-cache hits on the
         shared history prefix.

    The inter-turn delay models the time an RL environment takes to process
    the assistant response and produce the next observation/user turn.
    Delays are sampled from an exponential distribution (Poisson process)
    with the given mean.  A mean of 0 disables delays entirely.
    """
    chat_messages = _conversation_to_chat_messages(conversation)

    has_user = any(m["role"] == "user" for m in chat_messages)
    if not has_user:
        return ConversationResult(
            success=False,
            num_turns=0,
            total_latency=0.0,
            per_turn_latencies=[],
            total_input_tokens=0,
            total_output_tokens=0,
            error="No user turns in conversation",
        )

    history: list[dict[str, str]] = []
    per_turn_latencies: list[float] = []
    total_input_tokens = 0
    total_output_tokens = 0
    turn_idx = 0
    total_start = time.monotonic()

    try:
        i = 0
        while i < len(chat_messages):
            msg = chat_messages[i]

            if msg["role"] == "user":
                # Simulate environment delay before all turns after the first
                if turn_idx > 0 and mean_env_delay > 0:
                    delay = np.random.exponential(scale=mean_env_delay)
                    await asyncio.sleep(delay)

                # Append user message and build prompt
                history.append(msg)
                prompt_ids = tokenizer.apply_chat_template(
                    history,
                    add_generation_prompt=True,
                )
                prompt_len = len(prompt_ids)

                engine_input = InferenceEngineInput(
                    prompt_token_ids=[prompt_ids],
                    sampling_params={"max_tokens": max_tokens_per_turn},
                )

                turn_start = time.monotonic()
                output = await client.generate(engine_input)
                turn_latency = time.monotonic() - turn_start

                output_ids = output["response_ids"][0]

                per_turn_latencies.append(turn_latency)
                total_input_tokens += prompt_len
                total_output_tokens += len(output_ids)
                turn_idx += 1

                # Append ground-truth assistant response if it follows,
                # to keep conversation coherent and prompt lengths
                # deterministic across runs.
                if i + 1 < len(chat_messages) and chat_messages[i + 1]["role"] == "assistant":
                    history.append(chat_messages[i + 1])
                    i += 2  # skip past the assistant message
                else:
                    # No ground-truth reply: use model output
                    output_text = output["responses"][0]
                    if output_text.endswith(tokenizer.eos_token):
                        output_text = output_text[: -len(tokenizer.eos_token)]
                    history.append({"role": "assistant", "content": output_text})
                    i += 1
            else:
                # System or other messages: include in history as context
                history.append(msg)
                i += 1

    except Exception as e:
        return ConversationResult(
            success=False,
            num_turns=turn_idx,
            total_latency=time.monotonic() - total_start,
            per_turn_latencies=per_turn_latencies,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            error=str(e),
        )

    return ConversationResult(
        success=True,
        num_turns=turn_idx,
        total_latency=time.monotonic() - total_start,
        per_turn_latencies=per_turn_latencies,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
    )


async def run_multi_turn_benchmark(
    client: Union[RemoteInferenceClient, InferenceEngineClient],
    conversations: list[ShareGPTConversation],
    tokenizer: PreTrainedTokenizerBase,
    max_tokens_per_turn: int = 512,
    concurrency: int = 1,
    mean_env_delay: float = 0.0,
) -> Tuple[list[ConversationResult], float]:
    """Run multi-turn benchmark over a set of ShareGPT conversations.

    Each conversation is processed agent_loop style: user turns are sent
    sequentially within a conversation, but multiple conversations can
    run concurrently (controlled by ``concurrency``).

    Between turns, an exponentially-distributed delay simulates
    environment interaction time (reward computation, observation
    processing, etc.).  Set ``mean_env_delay=0`` to disable.

    Args:
        client: SkyRL inference client.
        conversations: Raw ShareGPT conversations.
        tokenizer: Tokenizer (must support apply_chat_template).
        max_tokens_per_turn: Max tokens to generate per assistant turn.
        concurrency: Number of conversations to run in parallel.
        mean_env_delay: Mean inter-turn delay in seconds (exponential
            distribution).  0 disables delays.

    Returns:
        (conversation_results, total_wall_clock_duration)
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def _run_with_limit(conv: ShareGPTConversation) -> ConversationResult:
        async with semaphore:
            return await _run_conversation(
                client,
                conv,
                tokenizer,
                max_tokens_per_turn,
                mean_env_delay,
            )

    benchmark_start = time.monotonic()
    results = list(await asyncio.gather(*[_run_with_limit(c) for c in conversations]))
    duration = time.monotonic() - benchmark_start
    return results, duration


def calculate_multi_turn_results(
    results: list[ConversationResult],
    duration: float,
    percentiles: list[float] | None = None,
) -> BenchmarkResult:
    """Compute aggregate metrics from multi-turn conversation results.

    Maps ConversationResult list to BenchmarkResult by treating each
    conversation as one "request" for throughput calculation.
    """
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
        )

    # Use total_latency per conversation as the E2EL
    e2els = [r.total_latency for r in successful]
    total_input = sum(r.total_input_tokens for r in successful)
    total_output = sum(r.total_output_tokens for r in successful)

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
    )


# ---------------------------------------------------------------------------
# Results calculation and printing
# ---------------------------------------------------------------------------


def calculate_results(
    results: list[RequestResult],
    duration: float,
    percentiles: list[float] | None = None,
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
        inference_router = InferenceRouter(server_urls=server_urls)
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


def run_single_turn_entrypoint(args: argparse.Namespace) -> None:
    """Run single-turn benchmark with the configured dataset and traffic pattern."""
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    requests = get_sample_requests(args, tokenizer)
    client, _server_group, _router = get_inference_client(args)

    async def _run() -> Tuple[list[RequestResult], float]:
        # Warmup
        if args.num_warmup > 0:
            warmup_reqs = requests[: args.num_warmup]
            logger.info(f"Warming up with {len(warmup_reqs)} requests...")
            warmup_tasks = [client.generate(sample_request_to_engine_input(r, tokenizer)) for r in warmup_reqs]
            await asyncio.gather(*warmup_tasks)
            logger.info("Warmup complete.")

        # Benchmark
        return await run_benchmark(
            client=client,
            requests=requests,
            tokenizer=tokenizer,
            request_rate=args.request_rate,
            burstiness=args.burstiness,
        )

    results, duration = asyncio.run(_run())
    bench_result = calculate_results(results, duration)
    print_results(bench_result)
    save_results(bench_result, args)


def run_multi_turn_entrypoint(args: argparse.Namespace) -> None:
    """Run multi-turn benchmark with ShareGPT conversations."""
    if args.dataset_name != "sharegpt":
        raise ValueError("Multi-turn benchmark requires --dataset-name sharegpt")
    if args.dataset_path is None:
        raise ValueError("--dataset-path is required for multi-turn benchmark")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    conversations = load_sharegpt_conversations(
        dataset_path=args.dataset_path,
        num_conversations=args.num_prompts,
        seed=args.seed,
    )
    logger.info(
        f"Loaded {len(conversations)} conversations " f"(avg {np.mean([len(c) for c in conversations]):.1f} turns)"
    )

    client, _server_group, _router = get_inference_client(args)

    async def _run() -> Tuple[list[ConversationResult], float]:
        return await run_multi_turn_benchmark(
            client=client,
            conversations=conversations,
            tokenizer=tokenizer,
            max_tokens_per_turn=args.output_len,
            concurrency=args.concurrency,
            mean_env_delay=args.mean_env_delay,
        )

    results, duration = asyncio.run(_run())
    bench_result = calculate_multi_turn_results(results, duration)
    print_results(bench_result)

    # Print multi-turn specific stats
    successful = [r for r in results if r.success]
    if successful:
        all_turn_latencies = [lat for r in successful for lat in r.per_turn_latencies]
        turns_per_conv = [r.num_turns for r in successful]
        print("Multi-turn Statistics:")
        print(f"  Conversations completed:   {len(successful)}")
        print(f"  Avg turns per conversation: {np.mean(turns_per_conv):.1f}")
        print(f"  Mean per-turn latency (ms): {np.mean(all_turn_latencies) * 1000:.2f}")
        print(f"  P50 per-turn latency (ms):  {np.percentile(all_turn_latencies, 50) * 1000:.2f}")
        print(f"  P99 per-turn latency (ms):  {np.percentile(all_turn_latencies, 99) * 1000:.2f}")
        print()

    save_results(bench_result, args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SkyRL Inference Performance Benchmark")

    # Test mode
    parser.add_argument(
        "--test",
        type=str,
        default="single-turn",
        choices=["single-turn", "multi-turn"],
        help="Benchmark type: single-turn (batch/online) or multi-turn (agent_loop)",
    )

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
        help="Number of requests (single-turn) or conversations (multi-turn)",
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
        help="Mean output token length / max tokens per turn (multi-turn)",
    )
    parser.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="Override output length for ShareGPT single-turn (None = use dataset lengths)",
    )

    # Traffic
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Requests/sec for single-turn (inf = batch mode)",
    )
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="Gamma distribution shape for request arrival (1.0 = Poisson)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent conversations for multi-turn benchmark",
    )
    parser.add_argument(
        "--mean-env-delay",
        type=float,
        default=1.0,
        help="Mean inter-turn environment delay in seconds (exponential distribution, "
        "simulates env processing time between turns). 0 disables delays.",
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
    if args.test == "multi-turn":
        run_multi_turn_entrypoint(args)
    else:
        run_single_turn_entrypoint(args)

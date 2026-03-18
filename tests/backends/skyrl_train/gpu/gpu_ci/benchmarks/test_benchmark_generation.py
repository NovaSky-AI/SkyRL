"""
Batched generation benchmark: old (Ray actors) vs new (HTTP servers) codepath.

Two CI scenarios:
  1. Single-engine, single-turn, GSM8K — verifies basic throughput parity.
  2. Multi-engine, multi-turn, ShareGPT — verifies multi-turn conversation
     throughput with concurrent engines.

Engines run sequentially within each test case so they never compete for GPU
memory.

Run:
    uv run --isolated --extra dev --extra fsdp \
        pytest -s -vv tests/backends/skyrl_train/gpu/gpu_ci/benchmarks/test_benchmark_generation.py
"""

import asyncio
import os

import ray
from ray.util.placement_group import placement_group
from transformers import AutoTokenizer

from skyrl.backends.skyrl_train.inference_engines.inference_engine_client import (
    InferenceEngineClient,
)
from skyrl.backends.skyrl_train.inference_engines.ray_wrapped_inference_engine import (
    create_ray_wrapped_inference_engines,
)
from skyrl.backends.skyrl_train.inference_engines.utils import (
    get_sampling_params_for_backend,
)
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    RemoteInferenceClient,
)
from skyrl.backends.skyrl_train.inference_servers.router import InferenceRouter
from skyrl.backends.skyrl_train.inference_servers.server_group import ServerGroup
from skyrl.backends.skyrl_train.inference_servers.utils import build_vllm_cli_args
from skyrl.benchmarks.benchmark_new_inference import (
    calculate_multi_turn_results,
    calculate_results,
    load_sharegpt_conversations,
    run_multi_turn_benchmark,
)
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.utils import get_ray_pg_ready_with_timeout

MODEL = "Qwen/Qwen2.5-0.5B"

# NOTE (sumanthrh): These thresholds are taken after estimating mean and standard deviation across 5 different runs.
# The regression threshold is estimated using t-distribution prediction interval.
# For n=5 and 95% confidence, t ≈ 2.13, and the sqrt factor is ~1.10 so it's roughly mean - 2 × std.
SCENARIOS = {
    "single_turn_gsm8k": {
        "batch_size": 32,
        "input_len": 128,
        "output_len": 128,
        "regression_threshold": 0.88,
    },
    "multi_turn_sharegpt": {
        "num_conversations": 8,
        "max_tokens_per_turn": 128,
        "concurrency": 4,
        "mean_env_delay": 1.0,
        "regression_threshold": 0.88,
    },
}

NUM_ITERATIONS = 5
WARMUP_ITERATIONS = 2


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _make_config(output_len: int, max_model_len: int, num_engines: int = 1) -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = MODEL
    cfg.generator.sampling_params.temperature = 0.0
    cfg.generator.sampling_params.top_p = 1.0
    cfg.generator.sampling_params.max_generate_length = output_len
    cfg.generator.backend = "vllm"
    cfg.generator.inference_engine.tensor_parallel_size = 1
    cfg.generator.inference_engine.pipeline_parallel_size = 1
    cfg.generator.inference_engine.data_parallel_size = 1
    cfg.generator.engine_init_kwargs = {"max_model_len": max_model_len}
    return cfg


def _make_placement_group(num_gpus: int = 1):
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    pg = placement_group(bundles, strategy="PACK")
    get_ray_pg_ready_with_timeout(pg, timeout=30)
    return pg


# ---------------------------------------------------------------------------
# Engine lifecycle — old codepath (Ray actors)
# ---------------------------------------------------------------------------


def _init_old_engine(cfg: SkyRLTrainConfig, max_model_len: int):
    """Return (client, pg)."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    pg = _make_placement_group()

    engines = create_ray_wrapped_inference_engines(
        num_inference_engines=1,
        tensor_parallel_size=1,
        model_dtype="bfloat16",
        pretrain=MODEL,
        seed=42,
        vllm_v1_disable_multiproc=True,
        enable_prefix_caching=True,
        enforce_eager=True,
        shared_pg=pg,
        gpu_memory_utilization=0.8,
        inference_engine_enable_sleep=True,
        async_engine=True,
        max_num_batched_tokens=8192,
        max_num_seqs=1024,
        tokenizer=tokenizer,
        backend="vllm",
        sleep_level=1,
        engine_init_kwargs={"max_model_len": max_model_len},
    )
    client = InferenceEngineClient(
        engines,
        tokenizer,
        cfg.trainer.policy.model.path,
        cfg.trainer.policy.model.lora,
        cfg.generator.inference_engine,
    )
    asyncio.run(client.wake_up())
    return client, pg


def _teardown_old_engine(client, pg):
    asyncio.run(client.teardown())


# ---------------------------------------------------------------------------
# Engine lifecycle — new codepath (HTTP servers)
# ---------------------------------------------------------------------------


def _init_new_engine(cfg: SkyRLTrainConfig, num_engines: int = 1):
    """Return (client, pg, router, server_group)."""
    pg = _make_placement_group(num_gpus=num_engines)

    server_group = ServerGroup(
        cli_args=build_vllm_cli_args(cfg),
        num_servers=num_engines,
        placement_group=pg,
    )
    server_infos = server_group.start()
    server_urls = [info.url for info in server_infos]

    router = InferenceRouter(server_urls=server_urls)
    proxy_url = router.start()

    client = RemoteInferenceClient(
        proxy_url=proxy_url,
        server_urls=server_urls,
        model_name=MODEL,
    )
    return client, pg, router, server_group


def _teardown_new_engine(client, pg, router, server_group):
    asyncio.run(client.teardown())
    router.shutdown()
    server_group.shutdown()
    ray.util.remove_placement_group(pg)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_comparison(scenario_name: str, old_result, new_result):
    print(f"\n{'=' * 72}")
    print(f"  Scenario: {scenario_name}")
    print(f"{'=' * 72}")
    print(f"  {'Metric':<30} {'Old (Ray)':>15} {'New (HTTP)':>15}")
    print(f"  {'-' * 60}")
    print(f"  {'output_tokens/s':<30} {old_result.output_throughput:>15.1f} {new_result.output_throughput:>15.1f}")
    print(f"  {'requests/s':<30} {old_result.request_throughput:>15.1f} {new_result.request_throughput:>15.1f}")
    print(f"  {'total_time (s)':<30} {old_result.duration_s:>15.2f} {new_result.duration_s:>15.2f}")
    print(f"  {'mean_e2el (ms)':<30} {old_result.mean_e2el_ms:>15.1f} {new_result.mean_e2el_ms:>15.1f}")
    ratio = new_result.output_throughput / old_result.output_throughput if old_result.output_throughput > 0 else 0
    print(f"  {'ratio (new/old)':<30} {ratio:>14.1%}")
    print(f"{'=' * 72}\n")


# ---------------------------------------------------------------------------
# Single-turn benchmark runner (iteration-based for stable measurements)
# ---------------------------------------------------------------------------


def _run_single_turn(
    client,
    prompt_token_ids,
    sampling_params,
    *,
    batch_size: int,
    input_len: int,
    output_len: int,
):
    """Run single-turn benchmark using iteration-based batching.

    Returns a BenchmarkResult from calculate_results.
    """
    import time

    from skyrl.backends.skyrl_train.inference_engines.base import InferenceEngineInput
    from skyrl.benchmarks.benchmark_new_inference import RequestResult

    input_batch = InferenceEngineInput(
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
    )

    async def _run():
        # Warmup
        for _ in range(WARMUP_ITERATIONS):
            await client.generate(input_batch)

        # Timed runs
        all_results = []
        overall_start = time.perf_counter()
        for _ in range(NUM_ITERATIONS):
            iter_start = time.perf_counter()
            output = await client.generate(input_batch)
            elapsed = time.perf_counter() - iter_start
            for ids in output["response_ids"]:
                all_results.append(
                    RequestResult(
                        success=True,
                        latency=elapsed,
                        prompt_len=input_len,
                        output_len=len(ids),
                    )
                )
        duration = time.perf_counter() - overall_start
        return all_results, duration

    results, duration = asyncio.run(_run())
    return calculate_results(results, duration)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBenchmarkRegression:
    """Benchmark regression tests comparing old vs new inference codepaths."""

    def test_single_turn_gsm8k(self, ray_init_fixture):
        """Single-engine, single-turn: old vs new codepath with synthetic prompts.

        Uses batch_size=32, short sequences (128 in, 128 out) to test
        throughput parity.
        """
        scene = SCENARIOS["single_turn_gsm8k"]
        batch_size = scene["batch_size"]
        input_len = scene["input_len"]
        output_len = scene["output_len"]
        max_model_len = input_len + output_len + 64

        cfg = _make_config(output_len, max_model_len)

        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        sampling_params = get_sampling_params_for_backend("vllm", cfg.generator.sampling_params)
        sampling_params["max_tokens"] = output_len
        sampling_params["temperature"] = 0.0

        # Generate synthetic prompts (similar to GSM8K length profile)
        from tests.backends.skyrl_train.gpu.gpu_ci.benchmarks.benchmark_utils import (
            generate_synthetic_prompt_token_ids,
        )

        prompt_token_ids = generate_synthetic_prompt_token_ids(tokenizer, input_len, batch_size)

        # --- old codepath (Ray actors) ---
        old_client, old_pg = _init_old_engine(cfg, max_model_len)
        try:
            old_result = _run_single_turn(
                old_client,
                prompt_token_ids,
                sampling_params,
                batch_size=batch_size,
                input_len=input_len,
                output_len=output_len,
            )
        finally:
            _teardown_old_engine(old_client, old_pg)

        # --- new codepath (HTTP servers) ---
        new_client, new_pg, router, server_group = _init_new_engine(cfg)
        try:
            new_result = _run_single_turn(
                new_client,
                prompt_token_ids,
                sampling_params,
                batch_size=batch_size,
                input_len=input_len,
                output_len=output_len,
            )
        finally:
            _teardown_new_engine(new_client, new_pg, router, server_group)

        # --- compare ---
        _print_comparison("single_turn_gsm8k", old_result, new_result)

        assert old_result.output_throughput > 0, "Old codepath produced zero output tok/s"
        assert new_result.output_throughput > 0, "New codepath produced zero output tok/s"

        threshold = scene["regression_threshold"]
        ratio = new_result.output_throughput / old_result.output_throughput
        assert ratio >= threshold, (
            f"Regression: new output tok/s ({new_result.output_throughput:.1f}) is "
            f"{ratio:.1%} of old ({old_result.output_throughput:.1f}), "
            f"threshold is {threshold:.0%}"
        )

    def test_multi_turn_sharegpt(self, ray_init_fixture):
        """Multi-engine, multi-turn: old vs new codepath with ShareGPT conversations.

        Uses synthetic multi-turn conversations to test the agent_loop pattern
        where each user turn is sent sequentially, building up chat context.
        """
        scene = SCENARIOS["multi_turn_sharegpt"]
        num_conversations = scene["num_conversations"]
        max_tokens_per_turn = scene["max_tokens_per_turn"]
        concurrency = scene["concurrency"]
        max_model_len = 2048

        cfg = _make_config(max_tokens_per_turn, max_model_len)
        tokenizer = AutoTokenizer.from_pretrained(MODEL)

        # Load ShareGPT conversations if available, otherwise use synthetic
        sharegpt_path = os.environ.get("SHAREGPT_PATH")
        if sharegpt_path and os.path.exists(sharegpt_path):
            conversations = load_sharegpt_conversations(
                dataset_path=sharegpt_path,
                num_conversations=num_conversations,
                min_turns=4,  # at least 2 user turns
                seed=42,
            )
        else:
            # Synthetic multi-turn conversations for CI
            conversations = []
            for i in range(num_conversations):
                conversations.append(
                    [
                        {"from": "human", "value": f"What is {i+1} + {i+2}?"},
                        {"from": "gpt", "value": f"The answer is {2*i+3}."},
                        {"from": "human", "value": "Can you explain your reasoning step by step?"},
                        {"from": "gpt", "value": "Sure, let me break it down..."},
                        {"from": "human", "value": "Now multiply the result by 3."},
                        {"from": "gpt", "value": "The result would be..."},
                    ]
                )

        # --- old codepath (Ray actors) ---
        old_client, old_pg = _init_old_engine(cfg, max_model_len)
        try:
            old_conv_results, old_duration = asyncio.run(
                run_multi_turn_benchmark(
                    client=old_client,
                    conversations=conversations,
                    tokenizer=tokenizer,
                    max_tokens_per_turn=max_tokens_per_turn,
                    concurrency=concurrency,
                    mean_env_delay=scene["mean_env_delay"],
                )
            )
            old_result = calculate_multi_turn_results(old_conv_results, old_duration)
        finally:
            _teardown_old_engine(old_client, old_pg)

        # --- new codepath (HTTP servers) ---
        new_client, new_pg, router, server_group = _init_new_engine(cfg)
        try:
            new_conv_results, new_duration = asyncio.run(
                run_multi_turn_benchmark(
                    client=new_client,
                    conversations=conversations,
                    tokenizer=tokenizer,
                    max_tokens_per_turn=max_tokens_per_turn,
                    concurrency=concurrency,
                    mean_env_delay=scene["mean_env_delay"],
                )
            )
            new_result = calculate_multi_turn_results(new_conv_results, new_duration)
        finally:
            _teardown_new_engine(new_client, new_pg, router, server_group)

        # --- compare ---
        _print_comparison("multi_turn_sharegpt", old_result, new_result)

        assert old_result.output_throughput > 0, "Old codepath produced zero output tok/s"
        assert new_result.output_throughput > 0, "New codepath produced zero output tok/s"

        threshold = scene["regression_threshold"]
        ratio = new_result.output_throughput / old_result.output_throughput
        assert ratio >= threshold, (
            f"Regression: new output tok/s ({new_result.output_throughput:.1f}) is "
            f"{ratio:.1%} of old ({old_result.output_throughput:.1f}), "
            f"threshold is {threshold:.0%}"
        )

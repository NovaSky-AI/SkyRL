"""
Performance benchmarks for the new inference codepath (HTTP servers).

Two CI scenarios with absolute thresholds:
  1. Single-engine, single-turn, GSM8K — verifies basic throughput.
  2. Multi-engine, multi-turn, ShareGPT — verifies multi-turn conversation
     throughput with concurrent engines.

Run:
    SHAREGPT_PATH=/path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
    uv run --isolated --extra dev --extra fsdp \
        pytest -s -vv tests/backends/skyrl_train/gpu/gpu_ci/benchmarks/test_benchmark_generation.py
"""

import asyncio
import os

from transformers import AutoTokenizer

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
from tests.backends.skyrl_train.gpu.gpu_ci.benchmarks.benchmark_utils import (
    generate_synthetic_prompt_token_ids,
)

MODEL = "Qwen/Qwen2.5-0.5B"

NUM_ITERATIONS = 5
WARMUP_ITERATIONS = 2

# ---------------------------------------------------------------------------
# Absolute performance thresholds
#
# Computed from 2 benchmark runs on H100 using mean ± 2×std, then rounded
# generously to avoid CI flakiness. Thresholds should be recalibrated if
# the CI GPU type changes (e.g. L4 vs H100).
# ---------------------------------------------------------------------------

SCENARIOS = {
    "single_turn_gsm8k": {
        "batch_size": 256,
        "input_len": 128,
        "output_len": 128,
        # Minimum throughput thresholds
        "min_output_throughput": 3500.0,  # tok/s (observed ~3944, threshold ~mean-2σ rounded down)
        # Maximum latency thresholds (ms)
        "max_mean_e2el_ms": 9500.0,  # ms (observed ~8276, threshold ~mean+2σ rounded up)
        "max_p99_e2el_ms": 10000.0,  # ms (observed ~8509, threshold ~mean+2σ rounded up)
    },
    "multi_turn_sharegpt": {
        "num_conversations": 64,
        "max_tokens_per_turn": 128,
        "concurrency": 4,
        "mean_env_delay": 1.0,
        # Minimum throughput thresholds
        "min_output_throughput": 180.0,  # tok/s (observed ~204, threshold ~mean-2σ rounded down)
        # Maximum latency thresholds (ms)
        "max_mean_e2el_ms": 12000.0,  # ms (observed ~9294, threshold ~mean+2σ rounded up)
        "max_p90_e2el_ms": 22000.0,  # ms (observed ~17313, threshold ~mean+2σ rounded up)
        "max_p99_e2el_ms": 35000.0,  # ms (observed ~28703, threshold ~mean+2σ rounded up)
    },
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _make_config(output_len: int, max_model_len: int) -> SkyRLTrainConfig:
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


# ---------------------------------------------------------------------------
# Engine lifecycle — new codepath (HTTP servers)
# ---------------------------------------------------------------------------


def _init_new_engine(cfg: SkyRLTrainConfig, num_engines: int = 1):
    """Return (client, router, server_group)."""
    server_group = ServerGroup(
        cli_args=build_vllm_cli_args(cfg),
        num_servers=num_engines,
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
    return client, router, server_group


def _teardown_new_engine(client, router, server_group):
    asyncio.run(client.teardown())
    router.shutdown()
    server_group.shutdown()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_result(scenario_name: str, result):
    print(f"\n{'=' * 72}")
    print(f"  Scenario: {scenario_name}")
    print(f"{'=' * 72}")
    print(f"  {'Metric':<35} {'Value':>15}")
    print(f"  {'-' * 50}")
    print(f"  {'output_tokens/s':<35} {result.output_throughput:>15.1f}")
    print(f"  {'requests/s':<35} {result.request_throughput:>15.1f}")
    print(f"  {'total_time (s)':<35} {result.duration_s:>15.2f}")
    print(f"  {'mean_e2el (ms)':<35} {result.mean_e2el_ms:>15.1f}")
    print(f"  {'median_e2el (ms)':<35} {result.median_e2el_ms:>15.1f}")
    for p, val in result.percentiles_e2el_ms:
        print(f"  {f'p{p}_e2el (ms)':<35} {val:>15.1f}")
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
    """Performance benchmarks for the new inference codepath (absolute thresholds)."""

    def test_single_turn_gsm8k(self, ray_init_fixture):
        """Single-engine, single-turn: new codepath with synthetic prompts.

        Uses batch_size=256, short sequences (128 in, 128 out) to test throughput.
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

        prompt_token_ids = generate_synthetic_prompt_token_ids(tokenizer, input_len, batch_size)

        # --- new codepath (HTTP servers) ---
        client, router, server_group = _init_new_engine(cfg)
        try:
            result = _run_single_turn(
                client,
                prompt_token_ids,
                sampling_params,
                batch_size=batch_size,
                input_len=input_len,
                output_len=output_len,
            )
        finally:
            _teardown_new_engine(client, router, server_group)

        _print_result("single_turn_gsm8k", result)

        # --- assert absolute thresholds ---
        assert result.output_throughput > 0, "New codepath produced zero output tok/s"
        assert result.output_throughput >= scene["min_output_throughput"], (
            f"Output throughput {result.output_throughput:.1f} tok/s below minimum "
            f"{scene['min_output_throughput']:.1f} tok/s"
        )
        assert result.mean_e2el_ms <= scene["max_mean_e2el_ms"], (
            f"Mean E2E latency {result.mean_e2el_ms:.1f} ms exceeds maximum " f"{scene['max_mean_e2el_ms']:.1f} ms"
        )
        for p, val in result.percentiles_e2el_ms:
            if p == 99:
                assert val <= scene["max_p99_e2el_ms"], (
                    f"P99 E2E latency {val:.1f} ms exceeds maximum " f"{scene['max_p99_e2el_ms']:.1f} ms"
                )

    def test_multi_turn_sharegpt(self, ray_init_fixture):
        """Multi-engine, multi-turn: new codepath with ShareGPT conversations.

        Uses real ShareGPT conversations to test the agent_loop pattern
        where each user turn is sent sequentially, building up chat context.
        """
        scene = SCENARIOS["multi_turn_sharegpt"]
        max_tokens_per_turn = scene["max_tokens_per_turn"]
        max_model_len = 2048

        cfg = _make_config(max_tokens_per_turn, max_model_len)
        tokenizer = AutoTokenizer.from_pretrained(MODEL)

        sharegpt_path = os.environ.get("SHAREGPT_PATH")
        assert sharegpt_path, "SHAREGPT_PATH env var must be set"
        assert os.path.exists(sharegpt_path), f"ShareGPT dataset not found at {sharegpt_path}"

        conversations = load_sharegpt_conversations(
            dataset_path=sharegpt_path,
            num_conversations=scene["num_conversations"],
            min_turns=4,
            seed=42,
        )

        # --- new codepath (HTTP servers) ---
        client, router, server_group = _init_new_engine(cfg)
        try:
            conv_results, duration = asyncio.run(
                run_multi_turn_benchmark(
                    client=client,
                    conversations=conversations,
                    tokenizer=tokenizer,
                    max_tokens_per_turn=max_tokens_per_turn,
                    concurrency=scene["concurrency"],
                    mean_env_delay=scene["mean_env_delay"],
                )
            )
            result = calculate_multi_turn_results(conv_results, duration)
        finally:
            _teardown_new_engine(client, router, server_group)

        _print_result("multi_turn_sharegpt", result)

        # --- assert absolute thresholds ---
        assert result.output_throughput > 0, "New codepath produced zero output tok/s"
        assert result.output_throughput >= scene["min_output_throughput"], (
            f"Output throughput {result.output_throughput:.1f} tok/s below minimum "
            f"{scene['min_output_throughput']:.1f} tok/s"
        )
        assert result.mean_e2el_ms <= scene["max_mean_e2el_ms"], (
            f"Mean E2E latency {result.mean_e2el_ms:.1f} ms exceeds maximum " f"{scene['max_mean_e2el_ms']:.1f} ms"
        )
        for p, val in result.percentiles_e2el_ms:
            if p == 90:
                assert val <= scene["max_p90_e2el_ms"], (
                    f"P90 E2E latency {val:.1f} ms exceeds maximum " f"{scene['max_p90_e2el_ms']:.1f} ms"
                )
            if p == 99:
                assert val <= scene["max_p99_e2el_ms"], (
                    f"P99 E2E latency {val:.1f} ms exceeds maximum " f"{scene['max_p99_e2el_ms']:.1f} ms"
                )

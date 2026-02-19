"""
Batched generation benchmark: old (Ray actors) vs new (HTTP servers) codepath.

Both engines are constructed explicitly in-process — no subprocess, no env var
toggling.  They run sequentially within each parametrized test case so they
never compete for GPU memory.

Run:
    uv run --isolated --extra dev --extra vllm \
        pytest -s tests/backends/skyrl_train/gpu/gpu_ci/benchmarks/test_benchmark_generation.py -m vllm
"""

import asyncio

import pytest
import ray
from ray.util.placement_group import placement_group
from transformers import AutoTokenizer

from skyrl.train.config import SkyRLConfig
from skyrl.backends.skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl.backends.skyrl_train.inference_engines.ray_wrapped_inference_engine import (
    create_ray_wrapped_inference_engines,
)
from skyrl.backends.skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import RemoteInferenceClient
from skyrl.backends.skyrl_train.inference_servers.router import InferenceRouter
from skyrl.backends.skyrl_train.inference_servers.server_group import ServerGroup
from skyrl.backends.skyrl_train.inference_servers.utils import build_vllm_cli_args
from skyrl.backends.skyrl_train.utils import get_ray_pg_ready_with_timeout

from tests.backends.skyrl_train.gpu.gpu_ci.benchmarks.benchmark_utils import (
    BenchmarkResult,
    generate_synthetic_prompt_token_ids,
    run_benchmark,
)

MODEL = "Qwen/Qwen2.5-0.5B"

SCENARIOS = {
    "short": {"input_len": 128, "output_len": 128, "regression_threshold": 0.86},
    "long": {"input_len": 128, "output_len": 2048, "regression_threshold": 0.96},
}

NUM_ITERATIONS = 5
WARMUP_ITERATIONS = 2


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _make_config(output_len: int, max_model_len: int) -> SkyRLConfig:
    cfg = SkyRLConfig()
    cfg.trainer.policy.model.path = MODEL
    cfg.generator.sampling_params.temperature = 0.0
    cfg.generator.sampling_params.top_p = 1.0
    cfg.generator.sampling_params.max_generate_length = output_len
    cfg.generator.backend = "vllm"
    cfg.generator.inference_engine_tensor_parallel_size = 1
    cfg.generator.inference_engine_pipeline_parallel_size = 1
    cfg.generator.inference_engine_data_parallel_size = 1
    cfg.generator.engine_init_kwargs = {"max_model_len": max_model_len}
    return cfg


def _make_placement_group():
    pg = placement_group([{"GPU": 1, "CPU": 1}], strategy="PACK")
    get_ray_pg_ready_with_timeout(pg, timeout=30)
    return pg


# ---------------------------------------------------------------------------
# Engine lifecycle — old codepath (Ray actors)
# ---------------------------------------------------------------------------


def _init_old_engine(cfg: SkyRLConfig, max_model_len: int):
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
    client = InferenceEngineClient(engines, tokenizer, cfg)
    asyncio.run(client.wake_up())
    return client, pg


def _teardown_old_engine(client, pg):
    asyncio.run(client.teardown())


# ---------------------------------------------------------------------------
# Engine lifecycle — new codepath (HTTP servers)
# ---------------------------------------------------------------------------


def _init_new_engine(cfg: SkyRLConfig):
    """Return (client, pg, router, server_group)."""
    pg = _make_placement_group()

    server_group = ServerGroup(
        cli_args=build_vllm_cli_args(cfg),
        num_servers=1,
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


def _print_comparison(old: BenchmarkResult, new: BenchmarkResult):
    hdr = (
        f"scenario={old.scenario}  batch_size={old.batch_size}  input_len={old.input_len}  output_len={old.output_len}"
    )
    print(f"\n{'=' * 72}")
    print(f"  {hdr}")
    print(f"{'=' * 72}")
    print(f"  {'Metric':<30} {'Old (Ray)':>15} {'New (HTTP)':>15}")
    print(f"  {'-' * 60}")
    print(f"  {'output_tokens/s':<30} {old.output_tokens_per_second:>15.1f} {new.output_tokens_per_second:>15.1f}")
    print(f"  {'requests/s':<30} {old.requests_per_second:>15.1f} {new.requests_per_second:>15.1f}")
    print(f"  {'total_time (s)':<30} {old.elapsed_time:>15.2f} {new.elapsed_time:>15.2f}")
    print(f"  {'avg_latency (s)':<30} {old.avg_latency:>15.3f} {new.avg_latency:>15.3f}")
    print(f"  {'p50_latency (s)':<30} {old.p50_latency:>15.3f} {new.p50_latency:>15.3f}")
    print(f"  {'p90_latency (s)':<30} {old.p90_latency:>15.3f} {new.p90_latency:>15.3f}")
    print(f"  {'p99_latency (s)':<30} {old.p99_latency:>15.3f} {new.p99_latency:>15.3f}")
    ratio = new.output_tokens_per_second / old.output_tokens_per_second if old.output_tokens_per_second > 0 else 0
    print(f"  {'ratio (new/old)':<30} {ratio:>14.1%}")
    print(f"{'=' * 72}\n")


# ---------------------------------------------------------------------------
# Benchmark runner (shared by both codepaths)
# ---------------------------------------------------------------------------


def _run_and_collect(
    client,
    prompt_token_ids,
    sampling_params,
    *,
    scenario: str,
    batch_size: int,
    input_len: int,
    output_len: int,
    codepath: str,
) -> BenchmarkResult:
    elapsed, total_output_tokens, latencies = asyncio.run(
        run_benchmark(client, prompt_token_ids, sampling_params, NUM_ITERATIONS, WARMUP_ITERATIONS)
    )
    return BenchmarkResult(
        scenario=scenario,
        batch_size=batch_size,
        input_len=input_len,
        output_len=output_len,
        codepath=codepath,
        num_iterations=NUM_ITERATIONS,
        elapsed_time=elapsed,
        total_requests=NUM_ITERATIONS * batch_size,
        total_input_tokens=NUM_ITERATIONS * batch_size * input_len,
        total_output_tokens=total_output_tokens,
        per_iteration_latencies=latencies,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.vllm
@pytest.mark.benchmark
@pytest.mark.parametrize(
    "scenario,batch_size",
    [
        ("short", 8),
        ("short", 32),
        ("long", 4),
        ("long", 8),
    ],
    ids=["short_bs8", "short_bs32", "long_bs4", "long_bs8"],
)
def test_benchmark_regression(ray_init_fixture, scenario: str, batch_size: int):
    """
    Run old and new codepaths back-to-back and assert the new path does not
    regress beyond REGRESSION_THRESHOLD of the old path's output token throughput.
    """
    scene = SCENARIOS[scenario]
    input_len = scene["input_len"]
    output_len = scene["output_len"]
    max_model_len = input_len + output_len + 64

    cfg = _make_config(output_len, max_model_len)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    sampling_params = get_sampling_params_for_backend("vllm", cfg.generator.sampling_params)
    sampling_params["max_tokens"] = output_len
    sampling_params["temperature"] = 0.0

    prompt_token_ids = generate_synthetic_prompt_token_ids(tokenizer, input_len, batch_size)

    common = dict(scenario=scenario, batch_size=batch_size, input_len=input_len, output_len=output_len)

    # --- old codepath (Ray actors) ---
    old_client, old_pg = _init_old_engine(cfg, max_model_len)
    try:
        old_result = _run_and_collect(old_client, prompt_token_ids, sampling_params, codepath="old", **common)
    finally:
        _teardown_old_engine(old_client, old_pg)

    # --- new codepath (HTTP servers) ---
    new_client, new_pg, router, server_group = _init_new_engine(cfg)
    try:
        new_result = _run_and_collect(new_client, prompt_token_ids, sampling_params, codepath="new", **common)
    finally:
        _teardown_new_engine(new_client, new_pg, router, server_group)

    # --- compare ---
    _print_comparison(old_result, new_result)

    assert old_result.output_tokens_per_second > 0, "Old codepath produced zero output tokens/s"
    assert new_result.output_tokens_per_second > 0, "New codepath produced zero output tokens/s"

    threshold = scene["regression_threshold"]
    ratio = new_result.output_tokens_per_second / old_result.output_tokens_per_second
    assert ratio >= threshold, (
        f"Regression: new output tok/s ({new_result.output_tokens_per_second:.1f}) is "
        f"{ratio:.1%} of old ({old_result.output_tokens_per_second:.1f}), "
        f"threshold is {threshold:.0%}"
    )

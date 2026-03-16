"""Unit tests for benchmark_new_inference pure functions.

These tests cover metrics calculation, request translation, and dataset loading
without requiring GPU or inference servers.
"""

import argparse
import asyncio
from unittest.mock import AsyncMock

import numpy as np
import pytest
from transformers import AutoTokenizer
from vllm.benchmarks.datasets import SampleRequest

from skyrl.benchmarks.benchmark_new_inference import (
    BenchmarkResult,
    RequestResult,
    calculate_results,
    get_sample_requests,
    print_results,
    run_benchmark,
    sample_request_to_engine_input,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")


@pytest.fixture
def sample_request():
    return SampleRequest(
        prompt="Hello, how are you?",
        prompt_len=5,
        expected_output_len=32,
    )


@pytest.fixture
def successful_results():
    return [
        RequestResult(success=True, latency=0.5, prompt_len=100, output_len=50),
        RequestResult(success=True, latency=1.0, prompt_len=200, output_len=100),
        RequestResult(success=True, latency=1.5, prompt_len=150, output_len=75),
        RequestResult(success=True, latency=2.0, prompt_len=300, output_len=120),
    ]


@pytest.fixture
def mixed_results():
    return [
        RequestResult(success=True, latency=0.5, prompt_len=100, output_len=50),
        RequestResult(success=False, latency=0.1, prompt_len=100, output_len=0, error="timeout"),
        RequestResult(success=True, latency=1.0, prompt_len=200, output_len=100),
    ]


# ---------------------------------------------------------------------------
# sample_request_to_engine_input
# ---------------------------------------------------------------------------


class TestSampleRequestToEngineInput:
    def test_basic_conversion(self, tokenizer, sample_request):
        engine_input = sample_request_to_engine_input(sample_request, tokenizer)

        assert "prompt_token_ids" in engine_input
        assert "sampling_params" in engine_input
        assert len(engine_input["prompt_token_ids"]) == 1
        assert isinstance(engine_input["prompt_token_ids"][0], list)
        assert all(isinstance(t, int) for t in engine_input["prompt_token_ids"][0])
        assert engine_input["sampling_params"]["max_tokens"] == 32

    def test_token_ids_match_tokenizer(self, tokenizer, sample_request):
        engine_input = sample_request_to_engine_input(sample_request, tokenizer)
        expected_ids = tokenizer.encode(sample_request.prompt)
        assert engine_input["prompt_token_ids"][0] == expected_ids

    def test_output_len_mapping(self, tokenizer):
        request = SampleRequest(prompt="test", prompt_len=1, expected_output_len=256)
        engine_input = sample_request_to_engine_input(request, tokenizer)
        assert engine_input["sampling_params"]["max_tokens"] == 256


# ---------------------------------------------------------------------------
# calculate_results
# ---------------------------------------------------------------------------


class TestCalculateResults:
    def test_all_successful(self, successful_results):
        result = calculate_results(successful_results, duration=5.0)

        assert result.completed == 4
        assert result.failed == 0
        assert result.total_input_tokens == 750  # 100+200+150+300
        assert result.total_output_tokens == 345  # 50+100+75+120
        assert result.duration_s == 5.0
        assert result.request_throughput == pytest.approx(0.8)  # 4/5
        assert result.output_throughput == pytest.approx(69.0)  # 345/5
        assert result.total_token_throughput == pytest.approx(219.0)  # 1095/5

    def test_latency_statistics(self, successful_results):
        result = calculate_results(successful_results, duration=5.0)
        latencies = [0.5, 1.0, 1.5, 2.0]

        assert result.mean_e2el_ms == pytest.approx(np.mean(latencies) * 1000)
        assert result.median_e2el_ms == pytest.approx(np.median(latencies) * 1000)
        assert result.std_e2el_ms == pytest.approx(np.std(latencies) * 1000)

    def test_percentiles(self, successful_results):
        result = calculate_results(successful_results, duration=5.0, percentiles=[50, 99])
        assert len(result.percentiles_e2el_ms) == 2
        assert result.percentiles_e2el_ms[0][0] == 50
        assert result.percentiles_e2el_ms[1][0] == 99

    def test_mixed_success_failure(self, mixed_results):
        result = calculate_results(mixed_results, duration=3.0)

        assert result.completed == 2
        assert result.failed == 1
        assert result.total_input_tokens == 300  # only successful
        assert result.total_output_tokens == 150

    def test_all_failed(self):
        results = [
            RequestResult(success=False, latency=0.1, prompt_len=100, output_len=0, error="err"),
            RequestResult(success=False, latency=0.2, prompt_len=100, output_len=0, error="err"),
        ]
        result = calculate_results(results, duration=1.0)

        assert result.completed == 0
        assert result.failed == 2
        assert result.request_throughput == 0.0
        assert result.mean_e2el_ms == 0.0
        assert result.percentiles_e2el_ms == []

    def test_empty_results(self):
        result = calculate_results([], duration=1.0)
        assert result.completed == 0
        assert result.failed == 0

    def test_single_result(self):
        results = [RequestResult(success=True, latency=0.5, prompt_len=100, output_len=50)]
        result = calculate_results(results, duration=1.0)
        assert result.completed == 1
        assert result.mean_e2el_ms == pytest.approx(500.0)
        assert result.std_e2el_ms == pytest.approx(0.0)

    def test_default_percentiles(self, successful_results):
        result = calculate_results(successful_results, duration=5.0)
        # Default percentiles: [50, 75, 90, 95, 99]
        assert len(result.percentiles_e2el_ms) == 5
        percentile_labels = [p for p, _ in result.percentiles_e2el_ms]
        assert percentile_labels == [50, 75, 90, 95, 99]


# ---------------------------------------------------------------------------
# get_sample_requests
# ---------------------------------------------------------------------------


class TestGetSampleRequests:
    def test_random_dataset(self, tokenizer):
        args = argparse.Namespace(
            dataset_name="random",
            num_prompts=5,
            input_len=64,
            output_len=32,
            seed=42,
        )
        requests = get_sample_requests(args, tokenizer)

        assert len(requests) == 5
        for r in requests:
            assert isinstance(r, SampleRequest)
            assert isinstance(r.prompt, str)
            assert r.prompt_len > 0
            assert r.expected_output_len > 0

    def test_sharegpt_missing_path(self, tokenizer):
        args = argparse.Namespace(
            dataset_name="sharegpt",
            dataset_path=None,
            num_prompts=5,
            seed=0,
        )
        with pytest.raises(ValueError, match="--dataset-path is required"):
            get_sample_requests(args, tokenizer)

    def test_gsm8k_dataset(self, tokenizer):
        args = argparse.Namespace(
            dataset_name="gsm8k",
            num_prompts=5,
            output_len=64,
            seed=42,
        )
        requests = get_sample_requests(args, tokenizer)

        assert len(requests) == 5
        for r in requests:
            assert isinstance(r, SampleRequest)
            assert isinstance(r.prompt, str)
            assert r.prompt_len > 0
            assert r.expected_output_len == 64

    def test_gsm8k_dataset_uses_answer_length(self, tokenizer):
        args = argparse.Namespace(
            dataset_name="gsm8k",
            num_prompts=3,
            output_len=None,
            seed=0,
        )
        requests = get_sample_requests(args, tokenizer)

        assert len(requests) == 3
        # With output_len=None, expected_output_len comes from answer token count
        for r in requests:
            assert r.expected_output_len > 0

    def test_unknown_dataset(self, tokenizer):
        args = argparse.Namespace(dataset_name="nonexistent")
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_sample_requests(args, tokenizer)


# ---------------------------------------------------------------------------
# print_results (smoke test)
# ---------------------------------------------------------------------------


class TestPrintResults:
    def test_print_does_not_raise(self, successful_results, capsys):
        result = calculate_results(successful_results, duration=5.0)
        print_results(result)
        captured = capsys.readouterr()
        assert "Benchmark Results" in captured.out
        assert "Request throughput" in captured.out
        assert "Mean E2EL" in captured.out

    def test_print_all_failed(self, capsys):
        result = calculate_results([], duration=1.0)
        print_results(result)
        captured = capsys.readouterr()
        assert "Completed requests:        0" in captured.out


# ---------------------------------------------------------------------------
# run_benchmark (with mocked client)
# ---------------------------------------------------------------------------


class TestRunBenchmark:
    def test_batch_mode(self, tokenizer):
        """Test batch mode (request_rate=inf) with a mock client."""
        mock_client = AsyncMock()
        mock_client.generate.return_value = {
            "responses": ["output text"],
            "response_ids": [[1, 2, 3, 4, 5]],
            "stop_reasons": ["stop"],
            "response_logprobs": None,
        }

        requests = [
            SampleRequest(prompt="Hello", prompt_len=3, expected_output_len=16),
            SampleRequest(prompt="World", prompt_len=3, expected_output_len=16),
        ]

        results, duration = asyncio.run(
            run_benchmark(
                client=mock_client,
                requests=requests,
                tokenizer=tokenizer,
                request_rate=float("inf"),
            )
        )

        assert len(results) == 2
        assert all(r.success for r in results)
        assert all(r.output_len == 5 for r in results)
        assert duration > 0
        assert mock_client.generate.call_count == 2

    def test_batch_mode_with_failure(self, tokenizer):
        """Test that exceptions in generate are captured as failed results."""
        mock_client = AsyncMock()
        mock_client.generate.side_effect = [
            {
                "responses": ["ok"],
                "response_ids": [[1, 2]],
                "stop_reasons": ["stop"],
                "response_logprobs": None,
            },
            RuntimeError("connection failed"),
        ]

        requests = [
            SampleRequest(prompt="Hello", prompt_len=3, expected_output_len=16),
            SampleRequest(prompt="World", prompt_len=3, expected_output_len=16),
        ]

        results, duration = asyncio.run(
            run_benchmark(
                client=mock_client,
                requests=requests,
                tokenizer=tokenizer,
                request_rate=float("inf"),
            )
        )

        assert len(results) == 2
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        assert len(successful) == 1
        assert len(failed) == 1
        assert "connection failed" in failed[0].error

    def test_online_mode(self, tokenizer):
        """Test online mode (finite request_rate) with a mock client."""
        mock_client = AsyncMock()
        mock_client.generate.return_value = {
            "responses": ["output"],
            "response_ids": [[10, 20, 30]],
            "stop_reasons": ["stop"],
            "response_logprobs": None,
        }

        requests = [
            SampleRequest(prompt="Test", prompt_len=2, expected_output_len=8),
            SampleRequest(prompt="Data", prompt_len=2, expected_output_len=8),
        ]

        # Use a high request rate to keep the test fast
        results, duration = asyncio.run(
            run_benchmark(
                client=mock_client,
                requests=requests,
                tokenizer=tokenizer,
                request_rate=1000.0,
            )
        )

        assert len(results) == 2
        assert all(r.success for r in results)
        assert all(r.output_len == 3 for r in results)


# ---------------------------------------------------------------------------
# RequestResult / BenchmarkResult dataclass sanity
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_request_result_defaults(self):
        r = RequestResult(success=True, latency=1.0, prompt_len=10, output_len=5)
        assert r.error == ""

    def test_request_result_with_error(self):
        r = RequestResult(success=False, latency=0.1, prompt_len=10, output_len=0, error="boom")
        assert r.error == "boom"
        assert not r.success

    def test_benchmark_result_fields(self):
        r = BenchmarkResult(
            completed=10,
            failed=2,
            total_input_tokens=1000,
            total_output_tokens=500,
            duration_s=5.0,
            request_throughput=2.0,
            output_throughput=100.0,
            total_token_throughput=300.0,
            mean_e2el_ms=250.0,
            median_e2el_ms=240.0,
            std_e2el_ms=30.0,
            percentiles_e2el_ms=[(50, 240.0), (99, 490.0)],
        )
        assert r.completed == 10
        assert r.percentiles_e2el_ms[1] == (99, 490.0)

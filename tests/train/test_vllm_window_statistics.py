"""Tests for vLLM counter windows and merged latency histograms."""

import math

import pytest

from skyrl.train.utils.vllm_window_statistics import (
    WindowStatistics,
    histogram_quantile,
)


def test_histogram_quantiles():
    buckets = {1.0: 2, 2.0: 8, math.inf: 10}
    assert histogram_quantile(0.5, buckets) == pytest.approx(1.5)
    assert histogram_quantile(0.9, buckets) == 2
    assert histogram_quantile(0.5, {1: 0, math.inf: 0}) is None
    assert histogram_quantile(0.5, {1: 3, 2: 2, math.inf: 4}) is None


def test_window_omits_missing_and_reset_counters():
    result = WindowStatistics.between(
        {"tokens_total": 10, "bad_total": 20}, {"tokens_total": 25, "bad_total": 1, "new_total": 4}, 3
    )
    assert result.deltas == {"tokens_total": 15}
    assert not result.valid
    assert result.duration_seconds == 3


def test_request_tpot_and_ttft_are_separate():
    current = {}
    for name, count, total in [
        ("time_to_first_token_seconds", 10, 12),
        ("request_time_per_output_token_seconds", 4, 2),
    ]:
        base = "ray_vllm_" + name
        current.update(
            {
                base + "_count": count,
                base + "_sum": total,
                base + "_bucket::1": count / 2,
                base + "_bucket::2": count,
                base + "_bucket::+Inf": count,
            }
        )
    result = WindowStatistics.between(dict.fromkeys(current, 0), current, 5)
    metrics = result.latency_metrics("vllm/")
    assert metrics["vllm/ttft_seconds_avg"] == 1.2
    assert metrics["vllm/request_tpot_seconds_avg"] == 0.5
    assert metrics["vllm/request_tpot_seconds_p50"] == 1
    assert metrics["vllm/ttft_seconds_p90"] == pytest.approx(1.8)


def test_lazy_histogram_buckets_use_confirmed_empty_baseline():
    base = "ray_vllm_time_to_first_token_seconds"
    previous = {base + "_count": 0, base + "_sum": 0}
    current = {
        base + "_count": 2,
        base + "_sum": 0.15,
        base + "_bucket::0.1": 1,
        base + "_bucket::0.2": 2,
        base + "_bucket::+Inf": 2,
    }
    metrics = WindowStatistics.between(previous, current, 1).latency_metrics("vllm/")
    assert metrics["vllm/ttft_seconds_avg"] == pytest.approx(0.075)
    assert metrics["vllm/ttft_seconds_p50"] == pytest.approx(0.1)
    assert metrics["vllm/ttft_seconds_p90"] == pytest.approx(0.18)
    # Absence without an explicit empty count is unknown.
    assert not WindowStatistics.between({}, current, 1).latency_metrics("vllm/")

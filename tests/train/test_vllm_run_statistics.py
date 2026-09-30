"""Tests for weighted summaries and checkpoint restoration."""

import pytest

from skyrl.train.utils.vllm_run_statistics import RunStatistics
from skyrl.train.utils.vllm_window_statistics import WindowStatistics


def test_weighted_rates_and_cache_hits_survive_resume():
    counter = "ray_vllm_generation_tokens_total"
    queries = "ray_vllm_external_prefix_cache_queries_total"
    hits = "ray_vllm_external_prefix_cache_hits_total"
    stats = RunStatistics()
    stats.add("train", WindowStatistics(2, {counter: 100, queries: 10, hits: 5}))
    restored = RunStatistics()
    restored.load_state_dict(stats.state_dict())
    restored.add("train", WindowStatistics(8, {counter: 200, queries: 90, hits: 9}))
    summary = restored.summary()
    assert summary["vllm_run/train/generation_throughput_tok_s"] == 30
    assert summary["vllm_run/train/external_prefix_cache_hit_rate"] == pytest.approx(0.14)
    assert summary["vllm_run/train/output_tokens_total"] == 300
    assert summary["vllm_run/covers_resumed_history"] is True


def test_missing_interval_is_not_zero_work():
    stats = RunStatistics()
    stats.add("combined", WindowStatistics(2, {"ray_vllm_generation_tokens_total": 100}))
    stats.add("combined", WindowStatistics(3, valid=False))
    summary = stats.summary()
    assert summary["vllm_run/combined/generation_throughput_tok_s"] == 50
    assert summary["vllm_run/combined/active_generation_seconds"] == 5
    assert summary["vllm_run/combined/observed_active_seconds"] == 2
    assert summary["vllm_run/combined/incomplete_windows"] == 1


def test_entirely_unobserved_run_reports_coverage_without_rates():
    stats = RunStatistics()
    stats.add("train", WindowStatistics(duration_seconds=3, valid=False))
    summary = stats.summary()
    assert summary["vllm_run/train/active_generation_seconds"] == 3
    assert summary["vllm_run/train/observed_active_seconds"] == 0
    assert summary["vllm_run/train/incomplete_windows"] == 1
    assert "vllm_run/train/generation_throughput_tok_s" not in summary


def test_run_tpot_is_request_weighted_and_tracker_metrics_are_pruned():
    stats = RunStatistics()
    base = "ray_vllm_request_time_per_output_token_seconds"
    ttft = "ray_vllm_time_to_first_token_seconds"
    for count, total in [(1, 0.8), (3, 0.6)]:
        deltas = {
            base + "_count": count,
            base + "_sum": total,
            base + "_bucket::1": count / 2,
            base + "_bucket::2": count,
            base + "_bucket::+Inf": count,
            ttft + "_count": count,
            ttft + "_sum": total,
            ttft + "_bucket::1": count / 2,
            ttft + "_bucket::2": count,
            ttft + "_bucket::+Inf": count,
            "ray_vllm_inter_token_latency_seconds_count": 100,
            "ray_vllm_inter_token_latency_seconds_sum": 0.1,
            "ray_vllm_kv_offload_store_bytes_total": 1000,
        }
        stats.add("train", WindowStatistics(2, deltas))
    summary = stats.summary()
    assert summary["vllm_run/train/tpot_seconds_avg"] == pytest.approx(1.4 / 4)
    assert summary["vllm_run/train/tpot_seconds_p90"] == pytest.approx(1.8)
    assert summary["vllm_run/train/ttft_seconds_p90"] == pytest.approx(1.8)
    assert summary["vllm_run/train/kv_offload_store_bytes_total"] == 2000
    assert not any(
        "itl" in key or "request_tpot" in key or "p50" in key or "store_throughput" in key for key in summary
    )

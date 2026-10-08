"""Tests for weighted summaries from observed counter windows."""

import pytest

from skyrl.train.utils.vllm_run_statistics import RunStatistics


def test_weighted_rates_and_cache_hits_use_raw_totals():
    counter = "ray_vllm_generation_tokens_total"
    queries = "ray_vllm_external_prefix_cache_queries_total"
    hits = "ray_vllm_external_prefix_cache_hits_total"
    stats = RunStatistics()
    first = {counter: 100, queries: 10, hits: 5}
    stats.add("train", dict.fromkeys(first, 0), first, 2)
    stats.add("train", first, {counter: 300, queries: 100, hits: 14}, 8)
    summary = stats.summary()
    assert summary["vllm_correct_aggregate/train/generation_throughput_tok_s"] == 30
    assert summary["vllm_correct_aggregate/train/external_prefix_cache_hit_rate"] == pytest.approx(0.14)
    assert summary["vllm_correct_aggregate/train/output_tokens_total"] == 300
    assert summary["vllm_correct_aggregate/train/measurement_seconds"] == 10


@pytest.mark.parametrize("terminal", [None, {"ray_vllm_generation_tokens_total": 1}])
def test_missing_or_reset_window_omits_scope(terminal):
    counter = "ray_vllm_generation_tokens_total"
    stats = RunStatistics()
    stats.add("train", {counter: 0}, {counter: 100}, 2)
    stats.add("train", {counter: 100}, terminal, 3)
    assert stats.summary() == {}


def test_missing_counter_is_not_reported_with_a_full_run_denominator():
    counter = "ray_vllm_generation_tokens_total"
    stats = RunStatistics()
    stats.add("train", {counter: 0}, {counter: 100}, 2)
    stats.add("train", {}, {counter: 300}, 3)
    assert "vllm_correct_aggregate/train/output_tokens_total" not in stats.summary()
    assert "vllm_correct_aggregate/train/generation_throughput_tok_s" not in stats.summary()


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
        stats.add("train", dict.fromkeys(deltas, 0), deltas, 2)
    summary = stats.summary()
    assert summary["vllm_correct_aggregate/train/tpot_seconds_avg"] == pytest.approx(1.4 / 4)
    assert summary["vllm_correct_aggregate/train/tpot_seconds_p90"] == pytest.approx(1.8)
    assert summary["vllm_correct_aggregate/train/ttft_seconds_p90"] == pytest.approx(1.8)
    assert summary["vllm_correct_aggregate/train/kv_offload_store_bytes_total"] == 2000
    assert not any(
        "itl" in key or "request_tpot" in key or "p50" in key or "store_throughput" in key for key in summary
    )

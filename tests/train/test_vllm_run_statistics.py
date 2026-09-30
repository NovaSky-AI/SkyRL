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


def test_missing_interval_is_not_zero_work():
    stats = RunStatistics()
    stats.add("combined", WindowStatistics(2, {"ray_vllm_generation_tokens_total": 100}))
    stats.add("combined", WindowStatistics(3, valid=False))
    summary = stats.summary()
    assert summary["vllm_run/combined/generation_throughput_tok_s"] == 50
    assert summary["vllm_run/combined/active_generation_seconds"] == 5
    assert summary["vllm_run/combined/observed_active_seconds"] == 2
    assert summary["vllm_run/combined/incomplete_windows"] == 1

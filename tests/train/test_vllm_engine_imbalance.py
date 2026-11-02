"""Tests for aggregate engine throughput imbalance."""

from skyrl.train.utils.vllm_metrics_scraper import engine_imbalance


def test_idle_engine_is_included():
    counter = "ray_vllm_generation_tokens_total"
    prev = {"a": {}, "b": {counter: 10}}
    cur = {"a": {}, "b": {counter: 110}}
    metrics = engine_imbalance(prev, cur, "vllm/")
    assert metrics["vllm/generation_throughput_cv"] == 1
    assert metrics["vllm/generation_throughput_cv_num_engines"] == 2


def test_equal_load_and_missing_engine():
    counter = "ray_vllm_prompt_tokens_total"
    previous = {"a": {counter: 5}, "b": {counter: 10}}
    current = {"a": {counter: 15}, "b": {counter: 20}}
    assert engine_imbalance(previous, current, "")["prompt_throughput_cv"] == 0
    assert "prompt_throughput_cv" not in engine_imbalance(previous, {"a": current["a"]}, "")
    assert "prompt_throughput_cv" not in engine_imbalance(previous, previous, "")

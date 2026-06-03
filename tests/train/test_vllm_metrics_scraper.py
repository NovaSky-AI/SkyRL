"""
uv run --isolated --extra dev pytest tests/train/test_vllm_metrics_scraper.py
"""

import asyncio
from unittest.mock import patch

import pytest

from skyrl.train.utils.vllm_metrics_scraper import (
    VLLMMetricsScraper,
    aggregate,
    discover_ray_metrics_urls,
    parse_metrics_text,
)


def _snapshot(
    *,
    running: float,
    waiting: float,
    kv: float,
    prefix_q: float,
    prefix_h: float,
    prompt_toks: float,
    gen_toks: float,
    ttft_sum: float,
    ttft_count: float,
    itl_sum: float,
    itl_count: float,
    replicas: int = 2,
    replica_id_offset: int = 0,
) -> str:
    """Build a Prometheus text payload split across N replica labels.

    ``replica_id_offset`` shifts the ``ReplicaId`` numbering (default 0 → r0,
    r1, ...).  Used by multi-node tests to give each simulated node a unique
    set of replica IDs so merges across nodes don't collide.
    """
    lines = []

    def per_replica_split(value: float):
        return [value / replicas] * replicas

    def rid(i: int) -> str:
        return f"r{i + replica_id_offset}"

    def emit_gauge(name: str, value: float):
        lines.append(f"# HELP {name} test")
        lines.append(f"# TYPE {name} gauge")
        for i, v in enumerate(per_replica_split(value)):
            lines.append(f'{name}{{ReplicaId="{rid(i)}"}} {v}')

    def emit_counter(name_base: str, value: float):
        lines.append(f"# HELP {name_base} test")
        lines.append(f"# TYPE {name_base} counter")
        for i, v in enumerate(per_replica_split(value)):
            lines.append(f'{name_base}_total{{ReplicaId="{rid(i)}"}} {v}')

    def emit_histogram_sumcount(base: str, total_sum: float, total_count: float):
        # Skip _bucket lines; the scraper only uses _sum / _count.
        lines.append(f"# HELP {base} test")
        lines.append(f"# TYPE {base} histogram")
        for i, (s, c) in enumerate(zip(per_replica_split(total_sum), per_replica_split(total_count))):
            lines.append(f'{base}_sum{{ReplicaId="{rid(i)}"}} {s}')
            lines.append(f'{base}_count{{ReplicaId="{rid(i)}"}} {c}')
            # An empty histogram still needs a +Inf bucket for parser sanity.
            lines.append(f'{base}_bucket{{ReplicaId="{rid(i)}",le="+Inf"}} {c}')

    emit_gauge("ray_vllm_num_requests_running", running)
    emit_gauge("ray_vllm_num_requests_waiting", waiting)
    emit_gauge("ray_vllm_kv_cache_usage_perc", kv)
    emit_counter("ray_vllm_prefix_cache_queries", prefix_q)
    emit_counter("ray_vllm_prefix_cache_hits", prefix_h)
    emit_counter("ray_vllm_prompt_tokens", prompt_toks)
    emit_counter("ray_vllm_generation_tokens", gen_toks)
    emit_histogram_sumcount("ray_vllm_time_to_first_token_seconds", ttft_sum, ttft_count)
    emit_histogram_sumcount("ray_vllm_inter_token_latency_seconds", itl_sum, itl_count)
    return "\n".join(lines) + "\n"


def test_parse_and_aggregate_sum_and_mean():
    text = _snapshot(
        running=4,
        waiting=2,
        kv=0.6,  # mean across 2 replicas should be 0.3
        prefix_q=100,
        prefix_h=80,
        prompt_toks=1000,
        gen_toks=500,
        ttft_sum=2.0,
        ttft_count=10,
        itl_sum=1.0,
        itl_count=200,
    )
    parsed = parse_metrics_text(text)

    sums = aggregate(parsed, ["ray_vllm_num_requests_running", "ray_vllm_prompt_tokens_total"], how="sum")
    assert sums["ray_vllm_num_requests_running"] == pytest.approx(4)
    assert sums["ray_vllm_prompt_tokens_total"] == pytest.approx(1000)

    means = aggregate(parsed, ["ray_vllm_kv_cache_usage_perc"], how="mean")
    # Two replicas at 0.3 each => mean is 0.3.
    assert means["ray_vllm_kv_cache_usage_perc"] == pytest.approx(0.3)


def test_aggregate_omits_missing_metric():
    parsed = parse_metrics_text(
        _snapshot(
            running=1,
            waiting=0,
            kv=0.1,
            prefix_q=0,
            prefix_h=0,
            prompt_toks=0,
            gen_toks=0,
            ttft_sum=0,
            ttft_count=0,
            itl_sum=0,
            itl_count=0,
        )
    )
    result = aggregate(parsed, ["does_not_exist"], how="sum")
    assert result == {}


@pytest.mark.asyncio
async def test_scraper_first_call_emits_only_gauges():
    scraper = VLLMMetricsScraper(urls=["http://stub/metrics"])
    text = _snapshot(
        running=4,
        waiting=2,
        kv=0.6,
        prefix_q=100,
        prefix_h=80,
        prompt_toks=1000,
        gen_toks=500,
        ttft_sum=2.0,
        ttft_count=10,
        itl_sum=1.0,
        itl_count=200,
    )

    async def fake_fetch_all():
        return parse_metrics_text(text)

    with patch.object(scraper, "_fetch_all", fake_fetch_all):
        out = await scraper.sample()

    assert out["vllm/num_requests_running"] == pytest.approx(4)
    assert out["vllm/num_requests_waiting"] == pytest.approx(2)
    assert out["vllm/kv_cache_usage_perc"] == pytest.approx(0.3)
    # No derived metrics yet — no previous snapshot.
    assert "vllm/generation_throughput_tok_s" not in out
    assert "vllm/prefix_cache_hit_rate" not in out
    assert "vllm/ttft_seconds_avg" not in out
    assert "vllm/tpot_seconds_avg" not in out


@pytest.mark.asyncio
async def test_scraper_second_call_derives_rates_and_averages():
    scraper = VLLMMetricsScraper(urls=["http://stub/metrics"])

    snap1 = _snapshot(
        running=4,
        waiting=2,
        kv=0.5,
        prefix_q=100,
        prefix_h=80,
        prompt_toks=1000,
        gen_toks=500,
        ttft_sum=2.0,
        ttft_count=10,
        itl_sum=1.0,
        itl_count=200,
    )
    # 1s later: 50 more queries with 40 hits, 250 more gen tokens, 100 more prompt
    # tokens, 5 more TTFT samples summing to 1.0 (avg 0.2s), 100 more ITL samples
    # summing to 0.5 (avg 5ms).
    snap2 = _snapshot(
        running=5,
        waiting=1,
        kv=0.7,
        prefix_q=150,
        prefix_h=120,
        prompt_toks=1100,
        gen_toks=750,
        ttft_sum=3.0,
        ttft_count=15,
        itl_sum=1.5,
        itl_count=300,
    )

    texts = iter([snap1, snap2])

    async def fake_fetch_all():
        return parse_metrics_text(next(texts))

    times = iter([1000.0, 1001.0])  # exactly 1s apart

    with (
        patch.object(scraper, "_fetch_all", fake_fetch_all),
        patch(
            "skyrl.train.utils.vllm_metrics_scraper.time.monotonic",
            side_effect=lambda: next(times),
        ),
    ):
        await scraper.sample()
        out = await scraper.sample()

    assert out["vllm/generation_throughput_tok_s"] == pytest.approx(250.0)
    assert out["vllm/prompt_throughput_tok_s"] == pytest.approx(100.0)
    assert out["vllm/prefix_cache_hit_rate"] == pytest.approx(40.0 / 50.0)
    assert out["vllm/ttft_seconds_avg"] == pytest.approx(1.0 / 5.0)
    assert out["vllm/tpot_seconds_avg"] == pytest.approx(0.5 / 100.0)
    # Gauges still flow through.
    assert out["vllm/num_requests_running"] == pytest.approx(5)
    assert out["vllm/kv_cache_usage_perc"] == pytest.approx(0.35)


@pytest.mark.asyncio
async def test_scraper_handles_counter_reset():
    scraper = VLLMMetricsScraper(urls=["http://stub/metrics"])
    big = _snapshot(
        running=0,
        waiting=0,
        kv=0.0,
        prefix_q=0,
        prefix_h=0,
        prompt_toks=10_000,
        gen_toks=10_000,
        ttft_sum=0,
        ttft_count=0,
        itl_sum=0,
        itl_count=0,
    )
    small = _snapshot(  # engine restart — counters dropped back to small values
        running=0,
        waiting=0,
        kv=0.0,
        prefix_q=0,
        prefix_h=0,
        prompt_toks=100,
        gen_toks=100,
        ttft_sum=0,
        ttft_count=0,
        itl_sum=0,
        itl_count=0,
    )
    texts = iter([big, small])

    async def fake_fetch_all():
        return parse_metrics_text(next(texts))

    with patch.object(scraper, "_fetch_all", fake_fetch_all):
        await scraper.sample()
        out = await scraper.sample()

    # Negative deltas are dropped, not emitted as garbage values.
    assert "vllm/generation_throughput_tok_s" not in out
    assert "vllm/prompt_throughput_tok_s" not in out


def test_scraper_with_no_urls_is_noop():
    scraper = VLLMMetricsScraper(urls=[])
    out = asyncio.run(scraper.sample())
    assert out == {}


@pytest.mark.asyncio
async def test_fetch_all_merges_across_nodes_and_aggregates_correctly():
    # Two nodes, each running two replicas with disjoint ReplicaId labels.
    # Node A → r0, r1.  Node B → r2, r3.  Merged set should have 4 entries
    # per metric with no collisions.
    urls = ["http://nodeA/metrics", "http://nodeB/metrics"]
    scraper = VLLMMetricsScraper(urls=urls)

    node_a = _snapshot(
        running=4,
        waiting=2,
        kv=0.6,  # split across r0, r1 → 0.3 each
        prefix_q=100,
        prefix_h=80,
        prompt_toks=1000,
        gen_toks=500,
        ttft_sum=2.0,
        ttft_count=10,
        itl_sum=1.0,
        itl_count=200,
    )
    node_b = _snapshot(
        running=6,
        waiting=4,
        kv=0.8,  # split across r2, r3 → 0.4 each
        prefix_q=200,
        prefix_h=140,
        prompt_toks=3000,
        gen_toks=1500,
        ttft_sum=4.0,
        ttft_count=20,
        itl_sum=3.0,
        itl_count=300,
        replica_id_offset=2,
    )
    url_to_text = {urls[0]: node_a, urls[1]: node_b}

    async def fake_fetch_one(_client, url):
        return url_to_text[url]

    with patch.object(scraper, "_fetch_one", fake_fetch_one):
        parsed = await scraper._fetch_all()

    # Merge sanity: each metric appears once per replica per node → 4 entries.
    running_entries = [v for (n, _l), v in parsed.items() if n == "ray_vllm_num_requests_running"]
    kv_entries = [v for (n, _l), v in parsed.items() if n == "ray_vllm_kv_cache_usage_perc"]
    assert len(running_entries) == 4
    assert len(kv_entries) == 4
    # ReplicaId labels should span all four ids with no collision.
    running_rids = {
        dict(labels)["ReplicaId"] for (n, labels), _ in parsed.items() if n == "ray_vllm_num_requests_running"
    }
    assert running_rids == {"r0", "r1", "r2", "r3"}

    # Sum aggregation reduces correctly across both nodes.
    sums = aggregate(
        parsed,
        ["ray_vllm_num_requests_running", "ray_vllm_prompt_tokens_total", "ray_vllm_prefix_cache_hits_total"],
        how="sum",
    )
    assert sums["ray_vllm_num_requests_running"] == pytest.approx(4 + 6)
    assert sums["ray_vllm_prompt_tokens_total"] == pytest.approx(1000 + 3000)
    assert sums["ray_vllm_prefix_cache_hits_total"] == pytest.approx(80 + 140)

    # Mean aggregation: 4 replicas at (0.3, 0.3, 0.4, 0.4) → 0.35.
    means = aggregate(parsed, ["ray_vllm_kv_cache_usage_perc"], how="mean")
    assert means["ray_vllm_kv_cache_usage_perc"] == pytest.approx(0.35)


def test_discover_ray_metrics_urls_filters_dead_and_missing(monkeypatch):
    fake_nodes = [
        {"Alive": True, "NodeManagerAddress": "10.0.0.1", "MetricsExportPort": 51001},
        {"Alive": False, "NodeManagerAddress": "10.0.0.2", "MetricsExportPort": 51002},
        {"Alive": True, "NodeManagerAddress": "10.0.0.3", "MetricsExportPort": None},
        {"Alive": True, "NodeManagerAddress": "10.0.0.4", "MetricsExportPort": 51004},
    ]
    monkeypatch.setattr(
        "skyrl.train.utils.vllm_metrics_scraper.ray.nodes",
        lambda: fake_nodes,
    )
    urls = discover_ray_metrics_urls()
    assert urls == [
        "http://10.0.0.1:51001/metrics",
        "http://10.0.0.4:51004/metrics",
    ]

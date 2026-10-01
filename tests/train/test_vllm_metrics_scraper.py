"""
uv run --isolated --extra dev pytest tests/train/test_vllm_metrics_scraper.py
"""

import asyncio
from unittest.mock import Mock, patch

import httpx
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


def _spec_snapshot(*, drafts: float, draft_tokens: float, accepted_tokens: float, per_pos, replicas: int = 2) -> str:
    """Prometheus payload with the four vLLM spec-decode counters split across ``replicas``.

    ``per_pos`` is a list of accepted-token counts indexed by draft position (0-based); each is
    emitted on the ``..._per_pos`` counter with a ``position`` label so ``sum_by_position`` can
    recover the per-depth breakdown.
    """
    lines = []

    def split(value: float):
        return [value / replicas] * replicas

    def emit_counter(name_base: str, value: float):
        lines.append(f"# TYPE {name_base} counter")
        for i, v in enumerate(split(value)):
            lines.append(f'{name_base}_total{{ReplicaId="r{i}"}} {v}')

    emit_counter("ray_vllm_spec_decode_num_drafts", drafts)
    emit_counter("ray_vllm_spec_decode_num_draft_tokens", draft_tokens)
    emit_counter("ray_vllm_spec_decode_num_accepted_tokens", accepted_tokens)
    lines.append("# TYPE ray_vllm_spec_decode_num_accepted_tokens_per_pos counter")
    for pos, count in enumerate(per_pos):
        for i, v in enumerate(split(count)):
            lines.append(
                f'ray_vllm_spec_decode_num_accepted_tokens_per_pos_total{{ReplicaId="r{i}",position="{pos}"}} {v}'
            )
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
    assert "vllm/tpot_seconds_avg" not in out
    assert "vllm/itl_seconds_avg" not in out
    # Gauges still flow through.
    assert out["vllm/num_requests_running"] == pytest.approx(5)
    assert out["vllm/kv_cache_usage_perc"] == pytest.approx(0.35)
    # No spec-decode counters in these snapshots => no draft metrics.
    assert not any(k.startswith("vllm/draft_") for k in out)


@pytest.mark.asyncio
async def test_scraper_derives_spec_decode_acceptance():
    """Spec-decode counters reduce to an overall acceptance rate plus a per-position (1-based) rate.

    snap1 -> snap2 delta: 100 drafts, 200 draft tokens, 150 accepted (0.75 overall);
    per position accepted 90 and 60 over 100 drafts => pos_1=0.9, pos_2=0.6. Counters are summed
    across the two replicas before the delta.
    """
    scraper = VLLMMetricsScraper(urls=["http://stub/metrics"])
    snap1 = _spec_snapshot(drafts=0, draft_tokens=0, accepted_tokens=0, per_pos=[0, 0])
    snap2 = _spec_snapshot(drafts=100, draft_tokens=200, accepted_tokens=150, per_pos=[90, 60])
    texts = iter([snap1, snap2])

    async def fake_fetch_all():
        return parse_metrics_text(next(texts))

    with patch.object(scraper, "_fetch_all", fake_fetch_all):
        await scraper.sample()
        out = await scraper.sample()

    assert out["vllm/draft_num_draft_tokens"] == pytest.approx(200)
    assert out["vllm/draft_num_accepted_tokens"] == pytest.approx(150)
    assert out["vllm/draft_acceptance_rate"] == pytest.approx(0.75)
    assert out["vllm/draft_acceptance_rate_pos_1"] == pytest.approx(0.9)
    assert out["vllm/draft_acceptance_rate_pos_2"] == pytest.approx(0.6)
    # Mean acceptance length = 1 + accepted/drafts = 1 + 150/100 = 2.5.
    assert out["vllm/draft_mean_acceptance_length"] == pytest.approx(2.5)


@pytest.mark.asyncio
async def test_spec_decode_acceptance_flows_through_windows():
    """Windowed start/stop nests the draft metrics under the window label (train vs eval)."""
    scraper = VLLMMetricsScraper(urls=["http://stub/metrics"])
    texts = iter(
        [
            _spec_snapshot(drafts=0, draft_tokens=0, accepted_tokens=0, per_pos=[0]),
            _spec_snapshot(drafts=50, draft_tokens=50, accepted_tokens=40, per_pos=[40]),
        ]
    )

    async def fake_fetch_all():
        return parse_metrics_text(next(texts))

    with patch.object(scraper, "_fetch_all", fake_fetch_all):
        await scraper.start("vllm/train")
        out = await scraper.stop()

    assert out["vllm/train/draft_acceptance_rate"] == pytest.approx(0.8)
    assert out["vllm/train/draft_acceptance_rate_pos_1"] == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_scraper_throughput_uses_generation_time_not_wall_clock():
    """Throughput divides token deltas by generation time, not the full step.

    The wall-clock gap between samples is 10s, but only 2s was spent
    generating. Throughput must use the 2s window; latency/hit-rate metrics
    (which don't depend on time) are unaffected.
    """
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
    # +250 gen tokens, +100 prompt tokens over a 10s wall-clock gap of which
    # only 2s was generation.
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

    times = iter([1000.0, 1010.0])  # 10s wall-clock apart

    with (
        patch.object(scraper, "_fetch_all", fake_fetch_all),
        patch(
            "skyrl.train.utils.vllm_metrics_scraper.time.monotonic",
            side_effect=lambda: next(times),
        ),
    ):
        await scraper.sample()
        out = await scraper.sample(generation_time_s=2.0)

    # 250 tokens / 2s generation == 125 tok/s (NOT 250/10 == 25).
    assert out["vllm/generation_throughput_tok_s"] == pytest.approx(125.0)
    assert out["vllm/prompt_throughput_tok_s"] == pytest.approx(50.0)
    # Time-independent derived metrics are unchanged by the window choice.
    assert out["vllm/prefix_cache_hit_rate"] == pytest.approx(40.0 / 50.0)
    assert out["vllm/ttft_seconds_avg"] == pytest.approx(1.0 / 5.0)
    assert "vllm/tpot_seconds_avg" not in out
    assert "vllm/itl_seconds_avg" not in out


@pytest.mark.asyncio
async def test_scraper_throughput_falls_back_to_wall_clock_without_gen_time():
    """With no generation_time_s (and with non-positive values), use dt."""
    scraper = VLLMMetricsScraper(urls=["http://stub/metrics"])

    common = dict(
        running=0,
        waiting=0,
        kv=0.0,
        prefix_q=0,
        prefix_h=0,
        ttft_sum=0,
        ttft_count=0,
        itl_sum=0,
        itl_count=0,
    )
    snap1 = _snapshot(prompt_toks=0, gen_toks=0, **common)
    snap2 = _snapshot(prompt_toks=100, gen_toks=200, **common)
    texts = iter([snap1, snap2])

    async def fake_fetch_all():
        return parse_metrics_text(next(texts))

    times = iter([1000.0, 1002.0])  # 2s wall-clock apart

    with (
        patch.object(scraper, "_fetch_all", fake_fetch_all),
        patch(
            "skyrl.train.utils.vllm_metrics_scraper.time.monotonic",
            side_effect=lambda: next(times),
        ),
    ):
        await scraper.sample()
        # generation_time_s=0 is non-positive -> fall back to wall-clock dt.
        out = await scraper.sample(generation_time_s=0.0)

    assert out["vllm/generation_throughput_tok_s"] == pytest.approx(100.0)
    assert out["vllm/prompt_throughput_tok_s"] == pytest.approx(50.0)


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


def _split_snap(gen, prompt):
    return _snapshot(
        running=1,
        waiting=0,
        kv=0.4,
        prefix_q=0,
        prefix_h=0,
        prompt_toks=prompt,
        gen_toks=gen,
        ttft_sum=0,
        ttft_count=0,
        itl_sum=0,
        itl_count=0,
    )


@pytest.mark.asyncio
async def test_window_separates_train_and_eval_rollouts():
    """Separate start/stop windows label train and eval rollouts independently.

      train start  gen=500  prompt=1000
      train stop   gen=600  prompt=1100   (train rollout: +100 gen, +100 prompt)
      eval start   gen=600  prompt=1100
      eval stop    gen=1100 prompt=1300   (eval rollout:  +500 gen, +200 prompt)
    The scraper owns timing: 2s of active train time, 5s of active eval time
    (with prep/scoring paused out).
    """
    scraper = VLLMMetricsScraper(urls=["http://stub/metrics"])

    # One fetch per start() and per stop().
    texts = iter([_split_snap(500, 1000), _split_snap(600, 1100), _split_snap(600, 1100), _split_snap(1100, 1300)])

    async def fake_fetch_all():
        return parse_metrics_text(next(texts))

    # monotonic() is read at: train start, train stop, eval start, eval pause,
    # eval resume, eval pause. Active train time = 102-100 = 2s; active eval
    # time = (215-210) = 5s (the 200->200 prep span is paused out).
    times = iter([100.0, 102.0, 200.0, 200.0, 210.0, 215.0])

    with (
        patch.object(scraper, "_fetch_all", fake_fetch_all),
        patch(
            "skyrl.train.utils.vllm_metrics_scraper.time.monotonic",
            side_effect=lambda: next(times),
        ),
    ):
        # Train rollout: a single generation spans the whole window.
        await scraper.start("vllm/train")
        train = await scraper.stop()

        # Eval rollout: paused for prep, resumed only around generation.
        await scraper.start("vllm/eval")
        scraper.pause()
        scraper.resume()
        scraper.pause()
        eval_out = await scraper.stop()

    # Train rollout: (600-500)/2 == 50, (1100-1000)/2 == 50.
    assert train["vllm/train/generation_throughput_tok_s"] == pytest.approx(50.0)
    assert train["vllm/train/prompt_throughput_tok_s"] == pytest.approx(50.0)
    assert not any(k.startswith("vllm/eval/") for k in train)
    # Eval rollout: (1100-600)/5 == 100, (1300-1100)/5 == 40.
    assert eval_out["vllm/eval/generation_throughput_tok_s"] == pytest.approx(100.0)
    assert eval_out["vllm/eval/prompt_throughput_tok_s"] == pytest.approx(40.0)


@pytest.mark.asyncio
async def test_window_accumulates_active_time_across_multiple_generations():
    """pause/resume around each generation sums only the active spans.

    Mirrors a dynamic-sampling step (or an eval loop) that generates more than
    once: the paused gap between generations is excluded from the denominator.
    """
    scraper = VLLMMetricsScraper(urls=["http://stub/metrics"])

    texts = iter([_split_snap(0, 0), _split_snap(300, 100)])

    async def fake_fetch_all():
        return parse_metrics_text(next(texts))

    # monotonic() reads: start(0), pause(0), resume(10), pause(12) -> gen1 = 2s,
    # resume(112), pause(114) -> gen2 = 2s. The 100s paused gap (12 -> 112) is
    # excluded, so active time is 4s, not 114s.
    times = iter([0.0, 0.0, 10.0, 12.0, 112.0, 114.0])

    with (
        patch.object(scraper, "_fetch_all", fake_fetch_all),
        patch(
            "skyrl.train.utils.vllm_metrics_scraper.time.monotonic",
            side_effect=lambda: next(times),
        ),
    ):
        await scraper.start("vllm/eval")
        scraper.pause()
        scraper.resume()  # generation 1
        scraper.pause()
        scraper.resume()  # generation 2
        scraper.pause()
        out = await scraper.stop()

    # gen 300 over 4s active == 75 tok/s; prompt 100 over 4s == 25 tok/s.
    assert out["vllm/eval/generation_throughput_tok_s"] == pytest.approx(75.0)
    assert out["vllm/eval/prompt_throughput_tok_s"] == pytest.approx(25.0)


@pytest.mark.asyncio
async def test_window_pause_resume_misuse_raises():
    scraper = VLLMMetricsScraper(urls=["http://stub/metrics"])

    async def fake_fetch_all():
        return parse_metrics_text(_split_snap(0, 0))

    with patch.object(scraper, "_fetch_all", fake_fetch_all):
        with pytest.raises(ValueError):
            scraper.pause()  # no open window
        with pytest.raises(ValueError):
            await scraper.stop()  # no open window
        await scraper.start("vllm/train")
        with pytest.raises(ValueError):
            await scraper.start("vllm/eval")  # window already open
        scraper.pause()
        with pytest.raises(ValueError):
            scraper.pause()  # double pause
        scraper.resume()
        with pytest.raises(ValueError):
            scraper.resume()  # double resume
        await scraper.stop()


@pytest.mark.asyncio
async def test_window_returns_empty_without_endpoints():
    scraper = VLLMMetricsScraper(urls=[])
    await scraper.start("vllm/train")
    out = await scraper.stop()
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


@pytest.mark.asyncio
async def test_worker_filter_and_merged_latency_buckets():
    from unittest.mock import AsyncMock

    scraper = VLLMMetricsScraper(urls=["http://test/metrics"], worker_ids=["ours"])
    text = """ray_vllm_generation_tokens_total{WorkerId="ours"} 10
ray_vllm_generation_tokens_total{WorkerId="other"} 900
ray_vllm_time_to_first_token_seconds_bucket{WorkerId="ours",le="1"} 2
ray_vllm_time_to_first_token_seconds_bucket{WorkerId="ours",le="+Inf"} 3
ray_vllm_time_to_first_token_seconds_bucket{WorkerId="other",le="1"} 900
"""
    scraper._fetch_all = AsyncMock(return_value=parse_metrics_text(text))
    snapshot = await scraper._read_snapshot()
    assert snapshot["ray_vllm_generation_tokens_total"] == 10
    assert snapshot["ray_vllm_time_to_first_token_seconds_bucket::1"] == 2


@pytest.mark.asyncio
async def test_failed_node_scrape_skips_partial_totals_and_reestablishes_baseline():
    step = 0

    def respond(request):
        worker = request.url.host
        if step == 1 and worker == "b":
            return httpx.Response(503)
        text = (
            f'ray_vllm_num_requests_running{{WorkerId="{worker}"}} 1\n'
            f'ray_vllm_generation_tokens_total{{WorkerId="{worker}"}} {step * 10}\n'
        )
        return httpx.Response(200, text=text)

    scraper = VLLMMetricsScraper(urls=["http://a/metrics", "http://b/metrics"], worker_ids=["a", "b"])
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        scraper._client = client
        await scraper.sample(generation_time_s=1)
        step = 1
        assert await scraper.sample(generation_time_s=1) == {}
        step = 2
        recovered = await scraper.sample(generation_time_s=1)
        assert "vllm/generation_throughput_tok_s" not in recovered
        step = 3
        metrics = await scraper.sample(generation_time_s=1)
    assert metrics["vllm/generation_throughput_tok_s"] == 20


def test_offload_and_preemption_scalar_reductions():
    current = {
        "ray_vllm_num_preemptions_total": 2,
        "ray_vllm_generation_tokens_total": 100,
        "ray_vllm_external_prefix_cache_queries_total": 30,
        "ray_vllm_external_prefix_cache_hits_total": 12,
        "ray_vllm_kv_offload_store_bytes_total": 1000,
        "ray_vllm_kv_offload_load_bytes_total": 500,
    }
    metrics = VLLMMetricsScraper._derive(current, dict.fromkeys(current, 0), 5, "vllm/")
    assert metrics["vllm/num_preemptions"] == 2
    assert metrics["vllm/preemptions_per_million_tokens"] == 20000
    assert metrics["vllm/external_prefix_cache_hit_rate"] == 0.4
    assert "vllm/kv_offload_store_throughput_bytes_s" not in metrics
    assert metrics["vllm/kv_offload_load_throughput_bytes_s"] == 100


def test_logged_tpot_uses_request_histogram_and_only_mean_p90():
    current = {}
    for name, count, total in [
        ("time_to_first_token_seconds", 10, 12),
        ("request_time_per_output_token_seconds", 4, 2),
        ("inter_token_latency_seconds", 100, 0.1),
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
    metrics = VLLMMetricsScraper._derive(current, dict.fromkeys(current, 0), 5, "vllm/")
    assert metrics["vllm/tpot_seconds_avg"] == pytest.approx(0.5)
    assert metrics["vllm/tpot_seconds_p90"] == pytest.approx(1.8)
    assert metrics["vllm/ttft_seconds_avg"] == pytest.approx(1.2)
    assert metrics["vllm/ttft_seconds_p90"] == pytest.approx(1.8)
    assert not any("itl" in key or "request_tpot" in key or "p50" in key for key in metrics)


@pytest.mark.parametrize("error", [None, RuntimeError("identity failed"), TimeoutError("identity timed out")])
def test_worker_identity_lookup_is_bounded_and_failure_does_not_abort_setup(tmp_path, monkeypatch, error):
    from skyrl.train.config import SkyRLTrainConfig
    from skyrl.train.entrypoints.main_base import BasePPOExp

    cfg = SkyRLTrainConfig()
    cfg.trainer.export_path = str(tmp_path / "export")
    cfg.trainer.ckpt_path = str(tmp_path / "checkpoints")
    cfg.trainer.fully_async.simulate_training = True
    experiment = BasePPOExp.__new__(BasePPOExp)
    experiment.cfg = cfg
    experiment.tokenizer = Mock()
    experiment.train_dataset = None
    experiment.eval_dataset = None
    experiment.colocate_pg = None
    actor = Mock()
    actor.get_metrics_worker_id.remote.return_value = "identity-ref"
    experiment._server_groups = [Mock(get_actors=Mock(return_value=[actor]))]
    trainer = Mock()
    experiment.get_trainer = Mock(return_value=trainer)
    for method in ("get_tracker", "get_inference_client", "get_generator", "get_trajectory_logger"):
        setattr(experiment, method, Mock())
    lookup = Mock(return_value=["owner"], side_effect=error)
    monkeypatch.setattr("skyrl.train.entrypoints.main_base.ray.get", lookup)

    assert experiment._setup_trainer() is trainer
    lookup.assert_called_once_with(["identity-ref"], timeout=10)
    trainer._vllm_metrics_scraper.set_worker_ids.assert_called_once_with([] if error else ["owner"])

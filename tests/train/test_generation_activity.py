"""Tests for generation duration with overlap and failures."""

import pytest

from skyrl.train.utils.generation_activity import GenerationActivity


def test_overlap_counts_union_and_closes_on_error():
    now = [0.0]
    published = []
    activity = GenerationActivity(clock=lambda: now[0], publish=published.append)
    with activity.active():
        now[0] = 2
        with pytest.raises(RuntimeError), activity.active():
            now[0] = 5
            raise RuntimeError("generation failed")
        now[0] = 7
    assert activity.seconds == 7
    now[0] = 10
    with activity.active():
        now[0] = 12
    assert activity.seconds == 9
    assert published == [1, 2, 1, 0, 1, 0]


@pytest.mark.asyncio
async def test_zero_active_time_does_not_fall_back_to_wall_time():
    from unittest.mock import AsyncMock

    from skyrl.train.utils.vllm_metrics_scraper import VLLMMetricsScraper

    scraper = VLLMMetricsScraper(urls=["test"])
    scraper._read_snapshot = AsyncMock(
        side_effect=[{"ray_vllm_generation_tokens_total": 0}, {"ray_vllm_generation_tokens_total": 10}]
    )
    await scraper.sample_active()
    metrics = await scraper.sample_active()
    assert "vllm/generation_throughput_tok_s" not in metrics
    assert scraper.last_window.duration_seconds == 0


@pytest.mark.asyncio
async def test_fresh_owned_engine_recovers_delayed_first_export():
    from unittest.mock import AsyncMock

    from skyrl.train.utils.vllm_metrics_scraper import VLLMMetricsScraper

    scraper = VLLMMetricsScraper(urls=["test"])
    scraper.set_worker_ids(["owned"])
    scraper._read_snapshot = AsyncMock(side_effect=[None, None, {"ray_vllm_generation_tokens_total": 100}])
    await scraper.sample(generation_time_s=0, allow_zero_duration=True)
    await scraper.sample(generation_time_s=2, allow_zero_duration=True)
    metrics = await scraper.sample(generation_time_s=0, allow_zero_duration=True)
    assert metrics["vllm/generation_throughput_tok_s"] == 50
    summary = scraper.run_statistics.summary()
    assert summary["vllm_run/combined/output_tokens_total"] == 100
    assert summary["vllm_run/combined/active_generation_seconds"] == 2
    assert summary["vllm_run/combined/observed_active_seconds"] == 2


@pytest.mark.asyncio
async def test_sync_terminal_export_recovers_missing_stop_snapshot():
    from unittest.mock import AsyncMock

    from skyrl.train.utils.vllm_metrics_scraper import VLLMMetricsScraper

    scraper = VLLMMetricsScraper(urls=["test"])
    scraper.set_worker_ids(["owned"])
    scraper._read_snapshot = AsyncMock(side_effect=[None, None, {"ray_vllm_generation_tokens_total": 100}])
    await scraper.start("vllm/train")
    scraper.pause()
    scraper._window_time_s = 2
    assert await scraper.stop() == {}
    await scraper.sample(generation_time_s=0, allow_zero_duration=True)
    summary = scraper.run_statistics.summary()
    assert summary["vllm_run/train/output_tokens_total"] == 100
    assert summary["vllm_run/train/generation_throughput_tok_s"] == 50
    assert summary["vllm_run/train/active_generation_seconds"] == 2


@pytest.mark.asyncio
async def test_sync_late_exports_between_windows_are_counted_once():
    from unittest.mock import AsyncMock

    from skyrl.train.utils.vllm_metrics_scraper import VLLMMetricsScraper

    counter = "ray_vllm_generation_tokens_total"
    scraper = VLLMMetricsScraper(urls=["test"])
    scraper._read_snapshot = AsyncMock(side_effect=[{counter: 0}, {counter: 100}, {counter: 150}, {counter: 200}])
    for _ in range(2):
        await scraper.start("vllm/train")
        scraper.pause()
        scraper._window_time_s = 1
        await scraper.stop()
    summary = scraper.run_statistics.summary()
    assert summary["vllm_run/train/output_tokens_total"] == 200
    assert summary["vllm_run/train/generation_throughput_tok_s"] == 100
    assert summary["vllm_run/train/active_generation_seconds"] == 2

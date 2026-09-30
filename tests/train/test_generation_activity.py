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

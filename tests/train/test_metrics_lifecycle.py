"""Tests for metrics finalization on success and controlled failures."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils.tracking import Tracking


@pytest.mark.parametrize("fail", [False, True])
def test_entrypoint_finalizes_before_exception_logging(fail):
    events = []
    tracker = Tracking("test", "test", backend="console")
    tracker.log_exception = Mock(side_effect=lambda *args, **kwargs: events.append("exception"))
    trainer = SimpleNamespace(tracker=tracker, global_step=0, flush_pending_metrics=Mock())

    async def train():
        if fail:
            raise ValueError("controlled training failure")

    async def finalize(status):
        events.append(status)
        tracker.run_status = status

    trainer.train = train
    trainer.finalize_vllm_metrics = finalize
    exp = BasePPOExp.__new__(BasePPOExp)

    def setup():
        exp.trainer = trainer
        exp.tracker = tracker
        return trainer

    exp._setup_trainer = setup
    if fail:
        with pytest.raises(ValueError, match="controlled training failure"):
            exp.run()
        assert events == ["failed", "exception"]
    else:
        exp.run()
        assert events == ["success"]
    assert tracker.run_status == ("failed" if fail else "success")


@pytest.mark.asyncio
async def test_trainer_finalization_is_idempotent(tmp_path):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer._metrics_finalized = False
    trainer.tracker = Tracking("test", "test", backend="console")
    trainer.cfg = SimpleNamespace(trainer=SimpleNamespace(ckpt_path=str(tmp_path)))
    scraper = SimpleNamespace(finalize=AsyncMock(), run_statistics=SimpleNamespace(summary=lambda: {"tokens": 10}))
    trainer._vllm_metrics_scraper = scraper
    await trainer.finalize_vllm_metrics("failed")
    await trainer.finalize_vllm_metrics("failed")
    scraper.finalize.assert_awaited_once()
    assert trainer.tracker.run_status == "failed"
    assert "failed" in (tmp_path / "vllm_run_summary.json").read_text()


@pytest.mark.parametrize("enable_pd", [False, True])
def test_pd_does_not_publish_double_counted_engine_summaries(enable_pd):
    from unittest.mock import patch

    from skyrl.train.config import SkyRLTrainConfig

    cfg = SkyRLTrainConfig()
    cfg.generator.inference_engine.enable_ray_prometheus_stats = True
    cfg.generator.inference_engine.enable_pd = enable_pd
    cfg.trainer.enable_ray_gpu_monitor = False
    with patch("skyrl.train.trainer.VLLMMetricsScraper") as scraper:
        trainer = RayPPOTrainer(
            cfg,
            Tracking("test", "test", backend="console"),
            tokenizer=Mock(),
            train_dataset=None,
            inference_engine_client=Mock(),
            generator=Mock(),
        )
    assert (trainer._vllm_metrics_scraper is None) == enable_pd
    assert scraper.call_count == (0 if enable_pd else 1)

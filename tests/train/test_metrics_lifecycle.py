"""Tests for metrics finalization on success and controlled failures."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
import torch

from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils.tracking import Tracking
from skyrl.train.utils.trainer_utils import ResumeMode


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
    trainer.finalize_metrics = finalize
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
@pytest.mark.parametrize("resumed,enable_pd", [(False, False), (True, False), (False, True)])
async def test_trainer_finalizes_once_and_omits_resumed_or_pd_aggregates(resumed, enable_pd):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer._metrics_finalized = False
    trainer.tracker = Tracking("test", "test", backend="console")
    trainer._resumed_from_checkpoint = resumed
    trainer.cfg = SimpleNamespace(generator=SimpleNamespace(inference_engine=SimpleNamespace(enable_pd=enable_pd)))
    trainer.tracker.update_summary = Mock()
    scraper = SimpleNamespace(finalize=AsyncMock(return_value={"tokens": 10}))
    trainer._vllm_metrics_scraper = scraper
    await trainer.finalize_metrics("failed")
    await trainer.finalize_metrics("failed")
    scraper.finalize.assert_awaited_once()
    assert trainer.tracker.run_status == "failed"
    if resumed or enable_pd:
        trainer.tracker.update_summary.assert_called_once_with({"run_status": "failed"})
    else:
        assert trainer.tracker.update_summary.call_args_list[0].args == ({"tokens": 10},)


@pytest.mark.asyncio
async def test_failed_terminal_collection_does_not_publish_aggregates():
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer._metrics_finalized = False
    trainer.tracker = Tracking("test", "test", backend="console")
    trainer._resumed_from_checkpoint = False
    trainer.tracker.update_summary = Mock()
    trainer._vllm_metrics_scraper = SimpleNamespace(
        finalize=AsyncMock(side_effect=RuntimeError("scrape failed")),
    )
    await trainer.finalize_metrics("failed")
    trainer.tracker.update_summary.assert_called_once_with({"run_status": "failed"})


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError("scrape failed"), asyncio.CancelledError()])
async def test_terminal_collection_always_closes_http_client(error):
    from skyrl.train.utils.vllm_metrics_scraper import VLLMMetricsScraper

    scraper = VLLMMetricsScraper(urls=["test"])
    scraper._prev_timestamp = 0
    client = SimpleNamespace(aclose=AsyncMock())
    scraper._client = client
    scraper.sample = AsyncMock(side_effect=error)
    with pytest.raises(type(error)):
        await scraper.finalize()
    client.aclose.assert_awaited_once()
    assert scraper._client is None


@pytest.mark.asyncio
async def test_final_collection_subtracts_reused_external_engine_baseline():
    from skyrl.train.utils.vllm_metrics_scraper import VLLMMetricsScraper

    counter = "ray_vllm_generation_tokens_total"
    scraper = VLLMMetricsScraper(urls=["test"])
    scraper._read_snapshot = AsyncMock(side_effect=[{counter: 10000}, {counter: 10100}])
    with patch("skyrl.train.utils.vllm_metrics_scraper.time.monotonic", side_effect=[10, 12]):
        await scraper.sample()
        summary = await scraper.finalize()
    assert summary["vllm_correct_aggregate/combined/output_tokens_total"] == 100
    assert summary["vllm_correct_aggregate/combined/generation_throughput_tok_s"] == 50
    assert scraper._read_snapshot.await_count == 2


@pytest.mark.asyncio
async def test_unavailable_initial_export_does_not_create_a_partial_run_total():
    from skyrl.train.utils.vllm_metrics_scraper import VLLMMetricsScraper

    counter = "ray_vllm_generation_tokens_total"
    scraper = VLLMMetricsScraper(urls=["test"])
    scraper._read_snapshot = AsyncMock(side_effect=[None, {counter: 100}, {counter: 200}])
    await scraper.sample()
    await scraper.sample(generation_time_s=2)
    await scraper.sample(generation_time_s=2)
    assert scraper.run_statistics.summary() == {}


@pytest.mark.parametrize("enable_pd", [False, True])
def test_pd_preserves_step_metrics_collection(enable_pd):
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
    assert trainer._vllm_metrics_scraper is not None
    assert scraper.call_count == 1


@pytest.mark.parametrize("checkpoint_exists", [False, True])
def test_resume_guard_distinguishes_missing_latest_from_loaded_step_zero(tmp_path, checkpoint_exists):
    from skyrl.train.config import SkyRLTrainConfig

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = SkyRLTrainConfig()
    trainer.cfg.trainer.ckpt_path = str(tmp_path)
    trainer.cfg.trainer.critic.model.path = ""
    trainer._resumed_from_checkpoint = False
    trainer.dispatch = Mock()
    if checkpoint_exists:
        checkpoint = tmp_path / "global_step_0"
        checkpoint.mkdir()
        torch.save({"global_step": 0}, checkpoint / "trainer_state.pt")
        trainer.cfg.trainer.resume_path = str(checkpoint)
        trainer.resume_mode = ResumeMode.FROM_PATH
    else:
        trainer.resume_mode = ResumeMode.LATEST

    step, path = trainer.load_checkpoints()
    assert step == 0
    assert trainer._resumed_from_checkpoint == checkpoint_exists
    assert (path is not None) == checkpoint_exists

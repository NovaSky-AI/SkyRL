"""Role-specific collection for prefill/decode inference servers."""

from unittest.mock import Mock

import httpx
import pytest

from skyrl.train.utils.vllm_metrics_scraper import VLLMMetricsScraper


def _exports(phase, missing_worker=None):
    lines = []
    for worker, prompt, output, latency in [
        ("p0", 60, 1, 0.1),
        ("p1", 40, 1, 0.1),
        ("d0", 60, 30, 0.3),
        ("d1", 40, 70, 0.3),
        ("unrelated", 10000, 10000, 100),
    ]:
        if worker == missing_worker:
            continue
        labels = f'WorkerId="{worker}",engine="0"'
        for name, value in {
            "num_requests_running": 2 if worker.startswith("p") else 3,
            "num_requests_waiting": 5 if worker.startswith("p") else 1,
            "generation_tokens_total": 1000 + phase * output,
            "prompt_tokens_total": 1000 + phase * prompt,
            "prefix_cache_queries_total": 1000 + phase * prompt,
            "prefix_cache_hits_total": 100 + phase * prompt / 2,
        }.items():
            lines.append(f"ray_vllm_{name}{{{labels}}} {value}")
        for name, mean, bound in [
            ("time_to_first_token_seconds", latency, 0.2 if worker.startswith("p") else 1),
            ("request_time_per_output_token_seconds", 0.03, 0.05),
        ]:
            count = 1 + phase
            lines.extend(
                [
                    f"ray_vllm_{name}_sum{{{labels}}} {mean * count}",
                    f"ray_vllm_{name}_count{{{labels}}} {count}",
                    f'ray_vllm_{name}_bucket{{{labels},le="{bound}"}} {count}',
                    f'ray_vllm_{name}_bucket{{{labels},le="+Inf"}} {count}',
                ]
            )
    return "\n".join(lines)


@pytest.mark.asyncio
@pytest.mark.parametrize("sync", [False, True])
async def test_pd_windows_keep_role_counters_histograms_and_reused_baselines_separate(monkeypatch, sync):
    clock = {"time": 0, "phase": 0}
    monkeypatch.setattr("skyrl.train.utils.vllm_metrics_scraper.time.monotonic", lambda: clock["time"])
    scraper = VLLMMetricsScraper(urls=["http://test/metrics"])
    scraper.set_worker_roles({"prefill": ["p0", "p1"], "decode": ["d0", "d1"]})

    def transport(request):
        if sync and clock["phase"] == 0 and request.headers["role"] == "decode":
            clock["time"] += 3
        return httpx.Response(200, text=_exports(clock["phase"]))

    for role, child in scraper._role_scrapers.items():
        child._client = httpx.AsyncClient(headers={"role": role}, transport=httpx.MockTransport(transport))
    try:
        if not sync:
            await scraper.sample()
        for phase, seconds in [(1, 2), (2, 8)]:
            if sync:
                await scraper.start("vllm/train")
                scraper.pause()
                clock["time"] += 5
                scraper.resume()
            clock["phase"] = phase
            clock["time"] += seconds
            step = await scraper.stop() if sync else await scraper.sample()
            prefix = "vllm/train/" if sync else "vllm/"
            assert step[prefix + "prefill/prompt_throughput_tok_s"] == pytest.approx(100 / seconds)
            assert step[prefix + "decode/generation_throughput_tok_s"] == pytest.approx(100 / seconds)
            assert step[prefix + "prefill/prompt_throughput_cv"] == pytest.approx(0.2)
            assert step[prefix + "decode/generation_throughput_cv"] == pytest.approx(0.4)
            assert step[prefix + "prefill/num_requests_waiting"] == 10
            assert step[prefix + "decode/num_requests_waiting"] == 2
        # The sync window is closed; finalization still closes both role clients.
        summary = await scraper.finalize()
    finally:
        await scraper.aclose()
    scope = "train" if sync else "combined"
    prefix = f"vllm_correct_aggregate/{scope}/"
    assert summary[prefix + "prefill/prompt_tokens_total"] == 200
    assert summary[prefix + "decode/output_tokens_total"] == 200
    assert summary[prefix + "prefill/prompt_throughput_tok_s"] == pytest.approx(20)
    assert summary[prefix + "decode/generation_throughput_tok_s"] == pytest.approx(20)
    assert summary[prefix + "prefill/ttft_seconds_avg"] == pytest.approx(0.1)
    assert summary[prefix + "decode/ttft_seconds_avg"] == pytest.approx(0.3)
    assert summary[prefix + "prefill/ttft_seconds_p90"] == pytest.approx(0.18)
    assert summary[prefix + "decode/ttft_seconds_p90"] == pytest.approx(0.9)
    assert summary[prefix + "decode/tpot_seconds_avg"] == pytest.approx(0.03)
    assert summary[prefix + "prefill/prefix_cache_hit_rate"] == pytest.approx(0.5)
    assert summary[prefix + "decode/prefix_cache_hit_rate"] == pytest.approx(0.5)
    assert summary[prefix + "prefill/output_tokens_total"] == 4
    assert summary[prefix + "prefill/tpot_seconds_avg"] == pytest.approx(0.03)
    assert summary[prefix + "decode/prompt_tokens_total"] == 200
    assert not any(key.rsplit("/", 2)[-2] not in {"prefill", "decode"} for key in summary)
    assert all(child._client is None for child in scraper._role_scrapers.values())


@pytest.mark.asyncio
async def test_missing_prefill_worker_omits_only_prefill_summary(monkeypatch):
    clock = {"phase": 0}
    monkeypatch.setattr("skyrl.train.utils.vllm_metrics_scraper.time.monotonic", lambda: clock["phase"])
    scraper = VLLMMetricsScraper(urls=["http://test/metrics"])
    scraper.set_worker_roles({"prefill": ["p0", "p1"], "decode": ["d0", "d1"]})
    for child in scraper._role_scrapers.values():
        child._client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, text=_exports(clock["phase"], "p1" if clock["phase"] == 2 else None)
                )
            )
        )
    await scraper.sample()
    clock["phase"] = 1
    await scraper.sample()
    clock["phase"] = 2
    summary = await scraper.finalize()
    assert not any("/prefill/" in key for key in summary)
    assert summary["vllm_correct_aggregate/combined/decode/output_tokens_total"] == 200


def test_setup_uses_server_groups_to_assign_worker_roles(tmp_path, monkeypatch):
    from skyrl.train.config import SkyRLTrainConfig
    from skyrl.train.entrypoints.main_base import BasePPOExp

    cfg = SkyRLTrainConfig()
    cfg.generator.inference_engine.enable_pd = True
    cfg.trainer.fully_async.simulate_training = True
    cfg.trainer.export_path = str(tmp_path / "export")
    cfg.trainer.ckpt_path = str(tmp_path / "checkpoints")
    exp = BasePPOExp.__new__(BasePPOExp)
    exp.cfg = cfg
    exp.tokenizer = Mock()
    exp.train_dataset = exp.eval_dataset = exp.colocate_pg = None
    actors = [Mock() for _ in range(3)]
    exp._server_groups = None
    exp._prefill_server_groups = [Mock(get_actors=Mock(return_value=actors[:2]))]
    exp._decode_server_groups = [Mock(get_actors=Mock(return_value=actors[2:]))]
    trainer = Mock()
    exp.get_trainer = Mock(return_value=trainer)
    for method in ("get_tracker", "get_inference_client", "get_generator", "get_trajectory_logger"):
        setattr(exp, method, Mock())
    lookup = Mock(return_value=["p0", "p1", "d0"])
    monkeypatch.setattr("skyrl.train.entrypoints.main_base.ray.get", lookup)
    assert exp._setup_trainer() is trainer
    lookup.assert_called_once_with([a.get_ray_worker_id.remote.return_value for a in actors], timeout=10)
    trainer._vllm_metrics_scraper.set_worker_roles.assert_called_once_with({"prefill": ["p0", "p1"], "decode": ["d0"]})
    trainer._vllm_metrics_scraper.set_worker_ids.assert_not_called()

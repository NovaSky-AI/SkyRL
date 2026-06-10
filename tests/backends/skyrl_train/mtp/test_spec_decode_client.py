"""CPU unit tests for InferenceEngineClient.get_spec_decode_metrics aggregation (MTP draft).

The client sums the per-engine cumulative spec-decode counters; the generator turns the per-step
delta into an acceptance rate (see test_spec_decode_metrics.py for that piece).

uv run --isolated --extra dev pytest tests/backends/skyrl_train/mtp/test_spec_decode_client.py
"""

import pytest

from skyrl.backends.skyrl_train.inference_engines.inference_engine_client import (
    InferenceEngineClient,
)
from skyrl.train.config import (
    GeneratorConfig,
    InferenceEngineConfig,
    ModelConfig,
    PolicyConfig,
    SkyRLTrainConfig,
    TrainerConfig,
)


def _make_min_cfg():
    return SkyRLTrainConfig(
        trainer=TrainerConfig(
            policy=PolicyConfig(model=ModelConfig(path="dummy-model")),
        ),
        generator=GeneratorConfig(
            inference_engine=InferenceEngineConfig(
                backend="vllm",
                enable_http_endpoint=False,
                http_endpoint_host="127.0.0.1",
                http_endpoint_port=0,
            ),
        ),
    )


def _make_spec_client(per_engine_stats):
    """Build a client whose engines report the given per-engine spec-decode stats."""

    class MockEngine:
        def __init__(self, stats):
            self._stats = stats

        def dp_size(self):
            return 1

        async def get_spec_decode_metrics(self):
            return self._stats

    cfg = _make_min_cfg()
    return InferenceEngineClient(
        engines=[MockEngine(s) for s in per_engine_stats],
        tokenizer=object(),
        model_path=cfg.trainer.policy.model.path,
        lora_cfg=cfg.trainer.policy.model.lora,
        inference_engine_cfg=cfg.generator.inference_engine,
    )


@pytest.mark.asyncio
async def test_get_spec_decode_metrics_sums_across_engines():
    client = _make_spec_client(
        [
            {"num_drafts": 3, "num_draft_tokens": 10, "num_accepted_tokens": 6},
            {"num_drafts": 1, "num_draft_tokens": 4, "num_accepted_tokens": 2},
        ]
    )
    totals = await client.get_spec_decode_metrics()
    assert totals == {"num_drafts": 4, "num_draft_tokens": 14, "num_accepted_tokens": 8}


@pytest.mark.asyncio
async def test_get_spec_decode_metrics_returns_none_when_no_engine_reports():
    # Speculative decoding disabled / unsupported on every engine -> None (not a zero-dict).
    client = _make_spec_client([None, None])
    assert await client.get_spec_decode_metrics() is None


@pytest.mark.asyncio
async def test_get_spec_decode_metrics_ignores_non_reporting_engines():
    # A mix of reporting and non-reporting engines still aggregates the reporters only.
    client = _make_spec_client([None, {"num_drafts": 2, "num_draft_tokens": 8, "num_accepted_tokens": 5}])
    totals = await client.get_spec_decode_metrics()
    assert totals == {"num_drafts": 2, "num_draft_tokens": 8, "num_accepted_tokens": 5}

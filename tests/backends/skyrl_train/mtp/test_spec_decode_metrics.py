"""CPU unit tests for vLLM spec-decode (MTP draft) acceptance accounting.

uv run --isolated --extra dev --extra megatron pytest tests/backends/skyrl_train/mtp/test_spec_decode_metrics.py
"""

import asyncio
import types

from skyrl.backends.skyrl_train.inference_engines.vllm.spec_decode_metrics import (
    make_spec_decode_stat_logger_class,
    sum_spec_decode_loggers,
)
from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator


def _sched(num_draft, num_accepted, num_drafts):
    return types.SimpleNamespace(
        spec_decoding_stats=types.SimpleNamespace(
            num_drafts=num_drafts, num_draft_tokens=num_draft, num_accepted_tokens=num_accepted
        )
    )


def test_logger_accumulates_across_iterations():
    cls = make_spec_decode_stat_logger_class()
    lg = cls(vllm_config=None, engine_index=0)
    lg.record(scheduler_stats=_sched(10, 7, 5))
    lg.record(scheduler_stats=_sched(8, 6, 4))
    lg.record(scheduler_stats=types.SimpleNamespace(spec_decoding_stats=None))  # no spec this iter
    assert (lg.num_draft_tokens, lg.num_accepted_tokens, lg.num_drafts) == (18, 13, 9)


def test_sum_spec_decode_loggers():
    cls = make_spec_decode_stat_logger_class()
    a = cls(None, 0)
    a.num_draft_tokens, a.num_accepted_tokens, a.num_drafts = 10, 6, 3
    b = cls(None, 1)
    b.num_draft_tokens, b.num_accepted_tokens, b.num_drafts = 4, 2, 1
    assert sum_spec_decode_loggers([a, b]) == {
        "num_drafts": 4,
        "num_draft_tokens": 14,
        "num_accepted_tokens": 8,
    }
    assert sum_spec_decode_loggers([]) is None


class _FakeClient:
    def __init__(self, values):
        self._values = list(values)
        self._i = 0

    async def get_spec_decode_metrics(self):
        v = self._values[min(self._i, len(self._values) - 1)]
        self._i += 1
        return v


def _make_generator(client):
    gen = SkyRLGymGenerator.__new__(SkyRLGymGenerator)
    gen.inference_engine_client = client
    gen._prev_spec_decode = None
    return gen


def test_generator_acceptance_rate_per_step_delta():
    # cumulative counters grow each step; the metric is the per-step delta ratio.
    client = _FakeClient(
        [
            {"num_drafts": 5, "num_draft_tokens": 10, "num_accepted_tokens": 6},
            {"num_drafts": 11, "num_draft_tokens": 22, "num_accepted_tokens": 15},
        ]
    )
    gen = _make_generator(client)

    m1 = asyncio.run(gen._spec_decode_rollout_metrics())
    assert m1["vllm/draft_num_draft_tokens"] == 10
    assert m1["vllm/draft_num_accepted_tokens"] == 6
    assert abs(m1["vllm/draft_acceptance_rate"] - 0.6) < 1e-9

    m2 = asyncio.run(gen._spec_decode_rollout_metrics())
    # deltas: drafted 22-10=12, accepted 15-6=9 -> 0.75
    assert m2["vllm/draft_num_draft_tokens"] == 12
    assert m2["vllm/draft_num_accepted_tokens"] == 9
    assert abs(m2["vllm/draft_acceptance_rate"] - 0.75) < 1e-9


def test_generator_no_spec_decode_returns_empty():
    gen = _make_generator(_FakeClient([None]))
    assert asyncio.run(gen._spec_decode_rollout_metrics()) == {}

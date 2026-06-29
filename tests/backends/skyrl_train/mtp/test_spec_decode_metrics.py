"""CPU unit tests for vLLM spec-decode (MTP draft) acceptance accounting.

uv run --isolated --extra dev --extra megatron pytest tests/backends/skyrl_train/mtp/test_spec_decode_metrics.py
"""

import types

import pytest

from skyrl.backends.skyrl_train.inference_engines.vllm.spec_decode_metrics import (
    acceptance_rate_metrics,
    make_spec_decode_stat_logger_class,
    merge_spec_decode_counters,
    sum_spec_decode_loggers,
)


def _logger_cls():
    # The stat-logger class factory lazily imports vLLM (it subclasses StatLoggerBase); skip the
    # logger-class tests in vLLM-less environments instead of failing. The pure aggregation /
    # rate-math tests below run everywhere.
    pytest.importorskip("vllm")
    return make_spec_decode_stat_logger_class()


def _sched(num_draft, num_accepted, num_drafts, per_pos=None):
    return types.SimpleNamespace(
        spec_decoding_stats=types.SimpleNamespace(
            num_drafts=num_drafts,
            num_draft_tokens=num_draft,
            num_accepted_tokens=num_accepted,
            num_accepted_tokens_per_pos=per_pos if per_pos is not None else [],
        )
    )


def test_logger_accumulates_across_iterations():
    cls = _logger_cls()
    lg = cls(vllm_config=None, engine_index=0)
    lg.record(scheduler_stats=_sched(10, 7, 5))
    lg.record(scheduler_stats=_sched(8, 6, 4))
    lg.record(scheduler_stats=types.SimpleNamespace(spec_decoding_stats=None))  # no spec this iter
    assert (lg.num_draft_tokens, lg.num_accepted_tokens, lg.num_drafts) == (18, 13, 9)


def test_logger_accumulates_per_position_and_grows():
    # The per-position list adapts to whatever depth vLLM reports (here it grows 2 -> 3, e.g. the
    # configured num_speculative_tokens), with missing tail positions treated as 0.
    cls = _logger_cls()
    lg = cls(vllm_config=None, engine_index=0)
    lg.record(scheduler_stats=_sched(10, 7, 5, per_pos=[5, 2]))
    lg.record(scheduler_stats=_sched(12, 9, 4, per_pos=[4, 3, 2]))
    lg.record(scheduler_stats=types.SimpleNamespace(spec_decoding_stats=None))
    assert lg.num_accepted_tokens_per_pos == [9, 5, 2]


def test_logger_tolerates_stats_without_per_position():
    # Older vLLM stats objects without num_accepted_tokens_per_pos must not break accounting.
    cls = _logger_cls()
    lg = cls(vllm_config=None, engine_index=0)
    stats = types.SimpleNamespace(
        spec_decoding_stats=types.SimpleNamespace(num_drafts=3, num_draft_tokens=6, num_accepted_tokens=4)
    )
    lg.record(scheduler_stats=stats)
    assert (lg.num_draft_tokens, lg.num_accepted_tokens, lg.num_drafts) == (6, 4, 3)
    assert lg.num_accepted_tokens_per_pos == []


def test_sum_spec_decode_loggers():
    # sum_spec_decode_loggers only reads counter attributes, so plain namespaces stand in for
    # logger instances and the test runs without vLLM.
    a = types.SimpleNamespace(
        num_draft_tokens=10, num_accepted_tokens=6, num_drafts=3, num_accepted_tokens_per_pos=[4, 2]
    )
    b = types.SimpleNamespace(
        num_draft_tokens=4, num_accepted_tokens=2, num_drafts=1, num_accepted_tokens_per_pos=[1, 1, 0]
    )
    assert sum_spec_decode_loggers([a, b]) == {
        "num_drafts": 4,
        "num_draft_tokens": 14,
        "num_accepted_tokens": 8,
        "num_accepted_tokens_per_pos": [5, 3, 0],
    }
    assert sum_spec_decode_loggers([]) is None


def test_merge_spec_decode_counters_across_engines():
    totals = {}
    merge_spec_decode_counters(
        totals,
        {"num_drafts": 3, "num_draft_tokens": 9, "num_accepted_tokens": 6, "num_accepted_tokens_per_pos": [3, 2, 1]},
    )
    merge_spec_decode_counters(
        totals,
        {"num_drafts": 2, "num_draft_tokens": 4, "num_accepted_tokens": 3, "num_accepted_tokens_per_pos": [2, 1]},
    )
    assert totals == {
        "num_drafts": 5,
        "num_draft_tokens": 13,
        "num_accepted_tokens": 9,
        "num_accepted_tokens_per_pos": [5, 3, 1],
    }


def test_acceptance_rate_per_step_delta():
    # cumulative counters grow each step; the metric is the per-step delta ratio.
    prev = None
    m1, prev = acceptance_rate_metrics({"num_drafts": 5, "num_draft_tokens": 10, "num_accepted_tokens": 6}, prev)
    assert m1["vllm/draft_num_draft_tokens"] == 10
    assert m1["vllm/draft_num_accepted_tokens"] == 6
    assert abs(m1["vllm/draft_acceptance_rate"] - 0.6) < 1e-9

    m2, prev = acceptance_rate_metrics({"num_drafts": 11, "num_draft_tokens": 22, "num_accepted_tokens": 15}, prev)
    # deltas: drafted 22-10=12, accepted 15-6=9 -> 0.75
    assert m2["vllm/draft_num_draft_tokens"] == 12
    assert m2["vllm/draft_num_accepted_tokens"] == 9
    assert abs(m2["vllm/draft_acceptance_rate"] - 0.75) < 1e-9


def test_acceptance_rate_per_position():
    # Per-position rate = drafts this step whose k-th speculated token was accepted / drafts this
    # step (1-based keys). One key per configured draft position; non-increasing in k.
    prev = None
    m1, prev = acceptance_rate_metrics(
        {
            "num_drafts": 10,
            "num_draft_tokens": 30,
            "num_accepted_tokens": 17,
            "num_accepted_tokens_per_pos": [8, 6, 3],
        },
        prev,
    )
    assert abs(m1["vllm/draft_acceptance_rate_pos_1"] - 0.8) < 1e-9
    assert abs(m1["vllm/draft_acceptance_rate_pos_2"] - 0.6) < 1e-9
    assert abs(m1["vllm/draft_acceptance_rate_pos_3"] - 0.3) < 1e-9

    # Second step: only the deltas count (drafts 10 -> 14, pos counts [8,6,3] -> [11,8,4]).
    m2, prev = acceptance_rate_metrics(
        {
            "num_drafts": 14,
            "num_draft_tokens": 42,
            "num_accepted_tokens": 24,
            "num_accepted_tokens_per_pos": [11, 8, 4],
        },
        prev,
    )
    assert abs(m2["vllm/draft_acceptance_rate_pos_1"] - 3 / 4) < 1e-9
    assert abs(m2["vllm/draft_acceptance_rate_pos_2"] - 2 / 4) < 1e-9
    assert abs(m2["vllm/draft_acceptance_rate_pos_3"] - 1 / 4) < 1e-9


def test_acceptance_rate_per_position_depth_one_single_key():
    # Depth 1 (the current default) emits exactly one per-position key.
    metrics, _ = acceptance_rate_metrics(
        {"num_drafts": 4, "num_draft_tokens": 4, "num_accepted_tokens": 3, "num_accepted_tokens_per_pos": [3]},
        None,
    )
    pos_keys = [k for k in metrics if "_pos_" in k]
    assert pos_keys == ["vllm/draft_acceptance_rate_pos_1"]
    assert abs(metrics["vllm/draft_acceptance_rate_pos_1"] - 0.75) < 1e-9


def test_acceptance_rate_prefix_separates_train_and_eval():
    # Mirrors the trainer's eval-step sequence: a train rollout records under vllm/*, then the eval
    # rollout records under vllm/eval/* using the SAME advancing baseline, so eval counts don't leak
    # into the next train step.
    prev = None
    # Train rollout: cumulative after train generate.
    train, prev = acceptance_rate_metrics(
        {"num_drafts": 10, "num_draft_tokens": 20, "num_accepted_tokens": 12}, prev, prefix="vllm/"
    )
    assert train["vllm/draft_num_draft_tokens"] == 20
    assert abs(train["vllm/draft_acceptance_rate"] - 0.6) < 1e-9

    # Eval rollout: cumulative grows a lot; reported under vllm/eval/* as the delta vs the train
    # baseline (drafted 120-20=100, accepted 90-12=78 -> 0.78).
    eval_m, prev = acceptance_rate_metrics(
        {"num_drafts": 60, "num_draft_tokens": 120, "num_accepted_tokens": 90}, prev, prefix="vllm/eval/"
    )
    assert eval_m["vllm/eval/draft_num_draft_tokens"] == 100
    assert abs(eval_m["vllm/eval/draft_acceptance_rate"] - 0.78) < 1e-9
    assert not any(k.startswith("vllm/draft_") for k in eval_m)

    # Next train step: delta is from the post-eval baseline, so eval is NOT counted again
    # (drafted 140-120=20, accepted 105-90=15 -> 0.75).
    train2, prev = acceptance_rate_metrics(
        {"num_drafts": 70, "num_draft_tokens": 140, "num_accepted_tokens": 105}, prev, prefix="vllm/"
    )
    assert train2["vllm/draft_num_draft_tokens"] == 20
    assert abs(train2["vllm/draft_acceptance_rate"] - 0.75) < 1e-9


def test_acceptance_rate_no_spec_decode_returns_empty():
    metrics, prev = acceptance_rate_metrics(None, None)
    assert metrics == {}
    assert prev is None


def test_acceptance_rate_no_drafts_omits_rate():
    # zero drafted tokens -> report counts but no (undefined) rate.
    metrics, _ = acceptance_rate_metrics({"num_drafts": 0, "num_draft_tokens": 0, "num_accepted_tokens": 0}, None)
    assert metrics == {"vllm/draft_num_draft_tokens": 0, "vllm/draft_num_accepted_tokens": 0}

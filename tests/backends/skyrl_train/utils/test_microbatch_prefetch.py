"""uv run --isolated --extra dev pytest tests/backends/skyrl_train/utils/test_microbatch_prefetch.py"""

import gc
import weakref

import pytest
import torch

from skyrl.backends.skyrl_train.utils.microbatch_prefetch import MicrobatchPrefetcher


def _prefetcher(n=4, enabled=True, stage=None):
    built = []

    def default_stage(item):
        built.append(item)
        return {"x": torch.full((2,), float(item))}

    host = list(range(n))
    prefetcher = MicrobatchPrefetcher(
        host, stage or default_stage, lambda item: item.values(), torch.device("cpu"), enabled=enabled
    )
    return prefetcher, built


@pytest.mark.parametrize(
    "schedule",
    [
        # no pipelining: next, prefetch per microbatch
        ["n", "p"] * 4,
        # a next before the previous prefetch, prefetch after exhaustion, repeated prefetches
        ["n", "n", "p", "p", "n", "p", "n", "p", "p"],
        # never prefetched
        ["n"] * 4,
    ],
)
def test_every_microbatch_is_yielded_in_order_and_built_once(schedule):
    prefetcher, built = _prefetcher()
    seen = []
    for op in schedule:
        if op == "n":
            seen.append(next(prefetcher)["x"][0].item())
        else:
            prefetcher.prefetch()

    assert seen == [0.0, 1.0, 2.0, 3.0]
    assert built == [0, 1, 2, 3]
    with pytest.raises(StopIteration):
        next(prefetcher)


def test_disabled_prefetching_stages_each_microbatch_when_consumed():
    prefetcher, built = _prefetcher(enabled=False)
    next(prefetcher)
    prefetcher.prefetch()
    assert built == [0]
    assert [item["x"][0].item() for item in prefetcher] == [1.0, 2.0, 3.0]


def test_a_failed_stage_names_its_microbatch():
    def stage(item):
        if item == 2:
            raise ValueError("bad microbatch")
        return {"x": torch.zeros(1)}

    prefetcher, _ = _prefetcher(stage=stage)
    next(prefetcher)
    next(prefetcher)
    with pytest.raises(ValueError, match="bad microbatch") as raised:
        prefetcher.prefetch()
    assert "while staging microbatch 2" in raised.value.__notes__


def test_an_exhausted_prefetcher_holds_no_microbatch():
    prefetcher, _ = _prefetcher(n=2)
    first = next(prefetcher)
    prefetcher.prefetch()
    ref = weakref.ref(first["x"])
    del first
    second = next(prefetcher)
    gc.collect()
    assert ref() is None and second is not None and next(prefetcher, None) is None


def test_staging_the_computation_waits_for_is_traced_apart(monkeypatch):
    from skyrl.utils import stage_trace

    monkeypatch.setattr(stage_trace, "_totals", {})
    prefetcher, _ = _prefetcher()
    next(prefetcher)  # nothing staged ahead: exposed
    prefetcher.prefetch()
    next(prefetcher)  # staged while the previous one computed
    next(prefetcher)  # consumed before its prefetch: exposed
    taken = stage_trace.take_totals("mb.")
    assert taken["mb.stage_exposed"][0] == 2
    assert taken["mb.stage"][0] == 1

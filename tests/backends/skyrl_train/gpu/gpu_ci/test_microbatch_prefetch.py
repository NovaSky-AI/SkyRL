"""The prefetcher's stream ordering on a GPU: a delayed copy is still waited for, and the waits matter.

uv run --isolated --extra dev pytest tests/backends/skyrl_train/gpu/gpu_ci/test_microbatch_prefetch.py
"""

import pytest
import torch

from skyrl.backends.skyrl_train.utils.microbatch_prefetch import MicrobatchPrefetcher

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

DELAY_CYCLES = 200_000_000  # ~0.1 s: the copy lands long after the consumer would read


def _delayed_upload(value: torch.Tensor) -> dict:
    """Upload ``value`` from pinned memory after a delay on the current (copy) stream."""
    torch.cuda._sleep(DELAY_CYCLES)
    return {"x": value.pin_memory().to("cuda", non_blocking=True)}


def _consume(prefetcher: MicrobatchPrefetcher, n: int) -> list:
    seen = []
    for _ in range(n):
        item = next(prefetcher)
        # Read on the compute stream right away, as a kernel would.
        seen.append(item["x"].clone())
        prefetcher.prefetch()
    torch.cuda.synchronize()
    return [t.cpu() for t in seen]


def test_the_consumer_waits_for_a_delayed_copy():
    host = [torch.full((1 << 20,), float(i)) for i in range(4)]
    prefetcher = MicrobatchPrefetcher(host, _delayed_upload, lambda item: item.values(), torch.device("cuda"))

    seen = _consume(prefetcher, len(host))

    for value, expected in zip(seen, host):
        assert torch.equal(value, expected)
    assert next(prefetcher, None) is None


def test_without_the_wait_the_consumer_reads_too_early(monkeypatch):
    """Negative control: proves the test above would catch a missing ``wait_event``."""
    monkeypatch.setattr(torch.cuda.Stream, "wait_event", lambda self, event: None)
    host = [torch.full((1 << 20,), float(i + 1)) for i in range(4)]
    prefetcher = MicrobatchPrefetcher(host, _delayed_upload, lambda item: item.values(), torch.device("cuda"))

    seen = _consume(prefetcher, len(host))

    assert any(not torch.equal(value, expected) for value, expected in zip(seen, host))


def test_staged_tensors_are_recorded_on_the_consuming_stream(monkeypatch):
    recorded = []
    original = torch.Tensor.record_stream
    monkeypatch.setattr(
        torch.Tensor, "record_stream", lambda self, stream: recorded.append(stream) or original(self, stream)
    )
    host = [torch.zeros(8) for _ in range(2)]
    prefetcher = MicrobatchPrefetcher(host, _delayed_upload, lambda item: item.values(), torch.device("cuda"))

    _consume(prefetcher, len(host))

    assert recorded == [torch.cuda.current_stream()] * 2

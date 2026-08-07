"""The live inference-server set is captured LATE in ``save_weights_for_sampler``.

``pause_generation`` fans out to the backends directly, so it is itself a detector:
when an engine dies outside a generation phase, that pause is the first call to
touch the corpse and it reconciles the fleet. Reading membership *before* it hands
the weight sync a set that still contains the dead consumer -- the producers then
block on a ``free_gather`` that can never arrive and the run dies at the stall
watchdog instead of degrading.

This is a pure ordering property, so it is tested by ordering: a recording stand-in
for the dispatch's collaborators, asserting where the read lands in the call
sequence. Everything here is CPU-only and touches no Ray.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch


class _RecordingClient:
    """Client whose control-plane calls append to a shared trace.

    ``active_server_urls`` shrinks the first time ``pause_generation`` runs, which
    is exactly the sequence a kill outside a generation phase produces.
    """

    def __init__(self, trace, urls, die_on_pause=False):
        self._trace = trace
        self.server_urls = list(urls)
        self._active = list(urls)
        self._die_on_pause = die_on_pause

    @property
    def active_server_urls(self):
        self._trace.append(f"read_active->{len(self._active)}")
        return list(self._active)

    async def pause_generation(self, *a, **kw):
        self._trace.append("pause")
        if self._die_on_pause:
            # The fan-out touched the dead engine and reconciled it away.
            self._active = self._active[:-1]
        return {}

    async def resume_generation(self, *a, **kw):
        self._trace.append("resume")
        return {}

    async def wake_up(self, *a, **kw):
        self._trace.append("wake_up")
        return {}

    def increment_weight_version(self):
        self._trace.append("bump_version")


def _dispatch(client, trace, colocate_all=False, backend="sharded_rdt"):
    d = WorkerDispatch.__new__(WorkerDispatch)
    d._inference_engine_client = client
    d.colocate_all = colocate_all
    d.cfg = SimpleNamespace(
        trainer=SimpleNamespace(
            strategy="fsdp",
            policy=SimpleNamespace(
                model=SimpleNamespace(lora=SimpleNamespace(rank=0)),
                megatron_config=SimpleNamespace(lora_config=SimpleNamespace(merge_lora=True)),
            ),
        ),
        generator=SimpleNamespace(inference_engine=SimpleNamespace(weight_sync_backend=backend)),
    )
    d._prepare_for_weight_sync = lambda: trace.append("prepare")
    d._finish_weight_sync = lambda: trace.append("finish")
    d.ensure_active_adapter = MagicMock()

    captured = {}

    def _broadcast(_client, model_id=None, live_server_urls=None):
        trace.append(f"broadcast(live={None if live_server_urls is None else len(live_server_urls)})")
        captured["live"] = live_server_urls

    d._broadcast_to_inference_engines = _broadcast
    return d, captured


URLS = ["http://a", "http://b", "http://c", "http://d"]


@pytest.mark.asyncio
async def test_live_set_is_read_after_pause():
    """The ordering itself: pause, THEN read membership, THEN broadcast."""
    trace = []
    client = _RecordingClient(trace, URLS)
    d, _ = _dispatch(client, trace)

    await d.save_weights_for_sampler()

    assert trace.index("pause") < trace.index("read_active->4"), trace
    assert [t for t in trace if t.startswith("broadcast")][0] == "broadcast(live=None)"


@pytest.mark.asyncio
async def test_a_death_detected_by_pause_reaches_the_same_sync():
    """The regression this guards. The engine dies outside a generation phase, so
    the pause fan-out is what discovers it -- and the sync that immediately follows
    must already be degraded, not one step behind."""
    trace = []
    client = _RecordingClient(trace, URLS, die_on_pause=True)
    d, captured = _dispatch(client, trace)

    await d.save_weights_for_sampler()

    assert captured["live"] == URLS[:3], f"sync ran against a stale live set: {captured['live']}"
    assert "read_active->3" in trace, trace


@pytest.mark.asyncio
async def test_a_whole_fleet_passes_none_through():
    """None is not cosmetic: it makes a non-degraded sync take the exact path it
    took before fault tolerance existed."""
    trace = []
    client = _RecordingClient(trace, URLS)
    d, captured = _dispatch(client, trace)

    await d.save_weights_for_sampler()

    assert captured["live"] is None


@pytest.mark.asyncio
async def test_colocated_reads_after_wake_up():
    trace = []
    client = _RecordingClient(trace, URLS)
    d, _ = _dispatch(client, trace, colocate_all=True)

    await d.save_weights_for_sampler()

    assert trace.index("wake_up") < trace.index("read_active->4"), trace


@pytest.mark.asyncio
async def test_a_degraded_fleet_refuses_a_non_rdt_broadcast():
    """NCCL and cuda_ipc broadcast over a communicator fixed at provision. A
    partial broadcast is not a degraded sync -- it is a hang, or silently stale
    weights on the survivors. validate_cfg already rejects the combination; this is
    the runtime backstop."""
    trace = []
    client = _RecordingClient(trace, URLS, die_on_pause=True)
    d, _ = _dispatch(client, trace, backend="nccl")

    with pytest.raises(RuntimeError, match="cannot sync to a subset"):
        await d.save_weights_for_sampler()


@pytest.mark.asyncio
async def test_a_client_without_the_ft_view_is_treated_as_whole():
    """Other InferenceEngineInterface implementations have no active_server_urls."""
    trace = []
    client = _RecordingClient(trace, URLS)
    del type(client).active_server_urls  # type: ignore[attr-defined]
    try:
        d, captured = _dispatch(client, trace)
        await d.save_weights_for_sampler()
        assert captured["live"] is None
    finally:
        type(client).active_server_urls = property(lambda self: list(self._active))  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_no_client_still_raises():
    trace = []
    d, _ = _dispatch(_RecordingClient(trace, URLS), trace)
    d._inference_engine_client = None
    with pytest.raises(RuntimeError, match="no inference_engine_client"):
        await d.save_weights_for_sampler()

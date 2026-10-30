"""build_snapshots.py: a reactivated snapshot is reported ready only once Daytona says it is active."""

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("daytona")

from examples.train_integrations.harbor_skycap.swebench_multimodal import (  # noqa: E402
    build_snapshots,
)


class FakeSnapshots:
    def __init__(self, states):
        self.states = list(states)
        self.activated = 0

    async def get(self, name):
        state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        return SimpleNamespace(state=state)

    async def activate(self, snapshot):
        self.activated += 1


async def _build(states, monkeypatch):
    real = build_snapshots.settled_state

    async def settled(daytona, name):
        return await real(daytona, name, timeout=0.05, interval=0.01)

    monkeypatch.setattr(build_snapshots, "settled_state", settled)
    daytona = SimpleNamespace(snapshot=FakeSnapshots(states))
    result = await build_snapshots.build(daytona, "snap", "image", None, asyncio.Semaphore(1), attempts=1)
    return result, daytona.snapshot


@pytest.mark.asyncio
async def test_a_reactivated_snapshot_is_ready_once_it_is_active(monkeypatch):
    result, snapshots = await _build(["inactive", "pending", "pending", "active"], monkeypatch)

    assert result == "snap: reactivated" and snapshots.activated == 1


@pytest.mark.asyncio
async def test_a_reactivated_snapshot_still_pending_is_not_ready(monkeypatch):
    result, _ = await _build(["inactive", "pending"], monkeypatch)

    assert result.startswith("snap: NOT READY (pending)")

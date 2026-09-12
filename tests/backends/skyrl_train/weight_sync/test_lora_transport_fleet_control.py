import pytest

from skyrl.backends.skyrl_train.weight_sync.lora_transport.fleet_control import (
    LoRATransportFleetTransaction,
)


@pytest.mark.asyncio
async def test_fleet_transaction_restores_every_server_after_activation_failure():
    events = []

    async def stage(url):
        events.append(("stage", url))

    async def pause():
        events.append(("pause",))

    async def activate(url):
        events.append(("activate", url))
        if url == "b":
            raise RuntimeError("forced activation failure")

    async def rollback(url):
        events.append(("rollback", url))

    async def commit(url):
        events.append(("commit", url))

    async def resume():
        events.append(("resume",))

    with pytest.raises(RuntimeError, match="forced activation failure"):
        await LoRATransportFleetTransaction(["a", "b"]).replace(
            stage, pause, activate, rollback, commit, resume
        )

    assert events == [
        ("stage", "a"),
        ("stage", "b"),
        ("pause",),
        ("activate", "a"),
        ("activate", "b"),
        ("rollback", "a"),
        ("rollback", "b"),
        ("resume",),
    ]


@pytest.mark.asyncio
async def test_fleet_transaction_does_not_pause_when_staging_fails():
    events = []

    async def stage(url):
        events.append(("stage", url))
        if url == "b":
            raise RuntimeError("forced stage failure")

    async def impossible(*_):
        raise AssertionError("phase should not run")

    async def rollback(url):
        events.append(("rollback", url))

    with pytest.raises(RuntimeError, match="forced stage failure"):
        await LoRATransportFleetTransaction(["a", "b"]).replace(
            stage, impossible, impossible, rollback, impossible, impossible
        )

    assert events == [
        ("stage", "a"),
        ("stage", "b"),
        ("rollback", "a"),
        ("rollback", "b"),
    ]


@pytest.mark.asyncio
async def test_failed_fleet_rollback_keeps_mixed_generations_paused():
    active = {"a": 1, "b": 1}
    resumed = False

    async def stage(url):
        pass

    async def pause():
        pass

    async def activate(url):
        if url == "b":
            raise RuntimeError("activation failed")
        active[url] = 2

    async def rollback(url):
        if url == "a":
            raise RuntimeError("rollback failed")
        active[url] = 1

    async def commit(url):
        raise AssertionError("commit must not run")

    async def resume():
        nonlocal resumed
        resumed = True

    with pytest.raises(RuntimeError, match="activation failed") as exc_info:
        await LoRATransportFleetTransaction(["a", "b"]).replace(
            stage, pause, activate, rollback, commit, resume
        )

    assert str(exc_info.value) == "activation failed"
    assert exc_info.value.__notes__ == [
        "LoRA fleet rollback also failed: LoRA fleet rollback failed"
    ]
    assert active == {"a": 2, "b": 1}
    assert not resumed


@pytest.mark.asyncio
async def test_failed_retirement_resumes_only_the_uniform_new_generation():
    active = {"a": 1, "b": 1}
    resumed = []

    async def stage(url):
        pass

    async def pause():
        pass

    async def activate(url):
        active[url] = 2

    async def rollback(url):
        raise AssertionError("cannot roll back after retirement started")

    async def commit(url):
        if url == "b":
            raise RuntimeError("retirement failed")

    async def resume():
        resumed.append(dict(active))

    with pytest.raises(RuntimeError, match="failed to retire"):
        await LoRATransportFleetTransaction(["a", "b"]).replace(
            stage, pause, activate, rollback, commit, resume
        )

    assert resumed == [{"a": 2, "b": 2}]


@pytest.mark.asyncio
async def test_fleet_transaction_rolls_back_before_activation_when_producer_fails():
    events = []

    async def stage(url):
        events.append(("stage", url))

    async def producer_ready():
        events.append(("producer_ready",))
        raise RuntimeError("producer send failed")

    async def rollback(url):
        events.append(("rollback", url))

    async def impossible(*_):
        raise AssertionError("activation phase must not run")

    with pytest.raises(RuntimeError, match="producer send failed"):
        await LoRATransportFleetTransaction(["a", "b"]).replace(
            stage,
            impossible,
            impossible,
            rollback,
            impossible,
            impossible,
            prepare_activation=producer_ready,
        )

    assert events == [
        ("stage", "a"),
        ("stage", "b"),
        ("producer_ready",),
        ("rollback", "a"),
        ("rollback", "b"),
    ]

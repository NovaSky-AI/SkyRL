"""The SDK path a training harness actually drives, end to end.

Three of the four bugs the 2026-09-15 Harbor run hit were on this path and none
was visible to the suite, because nothing joined the steps up:

* a trajectory created with a **caller-supplied name** was unroutable -- created
  successfully, handed back a working-looking base URL, then 404 on every turn;
* ``Trajectory.finish()`` dropped ``annotations``, which is how a reward reaches
  the export;
* ``Trajectory.export()`` never decompressed, so every call died in the JSON
  parser.

Each was individually covered. The join was not. So this drives the real SDK --
sync, over a real socket, exactly as a harness would -- through create, a turn,
finish with a reward, and export.

The SDK is synchronous and the server runs in this event loop, so every SDK call
goes through ``asyncio.to_thread``; calling it inline would deadlock.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from skyrl_capture.sdk import CaptureClient, create_trajectory


def _client(stack: Any) -> CaptureClient:
    return CaptureClient(stack.base_url)


# `0_1` is SkyRL's own trajectory id shape and is what broke: it does not start
# with the generated `tr_` prefix the dispatcher used to key on. `v1` is the
# harder case -- it used to collide with the control plane's own first path
# segment, and is legal now only because the data plane has its own prefix.
@pytest.mark.parametrize("name", ["0_1", "v1", "healthz", "run-3.step-2"])
async def test_a_caller_named_trajectory_is_routable(tokens_stack, name):
    """Naming a trajectory is the whole point of `trajectory_id=`."""
    client = _client(tokens_stack)
    trajectory = await asyncio.to_thread(
        create_trajectory,
        project="sdk-e2e",
        trajectory_id=name,
        client=client,
    )
    assert trajectory.id == name

    response = await tokens_stack.chat(
        {"base_url": trajectory.base_url},
        [{"role": "user", "content": "hello"}],
        model="mock-tokens-model",
    )
    assert response.status_code == 200, (
        f"a trajectory named {name!r} was created but its route answered "
        f"{response.status_code}: {response.text[:200]}"
    )


async def test_create_turn_finish_with_reward_then_export(tokens_stack):
    """The whole loop, in the order a harness runs it."""
    client = _client(tokens_stack)
    trajectory = await asyncio.to_thread(
        create_trajectory,
        project="sdk-e2e",
        trajectory_id="0_0",
        client=client,
    )

    handle = {"base_url": trajectory.base_url}
    first = await tokens_stack.chat(handle, [{"role": "user", "content": "hello"}],
                                    model="mock-tokens-model")
    assert first.status_code == 200, first.text
    reply = first.json()["choices"][0]["message"]

    # A second turn, so the export has a multi-node path rather than one reply.
    second = await tokens_stack.chat(
        handle,
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": reply["content"]},
            {"role": "user", "content": "and again"},
        ],
        model="mock-tokens-model",
    )
    assert second.status_code == 200, second.text
    await tokens_stack.settle()

    # Finishing and scoring in one call is what a harness does in its `finally`.
    result = await asyncio.to_thread(
        trajectory.finish,
        labels=["sdk-e2e"],
        annotations={"reward": 1.0, "verifier": "unit"},
        command_result="success",
    )
    assert result["status"] == "finished", result

    # `export()` runs a real export job. Nothing drains it in tests, so drive
    # the drain from this loop while the SDK blocks on it in a thread.
    task = asyncio.create_task(asyncio.to_thread(trajectory.export, "token-samples"))
    while not task.done():
        await tokens_stack.runtime.exports.drain()
        await asyncio.sleep(0.02)
    rows = await task

    assert rows, "export returned no rows"
    for row in rows:
        # Decompressed and parsed: before the fix this raised in orjson.
        assert isinstance(row, dict)
        assert row["trajectory_id"] == "0_0"
        assert len(row["input_ids"]) == len(row["loss_mask"])
        # The reward has to survive `finish()` -> storage -> export, on every
        # branch, because annotations belong to the whole tree.
        assert row["annotations"]["reward"] == 1.0
        assert row["annotations"]["verifier"] == "unit"
        assert "sdk-e2e" in row["labels"]
    assert any(sum(row["loss_mask"]) > 0 for row in rows), "nothing trainable in any row"


async def test_export_survives_an_uncompressed_artifact(tokens_stack, monkeypatch):
    """The tolerant path: a plain artifact must still parse.

    `export()` decides by the zstd magic number rather than by catching the
    failure, so this is the branch that proves the sniff, and the assertion in
    `test_create_turn_finish_with_reward_then_export` is the other one.
    """
    from skyrl_capture import compression

    client = _client(tokens_stack)
    trajectory = await asyncio.to_thread(
        create_trajectory, project="sdk-e2e",
        trajectory_id="plain-0", client=client,
    )
    handle = {"base_url": trajectory.base_url}
    assert (await tokens_stack.chat(handle, [{"role": "user", "content": "hi"}],
                                    model="mock-tokens-model")).status_code == 200
    await tokens_stack.settle()
    await asyncio.to_thread(trajectory.finish, annotations={"reward": 0.0})

    # Hand back the artifact uncompressed, as a sink or another store might.
    real_download = CaptureClient.download_export

    def plain(self, identifier):  # noqa: ANN001
        return compression.decompress(real_download(self, identifier))

    monkeypatch.setattr(CaptureClient, "download_export", plain)

    task = asyncio.create_task(asyncio.to_thread(trajectory.export, "token-samples"))
    while not task.done():
        await tokens_stack.runtime.exports.drain()
        await asyncio.sleep(0.02)
    rows = await task
    assert rows and all(row["trajectory_id"] == "plain-0" for row in rows)


# -- whose HTTP client it is ------------------------------------------------
# A trial per rollout means a connection pool per rollout when the SDK makes the
# client and nothing closes it. These pin the three cases apart: the SDK made
# it and hands you a `close()`, the SDK made it and the failure path releases
# it anyway, and you made it so the SDK must not touch it.
async def test_a_trajectory_closes_the_client_it_made(stack):
    created = await asyncio.to_thread(create_trajectory, project="owned", endpoint=stack.base_url)
    assert not created.client._client.is_closed

    await asyncio.to_thread(created.close)
    assert created.client._client.is_closed
    # And it is idempotent: teardown runs it, and so may a caller.
    await asyncio.to_thread(created.close)


async def test_a_caller_supplied_client_is_never_closed_for_you(stack):
    client = _client(stack)
    first = await asyncio.to_thread(create_trajectory, project="shared", client=client)

    await asyncio.to_thread(first.close)
    assert not client._client.is_closed, "closing it would break every later trial"

    # Which is the point: the same client still serves the next trial.
    second = await asyncio.to_thread(create_trajectory, project="shared", client=client)
    assert second.id != first.id
    await asyncio.to_thread(client.close)


def test_a_failed_creation_does_not_leak_the_client_it_made():
    """There is no `Trajectory` to close afterwards, so it closes here."""
    import httpx

    from skyrl_capture.sdk import CaptureError

    made: list[httpx.Client] = []
    real = CaptureClient.__init__

    def remember(self: CaptureClient, *args: Any, **kwargs: Any) -> None:
        real(self, *args, **kwargs)
        made.append(self._client)

    CaptureClient.__init__ = remember  # type: ignore[method-assign]
    try:
        with pytest.raises(CaptureError):
            create_trajectory(project="doomed", endpoint="http://127.0.0.1:9")
    finally:
        CaptureClient.__init__ = real  # type: ignore[method-assign]

    assert made, "the SDK made a client, since none was passed"
    assert made[-1].is_closed, "and released it when the create failed"


async def test_the_context_manager_releases_its_own_client(stack):
    from skyrl_capture.sdk import capture

    def run() -> Any:
        with capture(project="ctx", endpoint=stack.base_url, labels=["smoke"]) as trajectory:
            assert not trajectory.client._client.is_closed
            return trajectory

    trajectory = await asyncio.to_thread(run)
    assert trajectory.client._client.is_closed, "leaked once per `with capture(...)` otherwise"


# -- the error contract -----------------------------------------------------
def test_no_httpx_exception_escapes_the_client():
    """`health()` and the download used to bypass the wrapper and raise
    `httpx.ConnectError`, past a contract that says everything here raises
    `CaptureError`."""
    from skyrl_capture.sdk import CaptureError

    client = CaptureClient("http://127.0.0.1:9", timeout=1.0)
    try:
        for call in (client.health, lambda: client.download_export("exp_nope")):
            with pytest.raises(CaptureError):
                call()
    finally:
        client.close()

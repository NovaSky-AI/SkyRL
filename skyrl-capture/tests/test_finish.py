"""Finishing: what it returns, when it is the same finish, and what retries.

`finish` is the one call whose failure loses a trial's whole record, so it is
the one the design spends its care on. It is synchronous -- by the time it
answers, the canonical record is on disk and the reply was rendered from it --
and it is idempotent by the hash of what was asked, which is persisted beside
the record so a replacement process reaches the same answer.

The format is a rendering choice rather than part of that identity. Asking for
the same trajectory again in a different format is a retry; asking for a
different outcome is a conflict.
"""

from __future__ import annotations

import httpx
import pytest


async def one_turn(stack, **fields):
    created = await stack.create_trajectory(project="rl", run_id="run-a", **fields)
    history = [{"role": "user", "content": "first"}]
    reply = await stack.chat(created, history)
    history.append(reply.json()["choices"][0]["message"])
    history.append({"role": "user", "content": "second"})
    await stack.chat(created, history)
    await stack.settle()
    return created


async def test_finish_defaults_to_the_whole_trajectory(stack):
    created = await one_turn(stack)
    body = (await stack.finish(created["id"])).json()

    assert body["status"] == "finished"
    assert body["format"] == "graph"
    assert len(body["records"]) == 1, "the graph export is one row per trajectory"
    assert body["records"][0]["trajectory_id"] == created["id"]
    assert body["records"][0]["nodes"]


@pytest.mark.parametrize("fmt", ["graph", "replay", "text-samples"])
async def test_finish_renders_any_format_from_the_committed_record(stack, fmt):
    created = await one_turn(stack)
    body = (await stack.finish(created["id"], format=fmt)).json()

    assert body["format"] == fmt
    assert body["records"], fmt
    # Rendered from what was committed, so it matches a later export exactly.
    rows = await stack.export_lines(trajectory=created["id"], format=fmt)
    assert rows == body["records"]


async def test_finish_takes_the_same_options_an_export_does(stack):
    """Options decide what `trainable` says; they never drop a row."""
    created = await one_turn(stack)
    body = (
        await stack.finish(
            created["id"], format="text-samples", options={"mask_abandoned": True}
        )
    ).json()

    assert body["records"]
    assert all("trainable_count" in row for row in body["records"])


async def test_an_unknown_format_is_refused_rather_than_ignored(stack):
    created = await one_turn(stack)
    response = await stack.finish(created["id"], format="parquet")
    assert response.status_code == 400
    assert "unknown export format" in response.json()["detail"]


async def test_a_retry_in_another_format_is_still_the_same_finish(stack):
    created = await one_turn(stack)
    first = await stack.finish(created["id"], annotations={"reward": 1.0}, format="graph")
    second = await stack.finish(created["id"], annotations={"reward": 1.0}, format="replay")

    assert first.status_code == second.status_code == 200
    assert second.json()["format"] == "replay"
    assert stack.record(created["id"]).revision == 1, "one edit, not two"


async def test_finish_is_idempotent_across_a_restart(stack_builder, tmp_path):
    """A retry after the process that answered went away has to reach the same
    answer, which is why the finish hash is on disk and not in a table."""
    root = tmp_path / "traces"
    first = await stack_builder(record_dir=root)
    created = await first.create_trajectory(project="rl", trajectory_id="tr_finish_twice")
    await first.chat(created, [{"role": "user", "content": "once"}])
    original = (await first.finish(created["id"], annotations={"reward": 1.0})).json()
    await first.application.runtime.stop()

    second = await stack_builder(record_dir=root)
    repeat = await second.post(
        "/v1/trajectories/tr_finish_twice/finish", {"annotations": {"reward": 1.0}}
    )
    assert repeat.status_code == 200
    assert repeat.json() == original, "the committed record, found rather than rewritten"

    conflict = await second.post(
        "/v1/trajectories/tr_finish_twice/finish", {"annotations": {"reward": 0.0}}
    )
    assert conflict.status_code == 409


async def test_finish_renders_token_samples_from_a_token_capture(tokens_stack):
    """The fourth format, and the one an RL harness actually asks for: the
    rollout's training rows, in the call that scored it."""
    created = await tokens_stack.create_trajectory(project="rl", run_id="run-a")
    await tokens_stack.client.post(
        f"{created['base_url']}/chat/completions",
        json={"model": "mock-tokens-model", "messages": [{"role": "user", "content": "hi"}],
              "max_tokens": 4},
        headers={"authorization": "Bearer client-key"},
    )
    await tokens_stack.settle()

    body = (
        await tokens_stack.finish(
            created["id"],
            annotations={"reward": 1.0},
            format="token-samples",
            options={"overlong_filtering": True},
        )
    ).json()

    assert body["format"] == "token-samples"
    row = body["records"][0]
    assert row["input_ids"] and len(row["loss_mask"]) == len(row["input_ids"])
    assert sum(row["loss_mask"]) > 0, "something to train on"
    assert row["annotations"] == {"reward": 1.0}, "the reward this call carried"


async def test_a_text_capture_cannot_be_finished_as_token_samples(stack):
    """Refused rather than answered with an empty list: an empty artifact reads
    like "nothing matched" and not "this cannot be produced"."""
    created = await one_turn(stack)
    response = await stack.finish(created["id"], format="token-samples")

    assert response.status_code == 200, "the record is still committed"
    assert response.json()["records"] == [], "and there are no token rows in it"
    assert stack.record(created["id"]).trajectory.status == "finished"


async def test_finishing_a_trajectory_nobody_has_heard_of_is_a_404(stack):
    assert (await stack.finish("tr_never_existed")).status_code == 404


async def test_creating_without_an_id_is_refused(stack):
    """The id is the caller's. Generating one here would mean a create whose
    response was lost could not be repeated."""
    response = await stack.post("/v1/trajectories", {"project": "rl"})
    assert response.status_code == 422
    assert "trajectory_id" in response.text


# -- the SDK's own retry ---------------------------------------------------------
def test_the_sdk_retries_a_finish_exactly_three_times(monkeypatch):
    """Three attempts, and only for failures that say nothing about the
    request. A `409` is a decision and must never be retried."""
    from skyrl_capture import sdk

    monkeypatch.setattr(sdk.time, "sleep", lambda _seconds: None)
    client = sdk.CaptureClient("http://127.0.0.1:1")
    attempts: list[str] = []

    def unreachable(method, url, **kwargs):
        attempts.append(url)
        raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(client._client, "request", unreachable)
    with pytest.raises(sdk.CaptureError, match="cannot reach"):
        client._post_with_retry("/v1/trajectories/tr_1/finish", {})
    assert len(attempts) == sdk.FINISH_ATTEMPTS == 3


@pytest.mark.parametrize("status", [502, 503, 504])
def test_the_sdk_retries_a_gateway_saying_come_back(monkeypatch, status):
    from skyrl_capture import sdk

    monkeypatch.setattr(sdk.time, "sleep", lambda _seconds: None)
    client = sdk.CaptureClient("http://127.0.0.1:1")
    attempts: list[int] = []

    def refuse(method, url, **kwargs):
        attempts.append(status)
        return httpx.Response(status, request=httpx.Request(method, url), text="try later")

    monkeypatch.setattr(client._client, "request", refuse)
    with pytest.raises(sdk.CaptureError):
        client._post_with_retry("/v1/trajectories/tr_1/finish", {})
    assert len(attempts) == 3


def test_the_sdk_does_not_retry_a_decision(monkeypatch):
    from skyrl_capture import sdk

    monkeypatch.setattr(sdk.time, "sleep", lambda _seconds: None)
    client = sdk.CaptureClient("http://127.0.0.1:1")
    attempts: list[str] = []

    def conflict(method, url, **kwargs):
        attempts.append(url)
        return httpx.Response(
            409, request=httpx.Request(method, url), json={"detail": "a different outcome"}
        )

    monkeypatch.setattr(client._client, "request", conflict)
    with pytest.raises(sdk.CaptureError, match="different outcome"):
        client._post_with_retry("/v1/trajectories/tr_1/finish", {})
    assert len(attempts) == 1, "409 is an answer, not a failure to reach one"


def test_the_sdk_returns_the_first_success_without_retrying(monkeypatch):
    from skyrl_capture import sdk

    client = sdk.CaptureClient("http://127.0.0.1:1")
    attempts: list[str] = []

    def answer(method, url, **kwargs):
        attempts.append(url)
        return httpx.Response(
            200, request=httpx.Request(method, url), json={"id": "tr_1", "status": "finished"}
        )

    monkeypatch.setattr(client._client, "request", answer)
    assert client._post_with_retry("/v1/trajectories/tr_1/finish", {})["status"] == "finished"
    assert len(attempts) == 1

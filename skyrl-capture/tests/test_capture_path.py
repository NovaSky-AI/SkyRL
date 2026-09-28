"""Phase 1: capture and replay foundation."""

from __future__ import annotations

import asyncio

import pytest


async def test_unchanged_openai_workload_is_captured(stack):
    """Acceptance criterion 1 and 2.

    An unchanged OpenAI-compatible request, sent with nothing but a trajectory
    base URL, is captured and assigned to exactly one trajectory.
    """
    created = await stack.create_trajectory()
    assert created["base_url"].endswith("/v1")
    assert "api_key" not in created, "capture issues no credential of its own"

    response = await stack.chat(created, [{"role": "user", "content": "hello"}])
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "mock reply text"

    exchanges = await stack.exchanges(created["id"])
    assert len(exchanges) == 1
    exchange = exchanges[0]
    assert exchange["id"].startswith("ex_")
    assert exchange["endpoint_kind"] == "chat_completions"
    assert exchange["http_status"] == 200
    assert exchange["model"] == "mock-model"
    assert exchange["duration_ms"] > 0
    assert exchange["usage"]["prompt_tokens"] is not None

    # Whatever the client sent is replaced by the configured upstream
    # credential on the way out.
    upstream_request = stack.upstream.requests[-1]
    assert upstream_request["headers"]["authorization"] == "Bearer upstream-secret"


async def test_a_route_needs_no_credential_and_gates_on_the_trajectory(stack):
    """The route's trajectory id is correlation, not authorization.

    Capture authenticates nothing on the way in -- a deployment that needs to
    puts something in front -- so what decides whether a request is served is
    whether the trajectory exists and is still taking turns.
    """
    created = await stack.create_trajectory()

    # No header at all: served.
    response = await stack.data.post(
        f"{created['base_url']}/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "x"}]},
    )
    assert response.status_code == 200

    # An arbitrary one: served, and not recorded.
    response = await stack.data.post(
        f"{created['base_url']}/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "y"}]},
        headers={"authorization": "Bearer anything-at-all"},
    )
    assert response.status_code == 200

    # A trajectory that does not exist: 404, and nothing reached the upstream.
    before = len(stack.upstream.requests)
    unknown = await stack.data.post(
        f"{stack.base_url}/route/tr_nope/v1/chat/completions",
        json={"model": "m", "messages": []},
    )
    assert unknown.status_code == 404
    assert len(stack.upstream.requests) == before


def _anthropic(url: str):
    """An Anthropic-speaking upstream. The process serves one protocol, so a
    test that needs a different one builds a different process."""
    from skyrl_capture.config import TextUpstream

    return TextUpstream(type="anthropic", url=url, api_key="upstream-secret")


async def test_anthropic_workload_is_captured(stack_builder):
    stack = await stack_builder(upstream_for=_anthropic)
    created = await stack.create_trajectory()
    assert not created["base_url"].endswith("/v1"), "the Anthropic SDK appends /v1 itself"

    response = await stack.messages(created, [{"role": "user", "content": "hi"}], system="be terse")
    assert response.status_code == 200
    assert response.json()["content"][0]["text"] == "mock reply text"

    exchanges = await stack.exchanges(created["id"])
    assert exchanges[0]["endpoint_kind"] == "messages"
    assert exchanges[0]["usage"]["prompt_tokens"] is not None
    # The Anthropic system field is materialized as a leading system message.
    graph = await stack.graph(created["id"])
    roles = [node["role"] for node in graph["nodes"]]
    assert roles[:2] == ["system", "user"]
    system_node = graph["nodes"][0]
    assert system_node["derivation"]["materialized_from"] == "system_field"

    upstream_request = stack.upstream.requests[-1]
    assert upstream_request["headers"]["x-api-key"] == "upstream-secret"


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_streaming_records_chunk_timing(stack_builder, provider):
    stack = await stack_builder(upstream_for=None if provider == "openai" else _anthropic)
    created = await stack.create_trajectory()
    if provider == "openai":
        response = await stack.chat(created, [{"role": "user", "content": "stream"}], stream=True)
    else:
        response = await stack.messages(created, [{"role": "user", "content": "stream"}], stream=True)

    assert response.status_code == 200
    assert b"data:" in response.content

    exchanges = await stack.exchanges(created["id"])
    exchange = exchanges[0]
    assert exchange["streaming"] is True
    assert exchange["chunk_count"] > 1
    assert exchange["ttft_ms"] is not None
    assert exchange["duration_ms"] >= exchange["ttft_ms"]
    summary = exchange["stream_summary"]
    assert summary["chunk_count"] > 1
    assert summary["first_chunk_ms"] >= 0

    # The assembled assistant message reaches the graph.
    graph = await stack.graph(created["id"])
    outputs = [node for node in graph["nodes"] if node["author"] == "model"]
    assert len(outputs) == 1
    assert outputs[0]["derivation"]["assembled_from"].endswith("_stream")


async def test_replay_reproduces_the_request(stack):
    """Acceptance criterion 4: raw payloads can reproduce a same-provider request."""
    created = await stack.create_trajectory()
    sent = {
        "model": "mock-model",
        "messages": [{"role": "user", "content": "reproduce me"}],
        # A provider-specific field the parser does not interpret.
        "some_provider_extension": {"nested": [1, 2, 3]},
        "temperature": 0.42,
    }
    response = await stack.data.post(
        f"{created['base_url']}/chat/completions",
        json=sent,
        headers={"authorization": "Bearer client-key"},
    )
    assert response.status_code == 200

    await stack.exchanges(created["id"])
    row = stack.exchange_rows(created["id"])[0]
    stored = await stack.stored_object(row["id"], "request")
    assert stored == sent, "the raw exchange must round-trip provider-specific fields"

    lines = await stack.export_lines(trajectory=created["id"], format="replay")
    payload = lines[0]["turns"][0]["payload"]
    # Provider-specific fields survive: the payload is the request as sent.
    assert payload["some_provider_extension"] == sent["some_provider_extension"]
    assert payload["temperature"] == 0.42
    assert payload["messages"] == sent["messages"]


class _StuckStore:
    """A journal whose disk never answers. What a dead volume looks like."""

    def __init__(self, real) -> None:
        self.real = real
        self.released = asyncio.Event()
        self.attempts = 0

    async def create(self, header):
        return await self.real.create(header)

    async def append(self, trajectory_id, record):
        self.attempts += 1
        await self.released.wait()
        raise OSError("disk went away")

    async def recover(self, trajectory_id):
        return await self.real.recover(trajectory_id)

    async def remove(self, trajectory_id):
        return await self.real.remove(trajectory_id)


async def test_response_does_not_depend_on_the_persistence_write(stack):
    """Acceptance criterion 3, part one.

    The provider response is returned before anything reaches disk. With the
    journal stuck, the call still succeeds and the aggregate has the exchange
    -- the request path derived it and moved on -- while nothing is durable.
    """
    created = await stack.create_trajectory()
    commits = stack.runtime.commits
    stuck = _StuckStore(stack.runtime.active)
    real = commits._active  # noqa: SLF001
    commits._active = stuck  # noqa: SLF001
    try:
        response = await stack.chat(created, [{"role": "user", "content": "no disk"}])
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "mock reply text"
        assert len(stack.exchange_rows(created["id"])) == 1, "in the aggregate at once"
        await asyncio.sleep(0.05)
        assert commits.stats()["pending_commits"] == 1, "and not yet on disk"
        assert stack.journal(created["id"]).exchanges == [], "nothing in the journal"
    finally:
        stuck.released.set()
        await asyncio.sleep(0.05)
        commits._active = real  # noqa: SLF001

    # The failure is a capture gap, not a failed request, and it is written
    # down as soon as the disk comes back.
    assert stack.aggregate(created["id"]).integrity.calls_missing == 1
    await stack.chat(created, [{"role": "user", "content": "the disk is back"}])
    await stack.settle()
    replayed = stack.journal(created["id"])
    assert replayed.integrity.calls_missing == 1, "the gap reached the journal"
    assert len(replayed.exchanges) == 1, "and so did the turn after it"


async def test_a_full_commit_queue_refuses_and_counts_without_failing_the_request(stack_builder):
    """Acceptance criterion 3, part two: capture degrades, inference does not.

    The journal is stuck so nothing drains, and the bound is tiny -- the real
    back-pressure situation, rather than a stubbed one.
    """
    from skyrl_capture.config import RecordConfig

    stack = await stack_builder(record=RecordConfig(commit_capacity=2, fsync="never"))
    created = await stack.create_trajectory()
    commits = stack.runtime.commits
    stuck = _StuckStore(stack.runtime.active)
    real = commits._active  # noqa: SLF001
    commits._active = stuck  # noqa: SLF001
    try:
        for index in range(4):
            response = await stack.chat(created, [{"role": "user", "content": f"queue {index}"}])
            assert response.status_code == 200
            assert response.json()["choices"][0]["message"]["content"] == "mock reply text"
        assert commits.refused >= 1
        assert stack.aggregate(created["id"]).integrity.calls_missing >= 1

        # The gap is real and the disk cannot take it yet, so it is on health
        # rather than in the record -- the viewer reads what survived, and
        # this has not survived anything.
        health = (await stack.get("/healthz")).json()
        assert health["status"] == "degraded"
        assert health["commits"]["refused"] >= 1
        assert health["commits"]["unwritten_gaps"] >= 1
        trajectory = (await stack.get(f"/v1/trajectories/{created['id']}")).json()
        assert trajectory["capture"]["calls_missing"] == 0, "not on disk, so not in the record"
    finally:
        stuck.released.set()
        await asyncio.sleep(0.05)
        commits._active = real  # noqa: SLF001

    # And once the disk comes back, the gaps it refused are written down.
    await stack.chat(created, [{"role": "user", "content": "the disk is back"}])
    await stack.settle()
    trajectory = (await stack.get(f"/v1/trajectories/{created['id']}")).json()
    assert trajectory["capture"]["calls_missing"] >= 1
    assert trajectory["capture"]["complete"] is False


async def test_capture_exception_is_contained(stack):
    """A bug in exchange derivation must not surface to the workload."""
    import skyrl_capture.text.proxy as proxy_app

    created = await stack.create_trajectory()

    def explode(*_args, **_kwargs):
        raise RuntimeError("synthetic capture bug")

    original = proxy_app.derive_exchange
    proxy_app.derive_exchange = explode
    try:
        response = await stack.chat(created, [{"role": "user", "content": "capture explodes"}])
    finally:
        proxy_app.derive_exchange = original

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "mock reply text"
    assert stack.runtime.proxy.capture_errors >= 1
    health = (await stack.get("/healthz")).json()
    assert health["capture"]["capture_errors"] >= 1


async def test_upstream_error_is_passed_through_and_recorded(stack):
    created = await stack.create_trajectory()
    response = await stack.chat(
        created, [{"role": "user", "content": "fail"}], headers={"x-mock-status": "503"}
    )
    assert response.status_code == 503

    exchanges = await stack.exchanges(created["id"])
    assert exchanges[0]["http_status"] == 503
    assert exchanges[0]["provider_error"]["message"] == "mock provider error"
    # A failed call has no assistant message, so it commits no output node.
    assert exchanges[0]["output_node_id"] is None


async def test_provider_response_id_is_preserved_as_metadata(stack):
    created = await stack.create_trajectory()
    await stack.chat(created, [{"role": "user", "content": "ids"}])
    exchanges = await stack.exchanges(created["id"])
    assert exchanges[0]["provider_response_id"].startswith("chatcmpl-mock")
    assert exchanges[0]["provider_request_id"].startswith("chatcmpl-mock")


async def test_oversized_request_is_rejected(stack):

    created = await stack.create_trajectory()
    plane = stack.runtime.data_plane
    plane._max_request_bytes = 512  # noqa: SLF001
    response = await stack.chat(created, [{"role": "user", "content": "x" * 2048}])
    assert response.status_code == 413
    assert response.json()["error"]["type"] == "invalid_request_error"


async def test_non_graphed_endpoints_still_capture(stack):
    """Embeddings and model listings are captured but do not enter the graph."""
    created = await stack.create_trajectory()
    response = await stack.data.get(
        f"{created['base_url']}/models", headers={"authorization": "Bearer client-key"}
    )
    assert response.status_code == 200
    exchanges = await stack.exchanges(created["id"])
    assert exchanges[0]["endpoint_kind"] == "models"
    assert exchanges[0]["output_node_id"] is None
    graph = await stack.graph(created["id"])
    assert graph["nodes"] == []


async def test_the_journal_replays_to_the_same_graph(stack):
    """A journal read back converges on what the writing process holds -- which
    is the whole basis of recovery, and of the viewer reading files."""
    created = await stack.create_trajectory()
    for index in range(3):
        await stack.chat(created, [{"role": "user", "content": f"turn {index}"}])
    await stack.settle()
    assert len(await stack.exchanges(created["id"])) == 3

    replayed = stack.journal(created["id"])

    assert len(replayed.exchanges) == 3, "a replay must not duplicate exchanges"
    graph = await stack.graph(created["id"])
    assert [node.public() for node in replayed.graph.ordered()] == graph["nodes"]
    assert len(stack.node_rows(created["id"])) == len(graph["nodes"])


async def test_request_body_is_stored_verbatim(stack):
    created = await stack.create_trajectory()
    await stack.chat(created, [{"role": "user", "content": "verbatim"}])
    exchanges = await stack.exchanges(created["id"])
    document = await stack.stored_object(exchanges[0]["id"], "request")
    assert document["messages"][0]["content"] == "verbatim"

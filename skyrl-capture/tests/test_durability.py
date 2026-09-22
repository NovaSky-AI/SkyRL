"""What survives, what waits for it, and what capture admits it cannot vouch for.

The design's two halves meet here. Text capture is fail-open: a response is
never delayed by persistence and never altered by its failure, and what is lost
is counted rather than hidden. TITO is fail-closed: a response cannot close
cleanly until its exact exchange is on disk, so a cleanly closed response is
always one the record contains.

Between them sits the one ambiguity the design does not pretend away. Capture
and network delivery cannot be atomic, so a process that dies between the two
leaves an exchange that is durable and may never have reached the client.
Nothing here guesses: the exchange is kept, flagged, and resolved only by
evidence -- a later request whose own history contains that assistant output.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from skyrl_capture.persistence import journal
from skyrl_capture.persistence.active import read_journal


# -- helpers -------------------------------------------------------------------
class _SlowStore:
    """A journal whose appends take a measurable moment."""

    def __init__(self, real, delay: float = 0.15) -> None:
        self.real = real
        self.delay = delay
        self.finished_appends = 0

    async def create(self, header):
        return await self.real.create(header)

    async def append(self, trajectory_id, record):
        await asyncio.sleep(self.delay)
        await self.real.append(trajectory_id, record)
        self.finished_appends += 1

    async def recover(self, trajectory_id):
        return await self.real.recover(trajectory_id)

    async def remove(self, trajectory_id):
        return await self.real.remove(trajectory_id)


class _BrokenStore:
    """A journal that takes creations and refuses everything else."""

    def __init__(self, real) -> None:
        self.real = real

    async def create(self, header):
        return await self.real.create(header)

    async def append(self, trajectory_id, record):
        raise OSError("the volume went away")

    async def recover(self, trajectory_id):
        return await self.real.recover(trajectory_id)

    async def remove(self, trajectory_id):
        return await self.real.remove(trajectory_id)


def kinds(stack, trajectory_id: str) -> list[str]:
    return [type(record).__name__ for record in stack.journal_records(trajectory_id)]


def drop_delivery_confirmations(path: Path) -> None:
    """Rewrite a journal as a process that died during the close left it.

    The exchange is durable and the confirmation that would have followed it
    never got written, which is exactly the window the design calls out.
    """
    records = [
        record
        for record in read_journal(path).records
        if not isinstance(record, journal.ExchangeDeliveryConfirmed)
    ]
    path.write_bytes(
        journal.encode_header() + b"".join(journal.encode_record(r) for r in records)
    )


async def tito_turn(stack, route, messages, **body):
    return await stack.client.post(
        f"{route['base_url']}/chat/completions",
        json={"model": "mock-tokens-model", "messages": messages, "max_tokens": 4, **body},
        headers={"authorization": "Bearer client-key"},
    )


# -- text: the response never waits ---------------------------------------------
@pytest.mark.parametrize("stream", [False, True])
async def test_a_text_response_completes_without_waiting_for_persistence(stack, stream):
    """Acceptance criterion: capture is behind the response, not in front of it."""
    created = await stack.create_trajectory()
    commits = stack.runtime.commits
    slow = _SlowStore(stack.runtime.active, delay=0.3)
    real = commits._active  # noqa: SLF001
    commits._active = slow  # noqa: SLF001
    try:
        started = asyncio.get_running_loop().time()
        response = await stack.chat(created, [{"role": "user", "content": "go"}], stream=stream)
        elapsed = asyncio.get_running_loop().time() - started

        assert response.status_code == 200
        assert elapsed < 0.3, "the client did not wait for the append"
        assert slow.finished_appends == 0, "which had not finished"
        await stack.settle()
        assert slow.finished_appends == 1
    finally:
        commits._active = real  # noqa: SLF001


async def test_same_trajectory_commits_stay_in_the_order_they_were_made(stack):
    """The second turn's graph delta references nodes the first one created, so
    a journal that took them out of order would not replay."""
    created = await stack.create_trajectory()
    history: list[dict[str, str]] = []
    for turn in range(4):
        history.append({"role": "user", "content": f"turn {turn}"})
        reply = await stack.chat(created, history)
        history.append(reply.json()["choices"][0]["message"])
    await stack.settle()

    committed = [
        record
        for record in stack.journal_records(created["id"])
        if isinstance(record, journal.ExchangeCommitted)
    ]
    assert [record.exchange.sequence for record in committed] == [0, 1, 2, 3]
    # And the replay reaches the same graph, which is the thing order is for.
    assert len(stack.journal(created["id"]).graph) == len(stack.aggregate(created["id"]).graph)


async def test_finish_refuses_when_it_cannot_write_the_canonical_record(stack, monkeypatch):
    """A finish that reports success is a promise the record exists."""
    created = await stack.create_trajectory()
    await stack.chat(created, [{"role": "user", "content": "one"}])
    await stack.settle()

    async def refuse(record):
        raise OSError("the volume went away")

    monkeypatch.setattr(stack.runtime.committed, "put", refuse)
    response = await stack.finish(created["id"])

    assert response.status_code == 503, response.text
    assert "could not commit" in response.json()["detail"]
    assert stack.record(created["id"]) is None
    assert stack.aggregate(created["id"]) is not None, "still open, so a retry can finish it"

    monkeypatch.undo()
    assert (await stack.finish(created["id"])).status_code == 200
    assert stack.record(created["id"]) is not None


async def test_a_disk_that_comes_back_still_records_what_was_lost(stack):
    """The gap is marked in memory at once and written as soon as it can be,
    so the final record says the trajectory is incomplete either way."""
    created = await stack.create_trajectory()
    commits = stack.runtime.commits
    broken = _BrokenStore(stack.runtime.active)
    real = commits._active  # noqa: SLF001
    commits._active = broken  # noqa: SLF001
    try:
        assert (await stack.chat(created, [{"role": "user", "content": "lost"}])).status_code == 200
        await asyncio.sleep(0.05)
    finally:
        commits._active = real  # noqa: SLF001

    await stack.chat(created, [{"role": "user", "content": "kept"}])
    await stack.finish(created["id"])

    record = stack.record(created["id"])
    assert record.trajectory.integrity.calls_missing == 1
    assert record.trajectory.integrity.complete is False
    assert len(record.exchanges) == 1, "the turn after the outage was captured"


# -- TITO: the response waits, and says so --------------------------------------
async def test_a_tito_response_does_not_close_until_its_exchange_is_durable(tokens_stack):
    created = await tokens_stack.create_trajectory()
    commits = tokens_stack.runtime.commits
    slow = _SlowStore(tokens_stack.runtime.active, delay=0.2)
    real = commits._active  # noqa: SLF001
    commits._active = slow  # noqa: SLF001
    try:
        started = asyncio.get_running_loop().time()
        response = await tito_turn(tokens_stack, created, [{"role": "user", "content": "exact"}])
        elapsed = asyncio.get_running_loop().time() - started
    finally:
        commits._active = real  # noqa: SLF001

    assert response.status_code == 200
    assert elapsed >= 0.2, "the close waited for the append"
    assert slow.finished_appends >= 1
    assert kinds(tokens_stack, created["id"])[:2] == ["TrajectoryCreated", "ExchangeCommitted"]


async def test_a_tito_commit_failure_prevents_a_clean_close(tokens_stack):
    """No clean close, so the client sees a failed connection and retries the
    unchanged request -- which is the whole reason TITO needs no request id."""
    created = await tokens_stack.create_trajectory()
    commits = tokens_stack.runtime.commits
    real = commits._active  # noqa: SLF001
    commits._active = _BrokenStore(tokens_stack.runtime.active)  # noqa: SLF001
    try:
        with pytest.raises(httpx.HTTPError):
            await tito_turn(tokens_stack, created, [{"role": "user", "content": "unwritable"}])
    finally:
        commits._active = real  # noqa: SLF001

    assert tokens_stack.runtime.proxy.commit_failures >= 1
    assert "ExchangeCommitted" not in kinds(tokens_stack, created["id"])


async def test_delivery_is_confirmed_after_the_response_closes(tokens_stack):
    created = await tokens_stack.create_trajectory()
    await tito_turn(tokens_stack, created, [{"role": "user", "content": "delivered"}])
    await tokens_stack.settle()

    assert kinds(tokens_stack, created["id"]) == [
        "TrajectoryCreated", "ExchangeCommitted", "ExchangeDeliveryConfirmed"
    ]
    replayed = tokens_stack.journal(created["id"])
    assert replayed.exchanges[0].delivery_confirmed is True
    assert replayed.integrity.delivery_uncertain_exchange_ids == []


# -- the delivery ambiguity ------------------------------------------------------
async def _uncertain_trajectory(stack_builder, tmp_path, trajectory_id="tr_ambiguous"):
    """One TITO turn, durable, whose delivery the record cannot vouch for.

    Written by one process and picked up by another, with the confirmation
    record missing -- which is what a process that died during the close leaves.
    """
    root = tmp_path / "traces"
    first = await stack_builder(tokens=True, record_dir=root)
    created = await first.create_trajectory(project="rl", trajectory_id=trajectory_id)
    response = await tito_turn(first, created, [{"role": "user", "content": "did you get this"}])
    assistant = response.json()["choices"][0]["message"]["content"]
    await first.settle()
    await first.application.runtime.stop()
    drop_delivery_confirmations(first.runtime.active.path_for(trajectory_id))

    second = await stack_builder(tokens=True, record_dir=root)
    route = {"base_url": f"{second.base_url}/route/{trajectory_id}/v1"}
    return second, route, assistant


async def test_recovery_flags_an_exchange_whose_delivery_was_never_recorded(
    stack_builder, tmp_path
):
    stack, route, _assistant = await _uncertain_trajectory(stack_builder, tmp_path)
    # Any request is enough to make the replacement adopt the trajectory.
    await tito_turn(stack, route, [{"role": "user", "content": "something else entirely"}])

    active = stack.aggregate("tr_ambiguous")
    assert len(active.integrity.delivery_uncertain_exchange_ids) == 1
    assert active.exchanges[0].delivery_uncertain is True


async def test_a_history_containing_the_output_confirms_it_and_reuses_its_tokens(
    stack_builder, tmp_path
):
    """The client's own next request is the evidence. Nothing else is."""
    stack, route, assistant = await _uncertain_trajectory(stack_builder, tmp_path)

    response = await tito_turn(
        stack,
        route,
        [
            {"role": "user", "content": "did you get this"},
            {"role": "assistant", "content": assistant},
            {"role": "user", "content": "yes, carry on"},
        ],
    )
    assert response.status_code == 200
    await stack.settle()

    active = stack.aggregate("tr_ambiguous")
    assert active.integrity.delivery_uncertain_exchange_ids == []
    assert active.exchanges[0].delivery_confirmed is True
    assert "ExchangeDeliveryConfirmed" in kinds(stack, "tr_ambiguous")
    # And the turn continued the exact token path rather than starting a branch.
    assert len(active.graph) == 4
    assert stack.runtime.proxy.render_bridged >= 1


async def test_a_request_that_does_not_continue_from_it_is_a_separate_branch(
    stack_builder, tmp_path
):
    """The converse is not inferred. Repeated sampling is valid, so a request
    that does not build on the uncertain output says nothing about it."""
    stack, route, _assistant = await _uncertain_trajectory(stack_builder, tmp_path)

    await tito_turn(stack, route, [{"role": "user", "content": "a different opening"}])
    await stack.settle()

    active = stack.aggregate("tr_ambiguous")
    assert active.integrity.delivery_uncertain_exchange_ids != [], "still uncertain"
    roots = [node for node in active.graph.ordered() if node.parent_id is None]
    assert len(roots) == 2, "a separate branch, not a continuation"


async def test_exact_tokens_survive_a_restart_and_the_next_turn_extends_them(
    stack_builder, tmp_path
):
    """The trace is a cache; the journal is the record of it. A replacement
    process rebuilds the exact token deltas from the nodes, or prefix reuse
    would silently stop working and every turn would start a new branch."""
    root = tmp_path / "traces"
    first = await stack_builder(tokens=True, record_dir=root)
    created = await first.create_trajectory(project="rl", trajectory_id="tr_exact")
    response = await tito_turn(first, created, [{"role": "user", "content": "remember this"}])
    assistant = response.json()["choices"][0]["message"]["content"]
    await first.settle()
    before = [node.token_ids for node in first.aggregate("tr_exact").graph.ordered()]
    await first.application.runtime.stop()

    second = await stack_builder(tokens=True, record_dir=root)
    route = {"base_url": f"{second.base_url}/route/tr_exact/v1"}
    following = await tito_turn(
        second,
        route,
        [
            {"role": "user", "content": "remember this"},
            {"role": "assistant", "content": assistant},
            {"role": "user", "content": "and continue"},
        ],
    )
    assert following.status_code == 200
    await second.settle()

    recovered = second.aggregate("tr_exact")
    assert [node.token_ids for node in recovered.graph.ordered()][:2] == before
    assert len(recovered.graph) == 4, "it extended the trace rather than branching"
    assert second.runtime.proxy.render_bridged >= 1, "and reused the exact prefix"


async def test_finishing_releases_the_token_trace(tokens_stack):
    """The trace is the one thing a trajectory leaves in proxy memory."""
    created = await tokens_stack.create_trajectory()
    await tito_turn(tokens_stack, created, [{"role": "user", "content": "hold this"}])
    await tokens_stack.settle()
    sessions = tokens_stack.runtime.proxy.sessions
    assert sessions.stats()["cached_traces"] >= 1

    await tokens_stack.finish(created["id"])

    assert tokens_stack.aggregate(created["id"]) is None
    assert sessions.stats()["cached_traces"] == 0, "evicted with the aggregate"


async def test_an_identical_prompt_is_a_second_sample_and_not_a_retry(tokens_stack):
    """Two turns with the same messages are resampling, which is the reason
    the request fingerprint is diagnostic and never deduplicates."""
    created = await tokens_stack.create_trajectory()
    messages = [{"role": "user", "content": "sample me twice"}]
    first = await tito_turn(tokens_stack, created, messages)
    second = await tito_turn(tokens_stack, created, messages)
    assert first.status_code == second.status_code == 200
    await tokens_stack.settle()

    active = tokens_stack.aggregate(created["id"])
    assert len(active.exchanges) == 2, "both samples were kept"
    fingerprints = {
        record.request_fingerprint
        for record in tokens_stack.journal_records(created["id"])
        if isinstance(record, journal.ExchangeCommitted)
    }
    assert len(fingerprints) == 1, "identical requests, and two exchanges regardless"


async def test_what_each_reader_does_with_an_uncertain_exchange(stack_builder, tmp_path):
    """Three different answers, on purpose.

    The graph shows it flagged, because a branch that exists should be
    visible. Training exports refuse to train on it, because the model may
    never have seen it. A replay leaves it out, because reproducing traffic a
    client may never have received would replay a conversation that may never
    have happened.
    """
    from skyrl_capture.export.service import render_records
    from skyrl_capture.export.view import view_of

    stack, route, _assistant = await _uncertain_trajectory(stack_builder, tmp_path)
    await tito_turn(stack, route, [{"role": "user", "content": "adopt it"}])
    await stack.settle()
    await stack.finish("tr_ambiguous")

    record = stack.record("tr_ambiguous")
    assert record.trajectory.integrity.public()["delivery_uncertain"]
    view = view_of(record)

    graph = render_records(view, export_format="graph", options={})[0]
    flagged = [node for node in graph["nodes"] if node.get("delivery_uncertain")]
    assert len(flagged) == 1, "shown, and labelled"

    samples = render_records(view, export_format="token_samples", options={})
    uncertain_rows = [
        row for row in samples if row.get("masked_reason") == "delivery_uncertain"
    ]
    assert uncertain_rows, "untrainable, with the reason said out loud"
    assert all(sum(row["loss_mask"]) == 0 for row in uncertain_rows)

    replay = render_records(view, export_format="replay", options={})
    replayed_turns = sum(len(session["turns"]) for session in replay)
    assert len(record.exchanges) == 2
    assert replayed_turns == 1, "one exchange replayed, and it is not the uncertain one"
    replayed_text = str(replay)
    assert "did you get this" not in replayed_text

    # The viewer shows it too, over the same document.
    await stack.refresh()
    served = (await stack.get("/v1/trajectories/tr_ambiguous/graph")).json()
    assert served["nodes"]

"""Trajectory lifecycle, route gating, and finishing exactly once."""

from __future__ import annotations

import asyncio
import time


async def test_lifecycle_states(stack):
    created = await stack.create_trajectory()
    trajectory = (await stack.get(f"/v1/trajectories/{created['id']}")).json()
    assert trajectory["status"] == "created"
    assert trajectory["first_event_at"] is None

    await stack.chat(created, [{"role": "user", "content": "first"}])
    # Text capture commits behind the response and the viewer reads what was
    # written, so `active` appears when the turn reaches the journal rather
    # than when the client gets its bytes.
    await stack.settle()
    trajectory = (await stack.get(f"/v1/trajectories/{created['id']}")).json()
    assert trajectory["status"] == "active"
    assert trajectory["first_event_at"] is not None

    response = await stack.finish(created["id"], labels=["task-17"])
    assert response.status_code == 200
    trajectory = (await stack.get(f"/v1/trajectories/{created['id']}")).json()
    assert trajectory["status"] == "finished"
    assert trajectory["labels"] == ["task-17"]


async def test_finish_is_idempotent(stack):
    """Acceptance criterion 8: finishing twice must not duplicate or corrupt."""
    created = await stack.create_trajectory()
    await stack.chat(created, [{"role": "user", "content": "hello"}])

    labels = ["task-17", "success"]
    first = await stack.finish(
        created["id"], labels=labels, annotations={"rlvr_reward": 1.0}
    )
    second = await stack.finish(
        created["id"], labels=labels, annotations={"rlvr_reward": 1.0}
    )
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json(), "the committed record, rendered the same way"

    trajectory = (await stack.get(f"/v1/trajectories/{created['id']}")).json()
    assert trajectory["status"] == "finished"
    assert sorted(trajectory["labels"]) == sorted(labels)
    # The repeat merged onto the same value rather than accumulating.
    assert trajectory["annotations"] == {"rlvr_reward": 1.0}


async def test_metadata_stays_editable_after_finish(stack):
    """Metadata is not sealed: a label can be added once a reward is known."""
    created = await stack.create_trajectory()
    await stack.chat(created, [{"role": "user", "content": "hello"}])

    assert (await stack.finish(created["id"], labels=["draft"])).status_code == 200
    response = await stack.patch(
        f"/v1/trajectories/{created['id']}/metadata",
        {"annotations": {"rlvr_reward": 0.9}, "labels": ["success"], "remove_labels": ["draft"]},
    )
    assert response.status_code == 200

    trajectory = (await stack.get(f"/v1/trajectories/{created['id']}")).json()
    assert trajectory["labels"] == ["success"]
    assert trajectory["annotations"] == {"rlvr_reward": 0.9}


async def test_finish_closes_the_route_immediately(stack):
    """Finishing closes the route; it can never take another turn."""
    created = await stack.create_trajectory()
    assert (await stack.chat(created, [{"role": "user", "content": "before"}])).status_code == 200

    await stack.finish(created["id"])

    # No refresh, no waiting: the route closes at once.
    response = await stack.chat(created, [{"role": "user", "content": "after"}])
    assert response.status_code == 410
    assert "has finished" in response.json()["error"]["message"]


async def test_every_trial_gets_its_own_route(stack):
    """One route per trial, and an exchange belongs to exactly one of them."""
    first = await stack.create_trajectory()
    second = await stack.create_trajectory()
    assert first["id"] != second["id"]
    assert first["base_url"] != second["base_url"]

    # Both routes are independently usable while live.
    assert (await stack.chat(first, [{"role": "user", "content": "a"}])).status_code == 200
    assert (await stack.chat(second, [{"role": "user", "content": "b"}])).status_code == 200

    # Each exchange belongs to exactly one trajectory.
    assert len(await stack.exchanges(first["id"])) == 1
    assert len(await stack.exchanges(second["id"])) == 1


async def test_create_is_idempotent_by_the_id_and_the_body(stack):
    """No idempotency key. The trajectory id is the caller's, generated before
    the request goes out, and the hash of the creation body is persisted beside
    it -- so a repeat reaches the same answer after a restart, which a
    process-local key table could not."""
    payload = {"project": "idem", "trajectory_id": "trial-1"}
    first = await stack.post("/v1/trajectories", payload)
    second = await stack.post("/v1/trajectories", payload)
    assert first.status_code == 201
    assert first.json() == second.json(), "the original answer, not a second trajectory"

    # The same id with a different body is a caller bug, not a replay.
    conflict = await stack.post("/v1/trajectories", {"project": "other", "trajectory_id": "trial-1"})
    assert conflict.status_code == 409
    assert "different creation body" in conflict.json()["detail"]


async def test_creating_a_trajectory_that_already_finished_is_a_conflict(stack):
    created = await stack.create_trajectory(trajectory_id="tr_once")
    await stack.finish(created["id"])

    again = await stack.post("/v1/trajectories", {"project": "test-project", "trajectory_id": "tr_once"})
    assert again.status_code == 409
    assert "already finished" in again.json()["detail"]


async def test_an_abandoned_trajectory_stays_open_until_somebody_finishes_it(stack):
    """Nothing expires a trajectory on a clock.

    A caller that goes away leaves an unfinished journal, and that is the
    correct outcome: the work happened, and deciding it is over is the
    caller's to make -- possibly hours later, possibly from another process.
    A sweep that finalized it would be guessing, and would do so exactly when
    a slow harness was about to come back.
    """
    created = await stack.create_trajectory()
    await stack.chat(created, [{"role": "user", "content": "and then nothing"}])
    await stack.settle()

    assert stack.runtime.active.path_for(created["id"]).is_file()
    trajectory = (await stack.get(f"/v1/trajectories/{created['id']}")).json()
    assert trajectory["status"] == "active"
    assert (await stack.chat(created, [{"role": "user", "content": "still open"}])).status_code == 200


async def test_annotations_apply_to_a_branched_trajectory(stack):
    """One outcome belongs to the whole tree, however many branches it has."""
    created = await stack.create_trajectory()
    # Two divergent histories produce two leaves.
    await stack.chat(created, [{"role": "user", "content": "shared"}], headers={"x-mock-reply": "A"})
    await stack.chat(created, [{"role": "user", "content": "shared"}], headers={"x-mock-reply": "B"})
    await stack.settle()

    graph = await stack.graph(created["id"])
    assert len(graph["leaf_assistant_node_ids"]) == 2

    # Annotations are trajectory-scoped, so branching cannot reject them.
    response = await stack.finish(created["id"], annotations={"rlvr_reward": 1.0})
    assert response.status_code == 200

    trajectory = (await stack.get(f"/v1/trajectories/{created['id']}")).json()
    assert trajectory["status"] == "finished"
    assert trajectory["annotations"] == {"rlvr_reward": 1.0}


async def test_events_accepted_before_finish_are_not_late(stack):
    """Ingestion is asynchronous, so almost everything lands after finish.

    That must not mark a normal trajectory's tail as late.

    ``late`` compares two clocks: the exchange carries the proxy's wall time and
    ``finish_requested_at`` is the database's ``now()``. Any skew between them
    is added to the gap between these two calls, and a containerised database
    can lag a host process by a millisecond or two -- enough to flip the flag
    when they are issued back to back. The scenario under test is an exchange
    that genuinely precedes finish, so give it a gap larger than that skew.
    """
    created = await stack.create_trajectory()

    await stack.chat(created, [{"role": "user", "content": "in flight"}])
    await asyncio.sleep(0.05)
    await stack.finish(created["id"])

    exchanges = await stack.exchanges(created["id"])
    assert len(exchanges) == 1
    assert exchanges[0]["late"] is False
    trajectory = (await stack.get(f"/v1/trajectories/{created['id']}")).json()
    assert trajectory["capture"]["calls_after_close"] == 0


async def test_a_turn_in_flight_when_finish_arrives_still_lands(stack):
    """The race finish is built around: a turn that passed the gate before
    finish closed the route commits, and is in the trajectory it belongs to."""
    created = await stack.create_trajectory()
    turn = asyncio.create_task(
        stack.chat(created, [{"role": "user", "content": "slow"}], headers={"x-mock-delay": "0.2"})
    )
    await asyncio.sleep(0.05)
    finished = await stack.finish(created["id"])
    assert finished.status_code == 200
    assert (await turn).status_code == 200

    trajectory = (await stack.get(f"/v1/trajectories/{created['id']}")).json()
    assert trajectory["status"] == "finished"
    assert trajectory["capture"]["exchange_count"] == 1
    assert trajectory["capture"]["complete"] is True


async def test_a_commit_stamped_after_finish_is_marked_late(stack):
    """`late` is a microsecond race, now that the gate reads live status: a
    request passes it, finish lands, and the request's own clock reading is
    the later of the two. It is marked rather than refused, and it does not
    reopen the trajectory."""
    from datetime import UTC, datetime

    from skyrl_capture import upstream
    from skyrl_capture.writer.derive import derive_exchange
    from skyrl_capture.writer.exchange import Exchange

    created = await stack.create_trajectory()
    await stack.chat(created, [{"role": "user", "content": "in time"}])
    active = stack.aggregate(created["id"])
    # The route is closed, and a request that had already passed the gate is
    # about to commit. No request hash: `finish` below supplies the real one.
    active.request_finish(command_result=None, request_hash="", at=datetime.now(UTC))

    after = time.time_ns() + 1_000_000_000
    record = derive_exchange(
        active,
        Exchange(
            exchange_id="ex_late", trajectory_id=created["id"], project="test-project",
            provider="openai", endpoint_kind="chat_completions", method="POST",
            path="/chat/completions", query="",
            request_start_wall_ns=after, request_start_mono_ns=0,
            response_end_wall_ns=after, http_status=200,
        ),
        protocol=upstream.get("openai"),
        header_allowlist=(),
    )
    assert record.exchange.row["late"] is True
    active.add_exchange(record.exchange, record.graph, record.at)

    await stack.finish(created["id"])
    trajectory = (await stack.get(f"/v1/trajectories/{created['id']}")).json()
    assert trajectory["status"] == "finished", "a late commit does not reopen it"
    assert trajectory["capture"]["calls_after_close"] == 1


async def test_finish_drains_before_finalizing(stack):
    """Finish waits for accepted exchanges to land before reporting finished."""
    created = await stack.create_trajectory()
    for index in range(4):
        await stack.chat(created, [{"role": "user", "content": f"turn {index}"}])

    response = await stack.finish(created["id"], labels=["n4"])
    assert response.status_code == 200
    trajectory = (await stack.get(f"/v1/trajectories/{created['id']}")).json()
    assert trajectory["status"] == "finished"
    assert trajectory["capture"]["exchange_count"] == 4
    assert trajectory["capture"]["complete"] is True


async def test_there_is_no_delete_route(stack):
    """A record is the result of a run, and removing one is a file operation
    on the record directory rather than an API call. Leaving a delete here
    would mean a training run could lose its own data over HTTP."""
    created = await stack.create_trajectory()
    await stack.chat(created, [{"role": "user", "content": "keep me"}])
    await stack.finish(created["id"])

    response = await stack.delete(f"/v1/trajectories/{created['id']}")
    assert response.status_code in (404, 405)
    assert (await stack.get(f"/v1/trajectories/{created['id']}")).status_code == 200


async def test_the_upstream_credential_never_reaches_a_record(stack):
    """It is startup configuration, presented on the way out and nowhere else."""
    trajectory = await stack.create_trajectory()
    await stack.chat(trajectory, [{"role": "user", "content": "hi"}])
    exchanges = await stack.exchanges(trajectory["id"])
    assert "upstream-secret" not in str(exchanges)
    snapshot = str(stack.aggregate(trajectory["id"]).header.upstream)
    assert "upstream-secret" not in snapshot
    assert stack.upstream_url in snapshot, "the snapshot still names the server"


async def test_concurrent_requests_are_all_attributed(stack):
    """Overlapping calls are detected without confusing graph parentage."""
    created = await stack.create_trajectory()
    await asyncio.gather(
        *(
            stack.chat(
                created,
                [{"role": "user", "content": f"parallel {index}"}],
                headers={"x-mock-delay": "0.05"},
            )
            for index in range(4)
        )
    )
    exchanges = await stack.exchanges(created["id"])
    assert len(exchanges) == 4
    assert any(exchange["overlapping"] for exchange in exchanges)
    # Four distinct first messages means four root branches, not one chain.
    graph = await stack.graph(created["id"])
    assert len(graph["leaf_node_ids"]) == 4

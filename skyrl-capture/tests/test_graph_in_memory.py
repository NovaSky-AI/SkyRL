"""The graph is built in the writing process, and it is what the readers see.

Each proxy plans its exchange's change against the trajectory's own graph and
commits both together. There is no second copy: the aggregate's graph is the
graph, it goes into the journal as part of the exchange that made it, and it
comes back out of the committed record unchanged.

So the thing worth asserting is that the graph the writer holds agrees, node
for node, with what the read API serves off the disk -- on exactly the shapes
where prefix matching is interesting.
"""

from __future__ import annotations

import pytest


async def state_and_served(stack, trajectory_id):
    """The graph the writer holds, and the graph the viewer serves off disk."""
    await stack.refresh()
    held = stack.node_rows(trajectory_id)
    served = (await stack.get(f"/v1/trajectories/{trajectory_id}/graph")).json()["nodes"]
    return held, served


# -- the state and the route agree ------------------------------------------
async def test_a_linear_conversation_matches(stack):
    created = await stack.create_trajectory()
    history = [{"role": "system", "content": "be terse"}]
    for turn in range(3):
        history.append({"role": "user", "content": f"turn {turn}"})
        reply = await stack.chat(created, history)
        history.append(reply.json()["choices"][0]["message"])

    held, served = await state_and_served(stack, created["id"])
    assert held == served
    assert len(held) == 7
    assert len([node for node in held if node["parent_node_id"] is None]) == 1


# -- a replay converges -------------------------------------------------------
async def test_applying_a_committed_exchange_twice_adds_nothing(stack):
    """Recovery replays records this process may already have applied. Graph
    attribution is a fact about the first commit, so a second application must
    find the nodes it made and leave the counts alone."""
    created = await stack.create_trajectory()
    history = [{"role": "user", "content": "before the restart"}]
    reply = await stack.chat(created, history)
    history.append(reply.json()["choices"][0]["message"])
    await stack.settle()

    before, _ = await state_and_served(stack, created["id"])
    active = stack.aggregate(created["id"])
    replayed = stack.journal(created["id"])
    committed = next(
        record
        for record in stack.journal_records(created["id"])
        if type(record).__name__ == "ExchangeCommitted"
    )
    active.add_exchange(committed.exchange, committed.graph, committed.at)

    history.append({"role": "user", "content": "after the restart"})
    await stack.chat(created, history)
    after, served = await state_and_served(stack, created["id"])

    assert after == served
    assert [node["node_id"] for node in after[: len(before)]] == [node["node_id"] for node in before]
    assert len([node for node in after if node["parent_node_id"] is None]) == 1
    assert len(replayed.exchanges) == 1
    record = (await stack.get(f"/v1/trajectories/{created['id']}")).json()
    assert record["capture"]["exchange_count"] == 2


# -- what the write path counts ------------------------------------------------
async def test_the_commit_coordinator_counts_what_it_wrote(stack):
    created = await stack.create_trajectory()
    await stack.chat(created, [{"role": "user", "content": "one turn"}])
    await stack.settle()

    commits = stack.runtime.health.stats()["commits"]
    assert commits["commits"] >= 1
    assert commits["refused"] == 0
    assert commits["failures"] == 0
    assert commits["pending_commits"] == 0, "settled means durable"
    assert len(stack.journal(created["id"]).graph) == 2
    await stack.finish(created["id"])
    assert len(stack.record(created["id"]).nodes) == 2


@pytest.mark.parametrize("turns", [1, 5])
async def test_the_node_count_matches_what_was_built(stack, turns):
    created = await stack.create_trajectory()
    history: list[dict[str, str]] = []
    for turn in range(turns):
        history.append({"role": "user", "content": f"turn {turn}"})
        reply = await stack.chat(created, history)
        history.append(reply.json()["choices"][0]["message"])

    held, served = await state_and_served(stack, created["id"])
    assert len(held) == len(served) == turns * 2
    record = (await stack.get(f"/v1/trajectories/{created['id']}")).json()
    assert record["capture"]["node_count"] == len(held)

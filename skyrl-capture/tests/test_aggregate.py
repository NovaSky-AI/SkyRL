"""`ActiveTrajectory`: the guarantees one hot aggregate has to give.

There is no reducer and no global state any more, so what used to be a test of
a shared projection is a test of one trajectory's own object: the things that
have to hold for a single owner mutating a map between awaits to be a
sufficient substitute for a transaction.

The other half of the contract is that the same object comes back out of a
journal. A record written by one process and rebuilt by another has to reach
the same aggregate, or recovery is a different trajectory wearing the same id.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from skyrl_capture.domain.graph import GraphDelta, GraphNode
from skyrl_capture.domain.records import (
    ActiveTrajectory,
    CapturedExchange,
    TrajectoryHeader,
    metadata_update,
)
from skyrl_capture.persistence import journal, rebuild

NOW = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)


def header(identifier: str = "tr_1", **fields) -> TrajectoryHeader:
    return TrajectoryHeader(
        id=identifier,
        project=fields.pop("project", "rl"),
        run_id=fields.pop("run_id", None),
        task_id=fields.pop("task_id", None),
        step=fields.pop("step", None),
        mode=fields.pop("mode", "text"),
        upstream=fields.pop("upstream", {"protocol": "openai"}),
        labels=tuple(fields.pop("labels", ())),
        annotations=fields.pop("annotations", {}),
        bodies=fields.pop("bodies", "full"),
        source_metadata=fields.pop("source_metadata", {}),
        created_at=fields.pop("created_at", NOW),
        create_request_hash=fields.pop("create_request_hash", "h0"),
    )


def node(node_id: str, *, parent: str | None, author: str, exchange_id: str) -> GraphNode:
    return GraphNode(
        id=node_id,
        parent_id=parent,
        exchange_id=exchange_id,
        depth=0 if parent is None else 1,
        role="user" if author == "client" else "assistant",
        author=author,
        message_hash=f"m-{node_id}",
        delta_hash=f"d-{node_id}",
        context_hash=f"c-{node_id}",
        payload={"message": {"role": "user", "content": node_id}},
    )


def exchange(
    identifier: str, *, sequence: int, at: datetime = NOW, **row
) -> tuple[CapturedExchange, GraphDelta]:
    nodes = (
        node(f"nd-{identifier}-in", parent=None, author="client", exchange_id=identifier),
        node(f"nd-{identifier}-out", parent=f"nd-{identifier}-in", author="model", exchange_id=identifier),
    )
    delta = GraphDelta(
        exchange_id=identifier,
        nodes=nodes,
        input_prefix_node_id=None,
        input_node_ids=(nodes[0].id,),
        input_leaf_node_id=nodes[0].id,
        output_node_id=nodes[1].id,
        parent_output_node_id=None,
        matched_count=0,
        is_duplicate_retry=False,
    )
    captured = CapturedExchange(
        id=identifier,
        sequence=sequence,
        row={
            # The shape `writer/derive.py` produces. Kept complete rather than
            # minimal, because the exporters read it and a row missing a field
            # fails as a `KeyError` from inside a renderer rather than as
            # anything a reader of the test would recognise.
            "id": identifier,
            "sequence": sequence,
            "trajectory_id": "tr_1",
            "project": "rl",
            "provider": "openai",
            "model": "mock-model",
            "endpoint_kind": "chat_completions",
            "method": "POST",
            "path": "/chat/completions",
            "query": "",
            "request_start_at": at,
            "response_end_at": at,
            "http_status": 200,
            "streaming": False,
            "transport_error": None,
            "provider_error": None,
            "retry_attempt": 0,
            "completion_reason": "stop",
            "ttft_ms": 1.0,
            "duration_ms": 2.0,
            "chunk_count": 1,
            "stream_summary": {"chunk_count": 1},
            "usage": None,
            "sampling": {},
            "max_output_tokens": None,
            "tools": None,
            "tools_hash": "",
            "request_byte_count": 16,
            "response_byte_count": 16,
            "provider_request_id": None,
            "provider_response_id": f"resp-{identifier}",
            "previous_response_id": None,
            "previous_exchange_id": None,
            "source_metadata": {},
            "bodies_omitted": False,
            "has_payload": True,
            "late": False,
            "input_prefix_node_id": None,
            "input_node_ids": [nodes[0].id],
            "input_leaf_node_id": nodes[0].id,
            "output_node_id": nodes[1].id,
            "parent_output_node_id": None,
            "is_duplicate_retry": False,
            "schema_version": 4,
            "derivation_version": 2,
            **row,
        },
        request_body=b'{"messages": []}',
        response_body=b'{"choices": []}',
    )
    return captured, delta


# -- the aggregate ------------------------------------------------------------
def test_an_exchange_activates_the_trajectory_and_extends_its_graph():
    active = ActiveTrajectory.create(header())
    assert active.status == "created"

    captured, delta = exchange("ex_1", sequence=0)
    active.add_exchange(captured, delta, NOW)

    assert active.status == "active"
    assert active.first_event_at == NOW
    assert len(active.graph) == 2
    assert active.document().exchange_count == 1


def test_the_same_exchange_twice_is_one_exchange():
    """Recovery replays records this process may already have applied."""
    active = ActiveTrajectory.create(header())
    captured, delta = exchange("ex_1", sequence=0)
    active.add_exchange(captured, delta, NOW)
    active.add_exchange(captured, delta, NOW)

    assert len(active.exchanges) == 1
    assert len(active.graph) == 2


def test_sequences_are_per_trajectory_and_claimed_in_order():
    active = ActiveTrajectory.create(header())
    assert [active.next_sequence() for _ in range(3)] == [0, 1, 2]


def test_a_gap_makes_a_trajectory_incomplete_and_a_zero_gap_makes_it_uncertain():
    """The two doubts are different, and the document keeps them apart."""
    active = ActiveTrajectory.create(header())
    active.record_gap(2, "disk full", NOW)
    capture = active.document().public()["capture"]
    assert capture["calls_missing"] == 2
    assert capture["complete"] is False
    assert capture["recovery_uncertain"] is False

    active.record_gap(0, "recovered by a replacement process", NOW)
    capture = active.document().public()["capture"]
    assert capture["calls_missing"] == 2, "uncertainty is not a missing call"
    assert capture["recovery_uncertain"] is True


def test_annotations_merge_and_labels_add_and_remove():
    active = ActiveTrajectory.create(header(labels=["a"], annotations={"x": 1}))
    active.apply_metadata(metadata_update(annotations={"y": 2}, labels=["b"]))
    active.apply_metadata(metadata_update(remove_annotations=["x"], remove_labels=["a"]))

    assert active.labels == ["b"]
    assert active.annotations == {"y": 2}
    assert active.revision == 2


def test_finishing_closes_the_route_and_poisoning_is_terminal():
    active = ActiveTrajectory.create(header())
    assert active.accepting
    active.request_finish(command_result="success", request_hash="h", at=NOW)
    assert not active.accepting
    assert active.status == "finalizing"

    other = ActiveTrajectory.create(header("tr_2"))
    other.poison("a turn could not be attributed", NOW)
    assert other.status == "poisoned"
    assert not other.accepting
    # The trajectory stopped; capture did not lose anything. The turn that
    # caused this was captured exactly, so `complete` stays true and the
    # reason is on `errors`.
    assert other.integrity.complete is True
    assert other.integrity.errors == ["a turn could not be attributed"]


def test_payloads_leave_memory_once_they_are_durable():
    """The reason a long trajectory is bounded by its graph and not its bytes."""
    active = ActiveTrajectory.create(header())
    captured, delta = exchange("ex_1", sequence=0)
    active.add_exchange(captured, delta, NOW)
    assert active.exchanges[0].request_body

    active.shed_payloads()
    assert not isinstance(active.exchanges[0], CapturedExchange)
    assert active.exchanges[0].row is captured.row, "the derived row stays"


async def test_finish_waits_for_the_turns_already_in_flight():
    active = ActiveTrajectory.create(header())
    active.open_turn()
    assert await active.settle_turns(timeout=0.05) is False, "a turn is still open"
    active.close_turn()
    assert await active.settle_turns(timeout=0.05) is True


# -- journal round trip ---------------------------------------------------------
def test_a_journal_rebuilds_the_aggregate_it_recorded():
    """Written by one process, read by another, and the same trajectory."""
    records = [journal.TrajectoryCreated(header(labels=["seed"]))]
    for index in range(2):
        captured, delta = exchange(f"ex_{index}", sequence=index, at=NOW + timedelta(seconds=index))
        records.append(journal.ExchangeCommitted(captured, delta, NOW, "fingerprint"))
    records.append(journal.MetadataUpdated(metadata_update(annotations={"reward": 1.0}), NOW))

    rebuilt = rebuild(records, recovered=False)

    assert rebuilt is not None
    assert rebuilt.id == "tr_1"
    assert len(rebuilt.exchanges) == 2
    assert len(rebuilt.graph) == 4
    assert rebuilt.annotations == {"reward": 1.0}
    assert rebuilt.labels == ["seed"]
    assert rebuilt.exchanges[0].request_body == b'{"messages": []}'


def test_a_journal_with_no_header_rebuilds_nothing():
    """A file torn before creation became durable has nothing to attach to."""
    captured, delta = exchange("ex_0", sequence=0)
    assert rebuild([journal.ExchangeCommitted(captured, delta, NOW)], recovered=False) is None
    assert rebuild([], recovered=False) is None


@pytest.mark.parametrize(
    ("mode", "expect_recovery_uncertain", "expect_delivery_uncertain"),
    [("text", True, False), ("tokens", False, True)],
)
def test_recovery_marks_the_doubt_each_mode_actually_has(
    mode, expect_recovery_uncertain, expect_delivery_uncertain
):
    """Text capture cannot say which exchange it might have missed; token
    capture can say exactly which response it cannot vouch for."""
    captured, delta = exchange("ex_0", sequence=0)
    captured.delivery_confirmed = False
    rebuilt = rebuild(
        [
            journal.TrajectoryCreated(header(mode=mode)),
            journal.ExchangeCommitted(captured, delta, NOW),
        ],
        recovered=True,
    )

    assert rebuilt is not None
    assert rebuilt.integrity.recovery_uncertain is expect_recovery_uncertain
    assert bool(rebuilt.integrity.delivery_uncertain_exchange_ids) is expect_delivery_uncertain
    assert rebuilt.exchanges[0].delivery_uncertain is expect_delivery_uncertain


def test_a_confirmed_delivery_survives_recovery_as_certain():
    captured, delta = exchange("ex_0", sequence=0)
    captured.delivery_confirmed = False
    rebuilt = rebuild(
        [
            journal.TrajectoryCreated(header(mode="tokens")),
            journal.ExchangeCommitted(captured, delta, NOW),
            journal.ExchangeDeliveryConfirmed("ex_0", NOW),
        ],
        recovered=True,
    )

    assert rebuilt is not None
    assert rebuilt.integrity.delivery_uncertain_exchange_ids == []
    assert rebuilt.exchanges[0].delivery_confirmed is True

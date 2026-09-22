"""Acceptance criterion 11: trajectory metadata is editable post-hoc and queryable.

Labels and annotations are one surface with two shapes: labels are bare string
tags, annotations are a free-form key/value document. Both belong to the whole
trajectory, both stay mutable for its life, and neither keeps history.
"""

from __future__ import annotations


async def one_trajectory(stack):
    created = await stack.create_trajectory(project="annotated")
    await stack.chat(created, [{"role": "user", "content": "one"}])
    await stack.settle()
    return created


async def test_annotations_and_labels_are_editable_after_finish(stack):
    created = await one_trajectory(stack)
    await stack.finish(created["id"], labels=["draft"])

    response = await stack.patch(
        f"/v1/trajectories/{created['id']}/metadata",
        {"annotations": {"critique": "used the wrong tool on turn 2"}},
    )
    assert response.status_code == 200
    assert response.json()["annotations"]["critique"] == "used the wrong tool on turn 2"

    # Overwriting keeps only the latest value; there is no revision history.
    updated = await stack.patch(
        f"/v1/trajectories/{created['id']}/metadata",
        {"annotations": {"critique": "on reflection the tool choice was fine"}},
    )
    assert updated.json()["annotations"]["critique"] == "on reflection the tool choice was fine"

    # Tags add and remove freely, so a label can follow a reward.
    tagged = await stack.patch(
        f"/v1/trajectories/{created['id']}/metadata",
        {"labels": ["success"], "remove_labels": ["draft"]},
    )
    assert tagged.json()["labels"] == ["success"]


async def test_annotations_merge_rather_than_replace(stack):
    created = await one_trajectory(stack)
    await stack.patch(f"/v1/trajectories/{created['id']}/metadata", {"annotations": {"a": 1}})
    result = await stack.patch(
        f"/v1/trajectories/{created['id']}/metadata", {"annotations": {"b": 2}}
    )
    assert result.json()["annotations"] == {"a": 1, "b": 2}

    removed = await stack.patch(
        f"/v1/trajectories/{created['id']}/metadata", {"remove_annotations": ["a"]}
    )
    assert removed.json()["annotations"] == {"b": 2}


async def test_metadata_never_mutates_the_captured_data(stack):
    created = await one_trajectory(stack)
    graph = await stack.graph(created["id"])
    leaf = graph["leaf_assistant_node_ids"][0]
    before = await stack.node_payload(leaf)
    before_exchange = (await stack.exchanges(created["id"]))[0]

    await stack.patch(
        f"/v1/trajectories/{created['id']}/metadata",
        {"annotations": {"critique": "used the wrong tool"}, "labels": ["reviewed"]},
    )

    after = await stack.node_payload(leaf)
    after_exchange = (await stack.exchanges(created["id"]))[0]
    assert after == before
    assert after_exchange["output_node_id"] == before_exchange["output_node_id"]


async def test_numeric_annotations_are_queryable(stack):
    """No field is predeclared, so filtering reads the JSON document directly."""
    created = await one_trajectory(stack)
    await stack.patch(
        f"/v1/trajectories/{created['id']}/metadata",
        {"annotations": {"reward_A": 0.9, "reward_B": 0.2, "note": "text"}},
    )

    listing = (await stack.get("/v1/trajectories")).json()["data"]

    def above(key: str, threshold: float) -> list[str]:
        return [
            row["id"]
            for row in listing
            if float(row["annotations"].get(key, 0)) > threshold
        ]

    assert above("reward_A", 0.4) == [created["id"]]
    assert above("reward_B", 0.4) == []


async def test_labels_are_searchable_tags(stack):
    created = await one_trajectory(stack)
    other = await stack.create_trajectory(project="annotated")
    await stack.patch(f"/v1/trajectories/{created['id']}/metadata", {"labels": ["success", "reviewed"]})
    await stack.patch(f"/v1/trajectories/{other['id']}/metadata", {"labels": ["failed"]})

    listing = (await stack.get("/v1/trajectories")).json()["data"]
    rows = [row["id"] for row in listing if "success" in row["labels"]]
    assert rows == [created["id"]]


async def test_a_label_that_looks_like_a_pair_is_rejected(stack):
    """Labels are bare tags; a value belongs in an annotation."""
    created = await one_trajectory(stack)
    response = await stack.patch(
        f"/v1/trajectories/{created['id']}/metadata", {"labels": ["task=task-17"]}
    )
    assert response.status_code == 400
    assert "bare tags" in response.json()["detail"]


async def test_metadata_on_unknown_trajectory_is_rejected(stack):
    """404, not 400: nothing is wrong with the edit, there is nothing to edit.

    A trajectory is unknown when it is neither open on this process nor
    committed to the record, which is the same test `finish` makes.
    """
    response = await stack.patch(
        "/v1/trajectories/tr_does_not_exist/metadata", {"annotations": {"a": 1}}
    )
    assert response.status_code == 404


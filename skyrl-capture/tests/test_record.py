"""The record: a journal per trajectory in flight, a compiled record per one
that finished.

The claim being tested is narrow and strong: **what a reader answers off the
record is what the writing process held, and an export off the committed
record is byte-identical to one off the live aggregate.** That holds because
the committed record is compiled from the journal the live aggregate was
written to, and because both are read through one view -- so there is no second
implementation of a read path to keep in step, and this file is what would
notice if one appeared.

Everything else here is the layout, which is public: a reader in another
language walks a journal with `persistence/journal.py` as its only
specification, and opens a committed record with any JSON parser and zstd.
"""

from __future__ import annotations

import asyncio

import orjson
import pytest

from skyrl_capture.export.service import render_artifact
from skyrl_capture.export.view import view_of, view_of_active
from skyrl_capture.persistence.layout import (
    MANIFEST,
    active_path,
    active_paths,
    committed_path,
    committed_paths,
    shard,
)
from skyrl_capture.version import RECORD_FORMAT_VERSION


async def captured(
    stack, *, project="rl", run_id="run-a", task_id="t1", step=0, turns=2, reward=1.0,
    trajectory_id=None,
):
    """One finished trajectory with a branch in it, so the graph is not trivial."""
    extra = {"trajectory_id": trajectory_id} if trajectory_id else {}
    created = await stack.create_trajectory(
        project=project, run_id=run_id, task_id=task_id, step=step, **extra
    )
    history: list[dict[str, str]] = []
    for turn in range(turns):
        history.append({"role": "user", "content": f"turn {turn}"})
        reply = await stack.chat(created, history)
        history.append(reply.json()["choices"][0]["message"])
    # A branch: the same prefix with a different continuation.
    await stack.chat(created, [*history[:-1], {"role": "user", "content": "elsewhere"}])
    await stack.settle()
    await stack.finish(created["id"], annotations={"reward": reward})
    return created["id"]


# -- the round trip ---------------------------------------------------------
# The three fields finishing decides. Everything else an export carries has
# to be identical whether it was rendered from the live aggregate or from the
# committed record, or the record has grown a second read path.
LIFECYCLE_FIELDS = ("status", "finished_at")


def without_lifecycle(raw: bytes) -> list[dict]:
    rows = [orjson.loads(line) for line in raw.splitlines() if line.strip()]
    for row in rows:
        for field in LIFECYCLE_FIELDS:
            row.pop(field, None)
    return rows


@pytest.mark.parametrize("export_format", ["graph", "replay", "text_samples"])
async def test_an_export_off_the_record_matches_one_off_the_live_aggregate(stack, export_format):
    """The whole point. If these ever differ, the record has grown a second
    implementation of the export path."""
    created = await stack.create_trajectory(project="rl", run_id="run-a")
    history: list[dict[str, str]] = []
    for turn in range(2):
        history.append({"role": "user", "content": f"turn {turn}"})
        history.append((await stack.chat(created, history)).json()["choices"][0]["message"])
    await stack.settle()

    from_live = view_of_active(stack.aggregate(created["id"]))
    await stack.finish(created["id"])
    # The committed record is what the reader will hand every later caller.
    from_record = view_of(stack.record(created["id"]))

    options = {"allow_repeated_targets": False, "mask_abandoned": False}
    origin = min(exchange["request_start_at"] for exchange in from_live.exchanges)
    assert without_lifecycle(
        render_artifact([from_live], export_format=export_format, options=options, origin=origin)
    ) == without_lifecycle(
        render_artifact([from_record], export_format=export_format, options=options, origin=origin)
    )


async def test_a_token_export_off_the_record_matches(tokens_stack):
    """Token rows carry the arrays, so this is the case where a lossy record
    would show up."""
    created = await tokens_stack.create_trajectory(project="rl", run_id="run-a")
    await tokens_stack.client.post(
        f"{created['base_url']}/chat/completions",
        json={"model": "mock-tokens-model", "messages": [{"role": "user", "content": "hi"}],
              "max_tokens": 4},
        headers={"authorization": "Bearer client-key"},
    )
    await tokens_stack.settle()
    from_live = view_of_active(tokens_stack.aggregate(created["id"]))
    await tokens_stack.finish(created["id"])
    from_record = view_of(tokens_stack.record(created["id"]))

    options = {"allow_repeated_targets": False, "mask_abandoned": False, "overlong_filtering": False}
    assert without_lifecycle(
        render_artifact([from_live], export_format="token_samples", options=options, origin=None)
    ) == without_lifecycle(
        render_artifact([from_record], export_format="token_samples", options=options, origin=None)
    )


# -- the layout -------------------------------------------------------------
async def test_a_trajectory_in_flight_is_a_journal_under_its_shard(stack):
    created = await stack.create_trajectory()
    await stack.chat(created, [{"role": "user", "content": "one turn"}])
    await stack.settle()

    root = stack.runtime.config.record_dir
    path = active_path(root, created["id"])
    assert path.is_file()
    assert path.parent.name == shard(created["id"])
    assert [p.name for p in active_paths(root)] == [f"{created['id']}.capture"]
    assert committed_paths(root) == []


async def test_finishing_moves_it_from_active_to_committed(stack):
    trajectory_id = await captured(stack, turns=1)
    root = stack.runtime.config.record_dir

    assert committed_path(root, trajectory_id).is_file()
    assert committed_path(root, trajectory_id).name.endswith(".json.zst")
    assert not active_path(root, trajectory_id).exists(), "the journal is redundant now"


async def test_the_manifest_holds_the_format_and_nothing_that_moves(stack):
    await captured(stack, turns=1)
    root = stack.runtime.config.record_dir
    document = orjson.loads((root / MANIFEST).read_bytes())

    assert document["record_version"] == RECORD_FORMAT_VERSION
    assert document["upstream"]["type"] == "openai"
    assert "api_key" not in str(document), "provenance, never a credential"
    # Nothing in here changes while a run is in progress, so there is nothing
    # for two capture processes to race over rewriting.
    assert set(document) == {
        "record_version", "journal_format_version", "schema_version",
        "derivation_version", "created_at", "upstream",
    }


async def test_asking_for_what_is_not_there_says_so(stack):
    assert (await stack.get("/v1/trajectories/tr_nope")).status_code == 404
    assert (await stack.get("/v1/trajectories/tr_nope/graph")).status_code == 404


# -- what the reader sees ---------------------------------------------------
async def test_a_run_in_progress_is_readable_from_its_journal(stack):
    """A trajectory is in the record as soon as it is created, and every turn
    lands as it commits -- so the viewer shows work in progress without asking
    any proxy what it is holding."""
    created = await stack.create_trajectory(project="rl", run_id="run-a")
    await stack.chat(created, [{"role": "user", "content": "one turn"}])
    await stack.refresh()

    row = (await stack.get(f"/v1/trajectories/{created['id']}")).json()
    assert row["status"] == "active"
    assert row["capture"]["exchange_count"] == 1
    assert row["capture"]["complete"] is True


async def test_the_reward_a_finish_carries_reaches_the_record(stack):
    created = await stack.create_trajectory(project="rl", run_id="run-a")
    await stack.chat(created, [{"role": "user", "content": "one turn"}])
    await stack.finish(created["id"], annotations={"reward": 0.75})

    assert stack.record(created["id"]).trajectory.annotations == {"reward": 0.75}
    assert (await stack.get(f"/v1/trajectories/{created['id']}")).json()["annotations"] == {
        "reward": 0.75
    }


async def test_a_reward_annotated_afterwards_rewrites_the_committed_record(stack):
    """Nothing is sealed by finishing. A late reward loads the record, edits
    it, bumps its revision and replaces the file."""
    created = await stack.create_trajectory(project="rl", run_id="run-a")
    await stack.chat(created, [{"role": "user", "content": "one turn"}])
    await stack.finish(created["id"])
    assert stack.record(created["id"]).revision == 0

    result = await stack.patch(
        f"/v1/trajectories/{created['id']}/metadata",
        {"annotations": {"reward": 1.0}, "labels": ["scored"]},
    )
    assert result.status_code == 200
    assert result.json()["revision"] == 1
    await stack.refresh()

    row = (await stack.get(f"/v1/trajectories/{created['id']}")).json()
    assert row["annotations"] == {"reward": 1.0}
    assert row["labels"] == ["scored"]
    listing = (await stack.get("/v1/trajectories?run_id=run-a")).json()
    assert listing["total"] == 1, "a correction, not a duplicate"


async def test_the_bodies_are_in_the_record_and_not_in_memory(stack):
    """Raw request and response bytes leave hot memory once they are durable:
    a trajectory is bounded by its graph, not by its prompts."""
    created = await stack.create_trajectory()
    await stack.chat(created, [{"role": "user", "content": "verbatim"}])
    exchanges = await stack.exchanges(created["id"])
    row = stack.exchange_rows(created["id"])[0]
    assert row["has_payload"] is True
    assert "request_body" not in row and "request_uri" not in row
    assert not hasattr(stack.aggregate(created["id"]).exchanges[0], "request_body")

    bodies = await stack.stored_bodies(exchanges[0]["id"])
    assert orjson.loads(bodies["request"])["messages"][0]["content"] == "verbatim"
    assert b"mock reply text" in bodies["response"]


async def test_a_finished_trajectory_leaves_hot_memory(stack):
    created = await stack.create_trajectory()
    await stack.chat(created, [{"role": "user", "content": "hi"}])
    assert stack.aggregate(created["id"]) is not None

    await stack.finish(created["id"])

    assert stack.aggregate(created["id"]) is None
    assert stack.runtime.registry.count() == 0
    assert stack.record(created["id"]) is not None


# -- restart ----------------------------------------------------------------
async def test_a_restarted_process_serves_what_the_last_one_wrote(stack_builder, tmp_path):
    """No startup replay: the second process reads one journal, and only when
    a request for that trajectory arrives."""
    root = tmp_path / "traces"
    first = await stack_builder(record_dir=root)
    finished = await captured(first, turns=1)
    open_one = await first.create_trajectory(project="rl", trajectory_id="tr_still_open")
    await first.chat(open_one, [{"role": "user", "content": "before the restart"}])
    await first.settle()
    await first.application.runtime.stop()

    second = await stack_builder(record_dir=root)
    assert second.runtime.registry.count() == 0, "nothing was replayed at startup"

    detail = (await second.get(f"/v1/trajectories/{finished}")).json()
    assert detail["status"] == "finished"
    assert detail["capture"]["exchange_count"] == 2

    # The open one is recovered by the request that needs it, and continues.
    route = {"base_url": f"{second.base_url}/route/tr_still_open/v1"}
    response = await second.chat(route, [
        {"role": "user", "content": "before the restart"},
        {"role": "assistant", "content": "mock reply text"},
        {"role": "user", "content": "after it"},
    ])
    assert response.status_code == 200
    assert second.runtime.registry.stats()["recovered"] == 1
    assert len(second.aggregate("tr_still_open").exchanges) == 2


async def test_text_recovery_says_it_cannot_vouch_for_what_it_found(stack_builder, tmp_path):
    """Text capture commits behind the response, so a replacement process
    cannot know whether the last one served a turn it never wrote."""
    root = tmp_path / "traces"
    first = await stack_builder(record_dir=root)
    created = await first.create_trajectory(trajectory_id="tr_uncertain")
    await first.chat(created, [{"role": "user", "content": "one"}])
    await first.settle()
    await first.application.runtime.stop()

    second = await stack_builder(record_dir=root)
    route = {"base_url": f"{second.base_url}/route/tr_uncertain/v1"}
    await second.chat(route, [{"role": "user", "content": "two"}])
    await second.settle()

    assert second.aggregate("tr_uncertain").integrity.recovery_uncertain is True
    await second.refresh()
    capture = (await second.get("/v1/trajectories/tr_uncertain")).json()["capture"]
    assert capture["recovery_uncertain"] is True, "written down, not only in memory"
    assert capture["calls_missing"] == 0, "uncertainty is not a missing call"

    await second.finish("tr_uncertain")
    assert second.record("tr_uncertain").trajectory.integrity.recovery_uncertain is True


# -- the viewer over a record ----------------------------------------------
async def test_the_viewer_serves_the_same_reads_off_a_directory(stack_builder, tmp_path):
    """`skyrl-capture view --record` is the same app and the same /v1 paths,
    answered from files by a process that captures nothing."""
    import httpx
    from conftest import record_app

    root = tmp_path / "traces"
    stack = await stack_builder(record_dir=root)
    trajectory_id = await captured(stack, task_id="t1", step=0, reward=1.0)
    await captured(stack, task_id="t2", step=0, reward=0.0)
    open_one = await stack.create_trajectory(project="rl", run_id="run-a", task_id="t3")
    await stack.chat(open_one, [{"role": "user", "content": "still going"}])
    await stack.settle()

    served = record_app(root)
    transport = httpx.ASGITransport(app=served)
    async with httpx.AsyncClient(transport=transport, base_url="http://record") as client:
        assert (await client.get("/healthz")).json()["source"] == "record"

        # Nothing has indexed yet: this app was mounted on a transport with no
        # lifespan, which is what a manual refresh is for.
        runs = (await client.get("/v1/runs?refresh=true")).json()["data"]
        assert [run["id"] for run in runs] == ["run-a"]
        assert runs[0]["trajectory_count"] == 3

        by_step = (await client.get("/v1/trajectories?run_id=run-a&step=0")).json()["data"]
        assert {row["task_id"] for row in by_step} == {"t1", "t2"}
        assert {row["annotations"]["reward"] for row in by_step} == {0.0, 1.0}

        listing = (await client.get("/v1/trajectories")).json()
        assert listing["indexing"] is False
        assert listing["total"] == 3
        # Finished and unfinished in one listing, each exactly once.
        assert {item["status"] for item in listing["data"]} == {"finished", "active"}
        assert len({item["id"] for item in listing["data"]}) == 3

        detail = (await client.get(f"/v1/trajectories/{trajectory_id}")).json()
        assert detail["run_id"] == "run-a" and detail["task_id"] == "t1"
        assert detail["annotations"]["reward"] == 1.0

        graph = (await client.get(f"/v1/trajectories/{trajectory_id}/graph")).json()
        assert graph["nodes"] and graph["leaf_node_ids"]
        assert any(node["parent_node_id"] is None for node in graph["nodes"])

        # The one still running is readable too, off its journal.
        live_graph = (await client.get(f"/v1/trajectories/{open_one['id']}/graph")).json()
        assert len(live_graph["nodes"]) == 2

        exchanges = (await client.get(f"/v1/trajectories/{trajectory_id}/exchanges")).json()
        assert exchanges["data"] and exchanges["data"][0]["request_start_at"]

        paths = (await client.get(f"/v1/trajectories/{trajectory_id}/paths")).json()
        assert paths["mode"] == "text"
        assert paths["paths"]

        # `view` serves no HTML: the viewer is a separate program that reads
        # these same routes, so what matters is that the routes are all here.
        assert (await client.get("/ui/")).status_code == 404


async def test_the_viewer_decodes_a_tokens_record(stack_builder, tmp_path):
    """The record carries the text and offsets produced with its token IDs."""
    import httpx
    from conftest import record_app

    root = tmp_path / "traces"
    stack = await stack_builder(tokens=True, record_dir=root)
    created = await stack.create_trajectory(project="rl", run_id="run-a", task_id="t1", step=0)
    await stack.client.post(
        f"{created['base_url']}/chat/completions",
        json={"model": "mock-tokens-model", "messages": [{"role": "user", "content": "hello"}],
              "max_tokens": 4},
        headers={"authorization": "Bearer client-key"},
    )
    await stack.settle()
    await stack.finish(created["id"], annotations={"reward": 1.0})

    served = record_app(root)
    transport = httpx.ASGITransport(app=served)
    async with httpx.AsyncClient(transport=transport, base_url="http://record") as client:
        body = (await client.get(f"/v1/trajectories/{created['id']}/paths")).json()
    assert body["mode"] == "tokens"
    assert "decoder" not in body
    blocks = body["paths"][0]["blocks"]
    assert [block["kind"] for block in blocks] == ["given", "scaffold", "sampled"]
    assert "<|im_start|>" in blocks[0]["text"]


async def test_a_record_listing_pages_with_the_cursor_it_prints(stack_builder, tmp_path):
    """A record pages like the live service -- it is the same reader -- and
    the cursor is opaque: a caller hands it back and never reads it."""
    from typer.testing import CliRunner

    from skyrl_capture.cli.main import app

    root = tmp_path / "traces"
    stack = await stack_builder(record_dir=root)
    for task in ("t1", "t2", "t3"):
        await captured(stack, task_id=task, step=0, turns=1)

    # The CLI is a synchronous program with its own event loop, so it is run
    # on a thread rather than inside this test's.
    result = await asyncio.to_thread(
        CliRunner().invoke, app, ["list", "--record", str(root), "--limit", "2"]
    )
    assert result.exit_code == 0, result.output
    assert "more available (3 match)" in result.output, "it counts what it filtered"
    cursor = result.output.rsplit("--cursor ", 1)[1].strip()

    # And the cursor it printed opens the next page rather than repeating this
    # one, which is the part a printed cursor is worth nothing without.
    following = await asyncio.to_thread(
        CliRunner().invoke, app, ["list", "--record", str(root), "--limit", "2", "--cursor", cursor]
    )
    assert following.exit_code == 0, following.output
    first_page = {line.split()[0] for line in result.output.splitlines() if line.startswith("tr_")}
    second_page = {line.split()[0] for line in following.output.splitlines() if line.startswith("tr_")}
    assert len(first_page) == 2 and len(second_page) == 1
    assert not (first_page & second_page)

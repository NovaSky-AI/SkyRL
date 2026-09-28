"""Phase 3: benchmark adapters and export determinism."""

from __future__ import annotations

import asyncio
from pathlib import Path

import orjson
import pytest


async def linear_trajectory(stack, *, turns: int = 3, project: str = "export-linear"):
    """Build a straightforward multi-turn conversation."""
    created = await stack.create_trajectory(project=project)
    history: list[dict] = [{"role": "system", "content": "S"}]
    for index in range(turns):
        history.append({"role": "user", "content": f"turn {index}"})
        response = await stack.chat(
            created, list(history), headers={"x-mock-reply": f"reply {index}"}, max_tokens=128
        )
        history.append({"role": "assistant", "content": response.json()["choices"][0]["message"]["content"]})
        await stack.settle()
        # A measurable wait between calls, so delays are non-trivial.
        await asyncio.sleep(0.03)
    return created


async def forked_trajectory(stack, *, project: str = "export-forked"):
    """Build a trajectory that branches after an exact model output."""
    created = await stack.create_trajectory(project=project)
    shared = [{"role": "system", "content": "S"}, {"role": "user", "content": "shared question"}]
    await stack.chat(created, shared, headers={"x-mock-reply": "shared reply"})
    await stack.settle()
    history = shared + [{"role": "assistant", "content": "shared reply"}]
    await asyncio.sleep(0.03)
    await stack.chat(created, history + [{"role": "user", "content": "left branch"}],
                     headers={"x-mock-reply": "left reply"})
    await stack.settle()
    await asyncio.sleep(0.03)
    await stack.chat(created, history + [{"role": "user", "content": "right branch"}],
                     headers={"x-mock-reply": "right reply"})
    await stack.settle()
    return created


async def repaired_trajectory(stack, *, project: str = "export-repair"):
    """A client edits the sampled assistant message before replaying it."""
    created = await stack.create_trajectory(project=project)
    history = [{"role": "user", "content": "look up the weather"}]
    await stack.chat(created, history, headers={"x-mock-reply": 'CALL get_weather(city="Berlin"'})
    await stack.settle()
    await stack.chat(
        created,
        history
        + [
            {"role": "assistant", "content": 'CALL get_weather(city="Berlin")'},
            {"role": "user", "content": "tool result: 18C"},
        ],
        headers={"x-mock-reply": "It is 18C."},
    )
    await stack.settle()
    return created


# -- graph -----------------------------------------------------------------
async def test_graph_export_is_one_line_per_trace(stack):
    created = await linear_trajectory(stack, turns=2, project="graph-shape")
    await stack.finish(created["id"], labels=["shape"])
    records = await stack.export_lines(trajectory=created["id"], format="graph")

    assert len(records) == 1, "one line per trace, with the tree nested inside"
    line = records[0]
    assert line["trajectory_id"] == created["id"]
    assert line["labels"] == ["shape"]
    assert line["integrity"] == {"complete": True, "calls_missing": 0, "calls_after_close": 0}
    assert "capture" not in line, "the integrity block replaced it"

    nodes = line["nodes"]
    assert [node["author"] for node in nodes] == ["client", "client", "model", "client", "model"]
    assert nodes[0]["parent"] is None
    assert nodes[0]["children"] == [nodes[1]["node_id"]]
    # A call attaches to the node it produced, and nowhere else.
    assert all("call" not in node for node in nodes if node["author"] == "client")
    call = next(node["call"] for node in nodes if node["author"] == "model")
    assert call["endpoint"] == "/v1/chat/completions"
    assert call["timing"]["started_at"] and call["timing"]["ended_at"]
    assert call["usage"]["total_tokens"] > 0
    assert "gap_ms" not in call["timing"], "the wait is derivable from the timestamps"


async def test_graph_export_carries_sampling_and_a_chained_hash(stack):
    created = await stack.create_trajectory(project="graph-sampling")
    await stack.chat(created, [{"role": "user", "content": "hi"}], temperature=0.7, top_p=0.9)
    await stack.settle()
    line = (await stack.export_lines(trajectory=created["id"], format="graph"))[0]

    call = next(node["call"] for node in line["nodes"] if node["author"] == "model")
    assert call["sampling"]["temperature"] == 0.7
    assert call["sampling"]["top_p"] == 0.9

    # context_hash chains, message_hash does not. Both are needed: one answers
    # "same conversation", the other "same message".
    nodes = line["nodes"]
    assert nodes[0]["context_hash"] != nodes[0]["message_hash"]
    assert nodes[1]["context_hash"] != nodes[0]["context_hash"]


async def test_graph_export_shows_a_repair_as_two_authors(stack):
    created = await repaired_trajectory(stack)
    line = (await stack.export_lines(trajectory=created["id"], format="graph"))[0]
    nodes = {node["node_id"]: node for node in line["nodes"]}

    fork = next(node for node in line["nodes"] if len(node["children"]) > 1)
    children = [nodes[child] for child in fork["children"]]
    assert sorted(child["author"] for child in children) == ["client", "model"]
    sampled = next(child for child in children if child["author"] == "model")
    repaired = next(child for child in children if child["author"] == "client")
    assert sampled["message"]["content"] != repaired["message"]["content"]
    assert sampled["children"] == [], "the discarded original is a dead end"


# -- replay ----------------------------------------------------------------
async def test_replay_export_carries_arrival_and_delay(stack):
    created = await linear_trajectory(stack, turns=3, project="replay-linear")
    records = await stack.export_lines(trajectory=created["id"], format="replay")

    assert len(records) == 1
    session = records[0]
    assert session["arrival_ms"] == 0, "the first session starts the trace"
    delays = [turn["delay_ms"] for turn in session["turns"]]
    assert delays[0] is None, "nothing precedes the first turn"
    assert all(value >= 20 for value in delays[1:]), delays
    assert all(turn["endpoint"] == "/v1/chat/completions" for turn in session["turns"])
    assert all(turn["method"] == "POST" for turn in session["turns"])


async def test_replay_payload_is_the_request_and_nothing_else(stack):
    created = await stack.create_trajectory(project="replay-payload")
    await stack.chat(
        created, [{"role": "user", "content": "hi"}], temperature=0.4, seed=11, max_tokens=32
    )
    await stack.settle()
    turn = (await stack.export_lines(trajectory=created["id"], format="replay"))[0]["turns"][0]

    payload = turn["payload"]
    assert payload["messages"][0]["content"] == "hi"
    assert payload["temperature"] == 0.4 and payload["seed"] == 11
    assert payload["max_tokens"] == 32
    assert "response" not in turn, "replay does not need what came back"


async def test_a_fork_becomes_linked_sessions(stack):
    """A session is a maximal linear run between forks."""
    created = await forked_trajectory(stack, project="replay-fork")
    rows = await stack.export_lines(trajectory=created["id"], format="replay")

    # Shared prefix, then one session per branch -- nothing repeated.
    assert len(rows) == 3
    assert sum(len(r["turns"]) for r in rows) == 3, "one turn per real call, no inflation"

    root = next(r for r in rows if r["parent_session_id"] is None)
    children = [r for r in rows if r.get("parent_session_id") == root["session_id"]]
    assert len(children) == 2
    assert sorted(root["forks"]) == sorted(r["session_id"] for r in children)


async def test_a_child_session_is_scheduled_from_its_parents_end(stack):
    """Absolute offsets bake one rollout's latencies into the artifact.

    A child carries `delay_ms` measured from when its parent's last turn
    finished, so replaying against a different model still issues it after the
    work it continues from.
    """
    created = await forked_trajectory(stack, project="replay-relative")
    rows = await stack.export_lines(trajectory=created["id"], format="replay")

    root = next(r for r in rows if r["parent_session_id"] is None)
    assert root["arrival_ms"] == 0
    assert root["delay_ms"] is None, "nothing precedes a root session"
    assert root["forks"], "but it does fork"

    exchanges = await stack.exchanges(created["id"])
    parent = exchanges[0]
    assert parent["response_end_at"] is not None

    # The delay is the arrival minus everything that ran before this child --
    # here, the parent's one call. Asserting the difference rather than
    # `delay < arrival` is what makes this stable: against a mock upstream the
    # parent can finish in under half a millisecond, and both numbers are
    # reported as whole milliseconds, so the strict inequality was a coin flip
    # on a fast machine.
    for child in (r for r in rows if r["parent_session_id"] is not None):
        assert child["delay_ms"] >= 0
        assert child["arrival_ms"] - child["delay_ms"] >= round(parent["duration_ms"]) - 1, (
            "measured from the parent's end, not from the clock"
        )


async def test_trajectories_in_one_export_share_a_timeline(stack):
    """Otherwise replaying a project fires every trajectory at once."""
    import asyncio

    first = await linear_trajectory(stack, turns=1, project="timeline")
    await stack.finish(first["id"])
    await asyncio.sleep(0.2)
    second = await linear_trajectory(stack, turns=1, project="timeline")
    await stack.finish(second["id"])

    rows = await stack.export_lines(project="timeline", format="replay")
    arrivals = {
        r["trajectory_id"]: r["arrival_ms"] for r in rows if r["parent_session_id"] is None
    }
    assert len(arrivals) == 2
    assert sorted(arrivals.values())[0] == 0, "the earliest call in the export is the origin"
    assert sorted(arrivals.values())[1] >= 150, "the later trajectory keeps its offset"


# -- text samples ----------------------------------------------------------
async def test_text_samples_emit_one_row_per_path(stack):
    created = await forked_trajectory(stack, project="samples-fanout")
    rows = await stack.export_lines(trajectory=created["id"], format="text_samples")

    assert len(rows) == 2, "one row per root-to-leaf path"
    assert all(row["abandoned"] is False for row in rows), "both branches continued"
    # The shared ancestor appears in both rows, because each row is a whole
    # conversation -- but it is a training target in only one of them.
    shared = [
        message for row in rows for message in row["messages"]
        if message["message"]["content"] == "shared reply"
    ]
    assert len(shared) == 2
    assert [message["trainable"] for message in shared] == [True, False]


async def test_a_sampled_message_is_a_target_once_by_default(stack):
    """Branches share ancestors, so the safe behaviour is the default one."""
    created = await forked_trajectory(stack, project="samples-once")
    rows = await stack.export_lines(trajectory=created["id"], format="text_samples")

    # Three sampled generations in this fixture, each a target exactly once.
    assert sum(row["trainable_count"] for row in rows) == 3
    masked = next(r for r in rows if r["messages"][2]["trainable"] is False)
    assert masked["messages"][2]["author"] == "model", "it was a generation"


async def test_repeated_targets_are_opt_in(stack):
    """The opposite is a deliberate choice: the same generation weighted twice."""
    created = await forked_trajectory(stack, project="samples-repeat")
    rows = await stack.export_lines(
        trajectory=created["id"],
        format="text_samples",
        options={"allow_repeated_targets": True},
    )

    shared = [
        message for row in rows for message in row["messages"]
        if message["message"]["content"] == "shared reply"
    ]
    assert all(message["trainable"] for message in shared), "now a target in both rows"
    assert sum(row["trainable_count"] for row in rows) == 4, "one generation counted twice"


async def test_a_repaired_message_is_never_trainable(stack):
    """No model emitted it, so nothing may train on it."""
    created = await repaired_trajectory(stack, project="samples-repair")
    rows = await stack.export_lines(trajectory=created["id"], format="text_samples")

    repaired = [
        message for row in rows for message in row["messages"]
        if message["author"] == "client" and message["message"]["role"] == "assistant"
    ]
    assert repaired, "the fixture produced a client-authored assistant message"
    assert all(message["trainable"] is False for message in repaired)


async def test_mask_abandoned_keeps_the_row_and_zeroes_it(stack):
    created = await repaired_trajectory(stack, project="samples-abandoned")
    plain = await stack.export_lines(trajectory=created["id"], format="text_samples")
    masked = await stack.export_lines(
        trajectory=created["id"], format="text_samples", options={"mask_abandoned": True}
    )

    # The row count means the same thing under every flag combination.
    assert len(plain) == len(masked) == 2
    abandoned = next(row for row in masked if row["abandoned"])
    assert abandoned["trainable_count"] == 0
    assert abandoned["masked_reason"] == "abandoned"
    # The discarded sampled text is still there, still attributed to the model.
    sampled = [item for item in abandoned["messages"] if item["author"] == "model"]
    assert sampled and sampled[-1]["message"]["content"].startswith("CALL get_weather")
    assert any(
        item["trainable"] for row in plain if row["abandoned"] for item in row["messages"]
    )


# -- job record and determinism --------------------------------------------
async def test_a_project_export_selects_a_snapshot_and_records_it(stack):
    """Acceptance criterion 12, reshaped: the job record carries the facts.

    There is no manifest. Everything a consumer needs to judge the dataset --
    which trajectories were selected, what flags produced it, how many records
    and bytes, the checksum -- is on the export job, and per-trajectory capture
    integrity travels inside the data on each trajectory it describes.
    """
    first = await linear_trajectory(stack, turns=1, project="bulk")
    await stack.finish(first["id"], labels=["trial-1"])
    second = await linear_trajectory(stack, turns=2, project="bulk")
    await stack.finish(second["id"], labels=["trial-2"])
    # An unfinished trajectory in the same project must not be selected.
    unfinished = await linear_trajectory(stack, turns=1, project="bulk")

    job = await stack.run_export(
        project="bulk", format="graph", options={"mask_abandoned": True}
    )
    assert job["status"] == "ready"
    selected = set(job["selected_trajectory_ids"])
    assert selected == {first["id"], second["id"]}
    assert unfinished["id"] not in selected
    assert job["record_count"] == 2
    assert job["checksum"].startswith("sha256:")
    assert job["options"] == {"mask_abandoned": True}
    assert "manifest" not in job

    # The artifact is plain JSONL at every scope, and integrity rides along.
    from skyrl_capture.compression import decompress

    raw = decompress(stack.artifact(job))
    lines = [orjson.loads(line) for line in raw.splitlines() if line.strip()]
    assert {line["trajectory_id"] for line in lines} == selected
    assert all(line["integrity"]["complete"] for line in lines)


async def test_export_is_deterministic_for_one_snapshot(stack):
    created = await linear_trajectory(stack, turns=2, project="deterministic")
    await stack.finish(created["id"])
    first = await stack.run_export(trajectory=created["id"], format="graph")
    second = await stack.run_export(trajectory=created["id"], format="graph")
    assert first["checksum"] == second["checksum"]
    assert first["byte_count"] == second["byte_count"]


async def test_export_rejects_bad_input(stack):
    created = await stack.create_trajectory()
    assert (await stack.post("/v1/exports", {"format": "nope", "trajectory": created["id"]})).status_code == 400
    assert (await stack.post("/v1/exports", {"format": "graph"})).status_code == 400
    both = await stack.post(
        "/v1/exports", {"format": "graph", "project": "p", "trajectory": created["id"]}
    )
    assert both.status_code == 400
    unknown = await stack.post("/v1/exports", {"format": "graph", "trajectory": "tr_nope"})
    assert unknown.status_code == 404


async def test_the_artifact_is_written_beside_the_record(stack):
    """A record directory holds a run and what was exported off it, so copying
    the directory copies both."""
    from pathlib import Path

    created = await linear_trajectory(stack, turns=1, project="beside")
    await stack.finish(created["id"])
    job = await stack.run_export(project="beside", format="graph")

    path = Path(job["output_uri"].removeprefix("file://"))
    assert path.is_file()
    assert path.parent.parent == stack.runtime.config.record_dir / "exports" / "artifacts"
    assert path.name.endswith(".jsonl.zst")
    # There is one copy and one way to reach it: no destination, no signed URL.
    assert "delivered_uri" not in job and "delivery_error" not in job
    assert job["download_url"].endswith(f"/v1/exports/{job['id']}/download")


async def test_download_endpoint_serves_the_artifact(stack):
    created = await linear_trajectory(stack, turns=1, project="download")
    await stack.finish(created["id"])
    job = await stack.run_export(trajectory=created["id"], format="replay")
    assert job["download_url"]

    response = await stack.get(f"/v1/exports/{job['id']}/download")
    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]
    assert len(response.content) == job["byte_count"]


# -- local artifact encoding ------------------------------------------------
def test_compression_decides_what_lands_on_disk():
    """A stored artifact is compressed; a file at a terminal usually should not be."""
    import gzip
    import json

    from skyrl_capture.cli.main import artifact_extension, encode_artifact
    from skyrl_capture.compression import compress

    body = b'{"session_id": "tr_x-s0000"}\n'
    stored = compress(body)

    assert encode_artifact(stored, "none") == body, "readable by head and jq"
    assert json.loads(encode_artifact(stored, "none"))["session_id"] == "tr_x-s0000"
    assert gzip.decompress(encode_artifact(stored, "gzip")) == body
    assert encode_artifact(stored, "zst") == stored

    assert artifact_extension("none") == ".jsonl"
    assert artifact_extension("gzip") == ".jsonl.gz"
    assert artifact_extension("zst") == ".jsonl.zst"


def test_one_artifact_shape_at_every_scope():
    """A project export is a concatenation, so scope does not change the name.

    It used to be a tar, because the manifest was bundled into it. That made
    `--output turns.jsonl` fail on a project and succeed on a trajectory, for
    a reason no caller could see.
    """
    from skyrl_capture.cli.main import artifact_extension, check_output_name

    for compression in ("none", "gzip", "zst"):
        extension = artifact_extension(compression)
        assert extension.startswith(".jsonl")
        check_output_name(Path(f"turns{extension}"), compression)


def test_an_unknown_compression_is_refused():
    from skyrl_capture.cli.main import artifact_extension

    with pytest.raises(ValueError) as caught:
        artifact_extension("bz2")
    assert "expected one of" in str(caught.value)


def test_a_bare_output_name_is_accepted():
    """No extension means no claim about contents, so nothing to contradict."""
    from skyrl_capture.cli.main import check_output_name

    check_output_name(Path("turns"), "zst")


async def test_every_session_carries_every_field(stack):
    """Absence is null or an empty list, never a missing key.

    A consumer should be able to read `delay_ms` on any session without first
    testing whether it is there -- the same convention the turn level uses,
    where the first turn's delay is null rather than absent.
    """
    created = await forked_trajectory(stack, project="replay-shape-stable")
    rows = await stack.export_lines(trajectory=created["id"], format="replay")
    assert len(rows) > 1, "the fixture branches, so roots and children are both present"

    fields = {
        "schema_version", "session_id", "trajectory_id",
        "parent_session_id", "delay_ms", "arrival_ms", "forks", "turns",
    }
    for row in rows:
        assert set(row) == fields, row["session_id"]

    root = next(r for r in rows if r["parent_session_id"] is None)
    assert root["delay_ms"] is None
    child = next(r for r in rows if r["parent_session_id"] is not None)
    assert isinstance(child["delay_ms"], int)
    # A leaf session forks into nothing, and says so with an empty list.
    assert [] in [r["forks"] for r in rows]

    # Turn level uses the same convention.
    assert all(turn["delay_ms"] is None for r in rows for turn in r["turns"][:1])


# -- what comprises a sample ------------------------------------------------
async def test_a_sample_carries_the_conditions_it_was_produced_under(stack):
    """Tools define the data point, because they are rendered into the prompt."""
    tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
    created = await stack.create_trajectory(project="sample-conditions")
    history = [{"role": "user", "content": "go"}]
    response = await stack.chat(created, history, tools=tools, temperature=0.3, max_tokens=32)
    reply = response.json()["choices"][0]["message"]["content"]
    await stack.settle()
    await stack.chat(
        created,
        history + [{"role": "assistant", "content": reply}, {"role": "user", "content": "again"}],
        tools=tools,
        temperature=0.3,
        max_tokens=32,
    )
    await stack.settle()

    row = (await stack.export_lines(trajectory=created["id"], format="text_samples"))[0]
    assert row["model"] == "mock-model"
    assert row["tools"] == tools
    # Sampling parameters shape a replay, not a training row.
    assert "sampling" not in row and "tools_hash" not in row
    # And nothing needs a flag saying the conditions were consistent, because
    # tools are part of node identity: a change branches instead.
    assert "conditions_vary" not in row


async def test_changing_tools_starts_its_own_root(stack):
    """A chat template renders tool schemas into the prompt, so the same
    messages under a different tool set are a different context -- not a reuse
    of the old one. Without this the graph merges two generations sampled
    under different tools into a single node.
    """
    first = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
    second = [{"type": "function", "function": {"name": "write", "parameters": {}}}]
    created = await stack.create_trajectory(project="sample-tool-branch")
    history = [{"role": "user", "content": "go"}]
    # Identical messages *and* an identical reply: only the tools differ.
    await stack.chat(created, history, tools=first, headers={"x-mock-reply": "SAME"})
    await stack.settle()
    await stack.chat(created, history, tools=second, headers={"x-mock-reply": "SAME"})
    await stack.settle()

    graph = await stack.graph(created["id"])
    roots = [node for node in graph["nodes"] if node["parent_node_id"] is None]
    assert len(roots) == 2, "a tools change cannot reuse a prefix rendered under the old set"
    assert len(graph["nodes"]) == 4

    rows = await stack.export_lines(trajectory=created["id"], format="text_samples")
    assert len(rows) == 2
    # Each row has one tool set, so no row has to hedge about its conditions.
    assert sorted(r["tools"][0]["function"]["name"] for r in rows) == ["search", "write"]


async def test_a_message_is_not_flattened_beside_itself(stack):
    """`message` is the message as captured; role and content are inside it.

    Copying them out was lossy: for a tool call the flattened `content` is
    null, so a consumer reading it would have seen nothing at all.
    """
    created = await stack.create_trajectory(project="sample-message")
    await stack.chat(
        created,
        [{"role": "user", "content": "search"}],
        headers={"x-mock-tool-arguments": '{"q": "berlin"}'},
    )
    await stack.settle()

    row = (await stack.export_lines(trajectory=created["id"], format="text_samples"))[0]
    for message in row["messages"]:
        assert set(message) == {"node_id", "author", "trainable", "message"}

    sampled = next(m for m in row["messages"] if m["author"] == "model")
    assert sampled["message"]["content"] is None, "a tool call has no text content"
    assert sampled["message"]["tool_calls"][0]["function"]["name"] == "search"


def test_both_sample_formats_default_the_same_way():
    """They disagreed once, and the CLI turned the safe one off.

    `text_samples` defaulted to repeating targets and `token_samples` to not,
    but the CLI always sent the key explicitly -- so every command-line export
    of `token_samples` silently lost parity requirement 12, which says each
    sampled node is trainable at most once.
    """
    import inspect

    from skyrl_capture.export.formats import text_sample_records, token_sample_records

    defaults = {
        name: inspect.signature(fn).parameters["allow_repeated_targets"].default
        for name, fn in (("text", text_sample_records), ("tokens", token_sample_records))
    }
    assert defaults == {"text": False, "tokens": False}, defaults


async def test_token_samples_from_text_capture_is_refused(stack):
    """An empty artifact reads like "nothing matched", not "cannot be produced"."""
    created = await linear_trajectory(stack, turns=1, project="wrong-mode")
    await stack.finish(created["id"])

    response = await stack.post(
        "/v1/exports", {"format": "token_samples", "trajectory": created["id"]}
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "tokens" in detail and "text" in detail
    assert "text_samples" in detail, "says what to use instead"

    # Refused where the snapshot is chosen, so no job file is left to fail later.
    from skyrl_capture.persistence.layout import export_jobs_dir

    jobs = list(export_jobs_dir(stack.runtime.config.record_dir).glob("*.json"))
    assert jobs == []


async def test_zero_rows_is_not_itself_an_error(stack):
    """A trajectory that made no calls exports zero of anything, legitimately.

    The refusal above keys on capture mode rather than on the record count,
    because those are different things.
    """
    created = await stack.create_trajectory(project="no-calls")
    await stack.finish(created["id"])

    job = await stack.run_export(trajectory=created["id"], format="text_samples")
    assert job["status"] == "ready"
    assert job["record_count"] == 0


async def test_a_row_says_where_each_node_sits_in_it(tokens_stack):
    """`node_ids` and a flat `loss_mask` with nothing mapping one to the other
    meant turn boundaries lived only in the graph. Two claims rested on this
    field existing, and a verification check passed vacuously against it.

    With spans a row is self-describing: every node has its slice of
    `input_ids`, so which node a trainable token came from -- and therefore the
    train-once rule -- is answerable from the export alone.

    They must stay parallel, including for a node that contributed no tokens:
    an empty span rather than a missing one.
    """
    created = await tokens_stack.create_trajectory(project="spans")
    await tokens_stack.client.post(
        f"{created['base_url']}/chat/completions",
        json={"model": "mock-tokens-model", "messages": [{"role": "user", "content": "hi"}],
              "max_tokens": 4},
        headers={"authorization": "Bearer client-key"},
    )
    await tokens_stack.settle()
    await tokens_stack.finish(created["id"])

    from skyrl_capture.export import formats
    from skyrl_capture.export.view import view_of

    view = view_of(tokens_stack.record(created["id"]))
    rows = formats.token_sample_records(view)
    assert rows

    for row in rows:
        spans, ids = row["node_spans"], row["node_ids"]
        assert len(spans) == len(ids), "one span per node, in the same order"
        assert spans[0][0] == 0
        assert spans[-1][1] == len(row["input_ids"]), "they reach the end"
        for (start, end), (next_start, _) in zip(spans, spans[1:], strict=False):
            assert start <= end, "a span never runs backwards"
            assert end == next_start, "and they tile without a gap"

        # The thing the field is for: which node each trainable token came from.
        trained = {
            node_id
            for node_id, (start, end) in zip(ids, spans, strict=True)
            if any(row["loss_mask"][start:end])
        }
        assert trained, "something in this row is in the loss"

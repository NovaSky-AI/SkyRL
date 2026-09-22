"""The viewer's contract with the API.

The viewer is a separate program in a different language (`viewer/`, Node),
and it reads this service only over `/v1`. That is the arrangement R9 asks for
-- the UI has no private API, so anything on screen can be fetched with curl --
and it is also the arrangement that can break silently: nothing in Python
imports the viewer, so a renamed response field would not be caught until
someone opened the page.

So these tests pin the shapes the viewer reads, and check that every endpoint
it calls is one an external caller could call too. The viewer's own rendering
is tested on its own side, in `viewer/test/`, against fixtures captured from a
live `/v1`.
"""

from __future__ import annotations

import re
from pathlib import Path

VIEWER = Path("viewer")

# Fields each view reads. Kept explicit rather than scraped, because a field
# accessed through a variable would be missed by a regex and silently drop out
# of the contract.
TRAJECTORY_FIELDS = (
    "id", "project", "run_id", "task_id", "step", "upstream", "mode", "status",
    "labels", "annotations", "capture", "gap_distribution", "created_at",
)
# What a row in a run's attempt list carries. This replaces the grid's
# CELL_FIELDS: the viewer reads a flat listing filtered by step, so the
# contract is a trajectory row rather than a cell.
LISTING_FIELDS = (
    "id", "project", "run_id", "task_id", "step", "status", "labels",
    "annotations", "created_at", "capture",
)
# A run is derived from its trajectories, so it has counts and no metadata
# of its own: there is nothing to write to one.
RUN_FIELDS = (
    "id", "project", "trajectory_count", "task_count", "steps", "step_counts",
)
PATH_FIELDS = (
    "path_id", "node_ids", "leaf_node_id", "abandoned", "masked_reason",
    "stop_reason", "token_count", "trainable_count", "blocks", "messages",
)
BLOCK_FIELDS = ("kind", "role", "trainable", "start", "end", "token_count", "text")
CAPTURE_FIELDS = (
    "exchange_count", "node_count", "calls_missing", "calls_after_close",
    "complete", "recovery_uncertain", "delivery_uncertain", "errors",
)
EXCHANGE_FIELDS = (
    "id", "sequence", "endpoint_kind", "model", "http_status", "streaming",
    "transport_error", "retry_attempt", "request_start_at", "response_end_at",
    "ttft_ms", "duration_ms", "gap_ms", "overlapping", "chunk_count",
    "stream_summary", "usage", "output_node_id", "parent_output_node_id",
    "is_duplicate_retry", "late",
)
NODE_FIELDS = (
    "node_id", "parent_node_id", "exchange_id", "depth", "role", "author",
    "message_hash", "delta_hash", "char_count", "token_count", "sampled_start",
    "sampled_token_count", "has_logprobs", "has_routed_experts", "tokenizer",
    "derivation", "has_payload",
)
GRAPH_FIELDS = (
    "nodes", "leaf_node_ids", "leaf_assistant_node_ids", "branch_points",
)
HEALTH_FIELDS = (
    "status", "version", "schema_version", "mode", "record", "commits",
    "registry", "store", "capture",
)
# What a listing says about how much of the record has been read. `total` is
# null until the first scan finishes, so a pager is never handed a number that
# is about to change.
LISTING_ENVELOPE_FIELDS = ("data", "next_cursor", "has_more", "total", "indexing", "indexed_trajectories")


async def populated(stack):
    """A trajectory with a branch and an annotation to render."""
    created = await stack.create_trajectory(project="ui")
    shared = [{"role": "system", "content": "S"}, {"role": "user", "content": "shared"}]
    await stack.chat(created, shared, headers={"x-mock-reply": "reply-A"})
    await stack.settle()
    history = shared + [{"role": "assistant", "content": "reply-A"}]
    await stack.chat(created, history + [{"role": "user", "content": "left"}])
    await stack.settle()
    await stack.chat(created, history + [{"role": "user", "content": "right"}], stream=True)
    await stack.settle()
    await stack.patch(
        f"/v1/trajectories/{created['id']}/metadata",
        {"annotations": {"rlvr_reward": 0.5}, "labels": ["reviewed"]},
    )
    return created


def test_the_viewer_ships_with_the_package():
    """`skyrl-capture view` runs these files; a wheel without them has a
    command that cannot work."""
    for name in ("server.mjs", "package.json", "public/index.html", "public/app.mjs"):
        assert (VIEWER / name).is_file(), name


def test_the_viewer_calls_only_documented_v1_endpoints():
    """R9, from the other side: the viewer must not depend on a private API.

    Every endpoint it calls has to be one an external caller could use, so the
    viewer cannot drift onto behaviour the documented API does not have -- and
    so `skyrl-capture view` over a record, which serves the same routes from
    files, stays a complete substitute for the live service.
    """
    source = (VIEWER / "public/lib/api.mjs").read_text()
    called = set(re.findall(r"get\(`([^`]+)`", source)) | set(re.findall(r"get\('([^']+)'", source))
    # Template substitutions become path parameters.
    normalized = {re.sub(r"\$\{[^}]+\}", "{}", path).split("?")[0] for path in called}
    assert normalized, "no endpoints found; the extraction regex needs updating"
    for path in normalized:
        assert path.startswith("/"), path
    # And the only non-/v1 fetches are the two the proxy itself answers.
    raw = set(re.findall(r"fetch\('([^']+)'", source))
    assert raw <= {"/api/healthz", "/__viewer"}, raw


def test_the_viewer_has_no_dependencies():
    """Zero-dependency is a property worth failing a build over: it is what
    makes `viewer/` runnable from a checkout with no install step."""
    import json

    manifest = json.loads((VIEWER / "package.json").read_text())
    assert not manifest.get("dependencies")
    assert not manifest.get("devDependencies")
    assert not (VIEWER / "package-lock.json").exists()


async def test_health_shape_matches_what_the_header_renders(stack):
    body = (await stack.get("/healthz")).json()
    for field in HEALTH_FIELDS:
        assert field in body, field
    for field in ("pending_commits", "pending_high_water", "oldest_pending_age_s",
                  "commits", "refused", "failures", "unwritten_gaps"):
        assert field in body["commits"], field
    for field in ("hot_trajectories", "recovered", "evicted"):
        assert field in body["registry"], field
    for field in ("append_ms_mean", "fsync_ms_mean", "recovery_ms_mean", "torn_tails"):
        assert field in body["store"], field


async def test_trajectory_shape_matches_the_detail_view(stack):
    created = await populated(stack)
    body = (await stack.get(f"/v1/trajectories/{created['id']}")).json()
    for field in TRAJECTORY_FIELDS:
        assert field in body, field
    for field in CAPTURE_FIELDS:
        assert field in body["capture"], field


async def test_listing_shape_matches_the_sidebar(stack):
    await populated(stack)
    body = (await stack.get("/v1/trajectories?limit=50")).json()
    for field in ("data", "next_cursor", "has_more"):
        assert field in body, field
    item = body["data"][0]
    for field in ("id", "status", "project", "mode", "labels", "capture"):
        assert field in item, field


async def test_exchange_shape_matches_the_waterfall_and_table(stack):
    created = await populated(stack)
    rows = (await stack.get(f"/v1/trajectories/{created['id']}/exchanges?limit=500")).json()["data"]
    assert rows
    for row in rows:
        for field in EXCHANGE_FIELDS:
            assert field in row, field
    # The waterfall positions bars from these two, so they must be present and
    # parseable for every completed exchange.
    assert all(row["request_start_at"] for row in rows)
    assert all(row["response_end_at"] for row in rows)


async def test_graph_shape_matches_the_tree(stack):
    created = await populated(stack)
    graph = (await stack.get(f"/v1/trajectories/{created['id']}/graph")).json()
    for field in GRAPH_FIELDS:
        assert field in graph, field
    for node in graph["nodes"]:
        for field in NODE_FIELDS:
            assert field in node, field
    assert graph["branch_points"], "the fixture should produce a branch to mark"
    for branch in graph["branch_points"]:
        for field in ("node_id", "child_count", "child_ids"):
            assert field in branch, field
    # The tree builds parent links from these, so a root must be reachable.
    assert any(node["parent_node_id"] is None for node in graph["nodes"])


async def test_metadata_round_trip_matches_the_form(stack):
    created = await populated(stack)

    saved = (await stack.patch(
        f"/v1/trajectories/{created['id']}/metadata",
        {"annotations": {"critique": "note"}, "labels": ["checked"]},
    )).json()
    for field in ("labels", "annotations"):
        assert field in saved, field
    assert saved["annotations"]["critique"] == "note"
    assert "checked" in saved["labels"]

    # The detail view renders the same two fields, so the panel stays in sync.
    detail = (await stack.get(f"/v1/trajectories/{created['id']}")).json()
    assert detail["annotations"]["critique"] == "note"
    assert "checked" in detail["labels"]


async def test_export_flow_matches_the_export_panel(stack):
    created = await populated(stack)
    await stack.finish(created["id"])
    job = await stack.run_export(trajectory=created["id"], format="graph")
    # The panel polls until ready, then renders these.
    for field in ("id", "status", "record_count", "byte_count", "checksum", "download_url"):
        assert field in job, field
    download = await stack.get(f"/v1/exports/{job['id']}/download")
    assert download.status_code == 200


# -- a run's attempts, filtered -------------------------------------------
async def graded(stack, *, run_id="run-a"):
    """A run with two tasks over two steps, one of them resampled."""
    for task, step, reward in (
        ("t1", 0, 0.0), ("t1", 1, 1.0), ("t2", 0, 0.5), ("t2", 0, 1.0)
    ):
        created = await stack.create_trajectory(
            project="ui", run_id=run_id, task_id=task, step=step
        )
        await stack.chat(created, [{"role": "user", "content": f"{task}@{step}"}])
        await stack.finish(created["id"], annotations={"reward": reward})
    return run_id


async def test_run_shape_matches_the_sidebar(stack):
    await graded(stack)
    body = (await stack.get("/v1/runs")).json()
    assert body["data"]
    for field in RUN_FIELDS:
        assert field in body["data"][0], field


async def test_step_is_a_filter_on_a_flat_listing(stack):
    """What the viewer reads instead of a task x step grid.

    A run is a flat list of attempts; `step` narrows it. There is no matrix, so
    nothing has to decide what an empty cell means, and a resample is simply
    two rows rather than one averaged one.
    """
    run_id = await graded(stack)

    everything = (await stack.get(f"/v1/trajectories?run_id={run_id}")).json()["data"]
    assert len(everything) == 4
    for row in everything:
        for field in LISTING_FIELDS:
            assert field in row, field

    at_zero = (await stack.get(f"/v1/trajectories?run_id={run_id}&step=0")).json()["data"]
    assert {row["task_id"] for row in at_zero} == {"t1", "t2"}
    assert len(at_zero) == 3, "t2 was attempted twice at step 0"

    # t2 was never attempted at step 1, which shows up as its absence from the
    # list rather than as an empty cell that has to be told apart from a zero.
    at_one = (await stack.get(f"/v1/trajectories?run_id={run_id}&step=1")).json()["data"]
    assert [row["task_id"] for row in at_one] == ["t1"]

    # The run says which steps the filter can take, so the viewer can offer them.
    run = await stack.run(run_id)
    assert run["steps"] == [0, 1]

    # Opening a row has to resolve.
    assert (await stack.get(f"/v1/trajectories/{at_one[0]['id']}")).status_code == 200


# -- training: the loss mask over the decoded text -------------------------
async def test_path_shape_matches_the_training_panel(tokens_stack):
    created = await tokens_stack.create_trajectory()
    await tokens_stack.client.post(
        f"{created['base_url']}/chat/completions",
        json={"model": "mock-tokens-model", "messages": [{"role": "user", "content": "hello"}],
              "max_tokens": 4},
        headers={"authorization": "Bearer client-key"},
    )
    await tokens_stack.settle()

    body = (await tokens_stack.get(f"/v1/trajectories/{created['id']}/paths")).json()
    assert body["mode"] == "tokens"
    assert body["tokenizer"] == "builtin"
    assert body["paths"]
    path = body["paths"][0]
    for field in PATH_FIELDS:
        assert field in path, field
    assert path["blocks"], "a tokens path has to decode into blocks"
    for block in path["blocks"]:
        for field in BLOCK_FIELDS:
            assert field in block, field

    kinds = [block["kind"] for block in path["blocks"]]
    # The three that matter: context, the generation scaffold, and what the
    # model produced. `scaffold` is the boundary this panel exists to show.
    assert kinds == ["given", "scaffold", "sampled"]
    # Ranges index input_ids, so they have to tile the sequence exactly.
    assert path["blocks"][0]["start"] == 0
    assert path["blocks"][-1]["end"] == path["token_count"]
    for earlier, later in zip(path["blocks"], path["blocks"][1:], strict=False):
        assert earlier["end"] == later["start"]
    # Special tokens are shown, not stripped.
    assert "<|im_start|>" in path["blocks"][0]["text"]


async def test_replayed_assistant_text_is_its_own_kind(tokens_stack):
    """The case token capture exists for: assistant text the model did not
    produce. In the counts it is a node with no sampled tokens, which looks
    like a tool result; here it has to be visibly different from both."""
    created = await tokens_stack.create_trajectory()

    async def turn(messages):
        return await tokens_stack.client.post(
            f"{created['base_url']}/chat/completions",
            json={"model": "mock-tokens-model", "messages": messages, "max_tokens": 4},
            headers={"authorization": "Bearer client-key"},
        )

    history = [{"role": "user", "content": "call the tool"}]
    reply = (await turn(history)).json()["choices"][0]["message"]
    await tokens_stack.settle()
    # The client edits what the model said, then continues from its own version.
    await turn([
        *history,
        {"role": "assistant", "content": (reply.get("content") or "") + " (edited)"},
        {"role": "user", "content": "next"},
    ])
    await tokens_stack.settle()

    body = (await tokens_stack.get(f"/v1/trajectories/{created['id']}/paths")).json()
    kinds = {block["kind"] for path in body["paths"] for block in path["blocks"]}
    assert "replayed" in kinds, "edited assistant text must not read as context"
    assert "sampled" in kinds and "given" in kinds


async def test_a_text_trajectory_still_has_paths(stack):
    """Text mode has no mask, so the panel shows messages rather than blocks --
    the same shape, minus what does not exist."""
    created = await populated(stack)
    body = (await stack.get(f"/v1/trajectories/{created['id']}/paths")).json()
    assert body["mode"] == "text"
    assert len(body["paths"]) == 2, "the fixture branches, so there are two paths"
    for path in body["paths"]:
        assert path["blocks"] == []
        assert path["messages"], "and the messages are there instead"


async def test_paths_use_the_token_text_captured_with_the_ids(tokens_stack):
    """Viewing a TITO path needs no read-time tokenizer or decoder service."""
    created = await tokens_stack.create_trajectory()
    await tokens_stack.client.post(
        f"{created['base_url']}/chat/completions",
        json={"model": "mock-tokens-model", "messages": [{"role": "user", "content": "hi"}],
              "max_tokens": 4},
        headers={"authorization": "Bearer client-key"},
    )
    await tokens_stack.settle()

    body = (await tokens_stack.get(f"/v1/trajectories/{created['id']}/paths")).json()
    assert body["tokenizer"] == "builtin"
    assert "decoder" not in body
    blocks = body["paths"][0]["blocks"]
    assert blocks
    assert all(len(block["token_offsets"]) == block["token_count"] + 1 for block in blocks)


async def test_a_path_says_how_its_blocks_were_cut(tokens_stack):
    """Without a turn marker the classification degrades rather than just the
    readability -- `replayed` and `scaffold` become unreachable -- so a reader
    has to be able to tell which blocking it got."""
    created = await tokens_stack.create_trajectory()
    await tokens_stack.client.post(
        f"{created['base_url']}/chat/completions",
        json={"model": "mock-tokens-model", "messages": [{"role": "user", "content": "hi"}],
              "max_tokens": 4},
        headers={"authorization": "Bearer client-key"},
    )
    await tokens_stack.settle()

    body = (await tokens_stack.get(f"/v1/trajectories/{created['id']}/paths")).json()
    assert body["paths"][0]["blocking"] == "mask+turns"

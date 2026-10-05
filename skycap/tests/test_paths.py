"""Path rules: ``all``, ``final`` and custom ones pick the rows, skycap builds and checks the samples."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from skycap import CapturePool, CaptureService, record
from skycap.cli import build_parser, build_server
from skycap.graph import MessageGraph
from skycap.paths import (
    BUILTIN_RULES,
    Row,
    check_rows,
    final_path,
    import_rule,
    load_rule,
    rule_registry,
)
from skycap.samples import build_samples
from tests.test_tokens import client, converse, token_stack, user


async def _discard_then_retry(llm) -> list[dict]:
    """A harness that throws a reply away and asks again, as mini-swe-agent does on a format error:
    the discarded reply is a dead end, the conversation continues from the message before it."""
    history = await converse(llm, "hi", "more")
    discarded = await llm.chat.completions.create(model="policy", messages=[*history, user("act")])
    assert discarded.choices[0].message.content
    retry = [*history, user("act"), user("format error, try again")]
    reply = await llm.chat.completions.create(model="policy", messages=retry)
    return [*retry, reply.choices[0].message.model_dump(exclude_none=True)]


def last_reply(graph: MessageGraph) -> list[Row]:
    """A custom rule: only the final reply trains, with the conversation before it as context."""
    rows = final_path(graph)
    return [Row(row.path, row.targets[-1:]) for row in rows]


def dead_ends(graph: MessageGraph) -> list[tuple[list[int], list[int]]]:
    """A custom rule, as plain tuples: every path but the final one, training only its leaf."""
    return [(path, [path[-1]]) for path in graph.paths()[:-1]]


def not_a_rule() -> None:
    """Imported by name in a test, and not callable as a rule."""


NOT_CALLABLE = 1


# -- final ---------------------------------------------------------------------
async def test_final_trains_every_sampled_message_on_the_last_path_and_nothing_off_it() -> None:
    async with token_stack() as stack:
        created = await stack.create()
        await _discard_then_retry(client(created["base_url"]))
        graph = stack.server.trajectories[created["id"]].graph
        every = build_samples(graph)
        (final,) = build_samples(graph, final_path)

    model = [n.id for n in graph if n.author == "model"]
    assert len(model) == 4 and len(every) == 2
    dead_end, continued = every
    # "all": the first path (to the discarded reply) trains the shared turns and the discarded reply;
    # the continued path trains only its last reply.
    assert dead_end.targets == model[:3] and continued.targets == [model[3]]
    # "final": the continued path, training both shared turns and the last reply, never the discarded one.
    assert final.path == continued.path and final.leaf == graph.nodes[-1].id
    assert final.targets == [model[0], model[1], model[3]]
    assert final.input_ids == continued.input_ids
    discarded = graph.nodes[model[2]]
    assert model[2] not in final.path
    assert discarded.tokens is not None and discarded.tokens.token_ids
    # The loss mask is the continued path's own, plus the shared turns the dead-end row trained.
    shared = next(i for i, (a, b) in enumerate(zip(dead_end.input_ids, continued.input_ids)) if a != b)
    expected = [
        continued.loss_mask[i] or (i < shared and dead_end.loss_mask[i]) for i in range(len(continued.input_ids))
    ]
    assert final.loss_mask == [int(x) for x in expected]
    assert sum(final.loss_mask) == sum(dead_end.loss_mask[:shared]) + sum(continued.loss_mask)
    # Trained positions are exactly the sampled tokens of the targets, with their rollout logprobs.
    offset, sampled = 0, []
    for node_id in final.path:
        tokens = graph.nodes[node_id].tokens
        if node_id in final.targets:
            sampled += range(offset + tokens.sampled_start, offset + len(tokens.token_ids))
        offset += len(tokens.token_ids)
    assert [i for i, m in enumerate(final.loss_mask) if m] == sampled
    assert final.logprobs == continued.logprobs and len(final.logprobs) == len(final.input_ids)


async def test_final_on_a_linear_conversation_is_the_one_all_sample() -> None:
    async with token_stack() as stack:
        created = await stack.create()
        await converse(client(created["base_url"]), "hi", "more", "again")
        graph = stack.server.trajectories[created["id"]].graph
        (every,) = build_samples(graph)
        (final,) = build_samples(graph, final_path)

    assert (final.leaf, final.path, final.targets) == (every.leaf, every.path, every.targets)
    assert (final.input_ids, final.loss_mask, final.logprobs) == (every.input_ids, every.loss_mask, every.logprobs)


def test_final_of_an_empty_graph_is_no_sample() -> None:
    assert build_samples(MessageGraph(), final_path) == []


async def test_finish_takes_paths_and_refuses_an_unknown_one_without_ending_the_trajectory() -> None:
    async with token_stack() as stack:
        created = await stack.create()
        await _discard_then_retry(client(created["base_url"]))
        url = f"{stack.url}/trajectories/{created['id']}/finish"
        # Neither a name the server doesn't have nor an import path is loaded from a request.
        for paths in ("longest", "nowhere_skycap_test.rules:everything", ["final"]):
            async with stack.http.post(url, json={"annotations": {}, "paths": paths}) as response:
                assert response.status == 400
        assert "nowhere_skycap_test" not in sys.modules
        assert not stack.server.trajectories[created["id"]].ended
        async with stack.http.post(url, json={"annotations": {"reward": 1.0}, "paths": "final"}) as response:
            assert response.status == 200
            body = await response.json()

    (sample,) = body["samples"]
    assert len(sample["targets"]) == 3


async def test_the_pool_sends_paths_and_a_resent_finish_keeps_them() -> None:
    async with token_stack() as stack, CapturePool([stack.url]) as pool:
        async with pool.trajectory() as trajectory:
            await _discard_then_retry(client(trajectory.base_url))
            result = await trajectory.finish({"reward": 1.0}, paths="final")
        assert len(result.samples) == 1 and len(result.samples[0].targets) == 3

        with pytest.raises(RuntimeError):
            async with pool.trajectory() as failing:
                await _discard_then_retry(client(failing.base_url))
                failing.id, real = "tr_missing", failing.id
                try:
                    await failing.finish({"reward": 0.0}, paths="final")
                except Exception:
                    pass
                finally:
                    failing.id = real
                raise RuntimeError("harness crashed after its finish failed")
        # The context manager sent the failed finish again, with its paths.
        assert failing.result is not None and len(failing.result.samples) == 1


# -- custom rules ----------------------------------------------------------------
@pytest.mark.parametrize("rule", [last_reply, "tests.test_paths:last_reply"])
async def test_a_custom_rule_picks_the_rows_and_skycap_builds_their_tokens(rule) -> None:
    async with token_stack(path_rules={"last_reply": rule}) as stack, CapturePool([stack.url]) as pool:
        async with pool.trajectory() as trajectory:
            await _discard_then_retry(client(trajectory.base_url))
            result = await trajectory.finish({"reward": 1.0}, paths="last_reply")
        graph = stack.server.trajectories[trajectory.id].graph
        (final,) = build_samples(graph, final_path)

    (sample,) = result.samples
    assert sample.path == final.path and sample.targets == final.targets[-1:]
    assert sample.input_ids == final.input_ids and sample.logprobs == final.logprobs
    # Only the last reply's sampled tokens train.
    tokens = graph.nodes[sample.targets[0]].tokens
    trained = len(tokens.token_ids) - tokens.sampled_start
    assert sum(sample.loss_mask) == trained and sample.loss_mask[-trained:] == [1] * trained


async def test_a_rule_may_return_tuples_and_end_its_paths_anywhere() -> None:
    async with token_stack() as stack:
        created = await stack.create()
        await _discard_then_retry(client(created["base_url"]))
        graph = stack.server.trajectories[created["id"]].graph

    model = [n.id for n in graph if n.author == "model"]
    (dead_end,) = build_samples(graph, dead_ends)
    assert dead_end.leaf == model[2] and dead_end.targets == [model[2]]
    # A path may stop above a leaf, at a numpy id; its tokens are the prefix up to there.
    (prefix,) = build_samples(graph, lambda g: [(np.array(g.path_to(model[0])), [np.int64(model[0])])])
    assert prefix.leaf == model[0] and prefix.targets == [model[0]] and type(prefix.targets[0]) is int
    assert prefix.input_ids == dead_end.input_ids[: len(prefix.input_ids)]


@pytest.mark.parametrize(
    ("rows", "problem"),
    [
        ([([], [])], "empty"),
        ([([0, 1, 9], [])], "doesn't have"),
        ([([1, 2, 3], [3])], "not a root"),
        ([([0, 1, 4, 3], [3])], "not a child of node 4"),
        ([([0, 1, 2, 3], [5])], "target 5 is not on the path"),
        ([([0, 1, 2, 3], [2])], "not model-authored"),
        ([([0, 1, 2, 3], [3, 3])], "already trained"),
        ([([0, 1, 2, 3], [1, 3]), ([0, 1], [1])], "row 1: target 1 is already trained"),
    ],
)
def test_rows_that_are_not_paths_with_their_own_model_targets_are_refused(rows, problem) -> None:
    graph = MessageGraph()
    for parent, author in [(None, "client"), (0, "model"), (1, "client"), (2, "model"), (1, "client")]:
        graph.add(
            parent,
            role="user",
            author=author,
            message={"n": len(graph)},
            match_hash=str(len(graph)),
            delta_hash=str(len(graph)),
            created_at=0.0,
        )
    graph.add(4, role="assistant", author="model", message={}, match_hash="5", delta_hash="5", created_at=0.0)
    with pytest.raises(ValueError, match=problem):
        check_rows(graph, rows)
    # Rows sharing a prefix are fine as long as each target trains once.
    assert check_rows(graph, [([0, 1, 2, 3], [1, 3]), ([0, 1, 4, 5], [5])]) == [
        Row([0, 1, 2, 3], [1, 3]),
        Row([0, 1, 4, 5], [5]),
    ]


def test_rules_are_named_or_imported_and_the_built_in_names_are_reserved() -> None:
    assert load_rule("final") is BUILTIN_RULES["final"]
    assert load_rule("tests.test_paths:last_reply") is last_reply
    assert set(rule_registry({"mine": "tests.test_paths:last_reply"})) == {"all", "final", "mine"}
    service = CaptureService("http://engine/v1", path_rules={"mine": "tests.test_paths:last_reply"})
    assert service.server.path_rules["mine"] is last_reply
    with pytest.raises(ValueError, match="built-in"):
        rule_registry({"final": last_reply})
    with pytest.raises(ValueError, match="pkg.module:function"):
        load_rule("longest")
    with pytest.raises(ModuleNotFoundError):
        import_rule("nowhere_skycap_test.rules:everything")
    with pytest.raises(AttributeError):
        import_rule("tests.test_paths:missing")
    with pytest.raises(TypeError, match="not callable"):
        import_rule("tests.test_paths:NOT_CALLABLE")


def test_the_cli_registers_rules_by_name_or_by_import_path() -> None:
    args = build_parser().parse_args(
        [
            "serve",
            "--upstream-url",
            "http://engine:8000/v1",
            "--path-rule",
            "last=tests.test_paths:last_reply",
            "--path-rule",
            "tests.test_paths:dead_ends",
        ]
    )
    server = build_server(args)
    assert server.path_rules["last"] is last_reply
    assert server.path_rules["tests.test_paths:dead_ends"] is dead_ends
    assert set(server.path_rules) == {"all", "final", "last", "tests.test_paths:dead_ends"}


async def test_a_failing_rule_is_a_500_and_the_trajectory_is_still_written(tmp_path: Path) -> None:
    def broken(graph: MessageGraph) -> list[Row]:
        return [Row([0, 1], [0])]  # node 0 is the user's message

    async with token_stack(record_dir=tmp_path, path_rules={"broken": broken}) as stack:
        created = await stack.create()
        await converse(client(created["base_url"]), "hi")
        url = f"{stack.url}/trajectories/{created['id']}/finish"
        async with stack.http.post(url, json={"annotations": {"reward": 1.0}, "paths": "broken"}) as response:
            assert response.status == 500
            assert "not model-authored" in (await response.json())["error"]
        assert created["id"] not in stack.server.trajectories

    document = record.read_document(tmp_path, created["id"])
    assert document["status"] == "finished" and document["annotations"] == {"reward": 1.0}
    assert document["samples"] is None


# -- the record says what trained ------------------------------------------------
async def test_the_record_holds_the_samples_finish_returned(tmp_path: Path) -> None:
    async with token_stack(record_dir=tmp_path, path_rules={"last_reply": last_reply}) as stack:
        rows = {}
        for paths in ("all", "final", "last_reply"):
            created = await stack.create()
            await _discard_then_retry(client(created["base_url"]))
            url = f"{stack.url}/trajectories/{created['id']}/finish"
            async with stack.http.post(url, json={"annotations": {"reward": 1.0}, "paths": paths}) as response:
                body = await response.json()
            document = record.read_document(tmp_path, created["id"])
            rows[paths] = document["samples"]
            assert document["samples"] == {
                "paths": paths,
                "rows": [{"leaf": s["leaf"], "targets": s["targets"]} for s in body["samples"]],
            }
            # Loading the record brings them back, and they are the graph's samples under that rule.
            loaded = record.load(tmp_path, created["id"])
            assert loaded.samples == document["samples"]
            assert [s.leaf for s in build_samples(loaded.graph, stack.server.path_rules[paths])] == [
                r["leaf"] for r in document["samples"]["rows"]
            ]

    assert len(rows["all"]["rows"]) == 2 and len(rows["final"]["rows"]) == 1
    # The final row is the last path of "all", training the dead end's shared turns too.
    assert rows["final"]["rows"][0]["leaf"] == rows["all"]["rows"][1]["leaf"]
    assert len(rows["final"]["rows"][0]["targets"]) == 3
    (final,) = rows["final"]["rows"]
    assert rows["last_reply"]["rows"] == [{"leaf": final["leaf"], "targets": final["targets"][-1:]}]


async def test_a_repeat_finish_cannot_ask_for_other_paths_but_can_repeat_them(tmp_path: Path) -> None:
    async with token_stack(record_dir=tmp_path) as stack:
        created = await stack.create()
        await _discard_then_retry(client(created["base_url"]))
        url = f"{stack.url}/trajectories/{created['id']}/finish"
        async with stack.http.post(url, json={"annotations": {"reward": 1.0}, "paths": "final"}) as response:
            first = await response.json()
        async with stack.http.post(url, json={"annotations": {"reward": 1.0}, "paths": "all"}) as response:
            assert response.status == 409
        # The repeat is answered from disk, as the first finish was.
        async with stack.http.post(url, json={"annotations": {"reward": 1.0}, "paths": "final"}) as response:
            assert response.status == 200
            again = await response.json()

    assert again["samples"] == first["samples"]
    assert record.read_document(tmp_path, created["id"])["samples"]["paths"] == "final"


async def test_a_repeat_finish_is_answered_with_the_recorded_rows_not_the_rule_run_again() -> None:
    calls = []

    def changing(graph: MessageGraph) -> list[Row]:
        calls.append(1)
        return final_path(graph) if len(calls) == 1 else []

    async with token_stack(path_rules={"changing": changing}) as stack:
        created = await stack.create()
        await _discard_then_retry(client(created["base_url"]))
        url = f"{stack.url}/trajectories/{created['id']}/finish"
        answers = []
        for _ in range(2):
            async with stack.http.post(url, json={"annotations": {}, "paths": "changing"}) as response:
                answers.append(await response.json())

    assert len(calls) == 1 and len(answers[0]["samples"]) == 1
    assert answers[1]["samples"] == answers[0]["samples"]


async def test_a_trajectory_not_ended_by_finish_records_no_samples(tmp_path: Path) -> None:
    async with token_stack(record_dir=tmp_path) as stack:
        stack.server.ttl = 0.0
        created = await stack.create()
        await converse(client(created["base_url"]), "hi")
        assert await stack.server.sweep() == [created["id"]]

    document = record.read_document(tmp_path, created["id"])
    assert document["status"] == "abandoned" and document["samples"] is None

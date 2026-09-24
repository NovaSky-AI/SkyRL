"""The matching rule, one scenario per test.

Each scenario is a sequence of calls a harness could make; the assertions are
on the shape of the graph that results.
"""

from __future__ import annotations

import pytest

from skycap.graph import CallInfo, MessageGraph

SYS = {"role": "system", "content": "You are terse."}
SEARCH = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
EDIT = [{"type": "function", "function": {"name": "edit", "parameters": {}}}]


def user(text: str) -> dict:
    return {"role": "user", "content": text}


def assistant(text: str, **extra) -> dict:
    return {"role": "assistant", "content": text, **extra}


class Harness:
    """Drives a graph the way a harness drives a server: one call at a time."""

    def __init__(self) -> None:
        self.graph = MessageGraph()
        self.clock = 0.0

    def call(self, messages, reply, *, tools=None, model="policy", **sampling):
        self.clock += 1.0
        info = CallInfo(t_start=self.clock, t_end=self.clock + 0.5, model=model, sampling=sampling)
        return self.graph.commit_text(messages, reply, tools=tools, model=model, call=info)

    def shape(self) -> list[tuple[int | None, str, str]]:
        """``(parent, author, content)`` per node, in creation order."""
        return [(n.parent, n.author, n.message.get("content")) for n in self.graph]


def test_next_turn_extends_the_path() -> None:
    h = Harness()
    h.call([user("q1")], assistant("a1"))
    turn = h.call([user("q1"), assistant("a1"), user("q2")], assistant("a2"))

    assert turn.matched == 2
    assert h.shape() == [
        (None, "client", "q1"),
        (0, "model", "a1"),
        (1, "client", "q2"),
        (2, "model", "a2"),
    ]
    assert h.graph.paths() == [[0, 1, 2, 3]]


def test_identical_reply_is_one_node_with_two_calls() -> None:
    h = Harness()
    first = h.call([user("q")], assistant("a"))
    second = h.call([user("q")], assistant("a"))

    assert len(h.graph) == 2
    assert second.output == first.output
    assert not second.output_created
    assert second.created == ()
    assert len(h.graph.nodes[first.output].calls) == 2


def test_different_reply_forks() -> None:
    h = Harness()
    h.call([user("q")], assistant("a"))
    h.call([user("q")], assistant("b"))

    assert h.shape() == [(None, "client", "q"), (0, "model", "a"), (0, "model", "b")]
    assert h.graph.branch_points() == [0]
    assert h.graph.paths() == [[0, 1], [0, 2]]


def test_same_reply_under_different_sampling_is_two_samples() -> None:
    h = Harness()
    h.call([user("q")], assistant("a"), top_p=1.0)
    h.call([user("q")], assistant("a"), top_p=0.9)

    assert h.shape() == [(None, "client", "q"), (0, "model", "a"), (0, "model", "a")]


def test_replaying_a_sampled_reply_without_its_sampling_params_does_not_branch() -> None:
    h = Harness()
    h.call([user("q")], assistant("a"), top_p=1.0)
    turn = h.call([user("q"), assistant("a"), user("q2")], assistant("b"))

    assert turn.matched == 2
    assert h.shape() == [
        (None, "client", "q"),
        (0, "model", "a"),
        (1, "client", "q2"),
        (2, "model", "b"),
    ]
    assert h.graph.branch_points() == []


def test_non_support_sampling_params_do_not_split_nodes() -> None:
    h = Harness()
    h.call([user("q")], assistant("a"), max_tokens=10)
    h.call([user("q")], assistant("a"), max_tokens=20)

    assert len(h.graph) == 2


def test_history_continues_from_the_most_recent_matching_sample() -> None:
    h = Harness()
    h.call([user("q")], assistant("a"), top_p=1.0)
    h.call([user("q")], assistant("a"), top_p=0.9)
    turn = h.call([user("q"), assistant("a"), user("next")], assistant("done"))

    assert turn.matched == 2
    assert h.graph.nodes[turn.input_leaf].parent == 2


def test_the_sibling_not_picked_is_shadowed_but_still_trains() -> None:
    h = Harness()
    h.call([user("q")], assistant("a"), top_p=1.0)
    h.call([user("q")], assistant("a"), top_p=0.9)

    assert h.graph.shadowed_by(1) == 2
    assert h.graph.shadowed_by(2) is None
    assert h.graph.paths() == [[0, 1], [0, 2]]


def test_a_model_sibling_is_preferred_over_a_later_client_copy() -> None:
    graph = MessageGraph()
    root, _ = graph.add(
        None, role="user", author="client", message=user("q"), match_hash="q", delta_hash="q", created_at=0.0
    )
    sample, _ = graph.add(
        root.id,
        role="assistant",
        author="model",
        message=assistant("a"),
        match_hash="a",
        delta_hash="a/model",
        created_at=1.0,
    )
    copy, _ = graph.add(
        root.id,
        role="assistant",
        author="client",
        message=assistant("a"),
        match_hash="a",
        delta_hash="a/client",
        created_at=2.0,
    )

    assert graph.match(["q", "a"]) == [root.id, sample.id]
    assert graph.shadowed_by(copy.id) == sample.id
    assert graph.shadowed_by(sample.id) is None


def test_with_no_model_sibling_the_latest_client_one_is_picked() -> None:
    graph = MessageGraph()
    first, _ = graph.add(
        None, role="user", author="client", message=user("q"), match_hash="q", delta_hash="q1", created_at=0.0
    )
    second, _ = graph.add(
        None, role="user", author="client", message=user("q"), match_hash="q", delta_hash="q2", created_at=1.0
    )

    assert graph.match(["q"]) == [second.id]
    assert graph.shadowed_by(first.id) == second.id


def test_repaired_reply_forks_as_client_authored() -> None:
    h = Harness()
    h.call([user("how many retries?")], assistant("four"))
    h.call([user("how many retries?"), assistant("three"), user("ok")], assistant("noted"))

    assert h.shape() == [
        (None, "client", "how many retries?"),
        (0, "model", "four"),
        (0, "client", "three"),
        (2, "client", "ok"),
        (3, "model", "noted"),
    ]


def test_harness_written_message_never_absorbs_an_equal_sample() -> None:
    h = Harness()
    h.call([user("q"), assistant("x"), user("again")], assistant("y"))
    h.call([user("q")], assistant("x"))

    x_nodes = [n for n in h.graph if n.message.get("content") == "x"]
    assert [(n.parent, n.author) for n in x_nodes] == [(0, "client"), (0, "model")]
    assert x_nodes[0].calls == []
    assert len(x_nodes[1].calls) == 1


def test_compaction_forks_at_the_last_unchanged_message() -> None:
    h = Harness()
    h.call([SYS, user("task")], assistant("step1"))
    h.call([SYS, user("task"), assistant("step1"), user("summarize")], assistant("summary"))
    h.call([SYS, user("summary"), user("continue")], assistant("step2"))

    assert h.graph.roots() == [0]
    assert h.graph.children(0) == [1, 5]
    assert [h.graph.nodes[i].message["content"] for i in h.graph.paths()[1]] == [
        "You are terse.",
        "summary",
        "continue",
        "step2",
    ]
    summaries = [(n.author, n.role) for n in h.graph if n.message.get("content") == "summary"]
    assert summaries == [("model", "assistant"), ("client", "user")]


def test_subagent_with_its_own_system_prompt_is_a_new_root() -> None:
    h = Harness()
    h.call([SYS, user("task")], assistant("delegating"))
    h.call([{"role": "system", "content": "You are a searcher."}, user("find x")], assistant("found"))

    assert h.graph.roots() == [0, 3]
    assert h.graph.branch_points() == []


def test_subagent_sharing_the_prefix_forks_under_it() -> None:
    h = Harness()
    h.call([SYS, user("task")], assistant("plan"))
    h.call([SYS, user("task"), assistant("plan"), user("subtask A")], assistant("A done"))
    h.call([SYS, user("task"), assistant("plan"), user("subtask B")], assistant("B done"))

    assert h.graph.branch_points() == [2]
    assert len(h.graph.paths()) == 2


def test_stripped_reasoning_forks_as_client_authored() -> None:
    h = Harness()
    sampled = assistant("answer", reasoning_content="let me think")
    h.call([user("q")], sampled)
    h.call([user("q"), assistant("answer"), user("more")], assistant("more answer"))

    assert h.shape() == [
        (None, "client", "q"),
        (0, "model", "answer"),
        (0, "client", "answer"),
        (2, "client", "more"),
        (3, "model", "more answer"),
    ]


def test_absent_null_and_empty_are_the_same_message() -> None:
    h = Harness()
    h.call([user("q")], assistant("a", refusal=None, annotations=[], tool_calls=None))
    turn = h.call([user("q"), assistant("a"), user("next")], assistant("b"))

    assert turn.matched == 2


def test_empty_content_is_not_absent_content() -> None:
    h = Harness()
    h.call([user("q")], assistant(""))
    turn = h.call([user("q"), {"role": "assistant"}, user("next")], assistant("b"))

    assert turn.matched == 1


def test_changed_tool_set_starts_a_new_root() -> None:
    h = Harness()
    h.call([user("q")], assistant("a"), tools=SEARCH)
    h.call([user("q")], assistant("a"), tools=SEARCH + EDIT)

    assert h.graph.roots() == [0, 2]
    assert len(h.graph.tools) == 2


def test_tool_change_midway_duplicates_the_prefix_as_client() -> None:
    h = Harness()
    h.call([user("q")], assistant("a"), tools=SEARCH)
    h.call([user("q"), assistant("a"), user("edit it")], assistant("edited"), tools=SEARCH + EDIT)

    assert h.shape() == [
        (None, "client", "q"),
        (0, "model", "a"),
        (None, "client", "q"),
        (2, "client", "a"),
        (3, "client", "edit it"),
        (4, "model", "edited"),
    ]


def test_changed_model_starts_a_new_root() -> None:
    h = Harness()
    h.call([user("q")], assistant("a"), model="large")
    h.call([user("q")], assistant("a"), model="small")

    assert h.graph.roots() == [0, 2]


def test_tools_are_stored_once() -> None:
    h = Harness()
    h.call([user("q1")], assistant("a1"), tools=SEARCH)
    h.call([user("q1"), assistant("a1"), user("q2")], assistant("a2"), tools=SEARCH)

    assert list(h.graph.tools.values()) == [SEARCH]


def test_call_info_is_recorded_on_the_reply() -> None:
    h = Harness()
    turn = h.call([user("q")], assistant("a"), temperature=0.7)

    (info,) = h.graph.nodes[turn.output].calls
    assert info.model == "policy"
    assert info.sampling == {"temperature": 0.7}
    assert h.graph.nodes[turn.output].created_at == info.t_end


def test_add_rejects_an_unknown_parent() -> None:
    graph = MessageGraph()
    with pytest.raises(KeyError):
        graph.add(7, role="user", author="client", message=user("q"), match_hash="m", delta_hash="m", created_at=0.0)

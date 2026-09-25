"""Phase 2: the text context graph."""

from __future__ import annotations


def nodes_by_id(graph):
    return {node["node_id"]: node for node in graph["nodes"]}


def path_of(graph, node_id):
    index = nodes_by_id(graph)
    path = []
    current = node_id
    while current:
        node = index[current]
        path.append(node)
        current = node["parent_node_id"]
    return list(reversed(path))


async def test_first_call_commits_one_node_per_message(stack):
    created = await stack.create_trajectory()
    await stack.chat(
        created,
        [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "what is 2+2?"},
        ],
    )
    graph = await stack.graph(created["id"])
    assert len(graph["nodes"]) == 3

    roles = [(node["role"], node["author"], node["depth"]) for node in graph["nodes"]]
    assert roles == [
        ("system", "client", 0),
        ("user", "client", 1),
        ("assistant", "model", 2),
    ]
    # A single chain: every node has exactly one parent, one leaf.
    assert graph["nodes"][0]["parent_node_id"] is None
    assert len(graph["leaf_node_ids"]) == 1
    assert graph["leaf_assistant_node_ids"] == [graph["nodes"][-1]["node_id"]]


async def test_second_call_commits_only_the_new_tail(stack):
    """Shared prefixes are stored once."""
    created = await stack.create_trajectory()

    first = [{"role": "system", "content": "S"}, {"role": "user", "content": "one"}]
    response = await stack.chat(created, first)
    reply = response.json()["choices"][0]["message"]["content"]
    await stack.settle()
    after_first = await stack.graph(created["id"])
    assert len(after_first["nodes"]) == 3

    # A normal agent loop: same history, plus the assistant reply, plus a new turn.
    second = first + [
        {"role": "assistant", "content": reply},
        {"role": "user", "content": "two"},
    ]
    await stack.chat(created, second)
    graph = await stack.graph(created["id"])

    # Only the new user message and the new assistant reply were added.
    assert len(graph["nodes"]) == 5
    assert len(graph["leaf_node_ids"]) == 1
    assert graph["branch_points"] == []

    exchanges = await stack.exchanges(created["id"])
    second_exchange = exchanges[1]
    assert second_exchange["input_prefix_node_id"] == after_first["nodes"][-1]["node_id"]
    assert len(second_exchange["input_node_ids"]) == 1
    # The prefix matched exactly through the previous model output, which is the
    # only relationship exportable as an AIPerf fork.
    assert second_exchange["parent_output_node_id"] == after_first["nodes"][-1]["node_id"]

    # The whole conversation is one root-to-leaf path.
    leaf_path = path_of(graph, graph["leaf_node_ids"][0])
    assert [node["role"] for node in leaf_path] == [
        "system", "user", "assistant", "user", "assistant"
    ]


async def test_divergent_histories_render_a_visible_branch(stack):
    """Acceptance criterion 6: a visible prefix branch with its prefix evidence."""
    created = await stack.create_trajectory()

    shared = [{"role": "system", "content": "S"}, {"role": "user", "content": "shared"}]
    first = await stack.chat(created, shared, headers={"x-mock-reply": "reply-A"})
    await stack.settle()

    # Two continuations from the same assistant turn, diverging at the next
    # user message.
    history = shared + [{"role": "assistant", "content": "reply-A"}]
    await stack.chat(created, history + [{"role": "user", "content": "left"}])
    await stack.chat(created, history + [{"role": "user", "content": "right"}])
    graph = await stack.graph(created["id"])

    assert len(graph["branch_points"]) == 1
    branch = graph["branch_points"][0]
    assert branch["child_count"] == 2

    index = nodes_by_id(graph)
    fork_node = index[branch["node_id"]]
    assert fork_node["author"] == "model", "the fork is after the shared model reply"
    children = [index[child] for child in branch["child_ids"]]
    assert sorted(child["role"] for child in children) == ["user", "user"]
    assert len(graph["leaf_node_ids"]) == 2

    # The shared prefix is the path to the fork: system, user, assistant were
    # matched before diverging. The tree carries this; nothing else records it.
    assert len(path_of(graph, branch["node_id"])) == 3
    assert fork_node["depth"] == 2
    assert first.status_code == 200


async def test_identical_retry_is_not_a_branch(stack):
    """Rule 5: a retry is separate at the exchange layer, not in the graph."""
    created = await stack.create_trajectory()
    messages = [{"role": "user", "content": "flaky"}]

    await stack.chat(created, messages, headers={"x-mock-reply": "same"})
    await stack.settle()
    await stack.chat(
        created, messages, headers={"x-mock-reply": "same", "x-stainless-retry-count": "1"}
    )
    graph = await stack.graph(created["id"])

    # Two exchanges, but no duplicate nodes and no fork.
    exchanges = await stack.exchanges(created["id"])
    assert len(exchanges) == 2
    assert len(graph["nodes"]) == 2
    assert graph["branch_points"] == []
    assert len(graph["leaf_node_ids"]) == 1

    retry = exchanges[1]
    assert retry["is_duplicate_retry"] is True
    assert retry["retry_attempt"] == 1
    assert retry["input_node_ids"] == []
    # Both exchanges resolve to the same input leaf, which is what makes retry
    # detection exact.
    assert retry["input_leaf_node_id"] == exchanges[0]["input_leaf_node_id"]


async def test_resampled_reply_is_a_genuine_fork(stack):
    """A retry that produced a different reply really is a branch."""
    created = await stack.create_trajectory()
    messages = [{"role": "user", "content": "sample me"}]
    await stack.chat(created, messages, headers={"x-mock-reply": "first sample"})
    await stack.settle()
    await stack.chat(created, messages, headers={"x-mock-reply": "second sample"})
    graph = await stack.graph(created["id"])

    assert len(graph["nodes"]) == 3, "one user node, two assistant nodes"
    assert len(graph["branch_points"]) == 1
    assert len(graph["leaf_assistant_node_ids"]) == 2
    index = nodes_by_id(graph)
    assert index[graph["branch_points"][0]["node_id"]]["role"] == "user"


async def test_compacted_history_branches_at_the_last_unchanged_message(stack):
    """Rule 4: a rewritten history branches where it stopped matching."""
    created = await stack.create_trajectory()

    base = [{"role": "system", "content": "S"}, {"role": "user", "content": "turn one"}]
    await stack.chat(created, base, headers={"x-mock-reply": "R1"})
    await stack.settle()
    history = base + [{"role": "assistant", "content": "R1"}]
    await stack.chat(created, history + [{"role": "user", "content": "turn two"}],
                     headers={"x-mock-reply": "R2"})
    await stack.settle()

    # The agent compacts: same prefix through R1, then a summary instead.
    await stack.chat(
        created,
        history + [{"role": "user", "content": "SUMMARY of the conversation so far"}],
        headers={"x-mock-reply": "R3"},
    )
    graph = await stack.graph(created["id"])

    assert len(graph["branch_points"]) == 1
    fork = graph["branch_points"][0]
    index = nodes_by_id(graph)
    # It branched at R1, the last message that still matched.
    assert index[fork["node_id"]]["author"] == "model"
    assert index[fork["node_id"]]["depth"] == 2
    assert len(graph["leaf_node_ids"]) == 2

    exchanges = await stack.exchanges(created["id"])
    assert exchanges[2]["parent_output_node_id"] == fork["node_id"]


async def test_unmatched_history_branches_from_the_root(stack):
    """Rule 4: if no message matches, branch from the trajectory root."""
    created = await stack.create_trajectory()
    await stack.chat(created, [{"role": "user", "content": "first world"}])
    await stack.settle()
    await stack.chat(created, [{"role": "user", "content": "totally different"}])
    graph = await stack.graph(created["id"])

    roots = [node for node in graph["nodes"] if node["parent_node_id"] is None]
    assert len(roots) == 2
    assert graph["branch_points"] == [], "distinct roots are not a branch point"
    assert len(graph["leaf_node_ids"]) == 2

    exchanges = await stack.exchanges(created["id"])
    assert exchanges[1]["input_prefix_node_id"] is None
    assert exchanges[1]["parent_output_node_id"] is None


async def test_gap_is_measured_from_the_parent_not_the_previous_arrival(stack):
    """The wait before a call is measured from the call it continued from.

    Sequence order is arrival order. The moment a trajectory branches, the
    previous arrival is a sibling rather than a predecessor, so a gap taken
    from it is an invented number.
    """
    import asyncio

    created = await stack.create_trajectory()

    first = [{"role": "user", "content": "one"}]
    await stack.chat(created, first, headers={"x-mock-reply": "R1"})
    await stack.settle()
    await asyncio.sleep(0.08)
    await stack.chat(
        created,
        first + [{"role": "assistant", "content": "R1"}, {"role": "user", "content": "two"}],
    )
    exchanges = await stack.exchanges(created["id"])
    assert exchanges[0]["gap_ms"] is None, "the first call continued from nothing"
    assert exchanges[1]["gap_ms"] > 50, "measured from the parent's response end"


async def test_concurrent_siblings_have_no_gap_and_are_both_marked_overlapping(stack):
    """Rule 6: overlap is a timing fact, independent of graph parentage."""
    import asyncio

    overlapped = await stack.create_trajectory()
    await asyncio.gather(
        stack.chat(overlapped, [{"role": "user", "content": "a"}], headers={"x-mock-delay": "0.1"}),
        stack.chat(overlapped, [{"role": "user", "content": "b"}], headers={"x-mock-delay": "0.1"}),
    )
    concurrent = await stack.exchanges(overlapped["id"])

    # Neither call followed the other, so neither has a wait. Ordering by
    # arrival would have reported one as waiting on the other.
    assert [exchange["gap_ms"] for exchange in concurrent] == [None, None]
    # Overlap is still observed, because it is about wall-clock intervals.
    assert sum(1 for exchange in concurrent if exchange["overlapping"]) == 2

    distribution = (await stack.get(f"/v1/trajectories/{overlapped['id']}")).json()["gap_distribution"]
    assert distribution["samples"] == 0, "no parent, no sample"

async def test_explicit_parent_from_previous_response_id(stack):
    """A provider-supplied previous response ID becomes exact-evidence parentage."""
    created = await stack.create_trajectory()
    first = await stack.data.post(
        f"{created['base_url']}/responses",
        json={"model": "mock-model", "input": "start here"},
        headers={"authorization": "Bearer client-key"},
    )
    assert first.status_code == 200
    response_id = first.json()["id"]
    await stack.settle()

    second = await stack.data.post(
        f"{created['base_url']}/responses",
        json={"model": "mock-model", "input": "continue", "previous_response_id": response_id},
        headers={"authorization": "Bearer client-key"},
    )
    assert second.status_code == 200

    exchanges = await stack.exchanges(created["id"])
    assert exchanges[1]["previous_response_id"] == response_id
    assert exchanges[1]["previous_exchange_id"] == exchanges[0]["id"]


async def test_node_payload_holds_the_message_delta(stack):
    created = await stack.create_trajectory()
    await stack.chat(created, [{"role": "user", "content": "payload check"}])
    graph = await stack.graph(created["id"])
    user_node = graph["nodes"][0]

    payload = await stack.node_payload(user_node["node_id"])
    assert payload["message"] == {"role": "user", "content": "payload check"}
    assert payload["author"] == "client"
    assert payload["message_hash"] == user_node["message_hash"]


# -- repair -----------------------------------------------------------------
# A client that edits a sampled assistant message before replaying it is the
# case every other branch test misses: they all replay the assistant verbatim.
# It is also the one that matters most, because the difference between what the
# model produced and what a trainer would see is exactly one edit.
async def test_a_repaired_assistant_message_forks_at_the_user_node(stack):
    """Not at a model output, which is where every other branch case forks."""
    created = await stack.create_trajectory()
    history = [{"role": "user", "content": "look up the weather"}]
    await stack.chat(created, history, headers={"x-mock-reply": 'CALL weather(city="Berlin"'})
    await stack.settle()
    await stack.chat(
        created,
        history
        + [
            # One character different from what was sampled.
            {"role": "assistant", "content": 'CALL weather(city="Berlin")'},
            {"role": "user", "content": "tool result: 18C"},
        ],
    )
    graph = await stack.graph(created["id"])
    index = nodes_by_id(graph)

    assert len(graph["branch_points"]) == 1
    fork = index[graph["branch_points"][0]["node_id"]]
    assert fork["role"] == "user", "the repair diverged one level above itself"
    assert fork["author"] == "client"

    children = [index[child] for child in graph["branch_points"][0]["child_ids"]]
    assert sorted(child["role"] for child in children) == ["assistant", "assistant"]
    # The two siblings are the sampled reply and the client's substitute. The
    # author field is the only thing that separates them.
    assert sorted(child["author"] for child in children) == ["client", "model"]

    sampled = next(child for child in children if child["author"] == "model")
    repaired = next(child for child in children if child["author"] == "client")
    assert sampled["message_hash"] != repaired["message_hash"]


async def test_the_repaired_original_survives_as_an_abandoned_leaf(stack):
    """The dead leaf is the trainer/inference mismatch signal; it must persist."""
    created = await stack.create_trajectory()
    history = [{"role": "user", "content": "call the tool"}]
    await stack.chat(created, history, headers={"x-mock-reply": "malformed"})
    await stack.settle()
    await stack.chat(
        created,
        history + [{"role": "assistant", "content": "repaired"}, {"role": "user", "content": "next"}],
    )
    graph = await stack.graph(created["id"])
    index = nodes_by_id(graph)

    leaves = [index[node_id] for node_id in graph["leaf_node_ids"]]
    sampled_leaf = next(node for node in leaves if node["author"] == "model" and node["depth"] == 1)
    assert sampled_leaf["role"] == "assistant"

    # Childless is not the signal -- every path ends childless. What marks this
    # branch as abandoned is that its parent has another child that continued.
    siblings = [node for node in graph["nodes"] if node["parent_node_id"] == sampled_leaf["parent_node_id"]]
    assert len(siblings) == 2
    assert any(
        node["parent_node_id"] == sibling["node_id"]
        for sibling in siblings
        for node in graph["nodes"]
    ), "the other sibling carried on"


async def test_a_repair_is_not_exported_as_a_fork(stack):
    """AIPerf may only fork where the child matched through a model output.

    Here the matched leaf is a user node, so there is no parent response to
    replay from and the branch becomes an independent root session.
    """
    created = await stack.create_trajectory()
    history = [{"role": "user", "content": "start"}]
    await stack.chat(created, history, headers={"x-mock-reply": "sampled"})
    await stack.settle()
    await stack.chat(
        created,
        history + [{"role": "assistant", "content": "edited"}, {"role": "user", "content": "go on"}],
    )
    exchanges = await stack.exchanges(created["id"])

    assert exchanges[1]["parent_output_node_id"] is None, "no model output was matched through"
    # And so the wait cannot be attributed either: nothing preceded it.
    assert exchanges[1]["gap_ms"] is None


async def test_a_structured_tool_call_repair_forks_the_same_way(stack):
    """Real tool repair edits tool_calls[].arguments, not message text.

    The sampled and repaired messages are identical apart from one character
    inside a nested field, so this only works if the hash covers structure
    rather than a rendered string.
    """
    created = await stack.create_trajectory()
    history = [{"role": "user", "content": "search"}]
    await stack.chat(
        created,
        history,
        headers={"x-mock-tool-arguments": '{"q": "berlin"'},  # missing brace
    )
    await stack.settle()

    graph = await stack.graph(created["id"])
    sampled = next(node for node in graph["nodes"] if node["author"] == "model")
    sampled_payload = await stack.node_payload(sampled["node_id"])
    sampled_message = sampled_payload["message"]
    assert sampled_message["tool_calls"][0]["function"]["arguments"] == '{"q": "berlin"'

    # Repair only the arguments; everything else is replayed verbatim.
    repaired = {
        **sampled_message,
        "tool_calls": [
            {
                **sampled_message["tool_calls"][0],
                "function": {
                    **sampled_message["tool_calls"][0]["function"],
                    "arguments": '{"q": "berlin"}',
                },
            }
        ],
    }
    await stack.chat(
        created,
        history + [repaired, {"role": "tool", "tool_call_id": "call_1", "content": "ok"}],
    )
    graph = await stack.graph(created["id"])
    index = nodes_by_id(graph)

    assert len(graph["branch_points"]) == 1
    children = [index[child] for child in graph["branch_points"][0]["child_ids"]]
    assert sorted(child["author"] for child in children) == ["client", "model"]
    assert len({child["message_hash"] for child in children}) == 2, (
        "one character inside tool_calls[].arguments is still a different message"
    )


async def test_tools_are_part_of_node_identity(stack):
    """The same message under a different tool set is a different context.

    A chat template renders tool schemas into the prompt, so without this the
    graph merges two generations sampled under different tools into one node
    and reports a single call where there were two.
    """
    first = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
    second = [{"type": "function", "function": {"name": "write", "parameters": {}}}]
    created = await stack.create_trajectory()
    history = [{"role": "user", "content": "go"}]

    # Identical messages and an identical reply. Only the tools differ, so
    # nothing but the tool set can account for a divergence.
    await stack.chat(created, history, tools=first, headers={"x-mock-reply": "SAME"})
    await stack.settle()
    await stack.chat(created, history, tools=second, headers={"x-mock-reply": "SAME"})
    graph = await stack.graph(created["id"])

    roots = [node for node in graph["nodes"] if node["parent_node_id"] is None]
    assert len(roots) == 2, "the second call cannot reuse a prefix rendered under the old tools"
    assert len(graph["nodes"]) == 4
    assert len({node["delta_hash"] for node in roots}) == 2
    # The message itself is unchanged, and says so.
    assert len({node["message_hash"] for node in roots}) == 1


async def test_an_unchanged_tool_set_still_continues_the_path(stack):
    created = await stack.create_trajectory()
    tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
    history = [{"role": "user", "content": "go"}]
    response = await stack.chat(created, history, tools=tools)
    reply = response.json()["choices"][0]["message"]["content"]
    await stack.settle()
    await stack.chat(
        created,
        history + [{"role": "assistant", "content": reply}, {"role": "user", "content": "again"}],
        tools=tools,
    )
    graph = await stack.graph(created["id"])

    assert len([n for n in graph["nodes"] if n["parent_node_id"] is None]) == 1
    assert graph["branch_points"] == []
    assert len(graph["nodes"]) == 4


async def test_the_model_is_part_of_node_identity(stack):
    """Two models answering the same prompt the same way are still two samples.

    A dataset that means to distil a large model into a small one, or to train
    on one of them, has to be able to tell them apart -- and merging them would
    report one generation where there were two.
    """
    created = await stack.create_trajectory()
    history = [{"role": "user", "content": "same prompt"}]

    await stack.chat(created, history, model="big-model", headers={"x-mock-reply": "SAME"})
    await stack.settle()
    await stack.chat(created, history, model="small-model", headers={"x-mock-reply": "SAME"})
    graph = await stack.graph(created["id"])

    roots = [node for node in graph["nodes"] if node["parent_node_id"] is None]
    assert len(roots) == 2, "a model switch starts a new root"
    assert len({node["message_hash"] for node in roots}) == 1, "the message is unchanged"
    assert len({node["delta_hash"] for node in roots}) == 2, "the identity is not"

    rows = await stack.export_lines(trajectory=created["id"], format="text_samples")
    assert sorted(row["model"] for row in rows) == ["big-model", "small-model"]
    assert all(row["model"] is not None for row in rows), "a path has exactly one model"

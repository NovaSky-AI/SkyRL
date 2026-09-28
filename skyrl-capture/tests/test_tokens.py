"""Phase 4: token-in/token-out capture.

These tests check the invariants in docs/design/token-capture-parity.md end to end
through the real proxy, not just the trace data structure.
"""

from __future__ import annotations

import pytest


async def tokens_chat(stack, created, messages, **body):
    payload = {"model": "mock-tokens-model", "messages": messages, **body}
    return await stack.client.post(
        f"{created['base_url']}/chat/completions",
        json=payload,
        headers={"authorization": "Bearer client-key"},
    )


async def node_payload(stack, node_id):
    return await stack.node_payload(node_id)


def tokens_upstream(**fields):
    """A tokens upstream for ``stack_builder``.

    The upstream is startup configuration, so a test that needs a different
    model or context window builds a different capture process rather than
    registering a second target.
    """

    def build(url: str):
        from skyrl_capture.config import TitoUpstream

        return TitoUpstream(
            type="tokens",
            url=f"{url}/generate",
            tokenizer="builtin",
            api_key="upstream-secret",
            **fields,
        )

    return build


async def test_tokens_target_creates_a_tokens_trajectory(tokens_stack):
    created = await tokens_stack.create_trajectory()
    assert created["mode"] == "tokens"
    # `protocol` is what a *client* speaks to this route, which is OpenAI chat
    # completions whatever the engine underneath is. The engine's own wire is
    # capture's business, and a caller that had to know it could not point an
    # unchanged SDK at the route.
    assert created["protocol"] == "openai"
    assert created["base_url"].endswith("/v1")
    record = (await tokens_stack.get(f"/v1/trajectories/{created['id']}")).json()
    assert record["upstream"]["type"] == "tokens", "the engine is still on the record"


async def test_tokens_turn_stores_exact_tokens_and_logprobs(tokens_stack):
    created = await tokens_stack.create_trajectory()
    response = await tokens_chat(
        tokens_stack,
        created,
        [{"role": "system", "content": "be exact"}, {"role": "user", "content": "hello"}],
        max_tokens=6,
        logprobs=True,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["capture"]["tokens"] is True
    assert body["usage"]["completion_tokens"] == 6
    # Per-token logprobs are surfaced to the client from the exact sampled IDs.
    assert len(body["choices"][0]["logprobs"]["content"]) == 6

    exchanges = await tokens_stack.exchanges(created["id"])
    assert len(exchanges) == 1
    exchange = exchanges[0]
    assert exchange["provider"] == "tokens"
    assert exchange["completion_reason"] == "stop"

    graph = await tokens_stack.graph(created["id"])
    assert len(graph["nodes"]) == 3
    system, user, assistant = graph["nodes"]

    # Request messages introduce non-sampled token deltas.
    for node in (system, user):
        assert node["author"] == "client"
        assert node["token_count"] > 0
        assert node["sampled_start"] is None
        assert node["sampled_token_count"] == 0
        payload = await node_payload(tokens_stack, node["node_id"])
        assert all(flag is False for flag in payload["tokens"]["sampled_mask"])
        assert all(value == 0.0 for value in payload["tokens"]["logprobs"])

    # The assistant node holds its generation scaffold before the sampled
    # suffix -- parity requirement 3.
    assert assistant["author"] == "model"
    assert assistant["sampled_start"] > 0
    assert assistant["sampled_token_count"] == 6
    assert assistant["has_logprobs"] is True
    assert assistant["tokenizer"] == "builtin"
    assistant_payload = await node_payload(tokens_stack, assistant["node_id"])
    tokens = assistant_payload["tokens"]
    assert tokens["sampled_mask"][: tokens["sampled_start"]] == [False] * tokens["sampled_start"]
    assert tokens["sampled_mask"][tokens["sampled_start"] :] == [True] * 6
    assert all(value == 0.0 for value in tokens["logprobs"][: tokens["sampled_start"]])
    assert all(value != 0.0 for value in tokens["logprobs"][tokens["sampled_start"] :])

    # The prompt IDs sent to inference are exactly the concatenated node deltas
    # up to and including the assistant scaffold.
    stored = (await tokens_stack.stored_bodies(exchange["id"]))["tokens"]
    reconstructed = []
    for node in (system, user):
        reconstructed.extend((await node_payload(tokens_stack, node["node_id"]))["tokens"]["token_ids"])
    reconstructed.extend(tokens["token_ids"][: tokens["sampled_start"]])
    assert reconstructed == stored["prompt_token_ids"]
    assert tokens["token_ids"][tokens["sampled_start"] :] == stored["completion_ids"]


async def test_tokens_reuses_the_exact_prefix_on_the_next_turn(tokens_stack):
    """Parity requirements 7, 8 and 9: verified reuse, not assumed reuse."""
    created = await tokens_stack.create_trajectory()
    first = [{"role": "user", "content": "first question"}]
    response = await tokens_chat(tokens_stack, created, first, max_tokens=4)
    reply = response.json()["choices"][0]["message"]["content"]
    await tokens_stack.settle()
    after_first = await tokens_stack.graph(created["id"])
    assert len(after_first["nodes"]) == 2

    second = first + [
        {"role": "assistant", "content": reply},
        {"role": "user", "content": "second question"},
    ]
    response = await tokens_chat(tokens_stack, created, second, max_tokens=4)
    assert response.status_code == 200, response.text
    graph = await tokens_stack.graph(created["id"])

    # Only the new user message and the new assistant turn were committed.
    assert len(graph["nodes"]) == 4
    assert len(graph["leaf_node_ids"]) == 1

    exchanges = await tokens_stack.exchanges(created["id"])
    metadata = exchanges[1]["input_prefix_node_id"]
    assert metadata == after_first["nodes"][-1]["node_id"]

    # The reuse was bridged, and the bridge preserved the previous tokens.
    row = tokens_stack.get_exchange(exchanges[1]["id"])
    tokens_meta = row["source_metadata"]["tokens"]
    assert tokens_meta["reused_prefix_length"] > 0
    assert tokens_meta["bridge_transition_id"] == 0
    assert tokens_meta["matched_message_count"] == 2

    # Concatenating node deltas along the path reproduces the second prompt.
    stored = (await tokens_stack.stored_bodies(exchanges[1]["id"]))["tokens"]
    path_tokens: list[int] = []
    index = {node["node_id"]: node for node in graph["nodes"]}
    leaf = graph["leaf_node_ids"][0]
    chain = []
    current = leaf
    while current:
        chain.append(current)
        current = index[current]["parent_node_id"]
    chain.reverse()
    for node_id in chain[:-1]:
        path_tokens.extend((await node_payload(tokens_stack, node_id))["tokens"]["token_ids"])
    leaf_tokens = (await node_payload(tokens_stack, leaf))["tokens"]
    path_tokens.extend(leaf_tokens["token_ids"][: leaf_tokens["sampled_start"]])
    assert path_tokens == stored["prompt_token_ids"]


async def test_tokens_rewritten_history_branches(tokens_stack):
    """A compacted history branches at the last unchanged message."""
    created = await tokens_stack.create_trajectory()
    base = [{"role": "user", "content": "start"}]
    response = await tokens_chat(tokens_stack, created, base, max_tokens=3)
    reply = response.json()["choices"][0]["message"]["content"]
    await tokens_stack.settle()
    history = base + [{"role": "assistant", "content": reply}]

    await tokens_chat(tokens_stack, created, history + [{"role": "user", "content": "left"}], max_tokens=3)
    await tokens_stack.settle()
    await tokens_chat(tokens_stack, created, history + [{"role": "user", "content": "right"}], max_tokens=3)
    graph = await tokens_stack.graph(created["id"])

    assert len(graph["branch_points"]) == 1
    assert len(graph["leaf_node_ids"]) == 2
    index = {node["node_id"]: node for node in graph["nodes"]}
    assert index[graph["branch_points"][0]["node_id"]]["author"] == "model"


async def test_tokens_changed_tools_prevents_reuse(tokens_stack):
    """Parity requirement 6: tools participate in prefix identity."""
    created = await tokens_stack.create_trajectory()
    tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
    first = [{"role": "user", "content": "with tools"}]
    response = await tokens_chat(tokens_stack, created, first, tools=tools, max_tokens=3)
    reply = response.json()["choices"][0]["message"]["content"]
    await tokens_stack.settle()

    history = first + [{"role": "assistant", "content": reply}, {"role": "user", "content": "next"}]
    # Same messages, different tool schema: the rendered prompt differs, so the
    # previous prefix must not be reused.
    other_tools = [{"type": "function", "function": {"name": "browse", "parameters": {}}}]
    await tokens_chat(tokens_stack, created, history, tools=other_tools, max_tokens=3)
    await tokens_stack.settle()

    exchanges = await tokens_stack.exchanges(created["id"])
    row = tokens_stack.get_exchange(exchanges[1]["id"])
    tokens_meta = row["source_metadata"]["tokens"]
    assert tokens_meta["bridge_transition_id"] is None
    assert tokens_meta["reused_prefix_length"] == 0


async def test_tokens_identical_retry_reuses_nodes(tokens_stack):
    """Parity requirement 11: an identical retry is idempotent."""
    created = await tokens_stack.create_trajectory()
    messages = [{"role": "user", "content": "deterministic"}]
    # The mock derives its completion from the prompt token IDs, so an
    # identical prompt yields identical tokens -- which is exactly the retry
    # case this checks.
    for _ in range(2):
        response = await tokens_chat(tokens_stack, created, messages, max_tokens=3)
        assert response.status_code == 200, response.text
        await tokens_stack.settle()

    graph = await tokens_stack.graph(created["id"])
    assert len(graph["nodes"]) == 2, "the identical retry added no nodes"
    assert len(graph["leaf_node_ids"]) == 1
    assert graph["branch_points"] == []
    exchanges = await tokens_stack.exchanges(created["id"])
    assert len(exchanges) == 2
    assert exchanges[1]["is_duplicate_retry"] is True


async def test_tokens_routed_experts_are_stored(stack_builder):
    """Parity requirement 5: MoE routed experts, aligned to the full sequence."""
    tokens_stack = await stack_builder(
        upstream_for=tokens_upstream(model="mock-moe-model", max_model_len=4096)
    )
    created = await tokens_stack.create_trajectory()
    response = await tokens_chat(
        tokens_stack, created, [{"role": "user", "content": "moe"}], max_tokens=3
    )
    assert response.status_code == 200, response.text
    graph = await tokens_stack.graph(created["id"])
    assert all(node["has_routed_experts"] for node in graph["nodes"])

    total = 0
    for node in graph["nodes"]:
        payload = await node_payload(tokens_stack, node["node_id"])
        routed = payload["tokens"]["routed_experts"]
        assert routed is not None
        assert len(routed) == len(payload["tokens"]["token_ids"])
        assert all(len(layers) == 2 for layers in routed)
        total += len(routed)
    # Routed experts cover the full prompt plus completion.
    row = tokens_stack.exchange_rows(created["id"])[0]
    stored = (await tokens_stack.stored_bodies(row["id"]))["tokens"]
    assert total == len(stored["prompt_token_ids"]) + len(stored["completion_ids"])


async def test_tokens_max_tokens_is_clamped_to_the_context_window(stack_builder):
    """Parity requirement 13."""
    tokens_stack = await stack_builder(upstream_for=tokens_upstream(model="tiny", max_model_len=40))
    created = await tokens_stack.create_trajectory()
    response = await tokens_chat(
        tokens_stack, created, [{"role": "user", "content": "hello"}], max_tokens=10_000
    )
    assert response.status_code == 200, response.text
    # The mock echoes the sampling parameters it was asked for via token count.
    assert response.json()["usage"]["completion_tokens"] <= 16
    await tokens_stack.settle()
    row = tokens_stack.exchange_rows(created["id"])[0]
    sampling = row["source_metadata"]["tokens"]["sampling_params"]
    prompt_tokens = row["source_metadata"]["tokens"]["prompt_token_count"]
    assert sampling["max_tokens"] <= 40 - prompt_tokens
    # The renderer's own stop tokens are always passed through.
    assert sampling["stop_token_ids"]


async def test_tokens_prompt_that_exceeds_the_window_is_rejected(stack_builder):
    tokens_stack = await stack_builder(upstream_for=tokens_upstream(model="tiny", max_model_len=8))
    created = await tokens_stack.create_trajectory()
    response = await tokens_chat(
        tokens_stack, created, [{"role": "user", "content": "x" * 500}], max_tokens=4
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "context_length_exceeded"


async def test_tokens_missing_logprobs_fails_the_turn(tokens_stack):
    """Selected-token logprobs are required, not optional.

    A token capture exchange without them cannot serve the training it exists for, so it
    is an error rather than a partial success.
    """
    from skyrl_capture.tito import upstream as tito
    from skyrl_capture.tito.engine import TokenUpstreamError

    # Reading the response is the engine wire's job now, so the check runs
    # against the `tokens` protocol rather than a private engine helper.
    read = tito.get("tokens").response

    with pytest.raises(TokenUpstreamError, match="selected-token logprobs"):
        read({"response_ids": [[1, 2]], "stop_reasons": ["stop"]})
    with pytest.raises(TokenUpstreamError, match="different lengths"):
        read({"response_ids": [[1, 2]], "response_logprobs": [[-0.1]], "stop_reasons": ["stop"]})


async def test_an_unattributable_turn_poisons_the_trajectory(tokens_stack):
    """The deliberate divergence from text-mode fail-open capture.

    An attribution failure is not an inference failure. The engine produced this
    completion and the caller is owed it, so the turn returns 200 and says so in
    a header. What must not happen is a *next* turn: this turn is not in the
    graph, so the next one would render its assistant message out of the client's
    history and record model-generated tokens as ``author: client``. Stopping the
    trajectory is what prevents that, and it is durable because an evicted trace
    rebuilds from the graph, which has no record of the gap.
    """
    created = await tokens_stack.create_trajectory()
    service = tokens_stack.runtime.proxy.sessions
    tokens = tokens_stack.runtime.proxy
    trace = await service.trace_for(created["id"])

    from skyrl_capture.tito.types import TokenError

    def refuse(*_args, **_kwargs):
        raise TokenError("synthetic attribution failure")

    original = trace.commit
    trace.commit = refuse  # type: ignore[method-assign]
    try:
        response = await tokens_chat(
            tokens_stack, created, [{"role": "user", "content": "will fail"}], max_tokens=3
        )
    finally:
        trace.commit = original  # type: ignore[method-assign]

    # The caller still gets the completion the engine produced.
    assert response.status_code == 200
    assert response.headers["x-capture-status"] == "poisoned"
    assert response.json()["choices"][0]["message"]["role"] == "assistant"
    assert tokens.commit_failures >= 1
    assert tokens.trajectories_poisoned >= 1

    # Nothing partial was recorded.
    assert tokens_stack.node_rows(created["id"]) == []
    assert tokens_stack.aggregate(created["id"]).status == "poisoned"

    # And the trajectory takes no further turns, permanently.
    again = await tokens_chat(
        tokens_stack, created, [{"role": "user", "content": "and again"}], max_tokens=3
    )
    assert again.status_code == 410
    assert "client-authored" in again.text


async def test_tokens_streaming_is_synthesized_and_marked(tokens_stack):
    """A stream:true client still works; the exchange records the synthesis."""
    created = await tokens_stack.create_trajectory()
    chunks: list[bytes] = []
    async with tokens_stack.client.stream(
        "POST",
        f"{created['base_url']}/chat/completions",
        json={
            "model": "mock-tokens-model",
            "messages": [{"role": "user", "content": "stream please"}],
            "stream": True,
            "max_tokens": 4,
        },
        headers={"authorization": "Bearer client-key"},
    ) as response:
        assert response.status_code == 200
        async for chunk in response.aiter_raw():
            if chunk:
                chunks.append(chunk)
    body = b"".join(chunks)
    assert b"chat.completion.chunk" in body
    assert body.endswith(b"data: [DONE]\n\n")

    exchanges = await tokens_stack.exchanges(created["id"])
    assert exchanges[0]["streaming"] is True
    row = tokens_stack.get_exchange(exchanges[0]["id"])
    assert row["source_metadata"]["stream_synthesized"] is True


async def test_tokens_trace_is_rehydrated_from_storage(tokens_stack):
    """Prefix reuse survives a process losing its in-memory trace."""
    created = await tokens_stack.create_trajectory()
    first = [{"role": "user", "content": "before restart"}]
    response = await tokens_chat(tokens_stack, created, first, max_tokens=3)
    reply = response.json()["choices"][0]["message"]["content"]
    await tokens_stack.settle()

    # Simulate the trajectory moving to a process that has never seen it.
    tokens_stack.runtime.proxy.sessions.forget(created["id"])

    second = first + [
        {"role": "assistant", "content": reply},
        {"role": "user", "content": "after restart"},
    ]
    response = await tokens_chat(tokens_stack, created, second, max_tokens=3)
    assert response.status_code == 200, response.text
    await tokens_stack.settle()

    graph = await tokens_stack.graph(created["id"])
    # Reuse still worked: 4 nodes on one path, not a new root branch.
    assert len(graph["nodes"]) == 4
    assert len(graph["leaf_node_ids"]) == 1
    exchanges = await tokens_stack.exchanges(created["id"])
    row = tokens_stack.get_exchange(exchanges[1]["id"])
    assert row["source_metadata"]["tokens"]["reused_prefix_length"] > 0


async def test_tokens_export_produces_training_rows(tokens_stack):
    """The token_samples export shape a trainer consumes."""
    created = await tokens_stack.create_trajectory()
    first = [{"role": "user", "content": "train on me"}]
    response = await tokens_chat(tokens_stack, created, first, max_tokens=5)
    reply = response.json()["choices"][0]["message"]["content"]
    await tokens_stack.settle()
    await tokens_chat(
        tokens_stack,
        created,
        first + [{"role": "assistant", "content": reply}, {"role": "user", "content": "again"}],
        max_tokens=5,
    )
    await tokens_stack.settle()
    await tokens_stack.finish(
        created["id"], labels=["tokens"], annotations={"rlvr_reward": 1.0}
    )

    lines = await tokens_stack.export_lines(trajectory=created["id"], format="token_samples")
    assert len(lines) == 1
    row = lines[0]
    assert row["tokenizer"] == "builtin"
    # The reward is an annotation like any other, not a field of its own.
    assert row["annotations"]["rlvr_reward"] == 1.0
    assert row["abandoned"] is False

    # One flat sequence with a mask over it: everything aligns by index.
    assert len(row["loss_mask"]) == len(row["input_ids"])
    assert len(row["rollout_logprobs"]) == len(row["input_ids"])
    # Both sampled turns are trainable, and only sampled positions are.
    assert sum(row["loss_mask"]) == 10
    assert row["loss_mask"][0] == 0, "the opening message is context, not a target"


async def test_tokens_export_trains_each_sampled_node_at_most_once(tokens_stack):
    """Parity requirement 12.

    Forks share their ancestor assistant nodes. Without masking, the shared
    sampled tokens would be trained once per branch.
    """
    created = await tokens_stack.create_trajectory()
    base = [{"role": "user", "content": "shared root"}]
    response = await tokens_chat(tokens_stack, created, base, max_tokens=4)
    reply = response.json()["choices"][0]["message"]["content"]
    await tokens_stack.settle()
    history = base + [{"role": "assistant", "content": reply}]
    for tail in ("left", "right"):
        await tokens_chat(tokens_stack, created, history + [{"role": "user", "content": tail}], max_tokens=4)
        await tokens_stack.settle()

    lines = await tokens_stack.export_lines(trajectory=created["id"], format="token_samples")
    assert len(lines) == 2

    # The shared root completion is trainable in exactly one branch.
    graph = await tokens_stack.graph(created["id"])
    shared_assistant = graph["branch_points"][0]["node_id"]
    shared_tokens = (await node_payload(tokens_stack, shared_assistant))["tokens"]
    shared_tokens = shared_tokens["token_ids"][shared_tokens["sampled_start"] :]

    trainable_counts = []
    for row in lines:
        full = row["input_ids"]
        mask = row["loss_mask"]
        start = _find_subsequence(full, shared_tokens)
        assert start is not None
        trainable_counts.append(sum(mask[start : start + len(shared_tokens)]))
    assert sorted(trainable_counts) == [0, len(shared_tokens)]


def _find_subsequence(haystack: list[int], needle: list[int]) -> int | None:
    for index in range(len(haystack) - len(needle) + 1):
        if haystack[index : index + len(needle)] == needle:
            return index
    return None


async def test_mask_abandoned_zeroes_only_the_branch_that_lost(tokens_stack):
    """Both sides of a genuine fan-out are real generations worth training.

    Only a branch that lost a race is masked -- which is why --step-wise had
    nothing left to guard once training each target once became the default.
    """
    created = await tokens_stack.create_trajectory()
    base = [{"role": "user", "content": "root"}]
    response = await tokens_chat(tokens_stack, created, base, max_tokens=3)
    reply = response.json()["choices"][0]["message"]["content"]
    await tokens_stack.settle()
    history = base + [{"role": "assistant", "content": reply}]
    for tail in ("a", "b"):
        await tokens_chat(tokens_stack, created, history + [{"role": "user", "content": tail}], max_tokens=3)
        await tokens_stack.settle()

    lines = await tokens_stack.export_lines(
        trajectory=created["id"], format="token_samples", options={"mask_abandoned": True}
    )
    assert len(lines) == 2, "rows are never dropped"
    assert all(row["abandoned"] is False for row in lines), "neither branch lost a race"
    assert all(sum(row["loss_mask"]) > 0 for row in lines)


async def test_tokens_and_text_share_the_same_primitives(stack, tokens_stack):
    """Text and token capture use the same IDs, graph, routes and annotations.

    They are two capture processes now rather than two targets in one, because
    the mode follows the upstream and the upstream is startup configuration.
    What has to stay identical is everything above it.
    """
    text = await stack.create_trajectory(project="mixed")
    tokens = await tokens_stack.create_trajectory(project="mixed")

    await stack.chat(text, [{"role": "user", "content": "text mode"}])
    await tokens_chat(tokens_stack, tokens, [{"role": "user", "content": "tokens mode"}], max_tokens=3)
    await stack.settle()
    await tokens_stack.settle()

    for owner, created, mode in ((stack, text, "text"), (tokens_stack, tokens, "tokens")):
        trajectory = (await owner.get(f"/v1/trajectories/{created['id']}")).json()
        assert trajectory["mode"] == mode
        # The same metadata primitive works on both.
        result = await owner.patch(
            f"/v1/trajectories/{created['id']}/metadata", {"annotations": {"rlvr_reward": 0.5}}
        )
        assert result.status_code == 200
        assert result.json()["annotations"]["rlvr_reward"] == 0.5

    # What joins two capture processes is a shared record directory, not a
    # shared store: each holds its own state and sees only its own
    # trajectories, and both write into the same directory as they finish.
    for owner, created in ((stack, text), (tokens_stack, tokens)):
        listing = (await owner.get("/v1/trajectories?project=mixed")).json()["data"]
        assert [item["id"] for item in listing] == [created["id"]], "its own, and only its own"


# -- repair -----------------------------------------------------------------
async def test_tokens_repair_forks_and_leaves_the_sampled_reply_behind(tokens_stack):
    """The text-mode repair case, in token space.

    Worth pinning separately because ``delta_hash`` here covers the exact
    token delta and the sampled boundary, not just the message hash.
    """
    created = await tokens_stack.create_trajectory()
    history = [{"role": "user", "content": "call the tool"}]
    response = await tokens_chat(tokens_stack, created, history, max_tokens=4)
    sampled = response.json()["choices"][0]["message"]["content"]
    await tokens_stack.settle()

    await tokens_chat(
        tokens_stack,
        created,
        history
        + [
            {"role": "assistant", "content": sampled + "!"},  # the client edits it
            {"role": "user", "content": "next"},
        ],
        max_tokens=4,
    )
    await tokens_stack.settle()

    graph = await tokens_stack.graph(created["id"])
    index = {node["node_id"]: node for node in graph["nodes"]}
    assert len(graph["branch_points"]) == 1
    fork = index[graph["branch_points"][0]["node_id"]]
    assert fork["role"] == "user", "the repair diverged above itself, as in text mode"

    children = [index[child] for child in graph["branch_points"][0]["child_ids"]]
    assert sorted(child["author"] for child in children) == ["client", "model"]

    # The client's substitute carries no sampled tokens, so nothing can train
    # on text the model never produced.
    repaired = next(child for child in children if child["author"] == "client")
    assert repaired["sampled_start"] is None
    assert repaired["sampled_token_count"] in (None, 0)


async def test_tokens_repair_is_not_trainable(tokens_stack):
    created = await tokens_stack.create_trajectory()
    history = [{"role": "user", "content": "start"}]
    response = await tokens_chat(tokens_stack, created, history, max_tokens=4)
    sampled = response.json()["choices"][0]["message"]["content"]
    await tokens_stack.settle()
    await tokens_chat(
        tokens_stack,
        created,
        history + [{"role": "assistant", "content": sampled + "!"}, {"role": "user", "content": "go"}],
        max_tokens=4,
    )
    await tokens_stack.settle()

    rows = await tokens_stack.export_lines(trajectory=created["id"], format="token_samples")
    continued = next(row for row in rows if not row["abandoned"])
    repaired_tokens = await _sampled_tokens_of_client_assistant(tokens_stack, created["id"])
    full = continued["input_ids"]
    mask = continued["loss_mask"]
    start = _find_subsequence(full, repaired_tokens)
    assert start is not None, "the repaired message is present in the exported row"
    assert sum(mask[start : start + len(repaired_tokens)]) == 0, (
        "the client's substitute is context, never a training target"
    )


async def _sampled_tokens_of_client_assistant(stack, trajectory_id):
    graph = await stack.graph(trajectory_id)
    node = next(
        item
        for item in graph["nodes"]
        if item["author"] == "client" and item["role"] == "assistant"
    )
    return (await node_payload(stack, node["node_id"]))["tokens"]["token_ids"]


async def test_identical_text_can_still_diverge_in_token_space(tokens_stack):
    """The asymmetry that makes the tokens half worth its own test.

    Two messages that are byte-identical as text can carry different token
    deltas, and ``delta_hash`` covers the delta -- so they stay distinct nodes
    rather than being silently merged into one training record.
    """
    created = await tokens_stack.create_trajectory()
    messages = [{"role": "user", "content": "same words"}]
    first = await tokens_chat(tokens_stack, created, messages, max_tokens=3)
    await tokens_stack.settle()
    reply = first.json()["choices"][0]["message"]["content"]

    graph = await tokens_stack.graph(created["id"])
    assistant = next(node for node in graph["nodes"] if node["author"] == "model")
    payload = await node_payload(tokens_stack, assistant["node_id"])
    assert payload["tokens"]["token_ids"], "the sampled node carries its exact tokens"
    assert assistant["message_hash"], "and its message hash"

    # Replaying the identical text reuses the node: same message, same tokens.
    await tokens_chat(
        tokens_stack,
        created,
        messages + [{"role": "assistant", "content": reply}, {"role": "user", "content": "on"}],
        max_tokens=3,
    )
    await tokens_stack.settle()
    after = await tokens_stack.graph(created["id"])
    assert after["branch_points"] == [], "an exact replay is a continuation, not a fork"


async def test_a_bridge_that_rewrote_the_prefix_is_refused_under_audit(monkeypatch):
    """The audit path still catches a silently rewritten prefix.

    On the normal path the reused prefix is not re-read: it is the tokens these
    nodes were committed from, handed to the bridge and returned under a
    contract the library refuses rather than breaks. `TOKENS_AUDIT_PREFIX`
    re-proves that token for token, and this holds it honest by corrupting a
    prefix token that nothing upstream could corrupt on purpose.
    """
    from skyrl_capture.tito import trace as trace_module
    from skyrl_capture.tito.renderer import BuiltinRenderer
    from skyrl_capture.tito.trace import TokenTrace
    from skyrl_capture.tito.types import ModelTurnResult, TokenError

    monkeypatch.setattr(trace_module, "AUDIT_PREFIX", True)

    renderer = BuiltinRenderer()
    trace = TokenTrace("tr_bridge_guard")
    trace.set_text_renderer(renderer.decode, renderer.turn_start_token())

    def commit(messages, *, corrupt_at=None):
        pending = trace.prepare_turn(messages)
        rendered = None
        if pending.bridge_transition_id is not None:
            rendered = renderer.bridge(
                trace.transition_prompt_ids(pending.bridge_transition_id),
                trace.transition_completion_ids(pending.bridge_transition_id),
                list(messages)[len(pending.matched_node_ids) :],
            )
        if rendered is None:
            rendered = renderer.render(messages)
        token_ids = list(rendered.token_ids)
        if corrupt_at is not None:
            token_ids[corrupt_at] = 9_999
        completion = (72, 105, renderer.get_stop_token_ids()[0])
        return trace.commit(
            pending,
            ModelTurnResult(
                prompt_token_ids=tuple(token_ids),
                prompt_message_indices=tuple(rendered.message_indices),
                reused_prefix_length=rendered.reused_prefix_length,
                completion_ids=completion,
                completion_logprobs=(-0.1,) * len(completion),
                assistant_message={"role": "assistant", "content": "Hi"},
                stop_reason="stop",
                model="builtin",
            ),
            exchange_id="ex_guard",
        )

    first = [{"role": "user", "content": "hello"}]
    commit(first)
    second = first + [{"role": "assistant", "content": "Hi"}, {"role": "user", "content": "again"}]

    with pytest.raises(TokenError, match="do not match the renderer bridge prefix") as caught:
        commit(second, corrupt_at=4)

    # And it says *what* diverged, not just that something did: the class
    # points at which part of the machinery, the offset at where to look.
    message = str(caught.value)
    assert "given tokens diverge" in message, message
    assert "at prompt offset 4" in message, message
    assert trace.audit_failures == {"given": 1}

    # The same turn, uncorrupted, still commits.
    assert commit(second).assistant_node_id


def test_a_character_spanning_tokens_keeps_every_token_addressable():
    """A multi-byte character split across tokens.

    Two earlier contracts were both wrong. Raising poisoned the trajectory and
    threw away an exact, durable, perfectly trainable exchange -- along with
    every turn after it -- over a presentation detail. Voiding the segment's
    offsets kept the turn but cost every other token in the block its identity
    to pay for one character.

    The ambiguity is only ever *inside* the shared character, so that is where
    it stays: the tokens sharing it are grouped, the first gets the character,
    the rest get none. Offsets stop strictly increasing and stay one longer
    than the token count, so every token keeps an index, an id, a mask bit and
    a logprob.
    """
    from skyrl_capture.tito.trace import TokenNode

    def node_for(content, token_ids, decode):
        node = TokenNode(
            node_id="nd_mapping",
            parent_node_id=None,
            message={"role": "user", "content": content},
            token_ids=token_ids,
            sampled_start=None,
            completion_logprobs=(),
            routed_experts=None,
            depth=0,
            exchange_id="ex_mapping",
        )
        node.attach_text(decode, turn_start_token=None)
        return node.text_segments[0]

    # `é` is two bytes and the BPE put one in each token. Neither decodes.
    segment = node_for("é", (1, 2), lambda ids: "é" if len(ids) == 2 else "\ufffd")
    assert segment["text"] == "é", "the span decode, exactly"
    assert segment["offsets"] == [0, 1, 1], "the character goes to the first of them"

    # The realistic shape: clean tokens on both sides of a split character,
    # which is what a model emitting an emoji mid-sentence produces.
    pieces = {1: "ok ", 2: "\ufffd", 3: "\ufffd", 4: " done"}
    segment = node_for(
        "ok \U0001f680 done",
        (1, 2, 3, 4),
        lambda ids: "ok \U0001f680 done" if len(ids) == 4 else pieces[ids[0]],
    )
    assert segment["text"] == "ok \U0001f680 done"
    offsets = segment["offsets"]
    text = segment["text"]
    assert len(offsets) == 5, "one longer than the token count"

    # Counted in UTF-16 code units, which is how the JSON consumer that slices
    # this text indexes it. The emoji is one code point and two of these, so
    # code-point offsets would be [0, 3, 4, 4, 9] and would put every token
    # after it one character out.
    assert offsets == [0, 3, 5, 5, 10]

    units = text.encode("utf-16-le")

    def slice_of(index):
        return units[offsets[index] * 2 : offsets[index + 1] * 2].decode("utf-16-le")

    # Every token is addressable, the slices partition the text in order, and
    # the tokens sharing a character are the only empty ones.
    assert "".join(slice_of(i) for i in range(4)) == text
    assert [slice_of(i) for i in range(4)] == ["ok ", "\U0001f680", "", " done"]


async def test_the_audit_can_report_instead_of_refusing(monkeypatch):
    """For a long investigation, where stopping is worse than a wrong token.

    `TOKENS_AUDIT_PREFIX=report` classifies and counts the divergence and lets
    the turn commit. It is not a deployment setting: every class it can report
    is a bug.
    """
    from skyrl_capture.tito import trace as trace_module
    from skyrl_capture.tito.compare import compare_prefix

    monkeypatch.setattr(trace_module, "AUDIT_PREFIX", True)
    monkeypatch.setattr(trace_module, "AUDIT_REFUSES", False)

    class Node:
        def __init__(self, tokens, author, sampled_start=None):
            self.node_id = "nd_x"
            self.token_ids = tuple(tokens)
            self.author = author
            self.role = "assistant" if author == "model" else "user"
            self.sampled_start = sampled_start

    # The three classes, each pointing at a different part of the machinery.
    scaffold = compare_prefix([Node([1, 2, 3, 4], "model", sampled_start=2)], [1, 9, 3, 4], 4)
    assert scaffold.kind == "scaffold" and scaffold.offset == 1

    sampled = compare_prefix([Node([1, 2, 3, 4], "model", sampled_start=2)], [1, 2, 9, 4], 4)
    assert sampled.kind == "sampled" and sampled.offset == 2

    length = compare_prefix([Node([1, 2], "client")], [1, 2, 3], 3)
    assert length.kind == "length"
    assert "the graph holds 2 tokens" in length.describe()

    assert compare_prefix([Node([1, 2], "client")], [1, 2], 2).ok


async def test_a_finished_trajectory_does_not_keep_its_trace_in_memory(tokens_stack):
    """A cached trace is dead weight once no more turns can arrive.

    Nothing used to drop it, so a replica accumulated every trajectory it had
    ever served until the process died.
    """
    service = tokens_stack.runtime.proxy.sessions
    created = await tokens_stack.create_trajectory()
    await tokens_stack.chat(created, [{"role": "user", "content": "hello"}])
    await tokens_stack.settle()
    assert service.stats()["cached_traces"] >= 1

    await tokens_stack.finish(created["id"])
    assert created["id"] not in service._traces
    assert service.stats()["resident_tokens"] == 0


async def test_traces_are_evicted_once_they_outgrow_the_budget(tokens_stack):
    """The cache is bounded by tokens held, not by trajectory count.

    Evicting costs a rebuild from the graph on the next turn, which is the
    trade: a slow turn instead of a process that grows without limit.
    """
    service = tokens_stack.runtime.proxy.sessions
    service._trace_budget = 40  # a couple of short turns' worth

    ids = []
    for _ in range(4):
        created = await tokens_stack.create_trajectory()
        await tokens_stack.chat(created, [{"role": "user", "content": "hello there"}])
        await tokens_stack.settle()
        ids.append(created["id"])

    stats = service.stats()
    assert stats["traces_evicted"] > 0, "nothing was evicted under a tiny budget"
    assert stats["resident_tokens"] <= max(service._trace_budget, stats["resident_tokens"])
    # The most recent trajectory is the one still held.
    assert ids[-1] in service._traces
    assert len(service._traces) < len(ids)


async def test_an_evicted_trajectory_still_continues_correctly(tokens_stack):
    """Eviction must cost speed, never correctness."""
    service = tokens_stack.runtime.proxy.sessions
    created = await tokens_stack.create_trajectory()
    first = await tokens_stack.chat(created, [{"role": "user", "content": "first turn"}])
    reply = first.json()["choices"][0]["message"]
    await tokens_stack.settle()

    # Drop it the way the budget would, then continue the conversation.
    service._traces.pop(created["id"], None)

    second = await tokens_stack.chat(
        created,
        [
            {"role": "user", "content": "first turn"},
            {key: value for key, value in reply.items() if value is not None},
            {"role": "user", "content": "second turn"},
        ],
    )
    assert second.status_code == 200, second.text
    await tokens_stack.settle()

    graph = await tokens_stack.graph(created["id"])
    roots = [node for node in graph["nodes"] if node["parent_node_id"] is None]
    assert len(roots) == 1, "a rebuilt trace must continue the trajectory, not fork it"
    assert sum(1 for node in graph["nodes"] if node["author"] == "model") == 2


async def test_the_tokenizer_build_is_not_counted_as_a_per_turn_cost(tokens_stack):
    """A once-per-process cost inside a per-turn phase reads as a per-turn cost.

    Building the renderer takes seconds and happens once, charged to whichever
    turns arrive first. While it was summed into `trace` it showed up as
    130 ms/turn at 139 turns and 8 ms/turn at 2348 -- the same 18 s both times,
    divided by a growing turn count. It looked like a trajectory lookup that
    was somehow a third of engine time, and it was neither.

    So it has its own counter, and the raw totals are published beside the
    averages: a rate that means anything comes from differencing two scrapes,
    which the average cannot give you.
    """
    created = await tokens_stack.create_trajectory()
    for content in ("one", "two"):
        await tokens_stack.client.post(
            f"{created['base_url']}/chat/completions",
            json={"model": "mock-tokens-model", "messages": [{"role": "user", "content": content}],
                  "max_tokens": 4},
            headers={"authorization": "Bearer client-key"},
        )
    await tokens_stack.settle()

    health = (await tokens_stack.get("/healthz")).json()
    phases = health["capture"]["phase_ms_per_turn"]
    totals = health["capture"]["phase_ms_total"]
    turns = health["capture"]["turns_served"]

    assert "warmup" in phases, "the renderer build is its own phase"
    assert "trace" in phases
    assert set(totals) == set(phases), "every phase publishes a total beside its average"
    for name, total in totals.items():
        assert total == pytest.approx(phases[name] * turns, rel=0.02, abs=0.01), (
            f"{name}: the average must be the total over the turn count, so the two "
            "can be reconciled by whoever is differencing scrapes"
        )


# -- a character that spans tokens ------------------------------------------------
def test_a_block_without_offsets_still_carries_its_text(tokens_stack):
    """What a reader gets: the text, and no claim about where one token sits.

    `token_offsets` on the path says which it was, because "can I point at a
    token" is a question about this record rather than about the viewer.
    """
    from skyrl_capture.export.blocks import path_blocks
    from skyrl_capture.export.captured_text import CapturedText

    captured = CapturedText([(0, 3, "hi A", None)], turn_start_token=None)
    blocks = path_blocks([1, 7, 8], [1, 1, 1], captured_text=captured, turn_start_token=None)

    assert len(blocks) == 1
    assert blocks[0]["text"] == "hi A"
    assert blocks[0]["token_ids"] == [1, 7, 8]
    assert "token_offsets" not in blocks[0]


async def test_a_path_says_whether_every_token_is_addressable(tokens_stack):
    created = await tokens_stack.create_trajectory()
    await tokens_stack.client.post(
        f"{created['base_url']}/chat/completions",
        json={"model": "mock-tokens-model", "messages": [{"role": "user", "content": "hi"}],
              "max_tokens": 4},
        headers={"authorization": "Bearer client-key"},
    )
    await tokens_stack.refresh()

    body = (await tokens_stack.get(f"/v1/trajectories/{created['id']}/paths")).json()
    assert body["paths"][0]["token_offsets"] == "exact"

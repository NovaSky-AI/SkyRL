"""The behavioural contract, pinned to files.

Every scenario the product distinguishes -- linear, forked, repaired, retried,
streaming, tool-bearing, two providers, two capture modes, a poisoned turn,
metadata that arrives late -- is captured once through the real proxy, then
everything a consumer can read of it is normalised and compared to
`tests/golden/scenarios/<name>.json`: all four export formats, and the four
`/v1` documents the viewer draws from.

This exists for the refactor in `docs/design/refactor_2026-09-19.md`. The
storage and composition underneath are being replaced; these files are what
"the same product" means while that happens. A diff here is a behaviour change
and has to be read as one, whichever direction it goes.

Ids and clocks are normalised (`tests/golden.py`), so what is pinned is
structure and content: which nodes exist, who authored them, which path is
trainable, what a replay session contains, what a block's text is.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from golden import check

EXPORT_FORMATS = ("graph", "replay", "text_samples")
TOKEN_FORMATS = (*EXPORT_FORMATS, "token_samples")


# -- scenarios ----------------------------------------------------------------
# Each returns the trajectory id of one finished trajectory. Replies are pinned
# with `x-mock-reply` so the graph is a function of the scenario alone.
async def text_linear(stack) -> str:
    created = await stack.create_trajectory(project="golden", run_id="run-g", task_id="t1", step=0)
    history: list[dict[str, Any]] = [{"role": "system", "content": "be terse"}]
    for turn in range(3):
        history.append({"role": "user", "content": f"turn {turn}"})
        reply = await stack.chat(created, history, headers={"x-mock-reply": f"reply {turn}"})
        history.append(reply.json()["choices"][0]["message"])
        await asyncio.sleep(0.01)
    await stack.settle()
    await stack.finish(created["id"], labels=["ok"], annotations={"reward": 1.0})
    return created["id"]


async def text_forked(stack) -> str:
    created = await stack.create_trajectory(project="golden", run_id="run-g", task_id="t2", step=0)
    shared = [{"role": "user", "content": "shared"}]
    await stack.chat(created, shared, headers={"x-mock-reply": "shared reply"})
    await stack.settle()
    history = [*shared, {"role": "assistant", "content": "shared reply"}]
    for tail in ("left", "right"):
        await stack.chat(
            created, [*history, {"role": "user", "content": tail}], headers={"x-mock-reply": f"{tail} reply"}
        )
        await stack.settle()
    await stack.finish(created["id"], annotations={"reward": 0.5})
    return created["id"]


async def text_repaired(stack) -> str:
    """The client edits the sampled reply before replaying it: the original
    survives as an abandoned leaf and the edit is client-authored."""
    created = await stack.create_trajectory(project="golden", run_id="run-g", task_id="t3", step=1)
    history = [{"role": "user", "content": "look up the weather"}]
    await stack.chat(created, history, headers={"x-mock-reply": 'CALL get_weather(city="Berlin"'})
    await stack.settle()
    await stack.chat(
        created,
        [
            *history,
            {"role": "assistant", "content": 'CALL get_weather(city="Berlin")'},
            {"role": "user", "content": "tool result: 18C"},
        ],
        headers={"x-mock-reply": "It is 18C."},
    )
    await stack.settle()
    await stack.finish(created["id"], annotations={"reward": 0.0})
    return created["id"]


async def text_retry(stack) -> str:
    created = await stack.create_trajectory(project="golden", run_id="run-g", task_id="t4", step=1)
    messages = [{"role": "user", "content": "same"}]
    await stack.chat(created, messages, headers={"x-mock-reply": "same"})
    await stack.settle()
    await stack.chat(
        created, messages, headers={"x-mock-reply": "same", "x-stainless-retry-count": "1"}
    )
    await stack.settle()
    await stack.finish(created["id"])
    return created["id"]


async def text_streaming(stack) -> str:
    created = await stack.create_trajectory(project="golden", run_id="run-g", task_id="t5", step=2)
    await stack.chat(created, [{"role": "user", "content": "stream"}], stream=True)
    await stack.settle()
    await stack.finish(created["id"])
    return created["id"]


async def text_tools(stack) -> str:
    """Tools are part of node identity, so a tool change is its own root."""
    first = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
    second = [{"type": "function", "function": {"name": "write", "parameters": {}}}]
    created = await stack.create_trajectory(project="golden", run_id="run-g", task_id="t6", step=2)
    history = [{"role": "user", "content": "go"}]
    await stack.chat(created, history, tools=first, temperature=0.3, max_tokens=32, headers={"x-mock-reply": "SAME"})
    await stack.settle()
    await stack.chat(created, history, tools=second, temperature=0.3, max_tokens=32, headers={"x-mock-reply": "SAME"})
    await stack.settle()
    await stack.finish(created["id"])
    return created["id"]


async def text_late_metadata(stack) -> str:
    """Finish carries a reward; a label and a critique arrive afterwards."""
    created = await stack.create_trajectory(
        project="golden", run_id="run-g", task_id="t7", step=3, labels=["draft"], annotations={"attempt": 1}
    )
    await stack.chat(created, [{"role": "user", "content": "hi"}], headers={"x-mock-reply": "hello"})
    await stack.settle()
    await stack.finish(created["id"], labels=["scored"], annotations={"reward": 0.75})
    await stack.patch(
        f"/v1/trajectories/{created['id']}/metadata",
        {"annotations": {"critique": "fine"}, "labels": ["reviewed"], "remove_labels": ["draft"]},
    )
    return created["id"]


async def anthropic_linear(stack) -> str:
    created = await stack.create_trajectory(project="golden", run_id="run-a", task_id="a1", step=0)
    history = [{"role": "user", "content": "hi"}]
    reply = await stack.messages(created, history, system="be terse")
    history.append({"role": "assistant", "content": reply.json()["content"][0]["text"]})
    history.append({"role": "user", "content": "more"})
    await stack.messages(created, history, system="be terse", stream=True)
    await stack.settle()
    await stack.finish(created["id"], annotations={"reward": 1.0})
    return created["id"]


async def tokens_linear(stack) -> str:
    created = await stack.create_trajectory(project="golden", run_id="run-k", task_id="k1", step=0)
    history: list[dict[str, Any]] = [{"role": "system", "content": "be exact"}]
    for turn in range(2):
        history.append({"role": "user", "content": f"turn {turn}"})
        reply = await _tokens_chat(stack, created, history, max_tokens=4, logprobs=True)
        history.append({k: v for k, v in reply.items() if v is not None})
    await stack.settle()
    await stack.finish(created["id"], annotations={"reward": 1.0})
    return created["id"]


async def tokens_forked(stack) -> str:
    """A rewritten history branches; the loser of the fork is abandoned."""
    created = await stack.create_trajectory(project="golden", run_id="run-k", task_id="k2", step=0)
    history = [{"role": "user", "content": "shared"}]
    reply = await _tokens_chat(stack, created, history, max_tokens=4)
    kept = [*history, {k: v for k, v in reply.items() if v is not None}]
    await _tokens_chat(stack, created, [*kept, {"role": "user", "content": "left"}], max_tokens=4)
    await _tokens_chat(stack, created, [*kept, {"role": "user", "content": "right"}], max_tokens=4)
    await stack.settle()
    await stack.finish(created["id"], annotations={"reward": 0.5})
    return created["id"]


async def tokens_repaired(stack) -> str:
    """The client edits the sampled reply: a `replayed` block in the path."""
    created = await stack.create_trajectory(project="golden", run_id="run-k", task_id="k3", step=1)
    history = [{"role": "user", "content": "call the tool"}]
    reply = await _tokens_chat(stack, created, history, max_tokens=4)
    await _tokens_chat(
        stack,
        created,
        [*history, {"role": "assistant", "content": reply["content"] + "!"}, {"role": "user", "content": "next"}],
        max_tokens=4,
    )
    await stack.settle()
    await stack.finish(created["id"], annotations={"reward": 0.0})
    return created["id"]


async def tokens_streaming(stack) -> str:
    created = await stack.create_trajectory(project="golden", run_id="run-k", task_id="k4", step=1)
    chunks: list[bytes] = []
    async with stack.client.stream(
        "POST",
        f"{created['base_url']}/chat/completions",
        json={"model": "mock-tokens-model", "messages": [{"role": "user", "content": "stream please"}],
              "stream": True, "max_tokens": 4},
        headers={"authorization": "Bearer client-key"},
    ) as response:
        async for chunk in response.aiter_raw():
            chunks.append(chunk)
    await stack.settle()
    await stack.finish(created["id"])
    return created["id"]


async def tokens_poisoned(stack) -> str:
    """An unattributable turn: the caller gets its completion, the trajectory
    stops, and nothing partial is recorded."""
    created = await stack.create_trajectory(project="golden", run_id="run-k", task_id="k5", step=2)
    await _tokens_chat(stack, created, [{"role": "user", "content": "fine"}], max_tokens=3)
    await stack.settle()
    from skyrl_capture.tito.types import TokenError

    trace = await stack.token_trace(created["id"])
    original = trace.commit

    def refuse(*_args, **_kwargs):
        raise TokenError("synthetic attribution failure")

    trace.commit = refuse  # type: ignore[method-assign]
    try:
        await _tokens_chat(stack, created, [{"role": "user", "content": "will fail"}], max_tokens=3)
    finally:
        trace.commit = original  # type: ignore[method-assign]
    await stack.settle()
    return created["id"]


async def _tokens_chat(stack, created, messages, **body) -> dict[str, Any]:
    response = await stack.client.post(
        f"{created['base_url']}/chat/completions",
        json={"model": "mock-tokens-model", "messages": messages, **body},
        headers={"authorization": "Bearer client-key"},
    )
    assert response.status_code == 200, response.text
    return response.json()["choices"][0]["message"]


TEXT_SCENARIOS = {
    "text-linear": text_linear,
    "text-forked": text_forked,
    "text-repaired": text_repaired,
    "text-retry": text_retry,
    "text-streaming": text_streaming,
    "text-tools": text_tools,
    "text-late-metadata": text_late_metadata,
}
TOKENS_SCENARIOS = {
    "tokens-linear": tokens_linear,
    "tokens-forked": tokens_forked,
    "tokens-repaired": tokens_repaired,
    "tokens-streaming": tokens_streaming,
    "tokens-poisoned": tokens_poisoned,
}


# -- what a consumer reads ------------------------------------------------------
async def everything_readable(stack, trajectory_id: str, *, formats: tuple[str, ...]) -> dict[str, Any]:
    """The four export formats and the four `/v1` documents, for one trajectory."""
    exports = {}
    for export_format in formats:
        exports[export_format] = await stack.export_lines(trajectory=trajectory_id, format=export_format)
    api = {}
    for route in ("", "/exchanges", "/graph", "/paths"):
        response = await stack.get(f"/v1/trajectories/{trajectory_id}{route}")
        assert response.status_code == 200, (route, response.text)
        api[route or "/"] = response.json()
    return {"exports": exports, "api": api}


@pytest.mark.parametrize("name", sorted(TEXT_SCENARIOS))
async def test_text_scenarios_match_their_golden(stack, name):
    trajectory_id = await TEXT_SCENARIOS[name](stack)
    check(name, await everything_readable(stack, trajectory_id, formats=EXPORT_FORMATS), kind="scenarios")


@pytest.mark.parametrize("name", sorted(TOKENS_SCENARIOS))
async def test_tokens_scenarios_match_their_golden(tokens_stack, name):
    trajectory_id = await TOKENS_SCENARIOS[name](tokens_stack)
    formats = TOKEN_FORMATS if name != "tokens-poisoned" else ("graph", "replay")
    check(name, await everything_readable(tokens_stack, trajectory_id, formats=formats), kind="scenarios")


async def test_anthropic_scenario_matches_its_golden(stack_builder):
    from skyrl_capture.config import TextUpstream

    stack = await stack_builder(
        upstream_for=lambda url: TextUpstream(type="anthropic", url=url, api_key="upstream-secret")
    )
    trajectory_id = await anthropic_linear(stack)
    check("anthropic-linear", await everything_readable(stack, trajectory_id, formats=EXPORT_FORMATS), kind="scenarios")


async def test_the_run_listing_matches_its_golden(stack):
    """Two trajectories in one run: the counts, the steps, the listing rows."""
    await text_linear(stack)
    await text_forked(stack)
    runs = (await stack.get("/v1/runs")).json()
    listing = (await stack.get("/v1/trajectories?project=golden")).json()
    check("run-listing", {"runs": runs, "trajectories": listing}, kind="scenarios")

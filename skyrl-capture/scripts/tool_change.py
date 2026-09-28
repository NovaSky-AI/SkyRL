"""An agent that retries a turn with more tools, which is a different context.

A plausible loop: the agent asks with read-only tools, the model answers
without calling anything, so the agent widens the tool set and asks *the same
question again*. The messages are identical both times -- only the tools
differ.

That is the case where capture has to be careful. A chat template renders tool
schemas into the prompt, so the second call did not see the first call's
context; it saw a different one. If tools were not part of node identity the
graph would merge the two into a single node and report one generation where
there were two.

The replies are pinned identical on purpose, so nothing *except* the tool set
can explain a divergence. Run it and the graph shows two roots:

    skyrl-capture run --project toolchange \
      -- python scripts/tool_change.py

    skyrl-capture view
    skyrl-capture export --trajectory <id> --format text-samples --output samples.jsonl

Two rows come out, one per tool set, rather than one row hedging about which
tools it was produced under.
"""

from __future__ import annotations

import json
import os
import urllib.request

BASE = os.environ["OPENAI_BASE_URL"].rstrip("/")
KEY = os.environ["OPENAI_API_KEY"]

READ_ONLY = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the codebase.",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
    }
]
# The same tools, plus one that can change something.
WITH_WRITE = READ_ONLY + [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a file.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
    }
]

QUESTION = [{"role": "user", "content": "Fix the failing auth test."}]
# Pinned, so the only difference between the two calls is the tool set.
REPLY = "I need to modify the test file."


def ask(tools: list[dict[str, object]]) -> str:
    body = json.dumps(
        {"model": "gpt-4o-mini", "messages": QUESTION, "tools": tools, "max_tokens": 64}
    ).encode()
    request = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=body,
        headers={
            "authorization": f"Bearer {KEY}",
            "content-type": "application/json",
            "x-mock-reply": REPLY,
        },
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)["choices"][0]["message"]["content"]


def main() -> None:
    names = lambda tools: [tool["function"]["name"] for tool in tools]  # noqa: E731

    first = ask(READ_ONLY)
    print(f"with {names(READ_ONLY)}: {first}")

    # The model says it needs to write, so the agent widens the tool set and
    # asks again -- the same question, a different context.
    second = ask(WITH_WRITE)
    print(f"with {names(WITH_WRITE)}: {second}")

    assert first == second, "the replies are pinned, so only the tools differ"
    print("\nidentical messages, identical reply, different tools:")
    print("  the graph should show two roots, not one shared node.")


if __name__ == "__main__":
    main()

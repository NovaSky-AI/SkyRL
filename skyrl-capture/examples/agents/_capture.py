"""One model call, with nothing else in the way.

Every example in this directory imports `chat` from here so that the example
itself is only the story it is telling. There is no capture code in it: it
reads `OPENAI_BASE_URL` and `OPENAI_API_KEY`, the two variables the OpenAI SDK
already reads, which is all the proxy needs.

`reply` is a mock-only convenience. The bundled mock upstream returns whatever
the `x-mock-reply` header asks for, so every example produces the same graph
every time and the printed trees in `docs/graph.md` are reproducible. Against a
real provider the header is ignored and the same shapes appear with whatever
the model actually says.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

BASE = os.environ["OPENAI_BASE_URL"].rstrip("/")
KEY = os.environ["OPENAI_API_KEY"]
# What to ask for. The mock upstream ignores it; a real engine is asked for the
# model it serves, and the name is what the record says ran -- so it is worth
# being able to set without editing an example.
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


def chat(
    messages: list[dict[str, Any]],
    reply: str,
    *,
    model: str = MODEL,
    tools: list[dict[str, Any]] | None = None,
) -> str:
    """Send one chat completion and return the assistant's text."""
    payload: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": 128}
    if tools is not None:
        payload["tools"] = tools
    request = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "authorization": f"Bearer {KEY}",
            "content-type": "application/json",
            "x-mock-reply": reply,
        },
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)["choices"][0]["message"]["content"]


SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the codebase.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}

EDIT_TOOL = {
    "type": "function",
    "function": {
        "name": "edit",
        "description": "Edit a file.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "patch": {"type": "string"}},
            "required": ["path", "patch"],
        },
    },
}

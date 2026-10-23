"""The ``developer`` role for a chat template that doesn't know it, as vLLM handles it at render time.

OpenAI's newer clients (Codex, and any Responses client) send ``developer``
messages. Most open-weight chat templates have no such role: Qwen3's renderer
drops the message, and Qwen3.5's raises on it. vLLM's rule
(``vllm/renderers/hf.py``, ``_detect_developer_role_support``,
``_convert_developer_to_system``, ``_consolidate_system_messages``):

1. If any message is ``developer`` and the chat template's text never names the
   role, every ``developer`` message becomes ``system`` (its ``tools`` dropped).
2. If that leaves a ``system`` message anywhere but first, or more than one,
   all system text is merged into one leading message, joined by a blank line.

A template that names the role (gpt-oss) gets the messages untouched. Token
mode runs this before matching, for every dialect, so the graph stores the turns
as rendered and a replay converts to the same turns.

The merge assumes ``developer`` messages come only at the start. One that
arrives mid-session rewrites the leading message, so the next call can't extend
the previous call's tokens: it is re-rendered, and the trajectory splits into
another sample. That's faithful to vLLM, but each such message costs RL a
re-render and a split. Codex sends one only when a setting changes mid-session
(approval policy, model or collaboration mode, its tool list, or the off-by-default
time-reminder and token-budget features). Checked on 2026-10-17 on 64 ``codex exec``
rollouts (Codex 0.154.0, 604 calls, 11 compactions): one developer message per
session, before the first model call, and none after, compactions included
(Codex rebuilds the new window with the same leading message).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def template_knows_developer(chat_template: str) -> bool:
    """vLLM's test: the template's text names the role in quotes."""
    return '"developer"' in chat_template or "'developer'" in chat_template


def fold_developer(messages: Sequence[Mapping[str, Any]], chat_template: str | None) -> list[dict[str, Any]]:
    """``messages`` as vLLM renders them with ``chat_template``.

    With no template to test (a renderer that doesn't expose one), the messages are left as they are.
    """
    if chat_template is None or template_knows_developer(chat_template):
        return list(messages)  # type: ignore[arg-type]
    if not any(message.get("role") == "developer" for message in messages):
        return list(messages)  # type: ignore[arg-type]
    converted = []
    for message in messages:
        message = dict(message)
        if message.get("role") == "developer":
            message["role"] = "system"
            message.pop("tools", None)
        converted.append(message)
    return _consolidate_system(converted)


def _consolidate_system(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """All system text in one leading message, when a system message isn't first or isn't alone."""
    texts: list[str] = []
    others: list[dict[str, Any]] = []
    needed = False
    for index, message in enumerate(messages):
        if message.get("role") != "system":
            others.append(message)
            continue
        if index > 0 or texts:
            needed = True
        content = message.get("content") or ""
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    parts.append(part["text"])
                elif isinstance(part, str):
                    parts.append(part)
            content = "\n".join(parts)
        if content:
            texts.append(content)
    if not needed:
        return messages
    return [{"role": "system", "content": "\n\n".join(texts)}, *others]

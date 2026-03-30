"""
Tool call parser for task generation environment.

Parses <tool_call> and <function_call> tagged JSON from LLM responses.
Copied from skyrl-train/integrations/fleet/env.py to avoid cross-package imports.
"""

import json
import re
from typing import Any, Dict, List, Optional


def _try_parse_json(raw: str) -> Optional[Dict[str, Any]]:
    """Try to parse JSON, repairing missing trailing braces if needed."""
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Repair: models often drop trailing closing braces on nested JSON.
    # Try appending up to 3 closing braces.
    for extra in range(1, 4):
        try:
            parsed = json.loads(raw + "}" * extra)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            continue

    return None


def _parse_one(match_text: str) -> Optional[Dict[str, Any]]:
    """Parse a single tool call from matched text."""
    parsed = _try_parse_json(match_text)
    if parsed is None:
        return None
    name = parsed.get("name") or parsed.get("tool")
    args = parsed.get("arguments") or parsed.get("params", {})
    if name:
        return {"name": name, "arguments": args}
    return None


def parse_tool_call(action: str) -> Optional[Dict[str, Any]]:
    """Parse the first tool call from LLM response. Returns None if not found."""
    calls = parse_tool_calls(action)
    return calls[0] if calls else None


def parse_tool_calls(action: str) -> List[Dict[str, Any]]:
    """Parse all tool calls from LLM response.

    Supports tag-based formats:
    - <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    - <function_call>{"name": "...", "arguments": {...}}</function_call>

    Also handles cases where the closing tag is missing (e.g., when </tool_call>
    is used as the stop string and not included in the output).

    Returns list of dicts with "name" and "arguments" keys.
    """
    results: List[Dict[str, Any]] = []

    for tag in ["tool_call", "function_call"]:
        # Find all with closing tag
        for match in re.finditer(rf"<{tag}>(.*?)</{tag}>", action, re.DOTALL):
            parsed = _parse_one(match.group(1))
            if parsed:
                results.append(parsed)

        # If none found with closing tags, try without (stop string case)
        if not results:
            match = re.search(rf"<{tag}>(.*?)(?:<\||\Z)", action, re.DOTALL)
            if match:
                parsed = _parse_one(match.group(1))
                if parsed:
                    results.append(parsed)

    return results

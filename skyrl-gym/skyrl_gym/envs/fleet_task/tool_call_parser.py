"""Tool call parser for LLM-generated tool calls.

Parses tool calls from various tag-based formats commonly produced by LLMs:
- <tool_call>{"name": "...", "arguments": {...}}</tool_call>
- <function_call>{"name": "...", "arguments": {...}}</function_call>

Handles missing closing tags (e.g., when </tool_call> is the stop string)
and repairs common JSON issues like missing trailing braces.
"""

import json
import re
from typing import Any, Dict, Optional


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


def parse_tool_call(action: str) -> Optional[Dict[str, Any]]:
    """Parse tool call from LLM response.

    Supports tag-based formats:
    - <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    - <function_call>{"name": "...", "arguments": {...}}</function_call>

    Also handles cases where the closing tag is missing (e.g., when </tool_call>
    is used as the stop string and not included in the output).

    Returns:
        Dict with "name" and "arguments" keys, or None if no tool call found.
    """
    for tag in ["tool_call", "function_call"]:
        # First try with closing tag
        match = re.search(rf"<{tag}>(.*?)</{tag}>", action, re.DOTALL)
        if not match:
            # Try without closing tag (for when </tool_call> is the stop string)
            match = re.search(rf"<{tag}>(.*?)(?:<\||\Z)", action, re.DOTALL)
        if match:
            parsed = _try_parse_json(match.group(1))
            if parsed is None:
                continue
            # Normalize keys
            name = parsed.get("name") or parsed.get("tool")
            args = parsed.get("arguments") or parsed.get("params", {})
            if name:
                return {"name": name, "arguments": args}

    return None

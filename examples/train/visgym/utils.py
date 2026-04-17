import ast
import base64
import io
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


def encode_image_to_base64(rgb_array: np.ndarray, format: str = "png") -> str:
    """Convert an RGB numpy array to a base64-encoded string."""
    img = Image.fromarray(rgb_array)
    buffer = io.BytesIO()
    img.save(buffer, format=format.upper())
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def make_image_message(text: str, rgb_array: np.ndarray, role: str = "user") -> Dict[str, Any]:
    """Build an OpenAI-format multimodal message with text and an image.

    Returns a dict like:
        {"role": "user", "content": [
            {"type": "text", "text": "..."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ]}
    """
    b64 = encode_image_to_base64(rgb_array)
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ]
    return {"role": role, "content": content}


# ---------------------------------------------------------------------------
# Tuple action extraction (SFT recipe)
# ---------------------------------------------------------------------------

# Finds the opening of a tuple with a quoted action name: ('action_name',
_TUPLE_START_RE = re.compile(r"""\(\s*['"](\w+)['"]\s*,""")

# Same but with an unquoted action name: (action_name,
_TUPLE_START_UNQUOTED_RE = re.compile(r"""\(\s*(\w+)\s*,""")

# JSON action object: {"action": "...", "args": ...}
_JSON_ACTION_RE = re.compile(
    r"""\{\s*["']action["']\s*:\s*["'](\w+)["']\s*,\s*["']args?["']\s*:\s*(.+?)\}""",
    re.DOTALL,
)


def _find_balanced_tuple(text: str, start: int) -> Optional[str]:
    """Walk forward from ``text[start]`` (which must be ``(``) tracking
    bracket depth and string literals until the matching ``)`` is found.

    Returns the balanced substring including both delimiters, or ``None``
    if the text ends before the brackets balance.
    """
    depth = 0
    in_str: Optional[str] = None
    i = start
    while i < len(text):
        c = text[i]
        if in_str:
            if c == "\\" and i + 1 < len(text):
                i += 2
                continue
            if c == in_str:
                in_str = None
        else:
            if c in ("'", '"'):
                in_str = c
            elif c in ("(", "["):
                depth += 1
            elif c in (")", "]"):
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        i += 1
    return None


def _try_extract_balanced(text: str, pattern: re.Pattern) -> Optional[str]:
    """Find ``pattern`` in *text*, then extract the full balanced tuple."""
    for match in pattern.finditer(text):
        candidate = _find_balanced_tuple(text, match.start())
        if candidate is None:
            continue
        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, tuple) and len(parsed) == 2:
                return candidate
        except Exception:
            continue
    return None


def _strip_markdown(text: str) -> str:
    """Remove common markdown/code-fence wrappers that VLMs like to add."""
    text = re.sub(r"```[\w]*\n?", "", text)
    text = text.replace("`", "")
    return text


def _try_json_fallback(text: str) -> Optional[str]:
    """Look for ``{"action": "name", "args": ...}`` and convert to a tuple string."""
    match = _JSON_ACTION_RE.search(text)
    if not match:
        return None
    action_name = match.group(1)
    args_str = match.group(2).strip()
    try:
        args_val = ast.literal_eval(args_str)
    except Exception:
        try:
            args_val = json.loads(args_str)
        except Exception:
            return None
    candidate = repr((action_name, args_val))
    try:
        parsed = ast.literal_eval(candidate)
        if isinstance(parsed, tuple) and len(parsed) == 2:
            return candidate
    except Exception:
        return None
    return None


def extract_action(vlm_output: str) -> Tuple[str, bool]:
    """Extract the first action tuple from VLM output text.

    Uses a balanced-parenthesis walker (not a simple regex) so nested
    payloads like ``('mark', (0.5, 0.5))`` and ``('swap', ((0,0),(1,1)))``
    are extracted correctly.

    Falls back through several strategies:
      1. Balanced extraction on the raw text (quoted action name)
      2. Strip markdown formatting, retry balanced extraction
      3. Unquoted action name – inject quotes, retry
      4. JSON ``{"action": "...", "args": ...}`` -> tuple string

    Returns:
        ``(action_string, matched)``.  When ``matched`` is True the returned
        string is guaranteed to pass ``ast.literal_eval`` and produce a
        2-tuple.  When False the stripped raw input is returned.
    """
    # 1. Balanced extraction – quoted action name
    result = _try_extract_balanced(vlm_output, _TUPLE_START_RE)
    if result:
        return result, True

    # 2. Strip markdown formatting, retry
    stripped = _strip_markdown(vlm_output)
    if stripped != vlm_output:
        result = _try_extract_balanced(stripped, _TUPLE_START_RE)
        if result:
            return result, True

    # 3. Unquoted action name – e.g. (move, 0) instead of ('move', 0)
    for text in (vlm_output, stripped):
        for match in _TUPLE_START_UNQUOTED_RE.finditer(text):
            candidate = _find_balanced_tuple(text, match.start())
            if candidate is None:
                continue
            action_name = match.group(1)
            fixed = candidate.replace(action_name, f"'{action_name}'", 1)
            try:
                parsed = ast.literal_eval(fixed)
                if isinstance(parsed, tuple) and len(parsed) == 2:
                    return fixed, True
            except Exception:
                continue

    # 4. JSON fallback
    for text in (vlm_output, stripped):
        result = _try_json_fallback(text)
        if result:
            return result, True

    return vlm_output.strip(), False


# ---------------------------------------------------------------------------
# Keyword action extraction (instruct recipe)
# ---------------------------------------------------------------------------

VALID_ACTIONS = frozenset({"left", "right", "up", "down", "stop"})

_ACTION_TAG_RE = re.compile(r"<action>\s*(\w+)\s*</action>", re.IGNORECASE)

_KEYWORD_TO_TUPLE: Dict[str, str] = {
    "right": "('move', 0)",
    "up": "('move', 1)",
    "left": "('move', 2)",
    "down": "('move', 3)",
    "stop": "('stop', 'stop')",
}


def extract_relaxed_action(vlm_output: str) -> Tuple[str, bool]:
    """Extract a keyword action from ``<action>keyword</action>`` tags.

    Looks for the *last* ``<action>`` tag in the output (the model may
    reason about actions earlier in its response).

    Returns:
        ``(tuple_string, matched)``.  When ``matched`` is True the returned
        string is a valid tuple string for ``maze_2d.step()`` (e.g.
        ``"('move', 0)"``).  When False the raw input is returned.
    """
    matches = list(_ACTION_TAG_RE.finditer(vlm_output))
    if not matches:
        return vlm_output.strip(), False

    keyword = matches[-1].group(1).lower()
    if keyword not in VALID_ACTIONS:
        return vlm_output.strip(), False

    return _KEYWORD_TO_TUPLE[keyword], True

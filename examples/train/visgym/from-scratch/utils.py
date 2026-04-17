import base64
import io
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
# Relaxed action extraction
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

"""Normalise a capture document so two runs of one scenario compare equal.

Everything the exporters and `/v1` emit is deterministic given the same
inputs, except for three kinds of value that are fresh every run: generated
ids, clocks, and the numbers derived from clocks. This replaces each with a
stable placeholder so an export can be pinned to a file and compared -- which
is the behavioural contract the refactor is held to.

Ids are numbered by first appearance, per prefix, so `<tr#1>` is "the first
trajectory id this document mentioned" and the same node keeps the same
placeholder wherever it recurs, including inside longer strings such as a
path id or a session id.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

GOLDEN_ROOT = Path(__file__).parent / "golden"

# Every id `ids.py` mints, plus the mock upstream's own response ids.
_ID = re.compile(r"\b(tr|ex|nd|exp|bat)_[0-9A-HJKMNP-TV-Z]{26}\b")
_MOCK_ID = re.compile(r"\b(chatcmpl|resp|msg)[-_]mock\d{6}\b")
_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(\+00:00|Z)?$")
# The mock upstream binds a free port per test, and the object store lives in
# a temporary directory. Neither is behaviour.
_LOOPBACK_PORT = re.compile(r"127\.0\.0\.1:\d+")
# Keys whose values are clock readings, derived from them, or point at where
# this run happened to put something. Presence is kept; the value is not.
_VOLATILE_KEY_SUFFIXES = ("_at", "_ms", "_ns", "_uri", "_per_second")
# `indexed_trajectories` counts the whole record directory rather than the
# scenario, so it moves with whatever else a stack captured.
_VOLATILE_KEYS = frozenset(
    {"clock_epoch", "checksum", "byte_count", "created", "updated_at", "indexed_trajectories"}
)


class Normaliser:
    def __init__(self) -> None:
        self._ids: dict[str, str] = {}
        self._counts: dict[str, int] = {}

    def _placeholder(self, match: re.Match[str]) -> str:
        raw = match.group(0)
        if raw not in self._ids:
            prefix = match.group(1)
            self._counts[prefix] = self._counts.get(prefix, 0) + 1
            self._ids[raw] = f"<{prefix}#{self._counts[prefix]}>"
        return self._ids[raw]

    def text(self, value: str) -> str:
        value = _ID.sub(self._placeholder, value)
        value = _MOCK_ID.sub(lambda m: f"<{m.group(1)}-mock>", value)
        value = _LOOPBACK_PORT.sub("127.0.0.1:<port>", value)
        if _ISO.match(value):
            return "<time>"
        return value

    def __call__(self, value: Any, *, key: str | None = None) -> Any:
        if key is not None and (
            key in _VOLATILE_KEYS or key.endswith(_VOLATILE_KEY_SUFFIXES)
        ):
            # A timestamp, a duration, or a byte count: keep whether it was
            # present, drop what it said.
            return None if value is None else "<volatile>"
        if isinstance(value, dict):
            return {k: self(v, key=k) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [self(item) for item in value]
        if isinstance(value, str):
            return self.text(value)
        return value


def normalise(document: Any) -> Any:
    return Normaliser()(document)


def check(name: str, document: Any, *, kind: str = "exports") -> None:
    """Compare against `tests/golden/<kind>/<name>.json`, or write it.

    A missing golden is written and the test passes: that is how a new
    scenario is pinned. Set `CAPTURE_REGEN_GOLDEN=1` to rewrite them all,
    and read the diff before committing it -- a rewritten golden is a
    behaviour change being accepted.
    """
    path = GOLDEN_ROOT / kind / f"{name}.json"
    actual = normalise(document)
    rendered = json.dumps(actual, indent=1, sort_keys=False) + "\n"
    if os.environ.get("CAPTURE_REGEN_GOLDEN") or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered)
        return
    expected = json.loads(path.read_text())
    if actual != expected:
        entries = diff(expected, actual)
        shown = "\n  ".join(entries[:40]) + ("\n  ..." if len(entries) > 40 else "")
        raise AssertionError(
            f"{path.relative_to(GOLDEN_ROOT.parent)} differs from what this run produced "
            f"({len(entries)} leaves; golden != actual):\n  {shown}\n"
            "If the change is intended, rerun with CAPTURE_REGEN_GOLDEN=1 and review the diff."
        )


def diff(left: Any, right: Any, *, path: str = "$") -> list[str]:
    """Every leaf where two normalised documents disagree, as `path: a != b`."""
    if isinstance(left, dict) and isinstance(right, dict):
        out: list[str] = []
        for key in sorted(set(left) | set(right)):
            if key not in left:
                out.append(f"{path}.{key}: <absent> != {right[key]!r}")
            elif key not in right:
                out.append(f"{path}.{key}: {left[key]!r} != <absent>")
            else:
                out.extend(diff(left[key], right[key], path=f"{path}.{key}"))
        return out
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [f"{path}: {len(left)} items != {len(right)} items"]
        out = []
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            out.extend(diff(a, b, path=f"{path}[{index}]"))
        return out
    if left != right:
        return [f"{path}: {left!r} != {right!r}"]
    return []


Scenario = Callable[..., Any]

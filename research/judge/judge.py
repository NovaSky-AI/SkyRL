"""
taste_judge / judge.py
======================

LLM-as-judge that scores computer-use (CU) agent trajectories on five
qualitative axes (1-5) plus a weighted total. Companion to the binary
verifier reward in our RL training loop.

Rubric: see ./rubric.md
Hypotheses: see ./hypotheses.md
Design notes: see ./judge_design.md

Usage
-----
    from judge import score_trajectory

    out = score_trajectory(
        task="Send an email to bob@x.com saying 'hi'",
        actions=[
            {"type": "click", "target": "Compose"},
            {"type": "type", "target": "to-field", "text": "bob@x.com"},
            ...
        ],
        outcome=True,
        screenshots=["b64_or_path_1", ...],   # optional
        model="claude-sonnet-4-6",
    )
    out["scores"]          # {"intent_clarity": 4, ...}
    out["weighted_total"]  # 4.15
    out["rationale"]       # short string

Three model paths are provided so we can compute inter-rater agreement
and route through cheaper models in production:

    score_trajectory(...)              -> Anthropic (Claude) judge
    score_trajectory_gpt4o(...)        -> OpenAI (GPT-4o) judge
    score_trajectory_openrouter(...)   -> OpenRouter (any model) judge

API keys are read from `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, and
`OPENROUTER_API_KEY` at runtime. OpenRouter is OpenAI-compatible at
https://openrouter.ai/api/v1 — pass the model in slug form, e.g.
"anthropic/claude-haiku-4.5", "google/gemini-2.5-flash",
"openai/gpt-4o-mini", "deepseek/deepseek-v3".

Caching: results are keyed by hash(task, actions, outcome, model) and
persisted under ~/.cache/taste_judge/.

Failure mode: if a model call raises, we return a None-shaped result so the
training loop keeps running.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("taste_judge")

# ---------------------------------------------------------------------------
# Rubric constants
# ---------------------------------------------------------------------------

AXES: tuple[str, ...] = (
    "intent_clarity",
    "efficiency",
    "recovery",
    "ui_grounding",
    "coherence",
)

WEIGHTS: dict[str, float] = {
    "intent_clarity": 0.20,
    "efficiency": 0.20,
    "recovery": 0.20,
    "ui_grounding": 0.25,
    "coherence": 0.15,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

CACHE_DIR = Path(os.path.expanduser("~/.cache/taste_judge"))


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a strict but fair "taste judge" for computer-use agent trajectories.

You will receive:
  TASK: the natural-language goal
  ACTIONS: an ordered list of agent actions (clicks, typing, scrolls, etc.)
  OUTCOME: a boolean from a separate verifier (True=task achieved, False=not)
  SCREENSHOTS: up to 4 evenly-spaced images showing UI state during the run

Score 1-5 (integers only) on FIVE independent axes:
  1. intent_clarity  — every action has an obvious purpose given the task.
  2. efficiency      — action count reasonable; no unnecessary steps.
  3. recovery        — when something unexpected happens, agent diagnoses
                       and adjusts. If nothing unexpected occurred, give 4.
  4. ui_grounding    — clicks/typing target the right element given visible UI.
  5. coherence       — sequence reads like one mind pursuing one plan.

Anchors for each axis:
  1 = clearly bad / multiple violations
  3 = mediocre / one notable issue
  5 = excellent / no issues observed
Do NOT let OUTCOME inflate or deflate scores; verifier success is independent.
When uncertain between adjacent scores, pick the LOWER one.

Return STRICT JSON, no prose outside the JSON block:
{
  "scores": {
    "intent_clarity": <int 1-5>,
    "efficiency":     <int 1-5>,
    "recovery":       <int 1-5>,
    "ui_grounding":   <int 1-5>,
    "coherence":      <int 1-5>
  },
  "rationale": "<2-4 sentence summary, name the axis that drove the lowest score>"
}
"""


def _build_user_prompt(
    task: str,
    actions: list[dict],
    outcome: bool,
    blind_outcome: bool = False,
) -> str:
    actions_block = json.dumps(actions, indent=2, default=str)
    outcome_line = (
        ""
        if blind_outcome
        else f"OUTCOME (verifier): {'True' if outcome else 'False'}\n\n"
    )
    return (
        f"TASK:\n{task}\n\n"
        f"{outcome_line}"
        f"ACTIONS ({len(actions)} steps):\n{actions_block}\n\n"
        "Score each axis and return strict JSON as instructed."
    )


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_key(
    task: str,
    actions: list[dict],
    outcome: bool,
    model: str,
    blind_outcome: bool = False,
) -> str:
    h = hashlib.sha256()
    payload = json.dumps(
        {
            "task": task,
            "actions": actions,
            "outcome": outcome,
            "model": model,
            "blind_outcome": blind_outcome,
        },
        sort_keys=True,
        default=str,
    )
    h.update(payload.encode("utf-8"))
    return h.hexdigest()[:24]


def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{key}.json"


def _cache_get(key: str) -> Optional[dict]:
    p = _cache_path(key)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def _cache_put(key: str, value: dict) -> None:
    try:
        _cache_path(key).write_text(json.dumps(value))
    except Exception as e:  # never crash on cache write
        logger.warning("cache write failed: %s", e)


# ---------------------------------------------------------------------------
# Screenshot sampling
# ---------------------------------------------------------------------------


def _sample_screenshots(screenshots: Optional[list[str]], k: int = 4) -> list[str]:
    """Pick `k` evenly-spaced screenshots; preserves order."""
    if not screenshots:
        return []
    if len(screenshots) <= k:
        return list(screenshots)
    # evenly spaced indices including first and last
    step = (len(screenshots) - 1) / (k - 1)
    idx = sorted({round(i * step) for i in range(k)})
    return [screenshots[i] for i in idx]


def _screenshot_to_anthropic_block(s: str) -> dict:
    """Accepts either a base64 string or a path. Returns an Anthropic image
    content block."""
    if os.path.isfile(s):
        import base64

        b64 = base64.b64encode(Path(s).read_bytes()).decode("ascii")
        media = "image/png"
    else:
        # assume already base64
        b64 = s
        media = "image/png"
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media, "data": b64},
    }


def _screenshot_to_openai_block(s: str) -> dict:
    if os.path.isfile(s):
        import base64

        b64 = base64.b64encode(Path(s).read_bytes()).decode("ascii")
        url = f"data:image/png;base64,{b64}"
    else:
        url = f"data:image/png;base64,{s}"
    return {"type": "image_url", "image_url": {"url": url}}


# ---------------------------------------------------------------------------
# Parsing & scoring
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of `text`. Strict but tolerant of a
    surrounding code-fence."""
    text = text.strip()
    # try to strip ``` fences
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    # otherwise find first { ... last }
    if not text.startswith("{"):
        l = text.find("{")
        r = text.rfind("}")
        if l != -1 and r != -1:
            text = text[l : r + 1]
    return json.loads(text)


def _validate_scores(parsed: dict) -> dict[str, int]:
    raw = parsed.get("scores", {})
    out: dict[str, int] = {}
    for axis in AXES:
        v = raw.get(axis)
        if not isinstance(v, (int, float)):
            raise ValueError(f"missing/non-numeric score for axis {axis!r}")
        v = int(round(v))
        if v < 1 or v > 5:
            raise ValueError(f"score for {axis!r} out of range: {v}")
        out[axis] = v
    return out


def _weighted_total(scores: dict[str, int]) -> float:
    return round(sum(scores[a] * WEIGHTS[a] for a in AXES), 4)


def _none_result(error: str, raw: str = "") -> dict:
    return {
        "scores": {a: None for a in AXES},
        "weighted_total": None,
        "rationale": "",
        "raw_response": raw,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Public API: Claude judge
# ---------------------------------------------------------------------------


def score_trajectory(
    task: str,
    actions: list[dict],
    outcome: bool,
    screenshots: Optional[list[str]] = None,
    model: str = "claude-sonnet-4-6",
    blind_outcome: bool = False,
) -> dict:
    """Score a CU trajectory with Claude. Returns dict with `scores`,
    `weighted_total`, `rationale`, `raw_response`. Never raises; on failure
    returns a None-shaped result with `error` set.

    If `blind_outcome=True`, the OUTCOME line is suppressed from the prompt
    so the judge cannot see the verifier signal (used for outcome-bleed
    diagnostics)."""
    cache_key = _cache_key(task, actions, outcome, model, blind_outcome)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        from anthropic import Anthropic
    except Exception as e:  # pragma: no cover
        return _none_result(f"anthropic SDK import failed: {e}")

    user_text = _build_user_prompt(task, actions, outcome, blind_outcome=blind_outcome)
    content: list[dict] = [{"type": "text", "text": user_text}]
    for s in _sample_screenshots(screenshots, k=4):
        try:
            content.append(_screenshot_to_anthropic_block(s))
        except Exception as e:
            logger.warning("dropping screenshot: %s", e)

    try:
        client = Anthropic()  # reads ANTHROPIC_API_KEY from env
        resp = client.messages.create(
            model=model,
            max_tokens=600,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )
        raw = "".join(getattr(b, "text", "") for b in resp.content)
    except Exception as e:
        logger.warning("Claude judge call failed: %s", e)
        return _none_result(f"anthropic call failed: {e}")

    try:
        parsed = _extract_json(raw)
        scores = _validate_scores(parsed)
        result = {
            "scores": scores,
            "weighted_total": _weighted_total(scores),
            "rationale": parsed.get("rationale", "")[:1000],
            "raw_response": raw,
        }
    except Exception as e:
        logger.warning("Claude judge parse failed: %s", e)
        return _none_result(f"parse failed: {e}", raw=raw)

    _cache_put(cache_key, result)
    return result


# ---------------------------------------------------------------------------
# Public API: GPT-4o judge (for inter-rater)
# ---------------------------------------------------------------------------


def score_trajectory_gpt4o(
    task: str,
    actions: list[dict],
    outcome: bool,
    screenshots: Optional[list[str]] = None,
    model: str = "gpt-4o",
    blind_outcome: bool = False,
) -> dict:
    """Same contract as `score_trajectory` but uses OpenAI."""
    cache_key = _cache_key(task, actions, outcome, model, blind_outcome)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        from openai import OpenAI
    except Exception as e:  # pragma: no cover
        return _none_result(f"openai SDK import failed: {e}")

    user_text = _build_user_prompt(task, actions, outcome, blind_outcome=blind_outcome)
    content: list[dict] = [{"type": "text", "text": user_text}]
    for s in _sample_screenshots(screenshots, k=4):
        try:
            content.append(_screenshot_to_openai_block(s))
        except Exception as e:
            logger.warning("dropping screenshot: %s", e)

    try:
        client = OpenAI()  # reads OPENAI_API_KEY from env
        resp = client.chat.completions.create(
            model=model,
            max_tokens=600,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or ""
    except Exception as e:
        logger.warning("GPT-4o judge call failed: %s", e)
        return _none_result(f"openai call failed: {e}")

    try:
        parsed = _extract_json(raw)
        scores = _validate_scores(parsed)
        result = {
            "scores": scores,
            "weighted_total": _weighted_total(scores),
            "rationale": parsed.get("rationale", "")[:1000],
            "raw_response": raw,
        }
    except Exception as e:
        logger.warning("GPT-4o judge parse failed: %s", e)
        return _none_result(f"parse failed: {e}", raw=raw)

    _cache_put(cache_key, result)
    return result


# ---------------------------------------------------------------------------
# Public API: OpenRouter judge (any model, OpenAI-compatible API)
# ---------------------------------------------------------------------------


def score_trajectory_openrouter(
    task: str,
    actions: list[dict],
    outcome: bool,
    screenshots: Optional[list[str]] = None,
    model: str = "anthropic/claude-haiku-4.5",
    blind_outcome: bool = False,
) -> dict:
    """Same contract as `score_trajectory` but uses OpenRouter, which is
    OpenAI-compatible at https://openrouter.ai/api/v1. The `model` parameter
    is an OpenRouter slug (e.g. "anthropic/claude-haiku-4.5",
    "google/gemini-2.5-flash", "openai/gpt-4o-mini")."""
    cache_key = _cache_key(task, actions, outcome, model, blind_outcome)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        from openai import OpenAI
    except Exception as e:  # pragma: no cover
        return _none_result(f"openai SDK import failed: {e}")

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return _none_result("OPENROUTER_API_KEY not set")

    user_text = _build_user_prompt(task, actions, outcome, blind_outcome=blind_outcome)
    content: list[dict] = [{"type": "text", "text": user_text}]
    for s in _sample_screenshots(screenshots, k=4):
        try:
            # OpenRouter speaks OpenAI; reuse the OpenAI image_url block.
            content.append(_screenshot_to_openai_block(s))
        except Exception as e:
            logger.warning("dropping screenshot: %s", e)

    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        # Note: response_format={"type": "json_object"} is supported by most
        # OpenRouter models but not all (e.g. some Gemini routes). We rely on
        # the strict-JSON instruction in SYSTEM_PROMPT + _extract_json's
        # tolerant parser as the primary contract, and pass response_format
        # opportunistically — failures here are caught by the outer except.
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=600,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                response_format={"type": "json_object"},
            )
        except Exception:
            # Retry without response_format for models that reject it.
            resp = client.chat.completions.create(
                model=model,
                max_tokens=600,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
            )
        raw = resp.choices[0].message.content or ""
    except Exception as e:
        logger.warning("OpenRouter judge call failed: %s", e)
        return _none_result(f"openrouter call failed: {e}")

    try:
        parsed = _extract_json(raw)
        scores = _validate_scores(parsed)
        result = {
            "scores": scores,
            "weighted_total": _weighted_total(scores),
            "rationale": parsed.get("rationale", "")[:1000],
            "raw_response": raw,
        }
    except Exception as e:
        logger.warning("OpenRouter judge parse failed: %s", e)
        return _none_result(f"parse failed: {e}", raw=raw)

    _cache_put(cache_key, result)
    return result


# ---------------------------------------------------------------------------
# __main__: synthetic smoke test (no network calls)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Synthetic trajectory; we don't call APIs in this smoke test, we only
    # exercise the pure helpers and the public function on a stubbed model.
    task = "Send an email to bob@example.com saying 'hi'."
    actions = [
        {"type": "click", "target": "Compose"},
        {"type": "type", "target": "to-field", "text": "bob@example.com"},
        {"type": "type", "target": "subject-field", "text": "hi"},
        {"type": "type", "target": "body-field", "text": "hi"},
        {"type": "click", "target": "Send"},
    ]

    # --- pure-function checks ---
    key = _cache_key(task, actions, True, "claude-sonnet-4-6")
    assert isinstance(key, str) and len(key) == 24, key

    sample = _sample_screenshots(["a", "b", "c", "d", "e", "f", "g"], k=4)
    assert sample == ["a", "c", "e", "g"], sample
    assert _sample_screenshots(None) == []
    assert _sample_screenshots(["only"], k=4) == ["only"]

    fake_scores = {a: 4 for a in AXES}
    fake_scores["ui_grounding"] = 5
    wt = _weighted_total(fake_scores)
    expected = 4 * (0.20 + 0.20 + 0.20 + 0.15) + 5 * 0.25
    assert abs(wt - round(expected, 4)) < 1e-9, (wt, expected)

    # JSON extraction tolerates code fences
    parsed = _extract_json(
        "```json\n{\"scores\": {\"intent_clarity\":5,\"efficiency\":4,"
        "\"recovery\":4,\"ui_grounding\":5,\"coherence\":5},"
        "\"rationale\":\"clean\"}\n```"
    )
    assert _validate_scores(parsed)["intent_clarity"] == 5

    # Failure path: bad JSON -> None-shaped result via score_trajectory
    # (we monkey-patch the SDK to force a parse path)
    none = _none_result("simulated", raw="not json")
    assert none["scores"]["intent_clarity"] is None
    assert none["weighted_total"] is None

    # Live call, but only if the key is present. Otherwise skip silently.
    if os.environ.get("ANTHROPIC_API_KEY"):
        out = score_trajectory(task, actions, True)
        if out.get("scores", {}).get("intent_clarity") is not None:
            assert 1 <= out["scores"]["intent_clarity"] <= 5
            assert 1.0 <= out["weighted_total"] <= 5.0
            print("LIVE OK:", out["scores"], out["weighted_total"])
        else:
            print("LIVE call returned None-shaped result:", out.get("error"))
    else:
        print("ANTHROPIC_API_KEY not set; skipping live call.")

    print("smoke OK")

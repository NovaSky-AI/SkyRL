"""
score_dataset.py
================

CLI: load N trajectories from `FleetAI/fleet-cu-trajectories`, score each
with one or more judges in parallel, persist a parquet, and (when running
multiple judges) print inter-rater Cohen's kappa per axis.

Usage:
    python score_dataset.py --n 100 --out scored.parquet
    python score_dataset.py --n 100 --provider openrouter \
        --model anthropic/claude-haiku-4.5
    python score_dataset.py --n 100 --provider all   # claude + gpt + openrouter

Env:
    HF_TOKEN              required to load the gated dataset
    ANTHROPIC_API_KEY     required for Claude judge
    OPENAI_API_KEY        required for GPT-4o judge
    OPENROUTER_API_KEY    required for OpenRouter judge
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

# Allow `import judge` whether run from this dir or from elsewhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge import (  # noqa: E402
    AXES,
    score_trajectory,
    score_trajectory_gpt4o,
    score_trajectory_openrouter,
)


PROVIDER_DEFAULT_MODEL = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o",
    "openrouter": "anthropic/claude-haiku-4.5",
}


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_trajectories(n: int) -> list[dict]:
    """Pull N rows from FleetAI/fleet-cu-trajectories. Returns dicts with
    keys: trajectory_id, task, actions, outcome, screenshots."""
    from datasets import load_dataset

    ds = load_dataset(
        "FleetAI/fleet-cu-trajectories",
        split="train",
        token=os.environ.get("HF_TOKEN"),
        streaming=True,
    )
    rows: list[dict] = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        rows.append(_normalize_row(row, fallback_id=f"row_{i:05d}"))
    return rows


def _normalize_row(row: dict, fallback_id: str) -> dict:
    """Adapter for the FleetAI/fleet-cu-trajectories schema (and a few others).

    Real Fleet schema fields: session_id, env_key, task_key, model, score,
    outcome ('pass'/'fail'), num_turns, num_screenshots, conversation
    (JSON-encoded list of {role, text, has_image, position}), images
    (JSON-encoded list of paths).
    """
    tid = (
        row.get("trajectory_id")
        or row.get("session_id")
        or row.get("id")
        or row.get("episode_id")
        or fallback_id
    )

    # Outcome: prefer numeric score, then string outcome, then bool fields.
    outcome = None
    if row.get("score") is not None:
        try:
            outcome = float(row["score"]) > 0.5
        except Exception:
            outcome = None
    if outcome is None and isinstance(row.get("outcome"), str):
        outcome = row["outcome"].lower() in ("pass", "success", "true", "1", "win")
    if outcome is None:
        raw = row.get("verifier_score", row.get("success", 0))
        try:
            outcome = bool(int(raw))
        except Exception:
            outcome = bool(raw)

    # Conversation may be a JSON string OR an already-parsed list.
    conv = row.get("conversation") or row.get("messages") or row.get("trajectory")
    if isinstance(conv, str):
        try:
            conv = json.loads(conv)
        except Exception:
            conv = None
    msgs: list[dict] = conv if isinstance(conv, list) else []

    # Task: first user-message text. Fall back to top-level fields.
    task = row.get("task") or row.get("instruction") or row.get("goal") or ""
    if not task:
        for m in msgs:
            if isinstance(m, dict) and m.get("role") == "user":
                t = m.get("text") or m.get("content") or ""
                if isinstance(t, str) and len(t) > 5:
                    task = t
                    break

    # Actions: prefer top-level lists; otherwise extract from assistant turns.
    actions: list[dict] = []
    top_actions = row.get("actions") or row.get("steps")
    if isinstance(top_actions, list) and top_actions:
        actions = [a for a in top_actions if isinstance(a, dict)]
    else:
        TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            if role not in ("assistant", "agent", "model", "tool"):
                continue
            # 1. structured tool_calls field
            tcs = m.get("tool_calls") or []
            if isinstance(tcs, list):
                for tc in tcs:
                    if not isinstance(tc, dict):
                        continue
                    name = tc.get("name") or (tc.get("function", {}) or {}).get("name") or "unknown"
                    args = tc.get("arguments") or (tc.get("function", {}) or {}).get("arguments") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {"raw": args}
                    actions.append({"name": str(name), "arguments": args if isinstance(args, dict) else {}})
            # 2. <tool_call>{...}</tool_call> blocks inside text
            txt = m.get("text") or m.get("content") or ""
            if isinstance(txt, str) and "<tool_call>" in txt:
                for mm in TOOL_RE.finditer(txt):
                    raw = mm.group(1)
                    try:
                        obj = json.loads(raw)
                    except Exception:
                        obj = None
                    if isinstance(obj, dict):
                        name = obj.get("name") or obj.get("tool") or "unknown"
                        args = obj.get("arguments") or obj.get("params") or {}
                        actions.append({"name": str(name), "arguments": args if isinstance(args, dict) else {}})
            # 3. assistant text without tool_calls — keep as a "thought" pseudo-action
            elif role == "assistant" and isinstance(txt, str) and txt.strip():
                actions.append({"name": "think", "arguments": {"text": txt[:600]}})

    # Screenshots: dataset has `images` as a JSON string list of paths.
    screenshots = row.get("screenshots")
    if not screenshots:
        imgs = row.get("images")
        if isinstance(imgs, str):
            try:
                imgs = json.loads(imgs)
            except Exception:
                imgs = None
        if isinstance(imgs, list) and imgs:
            screenshots = imgs

    return {
        "trajectory_id": str(tid),
        "task": task[:2000] if isinstance(task, str) else "",
        "actions": actions,
        "outcome": bool(outcome),
        "screenshots": screenshots,
    }


# ---------------------------------------------------------------------------
# Scoring (parallel)
# ---------------------------------------------------------------------------


PROVIDER_PREFIX = {
    "anthropic": "claude",
    "openai": "gpt",
    "openrouter": "openrouter",
}


def _empty_provider_block(prefix: str) -> dict:
    return {
        f"{prefix}_total": None,
        f"{prefix}_rationale": "",
        f"{prefix}_error": None,
        **{f"{prefix}_{a}": None for a in AXES},
    }


def _score_one_provider(provider: str, model: str, row: dict) -> dict:
    """Run a single judge call. Errors swallow into a None-shaped block so
    we never let one bad row sink the whole batch."""
    prefix = PROVIDER_PREFIX[provider]
    fn_kwargs = dict(
        task=row["task"],
        actions=row["actions"],
        outcome=row["outcome"],
        screenshots=row["screenshots"],
    )
    try:
        if provider == "anthropic":
            res = score_trajectory(model=model, **fn_kwargs)
        elif provider == "openai":
            res = score_trajectory_gpt4o(model=model, **fn_kwargs)
        elif provider == "openrouter":
            res = score_trajectory_openrouter(model=model, **fn_kwargs)
        else:
            raise ValueError(f"unknown provider: {provider}")
    except Exception as e:
        block = _empty_provider_block(prefix)
        block[f"{prefix}_error"] = f"score raised: {e}"
        return block

    block = {
        f"{prefix}_total": res.get("weighted_total"),
        f"{prefix}_rationale": res.get("rationale", ""),
        f"{prefix}_error": res.get("error"),
    }
    for axis in AXES:
        block[f"{prefix}_{axis}"] = (res.get("scores") or {}).get(axis)
    return block


def _score_one(row: dict, providers: list[tuple[str, str]]) -> dict:
    """Score a single row with each (provider, model) tuple. Each judge call
    is itself blocking; we parallelize *across rows* below, so doing the
    judges sequentially per row keeps things simple and rate-limit friendly.
    """
    out: dict = {
        "trajectory_id": row["trajectory_id"],
        "verifier_score": int(bool(row["outcome"])),
    }
    for provider, model in providers:
        out.update(_score_one_provider(provider, model, row))
    return out


def score_all(
    rows: list[dict],
    providers: list[tuple[str, str]],
    workers: int = 8,
) -> pd.DataFrame:
    results: list[dict] = []
    empty_blocks: dict = {}
    for provider, _ in providers:
        empty_blocks.update(_empty_provider_block(PROVIDER_PREFIX[provider]))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_score_one, r, providers): r["trajectory_id"] for r in rows
        }
        for fut in as_completed(futures):
            tid = futures[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                # never let one row kill the whole run
                row_block = {
                    "trajectory_id": tid,
                    "verifier_score": None,
                    **empty_blocks,
                }
                for provider, _ in providers:
                    prefix = PROVIDER_PREFIX[provider]
                    row_block[f"{prefix}_error"] = f"score_one failed: {e}"
                results.append(row_block)
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Cohen's kappa
# ---------------------------------------------------------------------------


def cohen_kappa(a: list[int], b: list[int]) -> float:
    """Linear Cohen's kappa for ordinal labels (1-5). Returns NaN if there
    isn't enough data."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 2:
        return float("nan")
    # build label set
    labels = sorted({x for x, _ in pairs} | {y for _, y in pairs})
    if len(labels) < 2:
        return float("nan")
    n = len(pairs)
    po = sum(1 for x, y in pairs if x == y) / n
    # marginals
    pa: dict[int, float] = {l: 0.0 for l in labels}
    pb: dict[int, float] = {l: 0.0 for l in labels}
    for x, y in pairs:
        pa[x] += 1
        pb[y] += 1
    pa = {k: v / n for k, v in pa.items()}
    pb = {k: v / n for k, v in pb.items()}
    pe = sum(pa[l] * pb[l] for l in labels)
    if pe >= 1.0:
        return float("nan")
    return (po - pe) / (1.0 - pe)


def print_inter_rater(df: pd.DataFrame) -> None:
    """Print pairwise Cohen's kappa for any pair of judges actually present
    in the dataframe (claude vs gpt, claude vs openrouter, gpt vs openrouter).
    """
    available = [
        prefix for prefix in PROVIDER_PREFIX.values() if f"{prefix}_total" in df.columns
    ]
    if len(available) < 2:
        return

    pairs: list[tuple[str, str]] = []
    for i, p in enumerate(available):
        for q in available[i + 1 :]:
            pairs.append((p, q))

    for p, q in pairs:
        print(f"\n=== Inter-rater agreement: {p} vs {q} (Cohen's kappa) ===")
        for axis in AXES:
            a = df[f"{p}_{axis}"].tolist()
            b = df[f"{q}_{axis}"].tolist()
            k = cohen_kappa(a, b)
            n = sum(1 for x, y in zip(a, b) if x is not None and y is not None)
            print(f"  {axis:>16s}: kappa = {k:.3f}  (n={n})")

    # mean weighted_total per judge
    print()
    means = {p: df[f"{p}_total"].dropna().mean() for p in available}
    pretty = "  ".join(f"{p}={v:.3f}" for p, v in means.items())
    print(f"  mean weighted_total: {pretty}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="number of trajectories")
    ap.add_argument(
        "--out", type=str, default="scored.parquet", help="output parquet path"
    )
    ap.add_argument(
        "--workers", type=int, default=8, help="thread-pool size for scoring"
    )
    ap.add_argument(
        "--provider",
        type=str,
        default="anthropic",
        choices=["anthropic", "openai", "openrouter", "all"],
        help="judge backend to use; 'all' runs all three in parallel for inter-rater",
    )
    ap.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "model name for the chosen provider (default: provider-specific). "
            "Ignored when --provider all."
        ),
    )
    args = ap.parse_args()

    if not os.environ.get("HF_TOKEN"):
        print("WARNING: HF_TOKEN not set; dataset load will likely fail.", file=sys.stderr)

    if args.provider == "all":
        providers: list[tuple[str, str]] = [
            (p, PROVIDER_DEFAULT_MODEL[p])
            for p in ("anthropic", "openai", "openrouter")
        ]
    else:
        model = args.model or PROVIDER_DEFAULT_MODEL[args.provider]
        providers = [(args.provider, model)]

    print(f"Loading {args.n} trajectories from FleetAI/fleet-cu-trajectories...")
    rows = load_trajectories(args.n)
    print(f"Loaded {len(rows)} rows.")

    pretty = ", ".join(f"{p}({m})" for p, m in providers)
    print(f"Scoring with {args.workers} workers via: {pretty}")
    df = score_all(rows, providers=providers, workers=args.workers)

    out_path = Path(args.out)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")

    print_inter_rater(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

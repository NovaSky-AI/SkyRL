#!/usr/bin/env python3
"""Claude trajectory viewer — run with: python3 viewer.py [port]

Shows the 200 Claude trajectories from fleet-cu-claude-trajectories/
with ablation judge scores from research/judge/ablation_results.csv.
"""

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from flask import Flask, jsonify, request, send_file, abort, Response

BASE = Path(__file__).parent
CLAUDE_TRAJECTORIES_FILE = BASE / "fleet-cu-claude-trajectories" / "claude_trajectories.jsonl"
CLAUDE_IMAGES_DIR = BASE / "fleet-cu-claude-trajectories" / "images"
JUDGE_SCORES_CSV = BASE / "research" / "judge" / "ablation_results.csv"

JUDGE_AXES = ("intent_clarity", "efficiency", "recovery", "ui_grounding", "coherence")
JUDGE_CONFIGS = ("actions_only", "screenshots_only", "actions_and_screenshots")

# ---------------------------------------------------------------------------
# Verifier failure classification
# ---------------------------------------------------------------------------

_SUCCESS_KWS = (
    "successfully completed", "all tasks have been completed",
    "completed successfully", "i was able to successfully",
    "tasks have been completed", "i have completed",
    "i successfully completed", "all three tasks have been",
    "booking confirmed", "i created a wishlist", "i applied the filters",
    "i have adjusted", "i adjusted the timeline", "successfully added",
    "all items have been", "all 3 appliances",
    "i found and selected", "i found and booked",
)
_FAIL_KWS = (
    "unable to complete", "could not complete", "was not able to",
    "i was unable", "unable to find", "unable to access",
    "could not find", "the feature is not", "not available in this version",
    "the snooze feature", "the travel folder", "both drafts folders",
    "i cannot", "not able to",
    # structured grader responses that say fail
    '"overall_status": "fail"',
    # cloner-grader admits failure
    "the task could not be completed", "the task cannot be completed",
    "could not be completed", "cannot be completed",
    "does not have a", "there is no existing",
    "is not available in",
)

FAIL_CLASS_META = {
    "false_neg_claimed_success": {
        "label": "False Negative — Claimed Success",
        "signal": "disagrees",
        "signal_label": "Judge likely disagrees",
        "desc": "Agent submitted claiming the task was done; binary verifier rejected. "
                "A taste judge sees high-quality trajectory steps and would score 3–5, "
                "flagging verifier over-strictness (exact-match on field values, amounts, etc.).",
    },
    "partial_credit": {
        "label": "Partial Credit Missed",
        "signal": "disagrees",
        "signal_label": "Judge likely disagrees",
        "desc": "Verifier assigns a partial score (0 < score < 1) on multi-step tasks "
                "but still marks outcome=fail. A graduated taste judge would give proportional "
                "credit — the binary 0 is misleading for RL reward.",
    },
    "ambiguous_submit": {
        "label": "Ambiguous Submit",
        "signal": "partial",
        "signal_label": "Judge may disagree",
        "desc": "Agent submitted a structured answer or partial description without "
                "explicit success/failure language. May be a correct answer in the wrong "
                "format, or a partial attempt — judge can distinguish.",
    },
    "timeout_no_submit": {
        "label": "Timeout / No Submit",
        "signal": "partial",
        "signal_label": "Judge can rate progress",
        "desc": "Agent hit the turn limit without ever calling submit_final_answer. "
                "Verifier correctly fails, but a judge can assess whether the agent was "
                "making meaningful progress before running out of turns.",
    },
    "genuine_fail_admitted": {
        "label": "Genuine Fail (Admitted)",
        "signal": "agrees",
        "signal_label": "Judge agrees",
        "desc": "Agent explicitly admitted it could not complete the task. "
                "Verifier and judge both score this as a failure — but the judge can "
                "explain the quality of the attempt and identify the root cause.",
    },
    "potential_false_positive": {
        "label": "Potential False Positive",
        "signal": "disagrees",
        "signal_label": "Judge likely disagrees",
        "desc": "Verifier passes (score=1.0) but best judge score ≤ 0.45. "
                "Agent reached the correct final DB state via a poor path — "
                "excessive turns, wrong constraint, or a lucky completion.",
    },
}


FP_JUDGE_THRESHOLD = 0.45  # pass records with best judge score below this are flagged
_TOP_N_REPRESENTATIVE = 3   # how many top FN / FP cases to surface as "representative"

# Hand-written explanations for representative showcase cases.
VERIFIER_NOTES = {
    # FN: invoicing task — agent claimed success but verifier rejected
    "f9a5a048-a7e1-45c9-b994-5df202901e00": (
        "Verifier checks exact invoice-number-per-order assignment and status note text. "
        "Agent mapped SO-2300→INV-1200 … SO-2304→INV-1204 with 'SHIPPED' notes, but the DB "
        "validator likely requires a specific note format (e.g. 'Order marked shipped') or "
        "strict numbering sequence matching the seed state."
    ),
    # FN: receipt-creation task — agent saved file and uploaded but verifier rejected
    "6e57a372-cd0a-4dd5-96aa-9df46f25041f": (
        "Verifier checks the saved file path, ODT/PDF field positions, and Float upload linkage. "
        "Agent saved to ~/Desktop; verifier likely expects a specific directory, exact receipt "
        "line-item layout, or a Float attachment record that doesn't match the seeded expectation."
    ),
    # FN: 3-part finance task — agent completed all three steps but verifier rejected
    "b6de7e71-b2d1-44ee-96eb-996be53dee5f": (
        "Three independent verifier checks: (1) credit card payment from max-balance account "
        "excluding 8275 — verifier checks exact payment amount ($940.02) or confirmation number; "
        "(2) Funnel vendor entry — checks a specific field value; (3) QuickBooks bill payment — "
        "one check likely failed on exact amount or vendor name format."
    ),
    # Partial credit (score=0.90): QBR package with 10-field JSON
    "2f39c7de-c97c-4931-9ecf-36693840e6c3": (
        "9/10 sub-checks pass (verifier score=0.90). Pipeline value, CRM win/loss counts, and "
        "Jira epics all correct. The 1 missed check is likely an exact Outlook leadership-channel "
        "count or a specific CRM dollar-figure format — binary outcome=fail despite 90% completion."
    ),
    # Partial credit (score=0.90): win/loss analysis with 10-field JSON
    "6674eda8-a936-4afe-b628-4a3265c9aafb": (
        "9/10 sub-checks pass (verifier score=0.90). Win/loss counts and overall win rate correct. "
        "The 1 missed check is likely a rounding difference in deal-type win rates "
        "(Renewal 73.9%, New Business 65.5%) or a subtle date-range filter — "
        "deserves 90% reward, not binary 0."
    ),
}


def _best_judge_score(judge_scores: dict):
    scores = [v["score"] for v in judge_scores.values() if v.get("score") is not None]
    return max(scores) if scores else None


def _mark_representatives(rows: list) -> None:
    """Tag the most illustrative FN and FP cases with t['representative']."""
    # FN: fail records whose judge score is highest (verifier most wrong)
    fn_cands = [((_best_judge_score(t.get("judge_scores") or {})) or 0, t)
                for t in rows if t.get("fail_class") in {"false_neg_claimed_success", "partial_credit"}
                and t.get("judge_scores")]
    fn_cands.sort(key=lambda x: -x[0])
    seen: set = set()
    fn_count = 0
    for score, t in fn_cands:
        sid = t["session_id"]
        if sid not in seen and fn_count < _TOP_N_REPRESENTATIVE:
            t["representative"] = "fn"
            t["verifier_note"] = VERIFIER_NOTES.get(sid)
            seen.add(sid)
            fn_count += 1

    # Partial credit: top 2 by highest verifier partial score (binary 0 is most misleading)
    pc_cands = sorted(
        [t for t in rows if t.get("fail_class") == "partial_credit"],
        key=lambda x: -x.get("score", 0),
    )
    pc_count = 0
    for t in pc_cands:
        if t.get("representative") is None and pc_count < 2:
            t["representative"] = "fn"
            t["verifier_note"] = VERIFIER_NOTES.get(t["session_id"])
            pc_count += 1

    # FP: pass records whose judge score is lowest (most egregious lucky pass)
    fp_cands = [(((_best_judge_score(t.get("judge_scores") or {})) or 1), t)
                for t in rows if t.get("fail_class") == "potential_false_positive"
                and t.get("judge_scores")]
    fp_cands.sort(key=lambda x: x[0])
    seen = set()
    fp_count = 0
    for score, t in fp_cands:
        sid = t["session_id"]
        if sid not in seen and fp_count < _TOP_N_REPRESENTATIVE:
            t["representative"] = "fp"
            seen.add(sid)
            fp_count += 1


def _get_submit_answer(conv: list) -> "str | None":
    """Return the answer argument of the final submit_final_answer call, or None."""
    for msg in reversed(conv):
        if msg.get("role") != "assistant":
            continue
        for tc in (msg.get("tool_calls") or []):
            fn = (tc.get("function") or {}).get("name", "")
            if fn == "submit_final_answer":
                args_raw = (tc.get("function") or {}).get("arguments", "{}")
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except Exception:
                    args = {}
                return str(args.get("answer", ""))
    return None


def _classify_fail(t: dict) -> "str | None":
    """Classify a fail record into one of 5 verifier failure classes.
    Must be called on the raw (un-normalized) trajectory dict."""
    if t.get("outcome") != "fail":
        return None
    score = t.get("score") or 0.0
    if 0 < score < 1.0:
        return "partial_credit"
    ans = _get_submit_answer(t.get("conversation") or [])
    if ans is None:
        return "timeout_no_submit"
    ans_l = ans.lower()
    if any(kw in ans_l for kw in _SUCCESS_KWS):
        return "false_neg_claimed_success"
    if any(kw in ans_l for kw in _FAIL_KWS):
        return "genuine_fail_admitted"
    return "ambiguous_submit"

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Judge scores loader
# ---------------------------------------------------------------------------

def load_judge_scores() -> dict:
    """Load ablation_results.csv → {session_id: {config: {score, axes, rationale}}}"""
    index: dict = defaultdict(dict)
    if not JUDGE_SCORES_CSV.exists():
        return index
    with open(JUDGE_SCORES_CSV, newline="") as f:
        for row in csv.DictReader(f):
            sid = row["session_id"]
            cfg = row["config"]
            score_str = row.get("judge_score", "")
            try:
                score = float(score_str) if score_str not in ("", "None") else None
            except ValueError:
                score = None
            axes = {}
            for ax in JUDGE_AXES:
                v = row.get(ax, "")
                try:
                    axes[ax] = int(v) if v not in ("", "None") else None
                except (ValueError, TypeError):
                    axes[ax] = None
            index[sid][cfg] = {
                "score": score,
                "axes": axes,
                "rationale": row.get("rationale", ""),
            }
    return dict(index)


# ---------------------------------------------------------------------------
# Conversation normalisation (Claude → Qwen-compatible format)
# ---------------------------------------------------------------------------

def _normalize_claude_conv(conv: list) -> list:
    """Convert Claude conversation messages to the Qwen-like viewer format.

    Qwen format: {role, text, has_image, position}
    Claude format: {role, content: str|list, position, tool_calls, tool_call_id}
    """
    out = []
    for msg in conv:
        role = msg.get("role", "")
        content = msg.get("content")
        norm: dict = {"role": role, "position": msg.get("position", 0)}

        if isinstance(content, list):
            # User message with image_url blocks; text content may also be present
            texts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
            norm["text"] = " ".join(texts).strip() or None
            norm["has_image"] = any(
                isinstance(c, dict) and c.get("type") == "image_url"
                for c in content
            )
        elif isinstance(content, str):
            norm["text"] = content if content.strip() else None
            norm["has_image"] = False
        else:
            norm["text"] = None
            norm["has_image"] = False

        out.append(norm)
    return out


# ---------------------------------------------------------------------------
# Trajectory loaders
# ---------------------------------------------------------------------------

def load_claude_trajectories(judge_scores: dict) -> list:
    rows = []
    if not CLAUDE_TRAJECTORIES_FILE.exists():
        return rows
    with open(CLAUDE_TRAJECTORIES_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            # Map Claude outcome field to Qwen-like pass/fail
            outcome_raw = t.get("outcome", "")
            t["outcome"] = "pass" if outcome_raw == "success" else "fail"
            # Classify failure mode BEFORE normalising (raw conv has tool_calls)
            t["fail_class"] = _classify_fail(t)
            # Normalise conversation (strips tool_calls, extracts text/has_image)
            if isinstance(t.get("conversation"), list):
                t["conversation"] = _normalize_claude_conv(t["conversation"])
            # Attach judge scores
            t["judge_scores"] = judge_scores.get(t["session_id"], {})
            t["_dataset"] = "claude"
            rows.append(t)
    # Second pass: tag pass records whose best judge score is suspiciously low
    for t in rows:
        if t["outcome"] == "pass" and t["fail_class"] is None:
            js = t.get("judge_scores", {})
            scores = [v["score"] for v in js.values() if v.get("score") is not None]
            if scores and max(scores) < FP_JUDGE_THRESHOLD:
                t["fail_class"] = "potential_false_positive"
    # Third pass: mark the most illustrative cases
    _mark_representatives(rows)
    return rows


# ---------------------------------------------------------------------------
# Startup: load everything
# ---------------------------------------------------------------------------

print("Loading judge scores...", end="", flush=True)
JUDGE_SCORES = load_judge_scores()
print(f" {sum(len(v) for v in JUDGE_SCORES.values())} scored entries.")

print("Loading Claude trajectories...", end="", flush=True)
TRAJECTORIES = load_claude_trajectories(JUDGE_SCORES)
SESSION_INDEX = {t["session_id"]: t for t in TRAJECTORIES}
print(f" {len(TRAJECTORIES)} loaded.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_reasoning(conv: list) -> bool:
    return any(
        m.get("text") and not m["text"].strip().startswith(("{", "[", "<tool_call>"))
        for m in conv
        if m.get("role") == "assistant"
    )


# _best_judge_score is defined earlier alongside _mark_representatives


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.route("/api/trajectories")
def list_trajectories():
    page = int(request.args.get("page", 1))
    limit = int(request.args.get("limit", 50))
    env_filter = request.args.get("env", "")
    outcome_filter = request.args.get("outcome", "")
    fail_class_filter = request.args.get("fail_class", "")
    has_images_filter = request.args.get("has_images", "")
    has_reasoning_filter = request.args.get("has_reasoning", "")

    rows = TRAJECTORIES
    if env_filter:
        rows = [t for t in rows if t.get("env_key", "") == env_filter]
    if outcome_filter:
        rows = [t for t in rows if t.get("outcome", "") == outcome_filter]
    if fail_class_filter:
        rows = [t for t in rows if t.get("fail_class") == fail_class_filter]
    if has_images_filter == "true":
        rows = [t for t in rows if t.get("image_paths")]
    if has_reasoning_filter == "true":
        rows = [t for t in rows if _has_reasoning(t.get("conversation") or [])]

    total = len(rows)
    start = (page - 1) * limit
    slice_ = rows[start: start + limit]

    summaries = []
    for t in slice_:
        conv = t.get("conversation") or []
        judge_scores = t.get("judge_scores", {})
        s: dict = {
            "session_id": t["session_id"],
            "env_key": t.get("env_key", ""),
            "task_key": t.get("task_key", ""),
            "model": t.get("model", ""),
            "score": t.get("score", 0),
            "outcome": t.get("outcome", ""),
            "fail_class": t.get("fail_class"),
            "num_turns": t.get("num_turns", 0),
            "num_screenshots": t.get("num_screenshots", 0),
            "has_images": bool(t.get("image_paths")),
            "has_reasoning": _has_reasoning(conv),
            "representative": t.get("representative"),
        }
        if judge_scores:
            s["judge_scores"] = {
                cfg: {"score": v["score"]} for cfg, v in judge_scores.items()
            }
            s["best_judge_score"] = _best_judge_score(judge_scores)
        summaries.append(s)

    return jsonify({"total": total, "page": page, "limit": limit, "trajectories": summaries})


@app.route("/api/trajectories/envs")
def list_envs():
    envs = sorted(set(t.get("env_key", "") for t in TRAJECTORIES if t.get("env_key")))
    return jsonify(envs)


@app.route("/api/trajectories/<session_id>")
def get_trajectory(session_id):
    t = SESSION_INDEX.get(session_id)
    if not t:
        abort(404)
    return jsonify(t)


@app.route("/api/representative_cases")
def representative_cases_api():
    """Return the top representative FN and FP cases with full summary info."""
    fn_cases, fp_cases = [], []
    for t in TRAJECTORIES:
        rep = t.get("representative")
        if not rep:
            continue
        js = t.get("judge_scores", {})
        entry = {
            "session_id": t["session_id"],
            "env_key": t.get("env_key", ""),
            "task_key": t.get("task_key", ""),
            "outcome": t.get("outcome", ""),
            "score": t.get("score", 0),
            "fail_class": t.get("fail_class"),
            "num_turns": t.get("num_turns", 0),
            "best_judge_score": _best_judge_score(js),
            "judge_scores": {cfg: {"score": v["score"]} for cfg, v in js.items() if v.get("score") is not None},
            "verifier_note": t.get("verifier_note"),
        }
        (fn_cases if rep == "fn" else fp_cases).append(entry)
    # Sort: FN by highest judge score, FP by lowest
    fn_cases.sort(key=lambda x: -(x["best_judge_score"] or 0))
    fp_cases.sort(key=lambda x: (x["best_judge_score"] or 1))
    return jsonify({"fn": fn_cases, "fp": fp_cases})


@app.route("/api/failure_analysis")
def failure_analysis_api():
    fail_recs = [t for t in TRAJECTORIES if t.get("outcome") == "fail"]
    pass_recs = [t for t in TRAJECTORIES if t.get("outcome") == "pass"]
    class_counts: dict = {}
    class_env: dict = {}
    # Count fail classes (fails + potential_false_positive on passes)
    for r in TRAJECTORIES:
        cls = r.get("fail_class")
        if not cls:
            continue
        class_counts[cls] = class_counts.get(cls, 0) + 1
        env = r.get("env_key", "unknown")
        class_env.setdefault(cls, {})[env] = class_env.setdefault(cls, {}).get(env, 0) + 1
    return jsonify({
        "total": len(TRAJECTORIES),
        "pass": len(pass_recs),
        "fail": len(fail_recs),
        "fp": sum(1 for t in pass_recs if t.get("fail_class") == "potential_false_positive"),
        "classes": class_counts,
        "class_env": class_env,
        "meta": FAIL_CLASS_META,
    })


@app.route("/api/images/<path:image_path>")
def serve_image(image_path):
    full = (CLAUDE_IMAGES_DIR / image_path).resolve()
    if not str(full).startswith(str(CLAUDE_IMAGES_DIR.resolve())):
        abort(403)
    if not full.exists():
        abort(404)
    suffix = full.suffix.lower()
    mime = {"jpeg": "image/jpeg", "jpg": "image/jpeg", "png": "image/png",
            "gif": "image/gif", "webp": "image/webp"}.get(suffix.lstrip("."), "image/jpeg")
    return send_file(full, mimetype=mime)


# ---------------------------------------------------------------------------
# Frontend (SPA)
# ---------------------------------------------------------------------------

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Claude Trajectory Viewer</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0a0a0f; color: #e2e8f0; font-family: system-ui, sans-serif; font-size: 14px; }
  a { color: #60a5fa; text-decoration: none; }
  a:hover { color: #93c5fd; }

  #app { display: flex; flex-direction: column; min-height: 100vh; }
  .topbar { background: #111827; border-bottom: 1px solid #1f2937; padding: 12px 24px; display: flex; align-items: center; gap: 16px; position: sticky; top: 0; z-index: 100; }
  .topbar h1 { font-size: 16px; font-weight: 600; }
  .topbar .nav { display: flex; gap: 12px; margin-left: auto; }
  .topbar .nav a { font-size: 13px; color: #9ca3af; }
  .topbar .nav a:hover { color: #e2e8f0; }

  #list-view { padding: 24px; max-width: 1400px; margin: 0 auto; width: 100%; }
  .filters { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px; align-items: center; }
  .filters input, .filters select { background: #1f2937; border: 1px solid #374151; color: #e2e8f0; border-radius: 6px; padding: 6px 12px; font-size: 13px; outline: none; }
  .filters input:focus, .filters select:focus { border-color: #4b6bfb; }
  .filters label { display: flex; align-items: center; gap: 6px; font-size: 13px; color: #9ca3af; cursor: pointer; }
  .total { margin-left: auto; font-size: 12px; color: #6b7280; }

  table { width: 100%; border-collapse: collapse; border: 1px solid #1f2937; border-radius: 8px; overflow: hidden; }
  th { background: #111827; color: #6b7280; font-weight: 500; text-transform: uppercase; font-size: 11px; letter-spacing: 0.05em; padding: 10px 14px; text-align: left; }
  td { padding: 10px 14px; border-top: 1px solid #1f2937; vertical-align: middle; }
  tr:hover td { background: #111827; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 500; font-family: monospace; }
  .badge-pass { background: rgba(16,185,129,0.15); color: #34d399; }
  .badge-fail { background: rgba(239,68,68,0.15); color: #f87171; }
  .badge-env { background: #1f2937; color: #9ca3af; }
  .badge-claude { background: rgba(251,191,36,0.12); color: #fbbf24; }
  .badge-qwen { background: rgba(99,102,241,0.12); color: #818cf8; }
  .mono { font-family: monospace; font-size: 12px; }

  .judge-score { display: inline-block; font-family: monospace; font-size: 11px; padding: 2px 6px; border-radius: 3px; }
  .judge-score-hi { background: rgba(16,185,129,0.15); color: #34d399; }
  .judge-score-mid { background: rgba(251,191,36,0.12); color: #fbbf24; }
  .judge-score-lo { background: rgba(239,68,68,0.12); color: #f87171; }
  .judge-score-null { color: #374151; }

  .pagination { display: flex; align-items: center; justify-content: space-between; margin-top: 16px; }
  .btn { background: #1f2937; border: 1px solid #374151; color: #e2e8f0; border-radius: 6px; padding: 6px 16px; font-size: 13px; cursor: pointer; }
  .btn:hover:not(:disabled) { background: #374151; }
  .btn:disabled { opacity: 0.4; cursor: default; }

  #detail-view { display: none; }
  .detail-header { background: #111827; border-bottom: 1px solid #1f2937; padding: 12px 24px; position: sticky; top: 49px; z-index: 50; }
  .detail-header .meta { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
  .detail-header .session-id { font-family: monospace; font-size: 11px; color: #4b5563; margin-top: 4px; }
  .detail-body { display: flex; gap: 24px; padding: 24px; max-width: 1400px; margin: 0 auto; }
  .side-panel { width: 280px; flex-shrink: 0; display: flex; flex-direction: column; gap: 16px; }
  .task-card { background: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 16px; position: sticky; top: 110px; }
  .task-card h3 { font-size: 11px; font-weight: 600; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 10px; }
  .task-card p { font-size: 13px; color: #cbd5e1; line-height: 1.6; white-space: pre-wrap; }
  .task-card .task-key { font-family: monospace; font-size: 10px; color: #374151; margin-top: 12px; padding-top: 12px; border-top: 1px solid #1f2937; word-break: break-all; }

  .judge-card { background: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 16px; }
  .judge-card h3 { font-size: 11px; font-weight: 600; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px; }
  .judge-config-block { margin-bottom: 14px; }
  .judge-config-label { font-size: 11px; font-weight: 600; color: #9ca3af; margin-bottom: 6px; display: flex; align-items: center; justify-content: space-between; }
  .judge-axis-row { display: flex; justify-content: space-between; align-items: center; font-size: 11px; color: #6b7280; padding: 2px 0; }
  .judge-axis-name { text-transform: capitalize; }
  .axis-bar-wrap { flex: 1; margin: 0 8px; height: 4px; background: #1f2937; border-radius: 2px; }
  .axis-bar { height: 4px; border-radius: 2px; background: #4b6bfb; }
  .judge-axis-val { font-family: monospace; min-width: 14px; text-align: right; }
  .rationale-text { font-size: 11px; color: #4b5563; margin-top: 6px; line-height: 1.5; font-style: italic; }
  .judge-config-sep { border: none; border-top: 1px solid #1f2937; margin: 10px 0; }

  .turns { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 10px; }
  .turn { background: #111827; border: 1px solid #1f2937; border-radius: 8px; overflow: hidden; }
  .turn-header { display: flex; align-items: center; gap: 10px; padding: 10px 14px; cursor: pointer; user-select: none; }
  .turn-header:hover { background: #1a2332; }
  .turn-num { font-family: monospace; font-size: 11px; color: #4b5563; width: 28px; flex-shrink: 0; }
  .turn-summary { flex: 1; min-width: 0; display: flex; align-items: center; gap: 8px; font-size: 12px; overflow: hidden; }
  .turn-summary .action-preview { font-family: monospace; color: #86efac; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .screenshot-badge { color: #60a5fa; font-size: 11px; flex-shrink: 0; }
  .chevron { color: #4b5563; font-size: 10px; flex-shrink: 0; }
  .turn-body { border-top: 1px solid #1f2937; display: none; }
  .turn-body.open { display: block; }
  .turn-content { display: flex; gap: 16px; padding: 14px; }
  .screenshot-col { flex-shrink: 0; }
  .screenshot-col img { width: 280px; border-radius: 6px; border: 1px solid #374151; cursor: zoom-in; display: block; }
  .screenshot-col img:hover { opacity: 0.9; }
  .no-screenshot { width: 280px; height: 160px; background: #0f172a; border: 1px solid #1f2937; border-radius: 6px; display: flex; align-items: center; justify-content: center; color: #374151; font-size: 12px; }
  .text-col { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 12px; }
  .section-label { font-size: 11px; font-weight: 600; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }
  .action-block { background: #0a0f1e; border: 1px solid #1f3a6e; border-radius: 6px; padding: 10px 12px; font-family: monospace; font-size: 12px; color: #86efac; white-space: pre-wrap; word-break: break-all; max-height: 240px; overflow-y: auto; }
  .result-block { background: #0a0a0f; border: 1px solid #1f2937; border-radius: 6px; padding: 10px 12px; font-family: monospace; font-size: 12px; color: #94a3b8; }
  .reasoning-block { background: rgba(251,191,36,0.07); border: 1px solid rgba(251,191,36,0.22); border-radius: 6px; padding: 10px 12px; font-size: 13px; color: #fcd34d; line-height: 1.65; white-space: pre-wrap; word-break: break-word; }
  .reasoning-badge { display: inline-block; background: rgba(251,191,36,0.12); color: #fbbf24; border-radius: 3px; padding: 1px 6px; font-size: 10px; font-weight: 600; letter-spacing: 0.04em; margin-left: 6px; vertical-align: middle; }
  .loading { padding: 48px; text-align: center; color: #6b7280; }
  .error { padding: 48px; text-align: center; color: #f87171; }

  #lightbox { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.9); z-index: 1000; align-items: center; justify-content: center; cursor: zoom-out; }
  #lightbox.open { display: flex; }
  #lightbox img { max-width: 95vw; max-height: 95vh; border-radius: 4px; }

  /* Judge score column headers toggle */
  .th-judge { font-size: 10px; }

  /* Failure analysis view */
  #failures-view { display: none; padding: 24px; max-width: 1400px; margin: 0 auto; width: 100%; }
  .class-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin-bottom: 28px; }
  .class-card { background: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 16px; cursor: pointer; transition: border-color 0.15s; }
  .class-card:hover { border-color: #374151; }
  .class-card.selected { border-color: #4b6bfb; background: rgba(75,107,251,0.08); }
  .class-card.signal-disagrees { border-left: 3px solid #f87171; }
  .class-card.signal-partial   { border-left: 3px solid #fbbf24; }
  .class-card.signal-agrees    { border-left: 3px solid #34d399; }
  .class-count { font-size: 34px; font-weight: 700; color: #e2e8f0; line-height: 1; margin-bottom: 6px; }
  .class-label { font-size: 11px; font-weight: 600; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 8px; }
  .class-desc  { font-size: 12px; color: #6b7280; line-height: 1.55; margin-bottom: 10px; }
  .signal-badge { display: inline-block; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 3px; }
  .signal-badge.disagrees { background: rgba(248,113,113,0.15); color: #f87171; }
  .signal-badge.partial   { background: rgba(251,191,36,0.15);  color: #fbbf24; }
  .signal-badge.agrees    { background: rgba(52,211,153,0.12);  color: #34d399; }
  .env-chips { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 8px; }
  .env-chip  { font-size: 10px; color: #4b5563; background: rgba(255,255,255,0.04); border-radius: 3px; padding: 1px 5px; }
  .fcls-badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 500; font-family: monospace; }
  .fcls-false_neg_claimed_success { background: rgba(248,113,113,0.15); color: #f87171; }
  .fcls-partial_credit            { background: rgba(251,191,36,0.15);  color: #fbbf24; }
  .fcls-ambiguous_submit          { background: rgba(139,92,246,0.15);  color: #a78bfa; }
  .fcls-genuine_fail_admitted     { background: rgba(107,114,128,0.10); color: #6b7280; }
  .fcls-timeout_no_submit         { background: rgba(107,114,128,0.10); color: #6b7280; }
  .fcls-potential_false_positive  { background: rgba(99,102,241,0.15);  color: #818cf8; }
  .fail-stat-bar { display: flex; gap: 20px; margin-bottom: 24px; }
  .fail-stat { font-size: 13px; color: #6b7280; }
  .fail-stat .n { font-weight: 700; font-size: 16px; color: #e2e8f0; margin-right: 4px; }
  .fail-stat .n.green { color: #34d399; }
  .fail-stat .n.red   { color: #f87171; }
  /* Representative showcase */
  .showcase { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 28px; }
  .showcase-group { background: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 14px; }
  .showcase-group h4 { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 10px; display: flex; align-items: center; gap: 8px; }
  .showcase-group.fn-group h4 { color: #f87171; }
  .showcase-group.fp-group h4 { color: #818cf8; }
  .showcase-row { display: flex; flex-direction: column; gap: 5px; padding: 8px 0; border-bottom: 1px solid #1a2030; cursor: pointer; }
  .showcase-row:last-child { border-bottom: none; }
  .showcase-row:hover { background: rgba(255,255,255,0.02); border-radius: 4px; }
  .showcase-row-top { display: flex; align-items: center; gap: 10px; }
  .showcase-star { color: #fbbf24; font-size: 12px; flex-shrink: 0; }
  .showcase-env  { font-size: 11px; color: #9ca3af; font-family: monospace; flex-shrink: 0; min-width: 110px; }
  .showcase-meta { flex: 1; min-width: 0; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  .showcase-gap  { font-size: 11px; font-family: monospace; }
  .showcase-gap.fn-gap { color: #34d399; }
  .showcase-gap.fp-gap { color: #f87171; }
  .showcase-turns { font-size: 11px; color: #4b5563; }
  .showcase-note { font-size: 11px; color: #6b7280; line-height: 1.5; padding-left: 20px; border-left: 2px solid #1f2937; margin-left: 4px; }
  /* Star badge in table */
  .rep-star { color: #fbbf24; margin-right: 4px; font-size: 11px; }
  tr.is-rep td { background: rgba(251,191,36,0.04); }
  tr.is-rep:hover td { background: rgba(251,191,36,0.08); }
</style>
</head>
<body>
<div id="app">
  <div class="topbar">
    <h1>Claude Trajectory Viewer</h1>
    <div class="nav">
      <a href="#" onclick="showList(); return false;" id="nav-list">← List</a>
      <a href="#" onclick="showFailures(); return false;" style="color:#a78bfa">Failures</a>
    </div>
  </div>

  <div id="list-view">
    <div class="filters">
      <input id="env-filter" type="text" placeholder="Filter by env..." oninput="onFilterChange()" style="width:160px" />
      <select id="outcome-filter" onchange="onFilterChange()">
        <option value="">All outcomes</option>
        <option value="pass">pass</option>
        <option value="fail">fail</option>
      </select>
      <label>
        <input type="checkbox" id="images-only" onchange="onFilterChange()" />
        Has images
      </label>
      <label>
        <input type="checkbox" id="reasoning-only" onchange="onFilterChange()" />
        Has reasoning
      </label>
      <span class="total" id="total-label"></span>
    </div>
    <div id="table-container"><div class="loading">Loading…</div></div>
    <div class="pagination">
      <button class="btn" id="prev-btn" onclick="changePage(-1)" disabled>Previous</button>
      <span id="page-label" style="color:#6b7280;font-size:13px;"></span>
      <button class="btn" id="next-btn" onclick="changePage(1)">Next</button>
    </div>
  </div>

  <div id="failures-view">
    <div style="margin-bottom:20px">
      <h2 style="font-size:20px;font-weight:700;color:#e2e8f0;margin-bottom:6px">Verifier Failure Analysis</h2>
      <p style="font-size:13px;color:#6b7280">200 <span style="color:#9ca3af;font-family:monospace">claude-opus-4.7</span> trajectories — where the binary <code style="font-size:11px;background:#1f2937;padding:1px 6px;border-radius:3px">outcome</code> signal diverges from actual agent behavior (and where a taste judge adds signal)</p>
    </div>
    <div class="fail-stat-bar" id="fail-stat-bar"></div>
    <div class="class-grid" id="class-grid"></div>
    <div class="showcase" id="showcase-grid"></div>
    <div style="margin:4px 0 12px;display:flex;align-items:center;gap:12px">
      <h3 style="font-size:12px;font-weight:600;color:#4b5563;text-transform:uppercase;letter-spacing:0.05em">Trajectories</h3>
    </div>
    <div class="filters">
      <select id="fail-class-filter" onchange="onFailFilterChange()">
        <option value="">All fail classes</option>
        <option value="false_neg_claimed_success">false_neg_claimed_success</option>
        <option value="partial_credit">partial_credit</option>
        <option value="ambiguous_submit">ambiguous_submit</option>
        <option value="genuine_fail_admitted">genuine_fail_admitted</option>
        <option value="timeout_no_submit">timeout_no_submit</option>
        <option value="potential_false_positive">potential_false_positive</option>
      </select>
      <input id="fail-env-filter" type="text" placeholder="Filter by env…" oninput="onFailFilterChange()" style="width:150px" />
      <select id="fail-outcome-filter" onchange="onFailFilterChange()">
        <option value="">All outcomes</option>
        <option value="pass">pass</option>
        <option value="fail">fail</option>
      </select>
      <span class="total" id="fail-total-label"></span>
    </div>
    <div id="fail-table-container"><div class="loading">Loading…</div></div>
    <div class="pagination">
      <button class="btn" id="fail-prev-btn" onclick="changeFailPage(-1)" disabled>Previous</button>
      <span id="fail-page-label" style="color:#6b7280;font-size:13px"></span>
      <button class="btn" id="fail-next-btn" onclick="changeFailPage(1)">Next</button>
    </div>
  </div>

  <div id="detail-view">
    <div class="detail-header">
      <div class="meta" id="detail-meta"></div>
      <div class="session-id" id="detail-session-id"></div>
    </div>
    <div class="detail-body">
      <div class="side-panel">
        <div class="task-card">
          <h3>Task</h3>
          <p id="detail-task"></p>
          <div class="task-key" id="detail-task-key"></div>
        </div>
        <div class="judge-card" id="judge-card" style="display:none">
          <h3>Judge Scores</h3>
          <div id="judge-card-body"></div>
        </div>
      </div>
      <div class="turns" id="detail-turns"></div>
    </div>
  </div>
</div>

<div id="lightbox" onclick="closeLightbox()">
  <img id="lightbox-img" src="" alt="" />
</div>

<script>
const JUDGE_CONFIGS = ['actions_only', 'screenshots_only', 'actions_and_screenshots'];
const JUDGE_AXES = ['intent_clarity', 'efficiency', 'recovery', 'ui_grounding', 'coherence'];
const CONFIG_LABELS = {
  'actions_only': 'Actions only',
  'screenshots_only': 'Screenshots only',
  'actions_and_screenshots': 'Actions + Screenshots',
};

let state = {
  page: 1, limit: 50, total: 0,
  envFilter: '', outcomeFilter: '',
  imagesOnly: false, reasoningOnly: false
};

function onFilterChange() {
  state.page = 1;
  state.envFilter = document.getElementById('env-filter').value.trim();
  state.outcomeFilter = document.getElementById('outcome-filter').value;
  state.imagesOnly = document.getElementById('images-only').checked;
  state.reasoningOnly = document.getElementById('reasoning-only').checked;
  fetchList();
}

function changePage(delta) {
  state.page = Math.max(1, Math.min(Math.ceil(state.total / state.limit), state.page + delta));
  fetchList();
}

async function fetchList() {
  const params = new URLSearchParams({ page: state.page, limit: state.limit });
  if (state.envFilter)     params.set('env', state.envFilter);
  if (state.outcomeFilter) params.set('outcome', state.outcomeFilter);
  if (state.imagesOnly)    params.set('has_images', 'true');
  if (state.reasoningOnly) params.set('has_reasoning', 'true');
  document.getElementById('table-container').innerHTML = '<div class="loading">Loading…</div>';
  const res = await fetch('/api/trajectories?' + params);
  const data = await res.json();
  state.total = data.total;
  renderList(data);
}

const JUDGE_CONFIGS_ABBREV = {
  'actions_only': 'act',
  'screenshots_only': 'scr',
  'actions_and_screenshots': 'a+s',
};

function judgeScoreHtml(score) {
  if (score === null || score === undefined) return '<span class="judge-score judge-score-null">—</span>';
  const cls = score >= 0.65 ? 'judge-score-hi' : score >= 0.45 ? 'judge-score-mid' : 'judge-score-lo';
  return `<span class="judge-score ${cls}">${score.toFixed(2)}</span>`;
}

function renderList(data) {
  document.getElementById('total-label').textContent = data.total.toLocaleString() + ' trajectories';
  const totalPages = Math.ceil(data.total / state.limit) || 1;
  document.getElementById('page-label').textContent = `Page ${state.page} of ${totalPages}`;
  document.getElementById('prev-btn').disabled = state.page <= 1;
  document.getElementById('next-btn').disabled = state.page >= totalPages;

  if (!data.trajectories.length) {
    document.getElementById('table-container').innerHTML = '<div class="loading">No results.</div>';
    return;
  }

  const showJudge = data.trajectories.some(t => t.judge_scores);
  const judgeHeaders = showJudge
    ? JUDGE_CONFIGS.map(c => `<th class="th-judge">${JUDGE_CONFIGS_ABBREV[c]}</th>`).join('')
    : '';

  const rows = data.trajectories.map(t => {
    const judgeScoreCells = showJudge
      ? JUDGE_CONFIGS.map(c => {
          const sc = t.judge_scores && t.judge_scores[c] ? t.judge_scores[c].score : null;
          return `<td>${judgeScoreHtml(sc)}</td>`;
        }).join('')
      : '';
    return `
      <tr>
        <td><span class="badge badge-env">${esc(t.env_key)}</span></td>
        <td class="mono"><a href="#" onclick="showDetail('${t.session_id}'); return false;">${esc(t.task_key)}</a></td>
        <td class="mono" style="color:#94a3b8;font-size:11px">${esc(t.model)}</td>
        <td><span class="badge badge-${t.outcome}">${t.outcome}</span></td>
        <td style="color:#94a3b8">${t.score.toFixed(2)}</td>
        <td style="color:#94a3b8">${t.num_turns}</td>
        <td>${t.has_images ? '<span style="color:#34d399;font-size:13px">✓</span>' : '<span style="color:#374151">—</span>'}</td>
        <td>${t.has_reasoning ? '<span class="reasoning-badge">thinking</span>' : '<span style="color:#374151">—</span>'}</td>
        ${judgeScoreCells}
      </tr>`;
  }).join('');

  document.getElementById('table-container').innerHTML = `
    <table>
      <thead><tr>
        <th>Env</th><th>Task</th><th>Model</th><th>Outcome</th>
        <th>Score</th><th>Turns</th><th>Imgs</th><th>Reasoning</th>
        ${judgeHeaders}
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

async function showDetail(sessionId) {
  document.getElementById('list-view').style.display = 'none';
  document.getElementById('failures-view').style.display = 'none';
  document.getElementById('detail-view').style.display = 'block';
  const navList = document.getElementById('nav-list');
  navList.textContent = '← List';
  navList.onclick = (e) => { e.preventDefault(); showList(); };
  document.getElementById('detail-turns').innerHTML = '<div class="loading">Loading…</div>';

  const res = await fetch('/api/trajectories/' + sessionId);
  if (!res.ok) {
    document.getElementById('detail-turns').innerHTML = '<div class="error">Failed to load trajectory.</div>';
    return;
  }
  const traj = await res.json();
  renderDetail(traj);
}

function showList() {
  document.getElementById('detail-view').style.display = 'none';
  document.getElementById('failures-view').style.display = 'none';
  document.getElementById('list-view').style.display = 'block';
}

// ---------------------------------------------------------------------------
// Failures view
// ---------------------------------------------------------------------------

const CLASS_ORDER = [
  'false_neg_claimed_success', 'partial_credit', 'ambiguous_submit',
  'timeout_no_submit', 'genuine_fail_admitted', 'potential_false_positive',
];
const CLASS_LABELS_SHORT = {
  'false_neg_claimed_success': 'false neg: claimed success',
  'partial_credit':            'partial credit',
  'ambiguous_submit':          'ambiguous submit',
  'genuine_fail_admitted':         'genuine fail',
  'timeout_no_submit':             'timeout',
  'potential_false_positive':      'potential FP',
};
const CLASS_META_DEFAULTS = {
  'false_neg_claimed_success':  { label: 'False Negative — Claimed Success', signal: 'disagrees', signal_label: 'Judge likely disagrees', desc: 'Agent submitted claiming success; verifier rejected. A taste judge sees high-quality trajectory steps and rates 3–5.' },
  'partial_credit':             { label: 'Partial Credit Missed',            signal: 'disagrees', signal_label: 'Judge likely disagrees', desc: 'Verifier assigns a partial score (0–1) but still marks outcome=fail. Judge gives graduated credit instead of binary 0.' },
  'ambiguous_submit':           { label: 'Ambiguous Submit',                 signal: 'partial',   signal_label: 'Judge may disagree',    desc: 'Agent submitted a structured answer with no clear success/failure claim. May be a correct answer in the wrong format.' },
  'timeout_no_submit':          { label: 'Timeout / No Submit',              signal: 'partial',   signal_label: 'Judge can rate progress',desc: 'Agent hit the turn limit without submitting. Judge can assess whether meaningful progress was made.' },
  'genuine_fail_admitted':      { label: 'Genuine Fail (Admitted)',          signal: 'agrees',    signal_label: 'Judge agrees',          desc: 'Agent admitted failure. Both verifier and judge agree — but judge explains the quality and root cause of the attempt.' },
  'potential_false_positive':   { label: 'Potential False Positive',         signal: 'disagrees', signal_label: 'Judge likely disagrees', desc: 'Verifier passes (score=1.0) but best judge score ≤ 0.45. Agent reached correct final DB state via a poor or lucky trajectory.' },
};

let failState = { page: 1, limit: 50, total: 0, failClass: '', envFilter: '', outcomeFilter: '' };
let _failAnalysisData = null;
let _repData = null;

async function showFailures() {
  document.getElementById('list-view').style.display = 'none';
  document.getElementById('detail-view').style.display = 'none';
  document.getElementById('failures-view').style.display = 'block';
  if (!_failAnalysisData) {
    const [faRes, repRes] = await Promise.all([
      fetch('/api/failure_analysis'),
      fetch('/api/representative_cases'),
    ]);
    _failAnalysisData = await faRes.json();
    _repData = await repRes.json();
  }
  renderFailAnalysis(_failAnalysisData);
  renderShowcase(_repData);
  fetchFailList();
}

function renderFailAnalysis(data) {
  // Stat bar
  document.getElementById('fail-stat-bar').innerHTML = `
    <div class="fail-stat"><span class="n">${data.total}</span> total</div>
    <div class="fail-stat"><span class="n green">${data.pass}</span> pass</div>
    <div class="fail-stat"><span class="n red">${data.fail}</span> fail</div>
    <div style="width:1px;background:#1f2937;margin:0 4px"></div>
    <div class="fail-stat" title="pass records where best judge score ≤ 0.45"><span class="n" style="color:#818cf8">${data.fp||0}</span> potential FP</div>
    <div class="fail-stat" title="false_neg_claimed_success + partial_credit"><span class="n" style="color:#f87171">${(data.classes.false_neg_claimed_success||0)+(data.classes.partial_credit||0)}</span> false neg</div>`;

  // Class grid
  const meta = data.meta || CLASS_META_DEFAULTS;
  const counts = data.classes || {};
  const envBreakdown = data.class_env || {};
  const cards = CLASS_ORDER.map(key => {
    const m = meta[key] || CLASS_META_DEFAULTS[key] || {};
    const count = counts[key] || 0;
    const envs = envBreakdown[key] || {};
    const envChips = Object.entries(envs).sort((a,b)=>b[1]-a[1]).slice(0,5)
      .map(([e,n])=>`<span class="env-chip">${esc(e)} (${n})</span>`).join('');
    return `<div class="class-card signal-${m.signal||'agrees'}" id="card-${key}" onclick="filterByClass('${key}')">
      <div class="class-count">${count}</div>
      <div class="class-label">${esc(m.label||key)}</div>
      <div class="class-desc">${esc(m.desc||'')}</div>
      <div class="env-chips">${envChips}</div>
      <div style="margin-top:10px"><span class="signal-badge ${m.signal||'agrees'}">${esc(m.signal_label||'')}</span></div>
    </div>`;
  }).join('');
  document.getElementById('class-grid').innerHTML = cards;
}

function renderShowcase(data) {
  const fnCases = data.fn || [];
  const fpCases = data.fp || [];

  function makeRow(t, type) {
    const judgeScore = t.best_judge_score;
    const verifierScore = t.score;
    let gapHtml;
    if (type === 'fn') {
      // Judge high, verifier gives 0 (or partial)
      const judgeStr = judgeScore !== null ? judgeScore.toFixed(2) : '—';
      const verStr = verifierScore > 0 ? `partial ${verifierScore.toFixed(2)}` : 'binary 0';
      gapHtml = `<span class="showcase-gap fn-gap">judge ${judgeStr}</span><span class="showcase-turns">verifier ${verStr}</span>`;
    } else {
      // Judge low, verifier gives 1.0
      const judgeStr = judgeScore !== null ? judgeScore.toFixed(2) : '—';
      gapHtml = `<span class="showcase-gap fp-gap">judge ${judgeStr}</span><span class="showcase-turns">verifier 1.00</span>`;
    }
    const clsBadge = `<span class="fcls-badge fcls-${esc(t.fail_class)}">${esc(CLASS_LABELS_SHORT[t.fail_class] || t.fail_class)}</span>`;
    const noteHtml = t.verifier_note
      ? `<div class="showcase-note">${esc(t.verifier_note)}</div>` : '';
    return `<div class="showcase-row" onclick="showDetailFromFailures('${t.session_id}')">
      <div class="showcase-row-top">
        <span class="showcase-star">★</span>
        <span class="showcase-env">${esc(t.env_key)}</span>
        <div class="showcase-meta">
          ${gapHtml}
          <span class="showcase-turns">${t.num_turns} turns</span>
          ${clsBadge}
        </div>
      </div>
      ${noteHtml}
    </div>`;
  }

  const fnHtml = fnCases.map(t => makeRow(t, 'fn')).join('');
  const fpHtml = fpCases.map(t => makeRow(t, 'fp')).join('');

  document.getElementById('showcase-grid').innerHTML = `
    <div class="showcase-group fn-group">
      <h4>★ False Negatives <span style="font-weight:400;color:#4b5563;text-transform:none;letter-spacing:0">(verifier under-rewards)</span></h4>
      ${fnHtml || '<div style="color:#4b5563;font-size:12px">none scored yet</div>'}
    </div>
    <div class="showcase-group fp-group">
      <h4>★ False Positives <span style="font-weight:400;color:#4b5563;text-transform:none;letter-spacing:0">(verifier over-rewards)</span></h4>
      ${fpHtml || '<div style="color:#4b5563;font-size:12px">none scored yet</div>'}
    </div>`;
}

function filterByClass(cls) {
  document.getElementById('fail-class-filter').value = cls;
  failState.failClass = cls;
  failState.page = 1;
  CLASS_ORDER.forEach(k => {
    const c = document.getElementById('card-' + k);
    if (c) c.classList.toggle('selected', k === cls);
  });
  fetchFailList();
  document.getElementById('fail-table-container').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function onFailFilterChange() {
  failState.page = 1;
  failState.failClass = document.getElementById('fail-class-filter').value;
  failState.envFilter = document.getElementById('fail-env-filter').value.trim();
  failState.outcomeFilter = document.getElementById('fail-outcome-filter').value;
  // Sync card highlights
  CLASS_ORDER.forEach(k => {
    const c = document.getElementById('card-' + k);
    if (c) c.classList.toggle('selected', k === failState.failClass);
  });
  fetchFailList();
}

function changeFailPage(delta) {
  failState.page = Math.max(1, Math.min(Math.ceil(failState.total / failState.limit), failState.page + delta));
  fetchFailList();
}

async function fetchFailList() {
  const params = new URLSearchParams({ page: failState.page, limit: failState.limit });
  if (failState.failClass)    params.set('fail_class', failState.failClass);
  if (failState.envFilter)    params.set('env', failState.envFilter);
  if (failState.outcomeFilter) params.set('outcome', failState.outcomeFilter);
  document.getElementById('fail-table-container').innerHTML = '<div class="loading">Loading…</div>';
  const res = await fetch('/api/trajectories?' + params);
  const data = await res.json();
  failState.total = data.total;
  renderFailList(data);
}

function renderFailList(data) {
  document.getElementById('fail-total-label').textContent = data.total.toLocaleString() + ' trajectories';
  const totalPages = Math.ceil(data.total / failState.limit) || 1;
  document.getElementById('fail-page-label').textContent = `Page ${failState.page} of ${totalPages}`;
  document.getElementById('fail-prev-btn').disabled = failState.page <= 1;
  document.getElementById('fail-next-btn').disabled = failState.page >= totalPages;

  if (!data.trajectories.length) {
    document.getElementById('fail-table-container').innerHTML = '<div class="loading">No results.</div>';
    return;
  }

  const showJudge = data.trajectories.some(t => t.judge_scores);
  const judgeHeaders = showJudge
    ? JUDGE_CONFIGS.map(c => `<th class="th-judge">${JUDGE_CONFIGS_ABBREV[c]}</th>`).join('') : '';

  // Sort: representative rows first
  const sorted = [...data.trajectories].sort((a, b) => (b.representative ? 1 : 0) - (a.representative ? 1 : 0));

  const rows = sorted.map(t => {
    const cls = t.fail_class;
    const clsHtml = cls
      ? `<span class="fcls-badge fcls-${esc(cls)}">${esc(CLASS_LABELS_SHORT[cls] || cls)}</span>`
      : '<span style="color:#374151">—</span>';
    const judgeScoreCells = showJudge
      ? JUDGE_CONFIGS.map(c => {
          const sc = t.judge_scores && t.judge_scores[c] ? t.judge_scores[c].score : null;
          return `<td>${judgeScoreHtml(sc)}</td>`;
        }).join('') : '';
    const starHtml = t.representative ? `<span class="rep-star" title="Representative ${t.representative === 'fn' ? 'false negative' : 'false positive'}">★</span>` : '';
    const repClass = t.representative ? 'is-rep' : '';
    return `<tr class="${repClass}">
      <td><span class="badge badge-env">${esc(t.env_key)}</span></td>
      <td class="mono">${starHtml}<a href="#" onclick="showDetailFromFailures('${t.session_id}'); return false;">${esc(t.task_key)}</a></td>
      <td><span class="badge badge-${t.outcome}">${t.outcome}</span></td>
      <td style="color:#94a3b8">${(t.score||0).toFixed(2)}</td>
      <td>${clsHtml}</td>
      <td style="color:#94a3b8">${t.num_turns}</td>
      ${judgeScoreCells}
    </tr>`;
  }).join('');

  document.getElementById('fail-table-container').innerHTML = `
    <table>
      <thead><tr>
        <th>Env</th><th>Task</th><th>Outcome</th><th>Score</th><th>Fail Class</th><th>Turns</th>
        ${judgeHeaders}
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

async function showDetailFromFailures(sessionId) {
  document.getElementById('failures-view').style.display = 'none';
  document.getElementById('detail-view').style.display = 'block';
  document.getElementById('nav-list').textContent = '← Failures';
  document.getElementById('nav-list').onclick = (e) => { e.preventDefault(); showFailures(); };
  document.getElementById('detail-turns').innerHTML = '<div class="loading">Loading…</div>';
  const res = await fetch('/api/trajectories/' + sessionId);
  if (!res.ok) {
    document.getElementById('detail-turns').innerHTML = '<div class="error">Failed to load.</div>';
    return;
  }
  renderDetail(await res.json());
}

function renderDetail(traj) {
  document.getElementById('detail-session-id').textContent = traj.session_id;
  document.getElementById('detail-meta').innerHTML = `
    <span class="badge badge-env">${esc(traj.env_key || '')}</span>
    <span class="badge badge-${traj.outcome}">${traj.outcome}</span>
    <span style="color:#6b7280;font-size:12px">score: ${(traj.score||0).toFixed(2)}</span>
    <span style="color:#6b7280;font-size:12px">${traj.num_turns||0} turns</span>
    <span style="color:#4b5563;font-size:12px;font-family:monospace">${esc(traj.model||'')}</span>
  `;

  const conv = traj.conversation || [];
  const taskMsg = conv.find(m => m.role === 'user' && m.text && m.text.length > 10);
  document.getElementById('detail-task').textContent = taskMsg ? taskMsg.text : '(no task text)';
  document.getElementById('detail-task-key').textContent = traj.task_key || '';

  // Judge scores panel
  const judgeCard = document.getElementById('judge-card');
  const judgeBody = document.getElementById('judge-card-body');
  const judgeScores = traj.judge_scores || {};
  const hasJudge = Object.keys(judgeScores).length > 0;
  if (hasJudge) {
    judgeCard.style.display = 'block';
    judgeBody.innerHTML = renderJudgeCard(judgeScores);
  } else {
    judgeCard.style.display = 'none';
  }

  const imagePaths = traj.image_paths || [];
  const turns = buildTurns(conv, imagePaths);
  const container = document.getElementById('detail-turns');
  if (!turns.length) {
    container.innerHTML = '<div style="color:#6b7280;font-size:13px">No turns found.</div>';
    return;
  }
  container.innerHTML = turns.map((turn, i) => {
    const imgSrc = turn.screenshot_path
      ? '/api/images/' + turn.screenshot_path.replace(/^images\//, '')
      : null;
    const imgTag = imgSrc
      ? `<img src="${imgSrc}" alt="Step ${i+1}" onclick="openLightbox(this.src)" />`
      : `<div class="no-screenshot">No image</div>`;

    const reasoningBadge = turn.is_reasoning ? `<span class="reasoning-badge">thinking</span>` : '';
    const summaryAction = turn.assistant_text
      ? `<span class="action-preview">${esc(turn.assistant_text.slice(0, 100).replace(/\n/g, ' '))}</span>${reasoningBadge}`
      : `<span style="color:#4b5563;font-size:11px">tool call</span>`;
    const screenshotBadge = imgSrc ? `<span class="screenshot-badge">📸</span>` : '';
    const preBadge = turn.is_pre ? `<span style="color:#6366f1;font-size:10px;font-weight:600;margin-right:4px">PRE</span>` : '';

    let actionHtml;
    if (turn.is_reasoning) {
      actionHtml = `<div class="reasoning-block">${esc(turn.assistant_text)}</div>`;
    } else if (turn.assistant_text) {
      const pretty = tryPrettyJson(turn.assistant_text);
      actionHtml = `<div class="action-block">${esc(pretty)}</div>`;
    } else {
      actionHtml = `<div style="color:#4b5563;font-size:12px;padding:4px 0;">Tool call (no text recorded)</div>`;
    }

    const resultHtml = turn.tool_text
      ? `<div><div class="section-label">Result</div><div class="result-block">${esc(turn.tool_text)}</div></div>`
      : '';

    return `
      <div class="turn" id="turn-${i}">
        <div class="turn-header" onclick="toggleTurn(${i})">
          <span class="turn-num">${preBadge}#${i+1}</span>
          <div class="turn-summary">${screenshotBadge}${summaryAction}</div>
          <span class="chevron" id="chevron-${i}">▼</span>
        </div>
        <div class="turn-body ${i === 0 ? 'open' : ''}" id="turn-body-${i}">
          <div class="turn-content">
            <div class="screenshot-col">${imgTag}</div>
            <div class="text-col">
              <div>
                <div class="section-label">${turn.is_reasoning ? 'Thinking' : 'Agent Action'}</div>
                ${actionHtml}
              </div>
              ${resultHtml}
            </div>
          </div>
        </div>
      </div>`;
  }).join('');
}

function renderJudgeCard(judgeScores) {
  const configs = JUDGE_CONFIGS.filter(c => judgeScores[c]);
  if (!configs.length) return '<div style="color:#4b5563;font-size:12px">No scores available yet.</div>';
  return configs.map((cfg, idx) => {
    const d = judgeScores[cfg];
    const score = d.score;
    const scoreCls = score === null ? 'judge-score-null' : score >= 0.65 ? 'judge-score-hi' : score >= 0.45 ? 'judge-score-mid' : 'judge-score-lo';
    const scoreStr = score !== null && score !== undefined ? score.toFixed(3) : '—';
    const axesHtml = d.axes ? JUDGE_AXES.map(ax => {
      const v = d.axes[ax];
      const pct = v !== null && v !== undefined ? ((v - 1) / 4 * 100) : 0;
      return `<div class="judge-axis-row">
        <span class="judge-axis-name">${ax.replace(/_/g,' ')}</span>
        <div class="axis-bar-wrap"><div class="axis-bar" style="width:${pct}%"></div></div>
        <span class="judge-axis-val">${v !== null && v !== undefined ? v : '—'}</span>
      </div>`;
    }).join('') : '';
    const rationaleHtml = d.rationale
      ? `<div class="rationale-text">${esc(d.rationale)}</div>` : '';
    const sep = idx < configs.length - 1 ? '<hr class="judge-config-sep" />' : '';
    return `<div class="judge-config-block">
      <div class="judge-config-label">
        <span>${CONFIG_LABELS[cfg] || cfg}</span>
        <span class="judge-score ${scoreCls}">${scoreStr}</span>
      </div>
      ${axesHtml}
      ${rationaleHtml}
    </div>${sep}`;
  }).join('');
}

function isReasoningText(text) {
  if (!text) return false;
  const t = text.trim();
  return t.length > 0 && !t.startsWith('{') && !t.startsWith('[') && !t.startsWith('<tool_call>');
}

function buildTurns(conv, imagePaths) {
  const turns = [];
  let imageIndex = 0;
  let i = 0;

  while (i < conv.length && conv[i].role === 'system') i++;
  if (i < conv.length && conv[i].role === 'user' && !conv[i].has_image) i++;

  while (i < conv.length && !(conv[i].role === 'user' && conv[i].has_image)) {
    const msg = conv[i];
    if (msg.role === 'assistant') {
      const turn = { screenshot_path: null, assistant_text: msg.text, tool_text: null, is_pre: true };
      i++;
      if (i < conv.length && conv[i].role === 'tool') { turn.tool_text = conv[i].text; i++; }
      turn.is_reasoning = isReasoningText(turn.assistant_text);
      turns.push(turn);
    } else { i++; }
  }

  while (i < conv.length) {
    const msg = conv[i];
    if (msg.role === 'user' && msg.has_image) {
      const turn = { screenshot_path: imagePaths[imageIndex] || null, assistant_text: null, tool_text: null, is_pre: false };
      imageIndex++;
      let j = i + 1;
      if (j < conv.length && conv[j].role === 'assistant') { turn.assistant_text = conv[j].text; j++; }
      if (j < conv.length && conv[j].role === 'tool') { turn.tool_text = conv[j].text; j++; }
      turn.is_reasoning = isReasoningText(turn.assistant_text);
      turns.push(turn);
      i = j;
      continue;
    }
    i++;
  }
  return turns;
}

function toggleTurn(i) {
  const body = document.getElementById('turn-body-' + i);
  const chev = document.getElementById('chevron-' + i);
  body.classList.toggle('open');
  chev.textContent = body.classList.contains('open') ? '▲' : '▼';
}

function openLightbox(src) {
  document.getElementById('lightbox-img').src = src;
  document.getElementById('lightbox').classList.add('open');
}
function closeLightbox() {
  document.getElementById('lightbox').classList.remove('open');
}

function tryPrettyJson(text) {
  try {
    const cleaned = text.replace(/<\/tool_call>\s*$/, '').trim();
    return JSON.stringify(JSON.parse(cleaned), null, 2);
  } catch { return text; }
}

function esc(str) {
  return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

fetchList();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    print(f"Starting viewer at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)

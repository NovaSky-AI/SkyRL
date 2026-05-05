"""
trace_explorer.py
=================
Flask web app: browse trajectories where the taste judge scored low but the
verifier passed (verifier_score == 1, claude_total < threshold).

Usage:
    cd research/judge
    HF_TOKEN=... python trace_explorer.py
    # open http://localhost:5173

Args (env vars):
    HF_TOKEN          – HuggingFace token (required for dataset load)
    SCORED_PARQUET    – path to scored.parquet from score_dataset.py
                        (default: ~/Desktop/fleet/research/judge/scored.parquet)
    SCORE_THRESHOLD   – claude_total cutoff for "low score" (default: 3.0)
    PORT              – HTTP port (default: 5173)
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from flask import Flask, jsonify, render_template_string, request

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCORED_PARQUET = os.environ.get(
    "SCORED_PARQUET",
    str(Path.home() / "Desktop/fleet/research/judge/scored.parquet"),
)
SCORE_THRESHOLD = float(os.environ.get("SCORE_THRESHOLD", "3.0"))
PORT = int(os.environ.get("PORT", "5173"))

AXES = ("intent_clarity", "efficiency", "recovery", "ui_grounding", "coherence")
AXIS_LABELS = {
    "intent_clarity": "Intent Clarity",
    "efficiency": "Efficiency",
    "recovery": "Recovery",
    "ui_grounding": "UI Grounding",
    "coherence": "Coherence",
}

# ---------------------------------------------------------------------------
# Failure-mode classification
# ---------------------------------------------------------------------------

FAILURE_MODES = {
    "ghost_pass":    ("Ghost Pass",          "#6366f1"),  # purple – 0-1 turns, nothing visible
    "silent_acts":   ("Silent Actions",      "#8b5cf6"),  # violet – actions hidden, only "Action completed"
    "think_only":    ("Think-Only",          "#f59e0b"),  # amber – text reasoning, no actual tool calls
    "rage_click":    ("Rage Click / Spam",   "#ef4444"),  # red – visible repeated identical actions
    "low_effort":    ("Low Effort",          "#f97316"),  # orange – very few actions
    "partial":       ("Partial / Wrong Path","#3b82f6"),  # blue – long wandering trajectory
    "other":         ("Other",               "#6b7280"),  # gray
}


def _extract_actions_from_conv(conv: list[dict]) -> list[dict]:
    """Best-effort extraction of agent actions from a conversation list."""
    TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
    actions: list[dict] = []
    for m in conv:
        role = m.get("role", "")
        txt = m.get("text") or ""
        if not isinstance(txt, str):
            txt = ""
        if role in ("assistant", "agent", "model"):
            # structured tool_calls
            tcs = m.get("tool_calls") or []
            for tc in (tcs if isinstance(tcs, list) else []):
                if not isinstance(tc, dict):
                    continue
                name = tc.get("name") or (tc.get("function") or {}).get("name") or "unknown"
                args = tc.get("arguments") or (tc.get("function") or {}).get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                actions.append({"name": str(name), "arguments": args})
            # inline <tool_call> XML
            for mm in TOOL_RE.finditer(txt):
                try:
                    obj = json.loads(mm.group(1))
                    name = obj.get("name") or obj.get("tool") or "unknown"
                    args = obj.get("arguments") or obj.get("params") or {}
                    actions.append({"name": str(name), "arguments": args if isinstance(args, dict) else {}})
                except Exception:
                    pass
            # non-empty assistant text with no tool calls → think pseudo-action
            if txt.strip() and not tcs and "<tool_call>" not in txt:
                actions.append({"name": "think", "arguments": {"text": txt[:200]}})
    return actions


def _count_repeated_actions(actions: list[dict]) -> int:
    """Count how many consecutive pairs of actions are near-identical."""
    repeats = 0
    for i in range(1, len(actions)):
        a, b = actions[i - 1], actions[i]
        if a.get("name") == b.get("name") and a.get("name") != "think":
            # compare argument fingerprints
            fa = json.dumps(a.get("arguments", {}), sort_keys=True)[:100]
            fb = json.dumps(b.get("arguments", {}), sort_keys=True)[:100]
            if fa == fb:
                repeats += 1
    return repeats


def classify_failure_mode(
    conv: list[dict], actions: list[dict], num_turns: int
) -> str:
    n_actions = len(actions)
    think_only = all(a.get("name") == "think" for a in actions) if actions else False
    tool_results = [m for m in conv if m.get("role") == "tool"]
    n_tool_results = len(tool_results)
    asst_msgs = [m for m in conv if m.get("role") in ("assistant", "agent", "model")]
    empty_asst = sum(1 for m in asst_msgs if not (m.get("text") or "").strip())

    # Fraction of assistant messages with no visible text
    asst_silent_frac = empty_asst / max(len(asst_msgs), 1)

    # Ghost pass: 0-1 turns, virtually nothing happened
    if num_turns <= 1 or (n_actions == 0 and n_tool_results <= 2):
        return "ghost_pass"

    # Silent actions: the agent is executing tool calls (many "Action completed"
    # results) but the conversation text doesn't expose what the actions were.
    # The judge can't see them → scores 1 by default.  This is NOT rage-clicking;
    # the agent may be doing meaningful work that's just opaque in this log format.
    if n_actions == 0 and n_tool_results > 2:
        return "silent_acts"
    if asst_silent_frac >= 0.85 and n_tool_results > 5:
        return "silent_acts"

    # Think-only: all visible actions are text reasoning with no real tool calls
    if think_only and n_actions > 0:
        if n_actions <= 3:
            return "low_effort"
        return "think_only"

    # Rage-click: visibly repeated identical tool calls in the action list
    repeats = _count_repeated_actions(actions)
    if repeats >= 3:
        return "rage_click"

    # Low effort: very few real actions extracted
    real_actions = [a for a in actions if a.get("name") != "think"]
    if len(real_actions) <= 2 and n_tool_results <= 3:
        return "low_effort"

    # Partial / wandering: long trajectory that somehow still passed
    if num_turns > 30 and n_actions > 5:
        return "partial"

    return "other"


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _pil_to_b64(img: Any) -> str | None:
    try:
        from PIL import Image
        if isinstance(img, dict) and "bytes" in img:
            img = Image.open(io.BytesIO(img["bytes"]))
        elif isinstance(img, dict) and "path" in img:
            img = Image.open(img["path"])
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def _sample_images(images: list, k: int = 6) -> list[str]:
    """Return up to k evenly-spaced base64 PNGs."""
    if not images:
        return []
    step = max(1, len(images) // k)
    selected = images[::step][:k]
    result = []
    for img in selected:
        b64 = _pil_to_b64(img)
        if b64:
            result.append(b64)
    return result


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_DATA: list[dict] = []
_INDEX: dict[str, dict] = {}


def load_data() -> None:
    global _DATA, _INDEX
    print(f"Loading scored parquet: {SCORED_PARQUET}")
    df = pd.read_parquet(SCORED_PARQUET)
    print(f"  {len(df)} rows, columns: {list(df.columns)}")

    token = os.environ.get("HF_TOKEN")
    print("Loading HuggingFace dataset (from cache)...")
    from datasets import load_dataset
    ds = load_dataset(
        "FleetAI/fleet-cu-trajectories",
        split="train",
        token=token,
    )
    raw_rows: dict[str, Any] = {r["session_id"]: r for r in ds}
    print(f"  {len(raw_rows)} dataset rows")

    # Filter: verifier pass + low judge score
    passing = df[
        (df["verifier_score"] == 1)
        & (df["claude_total"].notna())
        & (df["claude_total"] < SCORE_THRESHOLD)
    ].copy()
    print(f"  {len(passing)} rows: verifier=pass, claude_total < {SCORE_THRESHOLD}")

    records: list[dict] = []
    for _, row in passing.iterrows():
        tid = row["trajectory_id"]
        raw = raw_rows.get(tid, {})

        conv_raw = raw.get("conversation", "[]")
        conv: list[dict] = (
            json.loads(conv_raw) if isinstance(conv_raw, str) else (conv_raw or [])
        )

        actions = _extract_actions_from_conv(conv)
        num_turns = int(raw.get("num_turns") or 0)
        failure_mode = classify_failure_mode(conv, actions, num_turns)

        # Task: first user message
        task = ""
        for m in conv:
            if m.get("role") == "user":
                t = (m.get("text") or "").strip()
                if len(t) > 5:
                    task = t
                    break
        if not task:
            task = raw.get("task_key", "")

        records.append(
            {
                "session_id": tid,
                "env_key": raw.get("env_key", ""),
                "task_key": raw.get("task_key", ""),
                "model": raw.get("model", ""),
                "num_turns": num_turns,
                "num_screenshots": int(raw.get("num_screenshots") or 0),
                "task": task[:600],
                "claude_total": float(row["claude_total"]),
                "gpt_total": float(row["gpt_total"]) if pd.notna(row.get("gpt_total")) else None,
                "claude_rationale": str(row.get("claude_rationale") or ""),
                "gpt_rationale": str(row.get("gpt_rationale") or ""),
                "scores": {
                    ax: {
                        "claude": (
                            int(row[f"claude_{ax}"])
                            if pd.notna(row.get(f"claude_{ax}"))
                            else None
                        ),
                        "gpt": (
                            int(row[f"gpt_{ax}"])
                            if pd.notna(row.get(f"gpt_{ax}"))
                            else None
                        ),
                    }
                    for ax in AXES
                },
                "failure_mode": failure_mode,
                # Conversation (without images to keep it light)
                "conv": [
                    {
                        "role": m.get("role", ""),
                        "text": (m.get("text") or "")[:2000],
                        "has_image": bool(m.get("has_image")),
                        "position": m.get("position"),
                    }
                    for m in conv
                ],
                # Raw HF images (PIL objects) – stored temporarily, not serialized
                "_images": raw.get("images") or [],
                "_actions": actions,
            }
        )

    # Sort: by failure mode then by score ascending
    MODE_ORDER = ["ghost_pass", "think_only", "rage_click", "low_effort", "partial", "other"]
    records.sort(
        key=lambda r: (
            MODE_ORDER.index(r["failure_mode"])
            if r["failure_mode"] in MODE_ORDER
            else 99,
            r["claude_total"],
        )
    )
    _DATA = records
    _INDEX = {r["session_id"]: r for r in records}
    print(f"Loaded {len(_DATA)} traces to display.")
    mode_counts = Counter(r["failure_mode"] for r in records)
    for mode, cnt in mode_counts.most_common():
        label = FAILURE_MODES[mode][0]
        print(f"  {label}: {cnt}")


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)

HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Trace Explorer — Low Judge / Verifier Pass</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f172a; color: #e2e8f0; height: 100vh; display: flex;
         flex-direction: column; }
  header { background: #1e293b; border-bottom: 1px solid #334155;
           padding: 12px 20px; display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 15px; font-weight: 600; color: #f1f5f9; }
  header .sub { font-size: 12px; color: #94a3b8; }
  .main { display: flex; flex: 1; overflow: hidden; }

  /* sidebar */
  #sidebar { width: 320px; min-width: 260px; background: #1e293b;
             border-right: 1px solid #334155; overflow-y: auto;
             display: flex; flex-direction: column; }
  #sidebar .filters { padding: 10px 12px; border-bottom: 1px solid #334155; }
  #sidebar .filters input { width: 100%; background: #0f172a; border: 1px solid #334155;
    color: #e2e8f0; border-radius: 6px; padding: 6px 10px; font-size: 12px; outline: none; }
  #sidebar .filters input:focus { border-color: #6366f1; }
  .mode-filters { display: flex; flex-wrap: wrap; gap: 4px; padding: 8px 12px;
                  border-bottom: 1px solid #334155; }
  .mode-btn { font-size: 10px; padding: 3px 8px; border-radius: 10px;
              border: none; cursor: pointer; opacity: 0.5; transition: opacity 0.15s; }
  .mode-btn.active { opacity: 1; }
  .trace-list { flex: 1; overflow-y: auto; }
  .trace-item { padding: 10px 12px; border-bottom: 1px solid #1e293b;
                cursor: pointer; transition: background 0.1s; }
  .trace-item:hover { background: #334155; }
  .trace-item.selected { background: #1d4ed8; }
  .trace-item .env { font-size: 10px; color: #94a3b8; margin-bottom: 2px; }
  .trace-item .task-snip { font-size: 11px; color: #cbd5e1; line-height: 1.4;
                           overflow: hidden; display: -webkit-box;
                           -webkit-line-clamp: 2; -webkit-box-orient: vertical; }
  .trace-item .meta { display: flex; align-items: center; gap: 6px; margin-top: 4px; }
  .badge { font-size: 9px; padding: 2px 7px; border-radius: 8px;
           font-weight: 600; color: #fff; white-space: nowrap; }
  .score-pill { font-size: 10px; background: #0f172a; color: #94a3b8;
                padding: 1px 6px; border-radius: 6px; }

  /* detail panel */
  #detail { flex: 1; overflow-y: auto; padding: 20px 24px; }
  .empty-state { display: flex; align-items: center; justify-content: center;
                 height: 100%; color: #475569; font-size: 14px; }
  .detail-header { display: flex; align-items: flex-start; gap: 16px;
                   margin-bottom: 20px; }
  .detail-header h2 { font-size: 13px; font-weight: 700; color: #f1f5f9;
                      flex: 1; line-height: 1.5; }
  .detail-meta { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 16px; }
  .chip { font-size: 11px; background: #1e293b; border: 1px solid #334155;
          color: #94a3b8; padding: 3px 10px; border-radius: 12px; }

  .scores-grid { display: grid; grid-template-columns: repeat(5, 1fr);
                 gap: 8px; margin-bottom: 20px; }
  .score-card { background: #1e293b; border-radius: 8px; padding: 10px 8px;
                text-align: center; }
  .score-card .axis { font-size: 9px; color: #64748b; margin-bottom: 4px;
                      text-transform: uppercase; letter-spacing: 0.05em; }
  .score-card .val { font-size: 22px; font-weight: 700; }
  .score-card .gpt-val { font-size: 11px; color: #64748b; margin-top: 2px; }
  .s1 { color: #ef4444; } .s2 { color: #f97316; } .s3 { color: #f59e0b; }
  .s4 { color: #84cc16; } .s5 { color: #22c55e; }

  .total-row { display: flex; align-items: center; gap: 10px; margin-bottom: 20px; }
  .total-label { font-size: 11px; color: #64748b; }
  .total-val { font-size: 20px; font-weight: 700; }
  .gpt-total { font-size: 12px; color: #64748b; }

  .section-title { font-size: 11px; font-weight: 600; color: #64748b;
                   text-transform: uppercase; letter-spacing: 0.08em;
                   margin-bottom: 8px; }
  .rationale-box { background: #1e293b; border-radius: 8px; padding: 12px 14px;
                   font-size: 12px; line-height: 1.6; color: #cbd5e1;
                   margin-bottom: 20px; border-left: 3px solid; }

  .screenshots { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 20px; }
  .screenshots img { max-height: 180px; border-radius: 6px;
                     border: 1px solid #334155; cursor: zoom-in; }

  .conv-list { margin-bottom: 20px; }
  .conv-msg { display: flex; gap: 10px; margin-bottom: 8px; }
  .conv-msg .role { font-size: 10px; font-weight: 600; width: 72px; min-width: 72px;
                    padding-top: 3px; text-align: right; }
  .role-system   { color: #475569; }
  .role-user     { color: #3b82f6; }
  .role-assistant{ color: #a78bfa; }
  .role-tool     { color: #34d399; }
  .conv-msg .bubble { background: #1e293b; border-radius: 6px; padding: 8px 10px;
                      font-size: 12px; line-height: 1.5; color: #cbd5e1;
                      flex: 1; white-space: pre-wrap; word-break: break-word; }
  .conv-msg .bubble.empty { color: #475569; font-style: italic; }

  /* lightbox */
  #lb { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.85);
        z-index: 100; align-items: center; justify-content: center; cursor: zoom-out; }
  #lb.open { display: flex; }
  #lb img { max-width: 92vw; max-height: 92vh; border-radius: 8px; }
</style>
</head>
<body>
<header>
  <h1>Trace Explorer</h1>
  <span class="sub">verifier pass · judge score &lt; {{ threshold }} · {{ total }} traces</span>
</header>
<div class="main">
  <nav id="sidebar">
    <div class="filters">
      <input type="text" id="search" placeholder="Search task / env / session…" oninput="filterList()">
    </div>
    <div class="mode-filters" id="modeFilters"></div>
    <div class="trace-list" id="traceList"></div>
  </nav>
  <div id="detail">
    <div class="empty-state">← Select a trace to inspect</div>
  </div>
</div>
<div id="lb" onclick="closeLb()"><img id="lbImg"></div>

<script>
const TRACES = {{ traces_json | safe }};
const MODES  = {{ modes_json | safe }};

let activeModes = new Set(Object.keys(MODES));
let currentId   = null;

const MODE_TIPS = {
  ghost_pass:   "0-1 turns; agent did virtually nothing yet verifier passed",
  silent_acts:  "Many 'Action completed' tool results but no visible action text — actions are opaque to the judge, which auto-scores them 1",
  think_only:   "Agent outputs reasoning text only; no actual tool calls were fired",
  rage_click:   "Repeated identical tool calls — agent loops on the same action",
  low_effort:   "Very few real actions taken before the session ended",
  partial:      "Long wandering trajectory that somehow still passed",
  other:        "Doesn't fit a clear pattern",
};

// Count per mode
const modeCounts = {};
for (const t of TRACES) modeCounts[t.failure_mode] = (modeCounts[t.failure_mode] || 0) + 1;

// Build mode filter buttons
const mf = document.getElementById('modeFilters');
for (const [key, [label, color]] of Object.entries(MODES)) {
  const cnt = modeCounts[key] || 0;
  if (!cnt) continue;
  const btn = document.createElement('button');
  btn.className = 'mode-btn active badge';
  btn.style.background = color;
  btn.title = MODE_TIPS[key] || '';
  btn.textContent = `${label} (${cnt})`;
  btn.dataset.mode = key;
  btn.onclick = () => toggleMode(key, btn);
  mf.appendChild(btn);
}

function toggleMode(key, btn) {
  if (activeModes.has(key)) {
    activeModes.delete(key);
    btn.classList.remove('active');
  } else {
    activeModes.add(key);
    btn.classList.add('active');
  }
  renderList();
}

function scoreClass(s) {
  if (s <= 1) return 's1';
  if (s <= 2) return 's2';
  if (s <= 3) return 's3';
  if (s <= 4) return 's4';
  return 's5';
}

function renderList() {
  const q = document.getElementById('search').value.toLowerCase();
  const list = document.getElementById('traceList');
  list.innerHTML = '';
  for (const t of TRACES) {
    if (!activeModes.has(t.failure_mode)) continue;
    if (q && !t.task.toLowerCase().includes(q) &&
        !t.env_key.toLowerCase().includes(q) &&
        !t.session_id.toLowerCase().includes(q)) continue;
    const [modeLabel, modeColor] = MODES[t.failure_mode];
    const div = document.createElement('div');
    div.className = 'trace-item' + (t.session_id === currentId ? ' selected' : '');
    div.dataset.id = t.session_id;
    div.onclick = () => showTrace(t.session_id);
    div.innerHTML = `
      <div class="env">${t.env_key} · ${t.session_id.slice(0,8)}</div>
      <div class="task-snip">${escHtml(t.task.slice(0,120))}</div>
      <div class="meta">
        <span class="badge" style="background:${modeColor}">${modeLabel}</span>
        <span class="score-pill ${scoreClass(t.claude_total)}">${t.claude_total.toFixed(2)}</span>
        <span class="score-pill">${t.num_turns}t</span>
      </div>`;
    list.appendChild(div);
  }
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

async function showTrace(id) {
  currentId = id;
  // Update sidebar selection
  document.querySelectorAll('.trace-item').forEach(el =>
    el.classList.toggle('selected', el.dataset.id === id));

  const t = TRACES.find(x => x.session_id === id);
  if (!t) return;
  const [modeLabel, modeColor] = MODES[t.failure_mode];

  // Load screenshots lazily
  document.getElementById('detail').innerHTML = '<div class="empty-state">Loading screenshots…</div>';
  let imgs = [];
  try {
    const res = await fetch('/screenshots/' + id);
    imgs = await res.json();
  } catch(e) { imgs = []; }

  const totalClass = scoreClass(t.claude_total);
  const axCards = Object.entries(t.scores).map(([ax, v]) => {
    const c = v.claude ?? '—';
    const cc = typeof c === 'number' ? scoreClass(c) : '';
    return `<div class="score-card">
      <div class="axis">${ax.replace(/_/g,' ')}</div>
      <div class="val ${cc}">${c}</div>
      <div class="gpt-val">GPT: ${v.gpt ?? '—'}</div>
    </div>`;
  }).join('');

  const convHtml = t.conv.map(m => {
    const roleClass = `role-${m.role}`;
    const txt = m.text ? escHtml(m.text) : '<em>( empty )</em>';
    const bubbleClass = m.text ? 'bubble' : 'bubble empty';
    return `<div class="conv-msg">
      <div class="role ${roleClass}">${m.role}</div>
      <div class="${bubbleClass}">${txt}${m.has_image ? ' 📷' : ''}</div>
    </div>`;
  }).join('');

  const imgHtml = imgs.length
    ? `<div class="screenshots">${imgs.map(b =>
        `<img src="data:image/png;base64,${b}" onclick="openLb(this.src)">`
      ).join('')}</div>`
    : `<div style="color:#475569;font-size:12px;margin-bottom:20px;">No screenshots</div>`;

  document.getElementById('detail').innerHTML = `
    <div class="detail-header">
      <h2>${escHtml(t.task.slice(0, 300))}${t.task.length > 300 ? '…' : ''}</h2>
    </div>
    <div class="detail-meta">
      <span class="badge" style="background:${modeColor};font-size:11px;padding:4px 12px">${modeLabel}</span>
      <span class="chip">${t.env_key}</span>
      <span class="chip">${t.task_key}</span>
      <span class="chip">${t.num_turns} turns</span>
      <span class="chip">${t.num_screenshots} screenshots</span>
      <span class="chip" style="font-family:monospace;font-size:10px">${t.session_id}</span>
    </div>
    <div class="total-row">
      <span class="total-label">Judge total (Claude)</span>
      <span class="total-val ${totalClass}">${t.claude_total.toFixed(2)}</span>
      ${t.gpt_total != null ? `<span class="gpt-total">GPT: ${t.gpt_total.toFixed(2)}</span>` : ''}
    </div>
    <div class="scores-grid">${axCards}</div>

    <div class="section-title">Claude rationale</div>
    <div class="rationale-box" style="border-color:${modeColor}">
      ${escHtml(t.claude_rationale || '(none)')}
    </div>
    ${t.gpt_rationale && t.gpt_rationale !== 'None' ? `
    <div class="section-title">GPT rationale</div>
    <div class="rationale-box" style="border-color:#10a37f">
      ${escHtml(t.gpt_rationale)}
    </div>` : ''}

    <div class="section-title">Screenshots (sampled)</div>
    ${imgHtml}

    <div class="section-title">Conversation (${t.conv.length} messages)</div>
    <div class="conv-list">${convHtml}</div>
  `;
}

function filterList() { renderList(); }

function openLb(src) {
  document.getElementById('lbImg').src = src;
  document.getElementById('lb').classList.add('open');
}
function closeLb() {
  document.getElementById('lb').classList.remove('open');
}

renderList();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    # Build a lightweight serializable copy (no PIL images)
    traces_out = [
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in _DATA
    ]
    modes_out = {k: list(v) for k, v in FAILURE_MODES.items()}
    return render_template_string(
        HTML,
        traces_json=json.dumps(traces_out),
        modes_json=json.dumps(modes_out),
        threshold=SCORE_THRESHOLD,
        total=len(_DATA),
    )


@app.route("/screenshots/<session_id>")
def screenshots(session_id: str):
    rec = _INDEX.get(session_id)
    if rec is None:
        return jsonify([])
    imgs = _sample_images(rec.get("_images", []), k=8)
    return jsonify(imgs)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    load_data()
    print(f"\nStarting on http://localhost:{PORT}\n")
    app.run(host="0.0.0.0", port=PORT, debug=False)

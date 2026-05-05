#!/usr/bin/env python3
"""
Viewer for WandB taste-trial trajectory data fetched by fetch_runs.py.

Usage:
    python3 wandb_viewer.py [port]
    # open http://localhost:8081
"""

import sys as _sys
_sys.path = [p for p in _sys.path if not p in ("", ".")]

import json
import sys
from pathlib import Path
from flask import Flask, jsonify, request, Response, abort

BASE = Path(__file__).parent
RUNS_DIR = BASE / "wandb-runs"

LABELS = {
    "zd3sk2db": "baseline",
    "gjfocn7r": "judge",
    "x09ot84k": "abl-screenshots",
    "s24aawb9": "rel-sonnet",
}

LABEL_COLORS = {
    "baseline":       "#6b7280",
    "judge":          "#4b6bfb",
    "abl-screenshots":"#8b5cf6",
    "rel-sonnet":     "#10b981",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_index() -> list:
    path = RUNS_DIR / "index.json"
    if path.exists():
        return json.loads(path.read_text())
    # Build minimal index from what's on disk
    index = []
    for run_id, label in LABELS.items():
        meta_path = RUNS_DIR / run_id / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            index.append({"run_id": run_id, "label": label, "name": meta.get("name", run_id),
                          "state": meta.get("state", "?"), "url": meta.get("url", "")})
    return index


def load_val_generations(run_id: str) -> list:
    path = RUNS_DIR / run_id / "val_generations.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_meta(run_id: str) -> dict:
    path = RUNS_DIR / run_id / "meta.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _detect_samples(row: dict) -> list[dict]:
    """Extract (input, output, score) triples from a val/generations table row."""
    samples = []
    i = 1
    while f"input_{i}" in row or f"output_{i}" in row:
        samples.append({
            "input":  row.get(f"input_{i}", ""),
            "output": row.get(f"output_{i}", ""),
            "score":  row.get(f"score_{i}"),
        })
        i += 1
    return samples


INDEX = load_index()
RUN_DATA: dict[str, list] = {}
RUN_META: dict[str, dict] = {}

for entry in INDEX:
    rid = entry["run_id"]
    RUN_DATA[rid] = load_val_generations(rid)
    RUN_META[rid] = load_meta(rid)

# Pre-build flat trajectory list: one entry per (step, sample_index)
ALL_TRAJS: list[dict] = []
for rid, rows in RUN_DATA.items():
    label = LABELS.get(rid, rid)
    name = RUN_META.get(rid, {}).get("name", rid)
    for row in rows:
        step = row.get("step", row.get("_step", "?"))
        samples = _detect_samples(row)
        for si, s in enumerate(samples):
            ALL_TRAJS.append({
                "id": f"{rid}_{step}_{si}",
                "run_id": rid,
                "label": label,
                "run_name": name,
                "step": step,
                "sample_idx": si,
                "score": s["score"],
                "input": s["input"],
                "output": s["output"],
            })

print(f"Loaded {len(INDEX)} runs, {len(ALL_TRAJS)} trajectory samples.")

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)


@app.route("/api/index")
def api_index():
    return jsonify([
        {**e, "n_samples": sum(1 for t in ALL_TRAJS if t["run_id"] == e["run_id"])}
        for e in INDEX
    ])


@app.route("/api/trajectories")
def api_trajectories():
    run_filter   = request.args.get("run_id", "")
    label_filter = request.args.get("label", "")
    min_score    = request.args.get("min_score", "")
    max_score    = request.args.get("max_score", "")
    q            = request.args.get("q", "").lower()
    page         = int(request.args.get("page", 1))
    limit        = int(request.args.get("limit", 50))

    rows = ALL_TRAJS
    if run_filter:
        rows = [r for r in rows if r["run_id"] == run_filter]
    if label_filter:
        rows = [r for r in rows if r["label"] == label_filter]
    if min_score:
        try:
            ms = float(min_score)
            rows = [r for r in rows if r["score"] is not None and float(r["score"]) >= ms]
        except ValueError:
            pass
    if max_score:
        try:
            ms = float(max_score)
            rows = [r for r in rows if r["score"] is not None and float(r["score"]) <= ms]
        except ValueError:
            pass
    if q:
        rows = [r for r in rows if q in (r["input"] or "").lower() or q in (r["output"] or "").lower()]

    total = len(rows)
    start = (page - 1) * limit
    page_rows = rows[start: start + limit]

    summaries = [{
        "id":       r["id"],
        "run_id":   r["run_id"],
        "label":    r["label"],
        "run_name": r["run_name"],
        "step":     r["step"],
        "sample_idx": r["sample_idx"],
        "score":    r["score"],
        "input_preview":  (r["input"] or "")[:120],
        "output_preview": (r["output"] or "")[:120],
    } for r in page_rows]

    return jsonify({"total": total, "page": page, "limit": limit, "trajectories": summaries})


@app.route("/api/trajectories/<traj_id>")
def api_trajectory(traj_id):
    for t in ALL_TRAJS:
        if t["id"] == traj_id:
            meta = RUN_META.get(t["run_id"], {})
            return jsonify({**t, "run_meta": meta.get("summary", {})})
    abort(404)


@app.route("/api/run_meta/<run_id>")
def api_run_meta(run_id):
    meta = RUN_META.get(run_id)
    if not meta:
        abort(404)
    return jsonify(meta)


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>WandB Run Viewer</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0a0a0f; color: #e2e8f0; font-family: system-ui, sans-serif; font-size: 14px; }

  .topbar { background: #111827; border-bottom: 1px solid #1f2937; padding: 12px 24px;
            display: flex; align-items: center; gap: 16px; position: sticky; top: 0; z-index: 100; }
  .topbar h1 { font-size: 16px; font-weight: 600; }
  .topbar .nav { display: flex; gap: 12px; margin-left: auto; }
  .topbar .nav a { font-size: 13px; color: #9ca3af; cursor: pointer; }
  .topbar .nav a:hover { color: #e2e8f0; }

  #list-view { padding: 24px; max-width: 1400px; margin: 0 auto; width: 100%; }
  #detail-view { display: none; }

  /* run cards */
  .run-cards { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px; }
  .run-card { background: #111827; border: 1px solid #1f2937; border-radius: 8px;
              padding: 12px 16px; cursor: pointer; transition: border-color 0.15s; flex: 1; min-width: 160px; }
  .run-card:hover { border-color: #374151; }
  .run-card.active { border-color: #4b6bfb; background: rgba(75,107,251,0.08); }
  .run-card .rc-label { font-size: 13px; font-weight: 600; color: #e2e8f0; margin-bottom: 4px; }
  .run-card .rc-name  { font-size: 10px; font-family: monospace; color: #4b5563; margin-bottom: 6px; word-break: break-all; }
  .run-card .rc-count { font-size: 11px; color: #6b7280; }

  /* filters */
  .filters { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 16px; align-items: center; }
  .filters input, .filters select { background: #1f2937; border: 1px solid #374151;
    color: #e2e8f0; border-radius: 6px; padding: 6px 12px; font-size: 13px; outline: none; }
  .filters input:focus, .filters select:focus { border-color: #4b6bfb; }
  .total { margin-left: auto; font-size: 12px; color: #6b7280; }

  table { width: 100%; border-collapse: collapse; border: 1px solid #1f2937; border-radius: 8px; overflow: hidden; }
  th { background: #111827; color: #6b7280; font-weight: 500; text-transform: uppercase;
       font-size: 11px; letter-spacing: 0.05em; padding: 10px 14px; text-align: left; }
  td { padding: 10px 14px; border-top: 1px solid #1f2937; vertical-align: middle; max-width: 360px; }
  tr:hover td { background: #111827; }
  .preview { font-size: 12px; color: #94a3b8; overflow: hidden; text-overflow: ellipsis;
             white-space: nowrap; max-width: 360px; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 500; }
  .mono  { font-family: monospace; font-size: 12px; }
  .score-hi  { background: rgba(16,185,129,0.15); color: #34d399; }
  .score-mid { background: rgba(251,191,36,0.12);  color: #fbbf24; }
  .score-lo  { background: rgba(239,68,68,0.12);   color: #f87171; }
  .score-null { color: #374151; }

  .pagination { display: flex; align-items: center; justify-content: space-between; margin-top: 16px; }
  .btn { background: #1f2937; border: 1px solid #374151; color: #e2e8f0; border-radius: 6px;
         padding: 6px 16px; font-size: 13px; cursor: pointer; }
  .btn:hover:not(:disabled) { background: #374151; }
  .btn:disabled { opacity: 0.4; cursor: default; }
  .loading { padding: 48px; text-align: center; color: #6b7280; }

  /* detail */
  .detail-header { background: #111827; border-bottom: 1px solid #1f2937; padding: 12px 24px;
                   position: sticky; top: 49px; z-index: 50; }
  .detail-meta { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }
  .detail-body { padding: 24px; max-width: 1200px; margin: 0 auto; display: flex; gap: 24px; }
  .col-left  { flex: 1; min-width: 0; }
  .col-right { flex: 1; min-width: 0; }
  .section-label { font-size: 11px; font-weight: 600; color: #6b7280;
                   text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; }
  .text-block { background: #111827; border: 1px solid #1f2937; border-radius: 8px;
                padding: 14px 16px; font-size: 13px; color: #cbd5e1; line-height: 1.65;
                white-space: pre-wrap; word-break: break-word; max-height: 70vh; overflow-y: auto; }
  .output-block { background: #0a0f1e; border: 1px solid #1f3a6e; border-radius: 8px;
                  padding: 14px 16px; font-size: 13px; color: #86efac; line-height: 1.65;
                  white-space: pre-wrap; word-break: break-word; max-height: 70vh; overflow-y: auto; }
  .meta-card { background: #111827; border: 1px solid #1f2937; border-radius: 8px;
               padding: 14px 16px; margin-top: 16px; }
  .meta-card h3 { font-size: 11px; font-weight: 600; color: #6b7280;
                  text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 10px; }
  .meta-row { display: flex; justify-content: space-between; font-size: 12px;
              padding: 3px 0; border-bottom: 1px solid #1f2937; }
  .meta-row:last-child { border-bottom: none; }
  .meta-key { color: #6b7280; }
  .meta-val { color: #e2e8f0; font-family: monospace; font-size: 11px; text-align: right;
              max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
</style>
</head>
<body>
<div class="topbar">
  <h1>WandB Run Viewer</h1>
  <div class="nav">
    <a id="nav-back" onclick="showList()" style="display:none">← List</a>
  </div>
</div>

<div id="list-view">
  <div class="run-cards" id="run-cards"></div>
  <div class="filters">
    <input id="q" type="text" placeholder="Search input / output…" oninput="onFilter()" style="width:220px" />
    <input id="min-score" type="number" step="0.01" placeholder="Min score" oninput="onFilter()" style="width:110px" />
    <input id="max-score" type="number" step="0.01" placeholder="Max score" oninput="onFilter()" style="width:110px" />
    <span class="total" id="total-label"></span>
  </div>
  <div id="table-container"><div class="loading">Loading…</div></div>
  <div class="pagination">
    <button class="btn" id="prev-btn" onclick="changePage(-1)" disabled>Previous</button>
    <span id="page-label" style="color:#6b7280;font-size:13px;"></span>
    <button class="btn" id="next-btn" onclick="changePage(1)">Next</button>
  </div>
</div>

<div id="detail-view">
  <div class="detail-header">
    <div class="detail-meta" id="detail-meta"></div>
  </div>
  <div class="detail-body">
    <div class="col-left">
      <div class="section-label">Input (task prompt)</div>
      <div class="text-block" id="detail-input"></div>
    </div>
    <div class="col-right">
      <div class="section-label">Output (model response)</div>
      <div class="output-block" id="detail-output"></div>
      <div class="meta-card" id="detail-run-meta" style="display:none">
        <h3>Run summary metrics</h3>
        <div id="detail-run-meta-body"></div>
      </div>
    </div>
  </div>
</div>

<script>
const LABEL_COLORS = {
  baseline:        '#6b7280',
  judge:           '#4b6bfb',
  'abl-screenshots':'#8b5cf6',
  'rel-sonnet':    '#10b981',
};

let state = { runId: '', page: 1, limit: 50, total: 0, q: '', minScore: '', maxScore: '' };
let INDEX = [];

function labelBadge(label) {
  const c = LABEL_COLORS[label] || '#374151';
  return `<span class="badge" style="background:${c}20;color:${c}">${esc(label)}</span>`;
}

function scoreHtml(score) {
  if (score === null || score === undefined) return '<span class="score-null">—</span>';
  const s = parseFloat(score);
  const cls = s >= 0.7 ? 'score-hi' : s >= 0.4 ? 'score-mid' : 'score-lo';
  return `<span class="badge ${cls}">${s.toFixed(3)}</span>`;
}

async function init() {
  const res = await fetch('/api/index');
  INDEX = await res.json();
  renderRunCards();
  fetchList();
}

function renderRunCards() {
  const container = document.getElementById('run-cards');
  // "All" card
  const allActive = !state.runId;
  const total = INDEX.reduce((s, e) => s + (e.n_samples || 0), 0);
  let html = `<div class="run-card ${allActive ? 'active' : ''}" onclick="selectRun('')">
    <div class="rc-label">All runs</div>
    <div class="rc-count">${total} samples</div>
  </div>`;
  for (const e of INDEX) {
    const active = state.runId === e.run_id;
    const c = LABEL_COLORS[e.label] || '#374151';
    html += `<div class="run-card ${active ? 'active' : ''}" onclick="selectRun('${e.run_id}')"
                  style="${active ? `border-color:${c}` : ''}">
      <div class="rc-label" style="color:${c}">${esc(e.label)}</div>
      <div class="rc-name">${esc(e.name || e.run_id)}</div>
      <div class="rc-count">${e.n_samples || 0} samples · ${esc(e.state || '?')}</div>
    </div>`;
  }
  container.innerHTML = html;
}

function selectRun(runId) {
  state.runId = runId;
  state.page = 1;
  renderRunCards();
  fetchList();
}

function onFilter() {
  state.q = document.getElementById('q').value.trim();
  state.minScore = document.getElementById('min-score').value;
  state.maxScore = document.getElementById('max-score').value;
  state.page = 1;
  fetchList();
}

function changePage(d) {
  state.page = Math.max(1, Math.min(Math.ceil(state.total / state.limit), state.page + d));
  fetchList();
}

async function fetchList() {
  const p = new URLSearchParams({ page: state.page, limit: state.limit });
  if (state.runId)   p.set('run_id', state.runId);
  if (state.q)       p.set('q', state.q);
  if (state.minScore) p.set('min_score', state.minScore);
  if (state.maxScore) p.set('max_score', state.maxScore);
  document.getElementById('table-container').innerHTML = '<div class="loading">Loading…</div>';
  const res = await fetch('/api/trajectories?' + p);
  const data = await res.json();
  state.total = data.total;
  renderTable(data);
}

function renderTable(data) {
  document.getElementById('total-label').textContent = data.total.toLocaleString() + ' samples';
  const pages = Math.ceil(data.total / state.limit) || 1;
  document.getElementById('page-label').textContent = `Page ${state.page} of ${pages}`;
  document.getElementById('prev-btn').disabled = state.page <= 1;
  document.getElementById('next-btn').disabled = state.page >= pages;

  if (!data.trajectories.length) {
    document.getElementById('table-container').innerHTML = '<div class="loading">No results.</div>';
    return;
  }

  const rows = data.trajectories.map(t => `
    <tr>
      <td>${labelBadge(t.label)}</td>
      <td class="mono" style="color:#6b7280">${esc(String(t.step))}</td>
      <td>${scoreHtml(t.score)}</td>
      <td class="preview"><a href="#" onclick="showDetail('${t.id}'); return false;">${esc(t.input_preview)}</a></td>
      <td class="preview">${esc(t.output_preview)}</td>
    </tr>`).join('');

  document.getElementById('table-container').innerHTML = `
    <table>
      <thead><tr>
        <th>Run</th><th>Step</th><th>Score</th><th>Input</th><th>Output</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

async function showDetail(id) {
  document.getElementById('list-view').style.display = 'none';
  document.getElementById('detail-view').style.display = 'block';
  document.getElementById('nav-back').style.display = 'inline';

  const res = await fetch('/api/trajectories/' + id);
  if (!res.ok) return;
  const t = await res.json();

  document.getElementById('detail-meta').innerHTML = `
    ${labelBadge(t.label)}
    <span style="color:#6b7280;font-size:12px">step ${t.step}</span>
    ${scoreHtml(t.score)}
    <span style="color:#4b5563;font-size:11px;font-family:monospace">${esc(t.run_name || t.run_id)}</span>
  `;

  document.getElementById('detail-input').textContent = t.input || '(empty)';
  document.getElementById('detail-output').textContent = t.output || '(empty)';

  // Run summary metrics
  const runMeta = t.run_meta || {};
  const metaKeys = Object.keys(runMeta).filter(k => typeof runMeta[k] !== 'object').slice(0, 20);
  if (metaKeys.length) {
    document.getElementById('detail-run-meta').style.display = 'block';
    document.getElementById('detail-run-meta-body').innerHTML = metaKeys.map(k =>
      `<div class="meta-row">
        <span class="meta-key">${esc(k)}</span>
        <span class="meta-val">${esc(String(runMeta[k]))}</span>
      </div>`
    ).join('');
  }
}

function showList() {
  document.getElementById('detail-view').style.display = 'none';
  document.getElementById('list-view').style.display = 'block';
  document.getElementById('nav-back').style.display = 'none';
}

function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

init();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")


if __name__ == "__main__":
    if not RUNS_DIR.exists():
        sys.exit(f"wandb-runs/ directory not found — run fetch_runs.py first")
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8081
    print(f"Starting viewer at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)

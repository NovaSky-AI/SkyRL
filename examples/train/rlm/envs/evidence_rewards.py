"""Evidence-based reward functions and per-trajectory child metrics.

F1 over retrieved text intervals vs. ground-truth evidence spans. Used by
``EvidenceRLMEnv`` for the env reward, and by ``RLMGymGenerator`` for
per-trajectory wandb metrics over the parent/child rollout tree.
"""

from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Interval metrics
# ---------------------------------------------------------------------------

def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def _union_size(intervals: List[Tuple[int, int]]) -> int:
    return sum(e - s for s, e in _merge_intervals(intervals))


def _intersection_size(a: List[Tuple[int, int]], b: List[Tuple[int, int]]) -> int:
    a, b = _merge_intervals(a), _merge_intervals(b)
    i = j = total = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if lo < hi:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def compute_metrics(
    retrieved_intervals: List[Tuple[int, int]],
    evidence_intervals: List[Tuple[int, int]],
) -> Dict[str, float]:
    covered = _intersection_size(retrieved_intervals, evidence_intervals)
    total_evidence = _union_size(evidence_intervals)
    total_retrieved = _union_size(retrieved_intervals)
    precision = covered / total_retrieved if total_retrieved else 0.0
    recall = covered / total_evidence if total_evidence else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_metrics_multipaper(
    retrieved: List[Tuple[str, int, int]],
    evidence: List[Tuple[str, int, int]],
) -> Dict[str, float]:
    """F1 over (paper_id, start, end) triples, computed per-paper then summed."""
    all_papers = set(p for p, _, _ in evidence) | set(p for p, _, _ in retrieved)
    total_evidence = total_retrieved = covered = 0
    for pid in all_papers:
        ev_ivs = [(s, e) for p, s, e in evidence if p == pid]
        re_ivs = [(s, e) for p, s, e in retrieved if p == pid]
        total_evidence += _union_size(ev_ivs)
        total_retrieved += _union_size(re_ivs)
        covered += _intersection_size(ev_ivs, re_ivs)
    precision = covered / total_retrieved if total_retrieved else 0.0
    recall = covered / total_evidence if total_evidence else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Reward function factories
# ---------------------------------------------------------------------------

def _parse_answer_substrings(final_answer: str) -> List[str]:
    """Parse a final answer (usually a Python list literal) into text substrings."""
    import ast
    try:
        substrings = ast.literal_eval(final_answer)
        if isinstance(substrings, str):
            substrings = [substrings]
        elif isinstance(substrings, (list, tuple)):
            substrings = [s if isinstance(s, str) else str(s) for s in substrings]
        else:
            substrings = [str(substrings)]
    except (ValueError, SyntaxError):
        substrings = [s.strip() for s in final_answer.split("\n\n") if s.strip()]
    return substrings


def make_reward_fn(ctx: str, evidence: List[str]):
    """Return a reward_fn(final_answer: str) -> float that scores F1.

    final_answer is expected to be a Python list literal of retrieved text
    snippets (as produced by FINAL_VAR on a list variable), or a plain string.
    Evidence intervals are located by substring search in ctx.
    """
    evidence_intervals: List[Tuple[int, int]] = []
    for ev in evidence:
        idx = ctx.find(ev.strip())
        if idx != -1:
            evidence_intervals.append((idx, idx + len(ev.strip())))

    def reward_fn(final_answer: str) -> float:
        substrings = _parse_answer_substrings(final_answer)
        retrieved_intervals: List[Tuple[int, int]] = []
        for s in substrings:
            idx = ctx.find(s)
            if idx != -1:
                retrieved_intervals.append((idx, idx + len(s)))
        metrics = compute_metrics(retrieved_intervals, evidence_intervals)
        return metrics["f1"]

    return reward_fn


def make_reward_fn_multipaper(ctx: Dict[str, str], evidence: List[Dict]):
    """Return a reward_fn(final_answer: str) -> float for multi-paper F1.

    evidence is a list of {paperId, selections: [{text: ...}]} dicts.
    final_answer is expected to be a Python list literal of retrieved text
    snippets (exact substrings from any paper in ctx).
    """
    evidence_intervals: List[Tuple[str, int, int]] = []
    for ev in evidence:
        paper_id = ev.get("paperId", "")
        paper_text = ctx.get(paper_id) or ""
        for sel in ev.get("selections", []):
            text = sel.get("text", "").strip()
            if not text:
                continue
            idx = paper_text.find(text)
            if idx != -1:
                evidence_intervals.append((paper_id, idx, idx + len(text)))

    def reward_fn(final_answer: str) -> float:
        substrings = _parse_answer_substrings(final_answer)
        retrieved_intervals: List[Tuple[str, int, int]] = []
        for s in substrings:
            for paper_id, paper_text in ctx.items():
                if not paper_text:
                    continue
                idx = paper_text.find(s)
                if idx != -1:
                    retrieved_intervals.append((paper_id, idx, idx + len(s)))
                    break
        metrics = compute_metrics_multipaper(retrieved_intervals, evidence_intervals)
        return metrics["f1"]

    return reward_fn


# ---------------------------------------------------------------------------
# Per-trajectory child RLM metrics (logged under environment/ on wandb)
# ---------------------------------------------------------------------------

def compute_child_rlm_metrics(
    child_call_records: List[Dict[str, Any]],
    evidence: List[Dict[str, Any]],
    parent_context: Dict[str, str],
) -> Dict[str, float]:
    """Compute per-trajectory child RLM metrics for wandb ``environment/`` logging.

    Args:
        child_call_records: one entry per child dispatch with keys
            ``paper_id`` (str | None), ``final_answer`` (str | None),
            ``had_final_answer`` (bool).
        evidence: ground-truth evidence list from ``reward_spec`` —
            ``[{paperId, selections: [{text}]}]``.
        parent_context: the parent's context dict ``{paper_id: paper_text}``.

    Returns:
        Dict with ``child_submission_rate``, ``paper_selection_f1``, and
        ``child_evidence_char_f1``.
    """
    if not child_call_records:
        return {
            "child_submission_rate": 0.0,
            "paper_selection_f1": 0.0,
            "child_evidence_char_f1": 0.0,
        }

    # --- 1. Child submission rate ---
    n_submitted = sum(1 for r in child_call_records if r["had_final_answer"])
    child_submission_rate = n_submitted / len(child_call_records)

    # --- 2. Paper selection F1 (set-level over paper IDs) ---
    gt_paper_ids: set = set()
    paper_evidence_map: Dict[str, List[str]] = {}
    for ev in (evidence or []):
        pid = ev.get("paperId", "")
        texts = [s.get("text", "").strip() for s in ev.get("selections", []) if s.get("text", "").strip()]
        if texts and pid:
            gt_paper_ids.add(pid)
            paper_evidence_map.setdefault(pid, []).extend(texts)

    selected_paper_ids = {r["paper_id"] for r in child_call_records if r["paper_id"] is not None}

    if gt_paper_ids or selected_paper_ids:
        intersection = gt_paper_ids & selected_paper_ids
        precision = len(intersection) / len(selected_paper_ids) if selected_paper_ids else 0.0
        recall = len(intersection) / len(gt_paper_ids) if gt_paper_ids else 0.0
        paper_selection_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    else:
        paper_selection_f1 = 0.0

    # --- 3. Child evidence char-level F1 (only children on GT papers) ---
    f1_scores: List[float] = []
    for record in child_call_records:
        pid = record["paper_id"]
        if pid is None or pid not in paper_evidence_map:
            continue

        if not record["had_final_answer"] or not record["final_answer"]:
            f1_scores.append(0.0)
            continue

        paper_text = parent_context.get(pid, "")
        evidence_intervals: List[Tuple[int, int]] = []
        for ev_text in paper_evidence_map[pid]:
            idx = paper_text.find(ev_text)
            if idx != -1:
                evidence_intervals.append((idx, idx + len(ev_text)))

        if not evidence_intervals:
            f1_scores.append(0.0)
            continue

        substrings = _parse_answer_substrings(record["final_answer"])
        retrieved_intervals: List[Tuple[int, int]] = []
        for s in substrings:
            idx = paper_text.find(s)
            if idx != -1:
                retrieved_intervals.append((idx, idx + len(s)))

        metrics = compute_metrics(retrieved_intervals, evidence_intervals)
        f1_scores.append(metrics["f1"])

    child_evidence_char_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    return {
        "child_submission_rate": child_submission_rate,
        "paper_selection_f1": paper_selection_f1,
        "child_evidence_char_f1": child_evidence_char_f1,
    }

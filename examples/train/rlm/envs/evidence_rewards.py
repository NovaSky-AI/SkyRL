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
    for ev in evidence or []:
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


# ---------------------------------------------------------------------------
# LLM-judge reward
# ---------------------------------------------------------------------------

_RUBRIC_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "RubricScore",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "precision_score": {
                    "type": "integer",
                    "description": "1-10: how tight and accurate the spans are (no extraneous sentences, no off-topic padding)",
                },
                "recall_score": {
                    "type": "integer",
                    "description": "1-10: how thoroughly the evidence covers what is needed to answer the question",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of the scores",
                },
            },
            "required": ["precision_score", "recall_score", "reasoning"],
            "additionalProperties": False,
        },
    },
}


def _extract_gt_strings(evidence: List[Any]) -> List[str]:
    out: List[str] = []
    for ev in evidence or []:
        if isinstance(ev, str):
            out.append(ev)
        elif isinstance(ev, dict):
            for sel in ev.get("selections", []):
                text = sel.get("text", "").strip()
                if text:
                    out.append(text)
    return out


def judge_reward(
    final_answer: str,
    question: str,
    evidence: List[Any],
    model: str = "gpt-4.1-nano",
    base_url: str = "https://api.openai.com/v1",
) -> Tuple[float, float, float]:
    """Score a final answer with an LLM judge.

    Returns ``(reward, precision, recall)`` where precision and recall are
    0-1 and reward is ``(precision + recall) / 2``.
    """
    import ast
    import json
    import os
    import textwrap
    import time

    import httpx
    from loguru import logger

    gt_strings = _extract_gt_strings(evidence)

    try:
        predicted = ast.literal_eval(final_answer)
        if isinstance(predicted, str):
            predicted = [predicted]
        elif isinstance(predicted, (list, tuple)):
            predicted = [s if isinstance(s, str) else str(s) for s in predicted]
        else:
            predicted = [str(predicted)]
    except (ValueError, SyntaxError):
        predicted = [s.strip() for s in final_answer.split("\n\n") if s.strip()]

    gt_block = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(gt_strings)) or "(none)"
    pred_block = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(predicted)) or "(none)"

    user_msg = textwrap.dedent(
        f"""\
        You are evaluating predicted evidence extractions against ground truth evidence for a question.
        Treat the ground truth evidence as a perfect 10/10 reference for all dimensions.

        Question: {question}

        Ground truth evidence (reference — treat as 10/10 on all dimensions):
        {gt_block}

        Predicted evidence:
        {pred_block}

        Score the predicted evidence on two dimensions (1-10 each):

        PRECISION SCORE — are the spans tight and free of off-topic padding?
          10 — every span contains only directly relevant sentences; no extraneous setup, headers, or filler
           9 — essentially tight; one trivially redundant phrase but no real noise
           8 — minor padding (1-2 extra sentences) but core content is accurate
           7 — a few extra sentences that are related but not strictly necessary
           6 — noticeable extraneous text in some spans, but the relevant parts are present
           5 — roughly half the content is relevant; half is filler or tangential
           4 — spans are significantly bloated with irrelevant surrounding text
           3 — small fraction of each span is on-topic; most is irrelevant context
           2 — most of each span is irrelevant; the relevant fragment is buried
           1 — extractions are almost entirely off-topic or wrong

        RECALL SCORE — do the predicted spans collectively cover what is needed to answer the question?
          10 — all key facts from the ground truth are present; nothing important missing
           9 — all key facts present; only a trivially minor detail absent
           8 — most key facts covered; one minor detail from the ground truth absent
           7 — core answer present; a couple of supporting details from the ground truth missing
           6 — core answer present but a meaningful portion of the ground truth evidence is missing
           5 — about half the ground truth evidence is covered; half is missing
           4 — partial answer only; several important aspects from the ground truth are absent
           3 — a few relevant facts retrieved but most of the ground truth evidence is missing
           2 — barely touches the question; most of the ground truth evidence is missing
           1 — no relevant content retrieved

        Provide a brief reasoning string (2-4 sentences) explaining the scores.
    """
    )

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable must be set when using a judge reward")

    base_url = base_url.rstrip("/")
    result = None
    for attempt in range(5):
        if attempt > 0:
            time.sleep(2**attempt)
        try:
            with httpx.Client(timeout=60) as client:
                resp = client.post(
                    f"{base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": user_msg}],
                        "temperature": 0,
                        "response_format": _RUBRIC_RESPONSE_FORMAT,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            if attempt == 4:
                logger.warning(f"Judge reward model failed after 5 attempts: {e}")
                return 0.0, 0.0, 0.0
            logger.warning(f"Judge reward model attempt {attempt + 1} failed: {e}, retrying...")
            continue

        try:
            result = json.loads(data["choices"][0]["message"]["content"])
            break
        except json.JSONDecodeError:
            if attempt == 4:
                return 0.0, 0.0, 0.0
            continue

    if result is None:
        return 0.0, 0.0, 0.0

    precision = result.get("precision_score", 0) / 10.0
    recall = result.get("recall_score", 0) / 10.0
    return (precision + recall) / 2.0, precision, recall

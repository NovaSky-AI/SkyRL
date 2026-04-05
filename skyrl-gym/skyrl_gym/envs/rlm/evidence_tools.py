"""
Evidence-based reward function and REPL tool factories for text-span retrieval tasks.

Reward is F1 over retrieved text intervals vs. ground-truth evidence spans
(ported from rlm/utils/evals.py).

Tools (search, extract_section) are per-example closures that capture the
context string; they are injected into the REPL when reward_spec contains
an "evidence" field.
"""

import re
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Interval metrics (from rlm/utils/evals.py)
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


# ---------------------------------------------------------------------------
# Reward function factory
# ---------------------------------------------------------------------------

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

        retrieved_intervals: List[Tuple[int, int]] = []
        for s in substrings:
            idx = ctx.find(s)
            if idx != -1:
                retrieved_intervals.append((idx, idx + len(s)))

        metrics = compute_metrics(retrieved_intervals, evidence_intervals)
        return metrics["f1"]

    return reward_fn


# ---------------------------------------------------------------------------
# Multi-paper reward function factory
# ---------------------------------------------------------------------------

def make_reward_fn_multipaper(ctx: Dict[str, str], evidence: List[Dict]):
    """Return a reward_fn(final_answer: str) -> float for multi-paper F1.

    evidence is a list of {paperId, selections: [{text: ...}]} dicts.
    final_answer is expected to be a Python list literal of retrieved text
    snippets (exact substrings from any paper in ctx).
    """
    evidence_intervals: List[Tuple[str, int, int]] = []  # (paper_id, start, end)
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
# REPL tool factory
# ---------------------------------------------------------------------------

def make_tools() -> Dict[str, Any]:
    """Build search/extract tools for dictionary-based paper context."""

    def list_papers(ctx: dict) -> list:
        """List all paper IDs with title and abstract."""
        print(f"Found {len(ctx)} papers:")
        titles = []
        for paper_id, content in ctx.items():
            lines = content.split("\n")
            title = lines[0].replace("### PAPER: ", "") if lines else "Unknown Title"
            abstract_match = re.search(r"<abstract>\n(.*?)\n</abstract>", content, re.DOTALL)
            abstract = abstract_match.group(1) if abstract_match else ""
            print(f"\nPaper ID: {paper_id}")
            print(f"Title: {title}")
            if abstract:
                print(f"Abstract: {abstract}")
            print("-" * 80)
            titles.append(title)
        return titles

    def _search_text(text: str, keyword: str, window: int) -> list:
        results = []
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        for m in pattern.finditer(text):
            left = max(0, m.start() - window // 2)
            right = min(len(text), m.end() + window // 2)
            while left > 0 and text[left - 1] not in ".!?\n":
                left -= 1
                if m.start() - left > window:
                    break
            while right < len(text) and text[right] not in ".!?\n":
                right += 1
                if right - m.end() > window:
                    break
            if right < len(text) and text[right] in ".!?\n":
                right += 1
            snippet = text[left:right]
            start_line = text[:left].count("\n") + 1
            snippet_lines = snippet.split("\n")
            numbered_lines = [f"L{start_line + i}: {line}" for i, line in enumerate(snippet_lines)]
            idx = len(results)
            print(f"--- snippet {idx} (L{start_line}) ---")
            print("\n".join(numbered_lines))
            results.append(snippet)
        return results

    def search(text, keyword: str, window: int = 300) -> list:
        """Keyword search within a text string or across all papers in a dict."""
        if isinstance(text, dict):
            results = []
            for paper_id, paper_text in text.items():
                title_line = paper_text.split("\n")[0].replace("### PAPER: ", "")
                paper_results = _search_text(paper_text, keyword, window)
                if paper_results:
                    print(f"\n=== Paper: {paper_id} — {title_line} ===")
                    results.extend(paper_results)
            if not results:
                print(f"(no hits for {keyword!r} in any paper)")
            return results
        else:
            results = _search_text(text, keyword, window)
            if not results:
                print(f"(no hits for {keyword!r})")
            return results

    def extract_lines(text: str, start_line: int, end_line: int) -> str:
        """Extract lines from start_line to end_line (inclusive, 1-indexed) from a text string."""
        all_lines = text.split("\n")
        s = max(0, start_line - 1)
        e = min(len(all_lines), end_line)
        if s >= len(all_lines):
            print(f"ERROR: start_line {start_line} is beyond end of text ({len(all_lines)} lines)")
            return ""
        result = "\n".join(all_lines[s:e])
        if len(result) > 2000:
            print(f"WARNING: extraction is {len(result)} chars (limit 2000). Truncating. Use a tighter line range.")
            result = result[:2000]
        print(f"extract_lines({start_line}, {end_line}):")
        print(result)
        return result

    def get_paper_abstract(ctx: dict, paper_id: str) -> str:
        """Return a formatted string with the paper ID, title, and abstract."""
        paper_text = ctx.get(paper_id, "")
        lines = paper_text.split("\n")
        title = lines[0].replace("### PAPER: ", "") if lines else "Unknown Title"
        abstract_match = re.search(r"<abstract>\n(.*?)\n</abstract>", paper_text, re.DOTALL)
        abstract = abstract_match.group(1) if abstract_match else ""
        return f"Paper ID: {paper_id}\nTitle: {title}\nAbstract: {abstract}"

    return {
        "list_papers": list_papers,
        "search": search,
        "extract_lines": extract_lines,
        "get_paper_abstract": get_paper_abstract,
    }

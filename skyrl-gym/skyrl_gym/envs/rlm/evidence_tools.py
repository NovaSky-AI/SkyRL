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
# REPL tool factory (search + extract_section, ported from rlm/examples/eval.py)
# ---------------------------------------------------------------------------

def make_tools(ctx: str) -> Dict[str, Any]:
    """Build search/extract_section closures that capture a per-example context."""

    def _merge(items: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
        if not items:
            return []
        intervals = sorted([(s, s + len(t)) for s, t in items])
        merged = [intervals[0]]
        for s, e in intervals[1:]:
            if s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        return [(s, ctx[s:e]) for s, e in merged]

    def search(
        keyword: str,
        window: int = 300,
        max_snippets: int = 10,
        bidirectional: bool = True,
    ) -> List[str]:
        results = []
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        for m in pattern.finditer(ctx):
            if bidirectional:
                left = max(0, m.start() - window // 2)
                right = min(len(ctx), m.end() + window // 2)
            else:
                left = m.start()
                right = min(len(ctx), m.start() + window)
            while left > 0 and ctx[left - 1] not in ".!?\n":
                left -= 1
                if m.start() - left > (window if bidirectional else 100):
                    break
            while right < len(ctx) and ctx[right] not in ".!?\n":
                right += 1
                if right - m.end() > window:
                    break
            if right < len(ctx) and ctx[right] in ".!?\n":
                right += 1
            results.append((left, ctx[left:right]))
        merged = _merge(results)
        shown = merged[:max_snippets]
        remaining = len(merged) - len(shown)
        snippets = []
        for _, snippet in shown:
            idx = len(snippets)
            print(f"--- snippet {idx} ---")
            print(snippet)
            snippets.append(snippet)
        if not shown:
            print(f"(no hits for {keyword!r})")
        if remaining > 0:
            print(f"(+{remaining} more)")
        return snippets

    def extract_section(snippet: str, start_phrase: str, end_phrase: str) -> str:
        si = snippet.lower().find(start_phrase.lower())
        if si == -1:
            si = 0
        ei = snippet.lower().find(end_phrase.lower(), si)
        if ei == -1:
            result = snippet[si:]
        else:
            result = snippet[si: ei + len(end_phrase)]
        print(result)
        return result

    return {
        "search": {
            "tool": search,
            "description": "search(keyword, window=300, max_snippets=10, bidirectional=True) -> list[str]: search context for keyword, returns surrounding snippets",
        },
        "extract_section": {
            "tool": extract_section,
            "description": "extract_section(snippet, start_phrase, end_phrase) -> str: extract substring from snippet between two phrases",
        },
    }


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
# Multi-paper REPL tool factory
# ---------------------------------------------------------------------------

def make_tools_multipaper(ctx: Dict[str, str]) -> Dict[str, Any]:
    """Build tools for a dict-based multi-paper context."""

    def list_papers(context: dict) -> List[str]:
        print(f"Found {len(context)} papers:")
        titles = []
        for paper_id, content in context.items():
            lines = content.split("\n")
            title = lines[0].replace("### PAPER: ", "") if lines else "Unknown Title"
            abstract_match = re.search(r"<abstract>\n(.*?)\n</abstract>", content, re.DOTALL)
            abstract = abstract_match.group(1) if abstract_match else ""
            print(f"\nPaper ID: {paper_id}")
            print(f"Title: {title}")
            if abstract:
                preview = abstract[:300] + ("..." if len(abstract) > 300 else "")
                print(f"Abstract: {preview}")
            print("-" * 80)
            titles.append(title)
        return titles

    def _search_single(text: str, keyword: str, window: int) -> List[str]:
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
            numbered = [f"L{start_line + i}: {line}" for i, line in enumerate(snippet_lines)]
            idx = len(results)
            print(f"--- snippet {idx} (L{start_line}) ---")
            print("\n".join(numbered))
            results.append(snippet)
        return results

    def search(text, keyword: str, window: int = 300) -> List[str]:
        """Keyword search within a paper string or across all papers in a dict."""
        if isinstance(text, dict):
            results = []
            for paper_id, paper_text in text.items():
                title_line = paper_text.split("\n")[0].replace("### PAPER: ", "")
                paper_results = _search_single(paper_text, keyword, window)
                if paper_results:
                    print(f"\n=== Paper: {paper_id} — {title_line} ===")
                    results.extend(paper_results)
            if not results:
                print(f"(no hits for {keyword!r} in any paper)")
            return results
        else:
            results = _search_single(text, keyword, window)
            if not results:
                print(f"(no hits for {keyword!r})")
            return results

    def extract_lines(text: str, start_line: int, end_line: int) -> str:
        """Extract lines from start_line to end_line (inclusive, 1-indexed)."""
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

    def get_paper_abstract(context: dict, paper_id: str) -> str:
        """Return paper ID, title, and abstract as a formatted string."""
        paper_text = context.get(paper_id, "")
        lines = paper_text.split("\n")
        title = lines[0].replace("### PAPER: ", "") if lines else "Unknown Title"
        abstract_match = re.search(r"<abstract>\n(.*?)\n</abstract>", paper_text, re.DOTALL)
        abstract = abstract_match.group(1) if abstract_match else ""
        return f"Paper ID: {paper_id}\nTitle: {title}\nAbstract: {abstract}"

    return {
        "list_papers": {
            "tool": list_papers,
            "description": "list_papers(context) -> list[str]: list all paper IDs with title and abstract preview",
        },
        "search": {
            "tool": search,
            "description": (
                "search(text, keyword, window=300) -> list[str]: keyword search. "
                "Pass context dict to search ALL papers (results grouped by paper), "
                "or pass context[paper_id] to search a single paper. "
                "Each line in snippets is prefixed with its line number (e.g. L42: ...)."
            ),
        },
        "extract_lines": {
            "tool": extract_lines,
            "description": "extract_lines(text, start_line, end_line) -> str: extract lines start_line..end_line (1-indexed) from a paper string; pass context[paper_id] as text",
        },
        "get_paper_abstract": {
            "tool": get_paper_abstract,
            "description": "get_paper_abstract(context, paper_id) -> str: return paper ID, title, and abstract",
        },
    }

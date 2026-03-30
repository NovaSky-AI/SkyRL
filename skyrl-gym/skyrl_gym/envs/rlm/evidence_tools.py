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
            elif not isinstance(substrings, list):
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

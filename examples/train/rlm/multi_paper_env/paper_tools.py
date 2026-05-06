"""REPL tool factory for QASPER-shaped paper context.

Builds search/extract helpers that operate on a dict-of-papers context, where
each paper text starts with ``### PAPER: <title>`` and contains an
``<abstract>...</abstract>`` block. Used by ``EvidenceRLMEnv``.
"""

import re
from typing import Any, Dict


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

    def _search_text(text: str, keyword: str, window: int, bidirectional: bool = True) -> list:
        results = []
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        for m in pattern.finditer(text):
            if bidirectional:
                left = max(0, m.start() - window // 2)
                right = min(len(text), m.end() + window // 2)
            else:
                left = m.start()
                right = min(len(text), m.start() + window)
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
            idx = len(results)
            print(f"--- snippet {idx} ---")
            print(snippet)
            results.append(snippet)
        return results

    def search(text, keyword: str, window: int = 300, bidirectional: bool = True) -> list:
        """Keyword search within a text string or across all papers in a dict."""
        if isinstance(text, dict):
            results = []
            for paper_id, paper_text in text.items():
                title_line = paper_text.split("\n")[0].replace("### PAPER: ", "")
                paper_results = _search_text(paper_text, keyword, window, bidirectional)
                if paper_results:
                    print(f"\n=== Paper: {paper_id} — {title_line} ===")
                    results.extend(paper_results)
            if not results:
                print(f"(no hits for {keyword!r} in any paper)")
            return results
        else:
            results = _search_text(text, keyword, window, bidirectional)
            if not results:
                print(f"(no hits for {keyword!r})")
            return results

    def extract_section(snippet: str, start_phrase: str, end_phrase: str) -> str:
        """Extract a substring from snippet starting at start_phrase and ending at end_phrase (inclusive)."""
        si = snippet.lower().find(start_phrase.lower())
        if si == -1:
            si = 0
        ei = snippet.lower().find(end_phrase.lower(), si)
        if ei == -1:
            result = snippet[si:]
        else:
            result = snippet[si : ei + len(end_phrase)]
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
        "extract_section": extract_section,
        "get_paper_abstract": get_paper_abstract,
    }

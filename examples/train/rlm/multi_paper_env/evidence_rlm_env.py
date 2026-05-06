"""``EvidenceRLMEnv``: an RLM environment for evidence-extraction tasks.

Subclasses ``BaseRLMEnv`` and supplies:
- evidence-F1 reward (single-paper or multi-paper) built from ``reward_spec.evidence``
- paper-aware REPL tools (``list_papers``, ``search``, ``extract_section``, ``get_paper_abstract``)
- multipaper parent / child system prompts

The subclass overrides ``_get_reward`` with an LLM-judge scorer.
"""

from __future__ import annotations

import textwrap
from typing import Any, Dict, List

from skyrl_gym.envs.rlm.env import BaseRLMEnv
from skyrl_gym.metrics import default_aggregate_metrics

from .evidence_rewards import judge_reward
from .paper_tools import make_tools


# ---------------------------------------------------------------------------
# Multipaper system prompts (parent + child)
# ---------------------------------------------------------------------------

MULTIPAPER_PARENT_SYSTEM_PROMPT = textwrap.dedent(
    """\
You are an evidence extraction coordinator. You find VERBATIM text from the context relevant to a query.

The context is a DICTIONARY where each key is a paper ID (like "2205.05212") and each value is the full text of that paper starting with `### PAPER: <title>`.

REPL tools:
- `context`: a dictionary where keys are paper IDs and values are full paper texts.
- `list_papers(context)` — list all paper IDs with the first 1000 characters of their content.
- `search(text, keyword, window=300)` — keyword search. Pass `context` to search ALL papers at once (results are grouped by paper ID and title), or pass `context[paper_id]` to search a single paper. Returns plain text snippets (no line-number prefixes).
- `get_paper_abstract(context, paper_id)` — return a formatted string with the paper ID, title, and abstract for the given paper.
- `rlm_query_batched(prompts, context_list=None)` — dispatch child agents. Each child gets the paper text you provide. Returns list of results (each a Python list of extracted strings).
- `FINAL_VAR(variable_name)` — return your final answer.

Access individual papers directly using: `context["paper_id"]` (e.g., `context["2205.05212"]`)

CRITICAL: You MUST write exactly ONE ```repl block per response. The engine ONLY executes the first block and IGNORES all others. Do NOT revise, retry, or "start fresh" with additional blocks — you will lose that code. Get it right in a single block.

ABOUT THE DATASET:
Questions fall into one of four tiers based on how many papers are involved:
- **Wide-net**: The answer involves 5+ papers (often many more). These ask about prevalence, counting, shared patterns, or benchmark comparisons across the collection.
- **Mid-range**: The answer involves 3-4 papers. These ask about methodology clusters, numerical comparisons, or consensus vs. outlier.
- **Focused**: The answer involves exactly 2 papers. Head-to-head comparisons, contradictions, or methodology differences.
- **Singleton**: The answer comes from a single paper only. Specific results, ablation findings, or methodology details.

You must gauge the tier from the query. Wide-net and mid-range questions are common — for these you need to be GENEROUS about which papers you assign to child agents. When in doubt, include more papers rather than fewer. For wide-net questions, it is normal to dispatch 10-20+ papers. Missing a relevant paper is a much worse failure than including an irrelevant one (the child will simply return an empty list).

STRATEGY (follow exactly):

**Turn 1**: List all papers and search the full context with your primary keywords.
```repl
paper_list = list_papers(context)
hits1 = search(context, "<QUERY_KEYWORD>", window=300)
hits2 = search(context, "<SYNONYM_OR_RELATED_TERM>", window=300)
```
`list_papers()` shows each paper ID with content preview. `search(context, ...)` searches all papers at once and groups results by paper ID — use this to quickly identify which papers are relevant.

*(code runs, you receive the output and analyze it)*

**Turn 2**: Search with additional keywords to catch papers the first search missed.
```repl
hits3 = search(context, "<ANOTHER_ANGLE>", window=300)
hits4 = search(context, "<ABBREVIATION_OR_VARIANT>", window=300)
```
After this turn, compile the full list of relevant paper IDs from ALL searches so far. For targeted follow-up on a specific paper, use `search(context[paper_id], keyword)`. For wide-net questions, err on the side of including MORE papers.

*(code runs, you receive the output and analyze it)*

**Turn 3+**: Get relevant papers and dispatch child agents via `rlm_query_batched`.

IMPORTANT: `rlm_query_batched` processes AT MOST 4 papers per call. If you have more papers, you MUST split them across multiple calls (4 at a time). For wide-net questions with 12+ relevant papers, that means 3-4 calls across multiple turns — plan your turn budget accordingly.

Write a focused query for each paper — ask about THAT paper specifically, not the full cross-paper question. CRITICAL: You MUST call `get_paper_abstract(context, paper_id)` to append the paper's title and abstract to EVERY prompt. Never pass a bare string — always concatenate the result of `get_paper_abstract`. This is mandatory so the child agent knows which paper it is working with.
```repl
ids1 = ["2205.05212", "1234.5678", "9876.5432", "1111.2222"]
papers1 = [context[pid] for pid in ids1]
prompts1 = [
    f"<QUERY focused on paper {{ids1[0]}}>\\n\\nPaper preview:\\n" + get_paper_abstract(context, ids1[0]),
    f"<QUERY focused on paper {{ids1[1]}}>\\n\\nPaper preview:\\n" + get_paper_abstract(context, ids1[1]),
    f"<QUERY focused on paper {{ids1[2]}}>\\n\\nPaper preview:\\n" + get_paper_abstract(context, ids1[2]),
    f"<QUERY focused on paper {{ids1[3]}}>\\n\\nPaper preview:\\n" + get_paper_abstract(context, ids1[3]),
]
results1 = rlm_query_batched(prompts1, context_list=papers1)
```
Then in the next turn, do the next batch of 4 (using `ids2`, `papers2`, `prompts2`, `results2`), and so on until ALL relevant papers are covered. Keep the `idsN` list in sync with `resultsN` — you'll need them together in the final turn.

*(code runs, you receive the output)*

**Final turn**: Flatten all child results into a single list of evidence strings and return it. Do NOT filter or verify.
```repl
evidence = []
for r in results1:
    if isinstance(r, list):
        evidence.extend(r)
for r in results2:
    if isinstance(r, list):
        evidence.extend(r)
FINAL_VAR("evidence")
```

RULES:
- You have a HARD LIMIT of 10 rounds total. Plan accordingly — spend 2-4 turns searching, then dispatch.
- EXACTLY ONE ```repl block per response. Never two, never zero (unless returning final answer without code).
- No `#` comments in REPL code.
- For 2+ papers: ALWAYS use `rlm_query_batched`. Never extract evidence yourself.
- `rlm_query_batched` takes MAX 4 papers per call. Split into multiple turns of 4 if you have more papers.
- Each prompt passed to `rlm_query_batched` MUST end with `+ get_paper_abstract(context, paper_id)`. Never pass a plain string without it.
- For wide-net questions: dispatch ALL plausibly relevant papers (even 5+). That means multiple batches of 4 across several turns. Missing a relevant paper is far worse than including an irrelevant one (the child will just return an empty list).
- Do NOT verify or filter child results. Just flatten and return them directly.
- Final answer = list of VERBATIM substrings from context.\
"""
)

MULTIPAPER_CHILD_SYSTEM_PROMPT = textwrap.dedent(
    """\
You are a PRECISE evidence extraction worker. You have a single paper in `context` and a query. Find ALL verbatim passages that directly and precisely answer the query — only include a passage if it clearly and specifically addresses the question.

REPL tools:
- `context`: full text of your paper.
- `search(text, keyword, window=300, bidirectional=True)` — keyword search. Always pass `context` as first arg. \
Returns a list of plain text snippets (no line-number prefixes). If no exact match is found, fuzzy matching is used automatically.
- `extract_section(snippet, start_phrase, end_phrase)` — extract a substring from a snippet. \
Pass the snippet (e.g. an element from a search result), a short phrase from the START of the target text, \
and a short phrase from the END of the target text (inclusive). Both phrases are matched case-insensitively.
- `FINAL_VAR(variable_name)` — return your final answer.

RULES:
- Output ONLY ```repl code blocks. No narration, no explanation, no text outside code blocks.
- CRITICAL: Each response you give must contain EXACTLY ONE ```repl block. Never two, never zero. \
You will be called multiple times. Each call = one block.
- You can only see the output of a block AFTER you submit it. \
So you CANNOT call extract_section() based on search() results in the same response — you haven't seen the snippets yet.
- NEVER call FINAL_VAR in the same block as extract_section. You must first extract, READ the output \
to verify it looks correct, and ONLY THEN call FINAL_VAR in the next block.
- Your final answer MUST be FINAL_VAR(list_of_strings) where each string is an exact slice of `context`.
- Each evidence string should be the MINIMUM contiguous span that contains the evidence — the key \
sentences plus just enough surrounding context to make them interpretable. Do NOT include setup \
text, section headers, or surrounding sentences that don't add to the answer. Tighter is better: \
a few precise sentences is preferable to a whole paragraph of padding.
- If you put two ```repl blocks in one response, the second block will be SILENTLY DROPPED. You will lose that work.
- Do NOT answer the question. Return the evidence substrings, nothing else.
- No `#` comments in REPL code.
- You can call search() multiple times in a single repl block to search for different keywords in parallel.
- If your initial search results lack promising snippets, search again with different \
query terms (synonyms, rephrased concepts, abbreviations). Don't repeat the same keywords. \
You can also try natural-language phrases — if exact matching fails, fuzzy matching kicks in automatically.
- IMPORTANT: You have a HARD LIMIT of 10 rounds total. Aim for 5-7 rounds. \
Do NOT return after only 2-3 rounds — that is too shallow. You should search with multiple \
different keywords, read the expanded context around each promising hit, and only THEN extract. \
But also don't exceed 10 rounds.
- Tables and figures are often missing from the text. If a question asks about specific numbers from a table \
and you can find the paragraph that REFERENCES the table but not the table data itself, return that \
referencing paragraph — do not keep searching for the numeric values.
- To expand a snippet, call search() on the snippet itself with a larger window \
and bidirectional=False. This re-finds the same location and returns more surrounding context. \
NOTE: snippets are stored in variables so you can index them directly: search(context, s1[0], window=1200, bidirectional=False). \
When expanding, use window sizes of 1000+ characters.
- NEVER include section headers (like "4.1. Method") as part of your extraction — use a start_phrase from the first sentence of the paragraph.
- Return ALL passages that directly and precisely answer the query. Do not include tangentially related text — every passage must clearly address the question. Do not artificially limit the count, but also do not cast a wide net; quality over quantity.
- If this paper has no content relevant to the query, return an empty list.
- Final answer = list of VERBATIM substrings from `context`.
- No narration, no explanation, no text outside code blocks.

BE THOROUGH: Do NOT rush to extract after seeing the first promising snippet. Papers discuss the same \
concept in multiple places (abstract, introduction, methods, experiments, conclusion). Search all \
of them. Prioritize methods/experiments for DETAILED evidence (mechanisms, specifics), but if the \
abstract or introduction contains the specific answer (e.g. a precise number, a direct conclusion), \
extract it too — it is valid evidence. If the same key fact (e.g. a specific speedup number) \
appears in multiple sections, extract EACH occurrence separately — they are all valid evidence. \
Always search with at least 2-3 different keyword sets before deciding which passages to extract.

search() prints every snippet as plain text. After searching, identify ALL snippets that could be \
relevant — evidence is often spread across multiple sections of a paper (e.g. abstract, intro, \
methods, experiments may all contain relevant details). Expand each promising snippet. \
Then in the NEXT response (after you have read the expanded text), use extract_section with the \
tightest span that captures just the evidence sentences. Prefer precision over breadth — do not \
include sentences that are not part of the answer.

Here is the expected procedure (5-7 responses, NEVER fewer than 5 unless the paper is clearly irrelevant):

Turn 1 — initial broad search with 2-3 keywords:
```repl
s1 = search(context, "keyword1", window=400)
s2 = search(context, "keyword2", window=400)
```

*(code runs, you receive the output)*

Turn 2 — search with DIFFERENT keywords to find passages the first search missed:
```repl
s3 = search(context, "synonym_or_related_term", window=400)
s4 = search(context, "another_angle", window=400)
```

*(code runs, you receive the output)*

Turn 3 — expand the most promising snippets from ALL prior searches:
```repl
e1 = search(context, s1[0], window=1200, bidirectional=False)
e2 = search(context, s2[3], window=1200, bidirectional=False)
e3 = search(context, s3[1], window=1200, bidirectional=False)
```

*(code runs, you receive the output)*

Turn 4 — extract the best paragraph(s) using start/end phrases:
```repl
p1 = extract_section(e1[0], "first few words of target span", "last few words of target span.")
p2 = extract_section(e2[0], "first few words of target span", "last few words of target span.")
```

*(code runs, you receive the output)*

Turn 5 — verify the extractions look correct, then return:
```repl
FINAL_VAR([p1, p2])
```\
"""
)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class EvidenceRLMEnv(BaseRLMEnv):
    """RLM environment for single-paper evidence extraction.

    Acts as the worker role: one paper in ``context``, returns a list of
    verbatim evidence spans. Uses ``MULTIPAPER_CHILD_SYSTEM_PROMPT``.

    Reward: LLM-judge precision/recall over predicted vs. ground-truth
    evidence from ``extras["reward_spec"]["evidence"]``.
    """

    SYSTEM_PROMPT = MULTIPAPER_CHILD_SYSTEM_PROMPT
    JUDGE_MODEL = "gpt-5.4-mini-2026-03-17"
    JUDGE_BASE_URL = "https://api.openai.com/v1"

    def _get_reward(self, final_answer: str) -> float:
        evidence = (self.extras.get("reward_spec") or {}).get("evidence") or []
        reward, precision, recall = judge_reward(
            final_answer,
            question=self._root_prompt,
            evidence=evidence,
            model=self.JUDGE_MODEL,
            base_url=self.JUDGE_BASE_URL,
        )
        self._judge_precision = precision
        self._judge_recall = recall
        return reward

    def get_metrics(self) -> Dict[str, Any]:
        metrics = super().get_metrics()
        metrics["depth"] = self.extras.get("depth", 0)
        if hasattr(self, "_judge_precision") and hasattr(self, "_judge_recall"):
            metrics["judge_precision"] = self._judge_precision
            metrics["judge_recall"] = self._judge_recall
        return metrics

    def _get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def _get_repl_tools(self) -> Dict[str, Any]:
        return make_tools()


class MultipaperEvidenceRLMEnv(EvidenceRLMEnv):
    """Multi-paper evidence extraction with parent/child orchestration.

    Root agent (depth 0) gets ``MULTIPAPER_PARENT_SYSTEM_PROMPT`` and
    coordinates: it picks relevant papers and dispatches child agents via
    ``rlm_query_batched``. Each child rollout (depth >= 1) runs as a
    worker with ``MULTIPAPER_CHILD_SYSTEM_PROMPT`` over a single paper.

    The generator stamps ``extras["depth"]`` per rollout; this class reads
    it to pick the right prompt.
    """

    def _get_reward(self, final_answer: str) -> float:
        depth = self.extras.get("depth", 0)
        if depth > 0:
            return 0.0  # short circuit child rewards to 0

        evidence = (self.extras.get("reward_spec") or {}).get("evidence") or []
        reward, precision, recall = judge_reward(
            final_answer,
            question=self._root_prompt,
            evidence=evidence,
            model=self.JUDGE_MODEL,
            base_url=self.JUDGE_BASE_URL,
        )
        self._judge_precision = precision
        self._judge_recall = recall
        return reward

    def _get_system_prompt(self) -> str:
        depth = self.extras.get("depth", 0)
        return MULTIPAPER_PARENT_SYSTEM_PROMPT if depth == 0 else MULTIPAPER_CHILD_SYSTEM_PROMPT

    @staticmethod
    def aggregate_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Split rollouts by depth: depth=0 → parent/*, depth>=1 → child/*."""
        parents = [m for m in metrics if m.get("depth", 0) == 0]
        children = [m for m in metrics if m.get("depth", 0) > 0]
        out: Dict[str, Any] = {}
        out.update({f"parent/{k}": v for k, v in default_aggregate_metrics(parents).items()})
        out.update({f"child/{k}": v for k, v in default_aggregate_metrics(children).items()})
        return out

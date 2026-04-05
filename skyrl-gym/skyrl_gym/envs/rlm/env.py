import json
import re
import textwrap
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from skyrl_gym.tools.repl import PersistentREPL, REPLResult


# ---------------------------------------------------------------------------
# System prompt (official RLM prompt from rlm/rlm/utils/prompts.py)
# ---------------------------------------------------------------------------

DEFAULT_RLM_SYSTEM_PROMPT = """\
You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment. You will be queried iteratively until you provide a final answer.

The REPL environment is initialized with:
1. A `context` variable that contains extremely important information about your query. You should check the content of the `context` variable to understand what you are working with. Make sure you look through it sufficiently as you answer your query.
2. A `SHOW_VARS()` function that returns all variables you have created in the REPL. Use this to check what variables exist before using FINAL_VAR.
3. The ability to use `print()` statements to view the output of your REPL code and continue your reasoning.
{custom_tools_section}
When you want to execute Python code in the REPL environment, wrap it in triple backticks with 'repl' language identifier:
```repl
# your code here
```

Use variables as buffers to build up your final answer. Make sure to explicitly look through the context in the REPL before answering your query.

IMPORTANT: When you are done with the iterative process, you MUST provide a final answer using one of:
1. FINAL(your final answer here) — provide the answer directly as text
2. FINAL_VAR(variable_name) — return a variable you have created in the REPL

WARNING: FINAL_VAR retrieves an EXISTING variable. You MUST create and assign the variable in a ```repl``` block FIRST, then call FINAL_VAR in a SEPARATE response.

Think step by step carefully, plan, and execute this plan immediately in your response -- do not just say "I will do this". Output to the REPL environment as much as possible.\
"""


# ---------------------------------------------------------------------------
# Multi-paper system prompts (parent + child)
# ---------------------------------------------------------------------------

MULTIPAPER_PARENT_SYSTEM_PROMPT = textwrap.dedent(
    """\
You are an evidence extraction coordinator. You find VERBATIM text from the context relevant to a query.

The context is a DICTIONARY where each key is a paper ID (like "2205.05212") and each value is the full text of that paper starting with `### PAPER: <title>`.

REPL tools:
- `context`: a dictionary where keys are paper IDs and values are full paper texts.
- `list_papers(context)` — list all paper IDs with the first 1000 characters of their content.
- `search(text, keyword, window=300)` — keyword search. Pass `context` to search ALL papers at once (results are grouped by paper ID and title), or pass `context[paper_id]` to search a single paper. Every line in each snippet is prefixed with its line number (e.g. `L42: ...`).
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
You are a PRECISE evidence extraction worker. You have a single paper in `context` and a query. Find the BEST verbatim passage(s) (at most 2) that directly answer the query.

REPL tools:
- `context`: full text of your paper.
- `search(text, keyword, window=300)` — keyword search. Always pass `context` as first arg. \
Every line in each snippet is prefixed with its line number (e.g. `L42: ...`). If no exact match is found, fuzzy matching is used automatically.
- `extract_lines(text, start_line, end_line)` — extract lines from `start_line` to `end_line` (inclusive, 1-indexed). \
Always pass `context` as first arg. Returns the verbatim text (without line number prefixes). \
Extractions over 2000 chars are truncated with a warning — if this happens, use a tighter line range.
- `FINAL_VAR(variable_name)` — return your final answer.

RULES:
- Output ONLY ```repl code blocks. No narration, no explanation, no text outside code blocks.
- CRITICAL: Each response you give must contain EXACTLY ONE ```repl block. Never two, never zero. \
You will be called multiple times. Each call = one block.
- You can only see the output of a block AFTER you submit it. \
So you CANNOT call extract_lines() based on search() results in the same response — you haven't seen the line numbers yet.
- NEVER call FINAL_VAR in the same block as extract_lines. You must first extract, READ the output \
to verify it looks correct, and ONLY THEN call FINAL_VAR in the next block.
- Your final answer MUST be FINAL_VAR(list_of_strings) where each string is an exact slice of `context`.
- Each evidence string should be exactly ONE complete paragraph — not a single sentence, but not \
multiple paragraphs either. Include the full paragraph that contains the key fact (topic sentence \
through final sentence). Never return isolated sentences, but also never return more than one \
paragraph per extraction. Be precise.
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
NOTE: you do not actually need to re-write out the snippet, it should be saved in an array/variable \
that you can just index. i.e. search(context, s1[0], window=1000, bidirectional=False). When specifically trying to expand, we encourage \
window sizes of 1000+ characters (in line count, that's roughly 30+ lines).
- NEVER include section headers (like "4.1. Method") as part of your extraction — start from the first sentence of the paragraph.
- AT MOST 2 passages in your final answer. Prefer 1 if one passage covers the query.
- If this paper has no content relevant to the query, return an empty list.
- Final answer = list of VERBATIM substrings from `context`.
- No narration, no explanation, no text outside code blocks.

BE THOROUGH: Do NOT rush to extract after seeing the first promising snippet. Papers discuss the same \
concept in multiple places (abstract, introduction, methods, experiments, conclusion). \
Your job is to find the MOST detailed and informative passage, which is usually in the methods or \
experiments section — not the abstract. The abstract gives a summary; the body gives the real evidence. \
Always search with at least 2-3 different keyword sets before deciding which passages to extract.

search() prints every snippet with line numbers prefixed on each line. Read the line numbers carefully. \
After searching, identify ALL snippets that could be relevant — evidence is often spread across multiple sections \
of a paper (e.g. intro, methods, experiments may all contain relevant details). Expand each promising snippet generously. \
Then in the NEXT response (after you have read the expanded text), use extract_lines with the start and end \
line numbers to return the full paragraph that contains the evidence. Prefer returning too much over too little.

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

Turn 4 — now you have full context; extract the best paragraph(s) by line number:
```repl
p1 = extract_lines(context, 142, 155)
p2 = extract_lines(context, 310, 322)
```

*(code runs, you receive the output)*

Turn 5 — verify the extractions look correct, then return:
```repl
FINAL_VAR([p1, p2])
```\
"""
)


# ---------------------------------------------------------------------------
# Per-turn user prompt injection (from rlm/rlm/utils/prompts.py)
# ---------------------------------------------------------------------------

_USER_PROMPT = (
    "Think step-by-step on what to do using the REPL environment (which contains the context) "
    "to answer the prompt.\n\n"
    "Continue using the REPL environment, which has the `context` variable, "
    "by writing to a ```repl``` tag, and determine your answer. Your next action:"
)
_USER_PROMPT_WITH_ROOT = (
    "Think step-by-step on what to do using the REPL environment (which contains the context) "
    "to answer the original prompt: \"{root_prompt}\".\n\n"
    "Continue using the REPL environment, which has the `context` variable, "
    "by writing to a ```repl``` tag, and determine your answer. Your next action:"
)


def _build_user_prompt(root_prompt: Optional[str], iteration: int) -> Dict[str, str]:
    """Build the per-turn user message injected before every model call."""
    if iteration == 0:
        safeguard = (
            "You have not interacted with the REPL environment or seen your prompt / context yet. "
            "Your next action should be to look through and figure out how to answer the prompt, "
            "so don't just provide a final answer yet.\n\n"
        )
        body = _USER_PROMPT_WITH_ROOT.format(root_prompt=root_prompt) if root_prompt else _USER_PROMPT
        content = safeguard + body
    else:
        prefix = "The history before is your previous interactions with the REPL environment. "
        body = _USER_PROMPT_WITH_ROOT.format(root_prompt=root_prompt) if root_prompt else _USER_PROMPT
        content = prefix + body
    return {"role": "user", "content": content}


# ---------------------------------------------------------------------------
# Parsing helpers (from rlm/rlm/utils/parsing.py)
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL)


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _find_code_block(text: str) -> Optional[str]:
    """Return the first ```repl ... ``` code block, or None."""
    text = _strip_thinking(text)
    match = re.search(r"```repl\s*\n(.*?)\n```", text, re.DOTALL)
    return match.group(1).strip() if match else None


def _find_final_answer(text: str, repl: Optional[PersistentREPL]) -> Optional[str]:
    """Parse FINAL_VAR(...) or FINAL(...) from the model's text response."""
    text = _strip_thinking(text)

    # FINAL_VAR — retrieves a variable from the REPL
    match = re.search(r"^\s*FINAL_VAR\((.*?)\)", text, re.MULTILINE | re.DOTALL)
    if match:
        variable_name = match.group(1).strip().strip('"').strip("'")
        if repl is not None:
            result = repl.execute(f"print(FINAL_VAR({variable_name!r}))")
            answer = result.stdout.strip()
            if answer == "":
                return None
            if "Variable '" in answer and "' not found" in answer and "FINAL_VAR" in answer:
                return None
            return answer
        return None

    # FINAL — inline literal
    match = re.search(r"^\s*FINAL\((.*)\)\s*$", text, re.MULTILINE | re.DOTALL)
    if match:
        return match.group(1).strip()

    return None


def _format_execution_result(result: REPLResult) -> str:
    """Format a REPLResult as a string for display in the conversation (from rlm/rlm/utils/parsing.py)."""
    parts = []
    if result.stdout:
        parts.append(f"\n{result.stdout}")
    if result.stderr:
        parts.append(f"\n{result.stderr}")
    important_vars = {
        k: ""
        for k, v in result.locals.items()
        if not k.startswith("_")
        and k not in ("__builtins__", "__name__", "__doc__")
        and isinstance(v, (str, int, float, bool, list, dict, tuple))
    }
    if important_vars:
        parts.append(f"REPL variables: {list(important_vars.keys())}\n")
    return "\n\n".join(parts) if parts else "No output"


_MAX_RESULT_LEN = 20_000


# ---------------------------------------------------------------------------
# QueryMetadata (from rlm/rlm/core/types.py)
# ---------------------------------------------------------------------------

class _QueryMetadata:
    def __init__(self, context_payload):
        if isinstance(context_payload, str):
            self.context_lengths = [len(context_payload)]
            self.context_type = "str"
        elif isinstance(context_payload, dict):
            self.context_type = "dict"
            self.context_lengths = []
            for chunk in context_payload.values():
                if isinstance(chunk, str):
                    self.context_lengths.append(len(chunk))
                else:
                    try:
                        self.context_lengths.append(len(json.dumps(chunk, default=str)))
                    except Exception:
                        self.context_lengths.append(len(repr(chunk)))
        elif isinstance(context_payload, list):
            self.context_type = "list"
            self.context_lengths = [len(str(c)) for c in context_payload]
        else:
            self.context_type = type(context_payload).__name__
            self.context_lengths = [len(repr(context_payload))]
        self.context_total_length = sum(self.context_lengths)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RLMEnvConfig:
    repl_timeout: float = 60.0
    parent_repl_timeout: float = 180.0  # timeout for parent REPL (with child RLM calls)
    custom_system_prompt: Optional[str] = None
    child_system_prompt: Optional[str] = None
    custom_tools: Optional[Dict[str, Any]] = field(default=None)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class RLMEnv(BaseTextEnv):
    """
    Recursive Language Model environment.

    init() returns:
        [system_msg, context_metadata_msg, turn_0_user_prompt_msg]

    step() returns observations:
        [repl_output_msg, turn_N_user_prompt_msg]

    The model always sees the per-turn user prompt as the last message before
    it generates, keeping root_prompt visible every turn.

    Context is loaded into the REPL via a temp file (add_context) so it
    appears as a genuine REPL variable in SHOW_VARS() and format_execution_result().

    Final answer is detected via:
      1. REPLResult.final_answer — set when FINAL_VAR() is called inside a repl block
      2. Text parsing of FINAL(...) / FINAL_VAR(...) in the model's response
    """

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = None):
        super().__init__()
        extras = extras or {}
        self.extras = extras

        assert "reward_spec" in extras, "reward_spec field is required"
        self.ground_truth = extras["reward_spec"].get("ground_truth")
        # reward_fn: optional callable(final_answer: str) -> float
        # Can be passed directly or will be built lazily in init() from reward_spec data.
        self.reward_fn = extras["reward_spec"].get("reward_fn")
        self.max_turns = extras.get("max_turns", 10)

        if isinstance(env_config, RLMEnvConfig):
            self.rlm_config = env_config
        elif isinstance(env_config, Mapping):
            self.rlm_config = RLMEnvConfig(**{k: v for k, v in env_config.items() if k in RLMEnvConfig.__dataclass_fields__})
        else:
            self.rlm_config = RLMEnvConfig()

        if extras.get("custom_system_prompt"):
            self.rlm_config.custom_system_prompt = extras["custom_system_prompt"]

        # Per-example custom tools (e.g. search/extract_section that close over context)
        # merged on top of any static config-level tools
        if extras.get("custom_tools"):
            merged = dict(self.rlm_config.custom_tools or {})
            merged.update(extras["custom_tools"])
            self.rlm_config.custom_tools = merged

        # LM query callbacks — passed via extras to stay serialization-friendly
        self.lm_callback = extras.get("lm_callback", None)
        self.subcall_fn = extras.get("subcall_fn", None)

        self.repl: Optional[PersistentREPL] = None
        self._final_answer: Optional[str] = None
        self._turn_index = 0  # iteration counter for build_user_prompt
        self._last_repl_exec_s: float = 0.0
        self._chat_history: Optional[List[Dict[str, str]]] = None

    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        extra_info = self.extras.get("extra_info", {}) if hasattr(self, "extras") else {}
        if not isinstance(extra_info, dict):
            extra_info = {}

        # root_prompt: the user question shown every turn
        # context_text: data payload loaded into the REPL `context` variable
        root_prompt = self._extract_prompt_text(prompt)
        context_payload = extra_info.get("context_text") or root_prompt
        if isinstance(context_payload, str):
            try:
                decoded = json.loads(context_payload)
                if isinstance(decoded, dict):
                    context_payload = decoded
            except (json.JSONDecodeError, ValueError):
                pass
        self._root_prompt = root_prompt

        # Build per-context REPL tools (search, extract_lines, etc.) and optionally
        # an evidence-based reward function from the serializable reward_spec data.
        reward_spec = self.extras.get("reward_spec", {})
        if self.reward_fn is None and reward_spec.get("evidence") is not None:
            evidence = reward_spec["evidence"]
            if isinstance(context_payload, dict):
                from skyrl_gym.envs.rlm.evidence_tools import make_reward_fn_multipaper
                self.reward_fn = make_reward_fn_multipaper(context_payload, evidence)
            else:
                from skyrl_gym.envs.rlm.evidence_tools import make_reward_fn
                self.reward_fn = make_reward_fn(context_payload, evidence)

        if not self.rlm_config.custom_tools or "search" not in self.rlm_config.custom_tools:
            from skyrl_gym.envs.rlm.evidence_tools import make_tools
            tools = make_tools()
            merged = dict(self.rlm_config.custom_tools or {})
            merged.update(tools)
            self.rlm_config.custom_tools = merged

        repl_timeout = self.rlm_config.parent_repl_timeout if self.subcall_fn is not None else self.rlm_config.repl_timeout
        self.repl = PersistentREPL(
            timeout=repl_timeout,
            custom_tools=self.rlm_config.custom_tools or {},
            lm_callback=self.lm_callback,
            subcall_fn=self.subcall_fn,
        )
        self.repl.add_context(context_payload, context_index=0)

        # Compute context metadata for the first user message
        meta = _QueryMetadata(context_payload)
        metadata_text = (
            f"Your context is a {meta.context_type} with {meta.context_total_length} total characters, "
            f"and is broken up into chunks of char lengths: {meta.context_lengths}."
        )

        system_content = self._build_system_prompt()

        # Turn-0 user prompt injection
        self._turn_index = 0
        turn0_prompt = _build_user_prompt(root_prompt, iteration=0)

        init_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": metadata_text},
        ]
        return init_messages, {"next_user_message": turn0_prompt}

    def _build_system_prompt(self) -> str:
        prompt_key = self.rlm_config.custom_system_prompt
        if prompt_key == "multipaper":
            return MULTIPAPER_PARENT_SYSTEM_PROMPT
        if prompt_key == "multipaper_child":
            return MULTIPAPER_CHILD_SYSTEM_PROMPT
        template = prompt_key or DEFAULT_RLM_SYSTEM_PROMPT

        custom_tools_section = ""
        if self.lm_callback is not None:
            custom_tools_section += (
                "\n4. LM query tools available in the REPL:\n"
                "- `llm_query(prompt)` — make a direct LLM call, returns str\n"
                "- `llm_query_batched(prompts)` — batch LLM calls, returns list[str]\n"
                "- `rlm_query(prompt)` — recursive LM call that spawns a child agent with its own REPL, returns str\n"
                "- `rlm_query_batched(prompts)` — batch recursive calls in parallel, returns list[str]"
            )
        if self.rlm_config.custom_tools:
            from skyrl_gym.tools.repl import format_tools_for_prompt
            tools_formatted = format_tools_for_prompt(self.rlm_config.custom_tools)
            if tools_formatted:
                section_num = 5 if self.lm_callback is not None else 4
                custom_tools_section += f"\n{section_num}. Custom tools and data available in the REPL:\n{tools_formatted}"

        return template.replace("{custom_tools_section}", custom_tools_section)

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        self._turn_index += 1

        done = self.turns >= self.max_turns
        code = _find_code_block(action)

        if code is None:
            obs_text = "[No ```repl``` code block found. Wrap your code in ```repl\\n...\\n``` blocks.]"
            if not done:
                next_prompt = _build_user_prompt(self._root_prompt, self._turn_index)
                return BaseTextEnvStepOutput(
                    observations=[{"role": "user", "content": obs_text}],
                    next_user_message=next_prompt,
                    reward=self._get_reward(done, None),
                    done=done,
                    metadata={},
                )
            return BaseTextEnvStepOutput(
                observations=[], next_user_message=None, reward=self._get_reward(done, None), done=done, metadata={}
            )

        _t_repl = time.perf_counter()
        result = self.repl.execute(code)
        self._last_repl_exec_s = time.perf_counter() - _t_repl

        # Two-stage final answer detection
        final_answer = result.final_answer  # set by FINAL_VAR() callable during execution
        if final_answer is None:
            final_answer = _find_final_answer(action, self.repl)
        if final_answer is not None:
            self._final_answer = final_answer
            done = True

        reward = self._get_reward(done, final_answer)

        if done:
            return BaseTextEnvStepOutput(
                observations=[], next_user_message=None, reward=reward, done=True, metadata=self._build_metadata()
            )

        # Format REPL output observation
        result_str = _format_execution_result(result)
        if len(result_str) > _MAX_RESULT_LEN:
            result_str = result_str[:_MAX_RESULT_LEN] + f"... + [{len(result_str) - _MAX_RESULT_LEN} chars...]"
        repl_obs_text = f"Code executed:\n```python\n{code}\n```\n\nREPL output:\n{result_str}"

        next_prompt = _build_user_prompt(self._root_prompt, self._turn_index)
        return BaseTextEnvStepOutput(
            observations=[{"role": "user", "content": repl_obs_text}],
            next_user_message=next_prompt,
            reward=reward,
            done=False,
            metadata=self._build_metadata(),
        )

    def _extract_prompt_text(self, prompt: ConversationType) -> str:
        parts = [msg["content"] for msg in prompt if msg.get("content")]
        return "\n".join(parts)

    def _get_reward(self, done: bool, final_answer: Optional[str]) -> float:
        if not done:
            return 0.0
        if final_answer is None:
            return 0.0

        if self.reward_fn is not None:
            return float(self.reward_fn(final_answer))

        final_str = str(final_answer).strip()
        gt_str = str(self.ground_truth).strip()

        if final_str == gt_str:
            return 1.0
        try:
            if abs(float(final_str) - float(gt_str)) < 1e-6:
                return 1.0
        except (ValueError, TypeError):
            pass
        if gt_str.lower() in final_str.lower():
            return 0.5
        return 0.0

    def _build_metadata(self) -> Dict[str, Any]:
        return {"turns": self.turns, "repl_exec_s": self._last_repl_exec_s}

    def set_chat_history(self, chat_history: List[Dict[str, str]]) -> None:
        self._chat_history = chat_history

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "turns_used": self.turns,
            "final_value_set": self._final_answer is not None,
            "final_answer": self._final_answer,
            "reward": self._get_reward(True, self._final_answer),
            "chat_history": self._chat_history,
        }

    def close(self):
        if self.repl is not None:
            self.repl.cleanup()
            self.repl = None


if __name__ == "__main__":
    direct_prompt = "What is the capital of France? Extract it from the context."
    context_text = (
        "The population of France is approximately 67 million people. "
        "The capital city is Paris. France is known for the Eiffel Tower, "
        "fine cuisine, and its contributions to art and philosophy."
    )

    env = RLMEnv(
        env_config=RLMEnvConfig(repl_timeout=30.0),
        extras={
            "reward_spec": {"ground_truth": "Paris"},
            "max_turns": 5,
            "extra_info": {"context_text": context_text},
        },
    )

    prompt = [{"role": "user", "content": direct_prompt}]
    init_messages, info = env.init(prompt)

    print("=== Init Messages ===")
    for msg in init_messages:
        preview = msg["content"][:300].replace("\n", "\\n")
        print(f"  [{msg['role']}] {preview}...")

    action_1 = (
        "Let me inspect the context.\n\n"
        "```repl\nprint(context[:100])\n```"
    )
    step_out = env.step(action_1)
    print("\n=== Turn 1 ===")
    print(f"  Reward: {step_out['reward']}, Done: {step_out['done']}")
    for obs in step_out["observations"]:
        preview = obs["content"][:200].replace("\n", "\\n")
        print(f"  [{obs['role']}] {preview}")

    action_2 = (
        "I can see the answer is Paris.\n\n"
        "```repl\nmy_answer = \"Paris\"\nprint(my_answer)\n```\n\n"
        "FINAL_VAR(my_answer)"
    )
    step_out = env.step(action_2)
    print("\n=== Turn 2 ===")
    print(f"  Reward: {step_out['reward']}, Done: {step_out['done']}")

    print("\n=== Metrics ===")
    for k, v in env.get_metrics().items():
        print(f"  {k}: {v}")

    env.close()

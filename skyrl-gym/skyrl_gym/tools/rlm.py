from __future__ import annotations

import io
import itertools
import json
import os
import re
import tempfile
import textwrap
import threading
import time
import uuid
import shutil
import sys
from urllib.parse import urlparse
from typing import Any, Optional
from openai import OpenAI

from skyrl_gym.tools.core import ToolGroup, tool

# =========================================================================
#   Initial Library and Utils Setup
# =========================================================================

# TODO(devpatel): update prompt to be better and more general. Currently, it's copied and modified from the rlm repo ()

DEFAULT_INIT_PROMPT = """
You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs, which you are strongly encouraged to use as much as possible. You will be queried iteratively until you provide a final answer.

The REPL environment is initialized with:
1. A `context` variable that contains extremely important information about your query. You should check the content of the `context` variable to understand what you are working with. Make sure you look through it sufficiently as you answer your query.
2. A `llm_query` function that allows you to query an LLM (that can handle around 500K chars) inside your REPL environment.
3. The ability to use `print()` statements to view the output of your REPL code and continue your reasoning. You must print any important information to the REPL environment for subsequent queries and calls.

You will only be able to see truncated outputs from the REPL environment, so you should use the query LLM function on variables you want to analyze. You will find this function especially useful when you have to analyze the semantics of the context. Use these variables as buffers to build up your final answer.
Make sure to explicitly look through the entire context in REPL before answering your query. An example strategy is to first look at the context and figure out a chunking strategy, then break up the context into smart chunks, and query an LLM per chunk with a particular question and save the answers to a buffer, then query an LLM with all the buffers to produce your final answer.

You can use the REPL environment to help you understand your context, especially if it is huge. Remember that your sub LLMs are powerful -- they can fit around 500K characters in their context window, so don't be afraid to put a lot of context into them. For example, a viable strategy is to feed 10 documents per sub-LLM query. Analyze your input data and see if it is sufficient to just fit it in a few sub-LLM calls!

When you want to execute Python code in the REPL environment, wrap it in <repl> </repl> language tags. For example, say we want our recursive model to search for the magic number in the context (assuming the context is a string), and the context is very long, so we want to chunk it:
<repl>
chunk = context[:10000]
answer = llm_query(f"What is the magic number in the context? Here is the chunk: {{chunk}}")
print(answer)
</repl>

As an example, suppose you're trying to answer a question about a book. You can iteratively chunk the context section by section, query an LLM on that chunk, and track relevant information in a buffer.
<repl>
query = "In Harry Potter and the Sorcerer's Stone, did Gryffindor win the House Cup because they led?"
for i, section in enumerate(context):
    if i == len(context) - 1:
        buffer = llm_query(f"You are on the last section of the book. So far you know that: {{buffers}}. Gather from this last section to answer {{query}}. Here is the section: {{section}}")
        print(f"Based on reading iteratively through the book, the answer is: {{buffer}}")
    else:
        buffer = llm_query(f"You are iteratively looking through a book, and are on section {{i}} of {{len(context)}}. Gather information to help answer {{query}}. Here is the section: {{section}}")
        print(f"After section {{i}} of {{len(context)}}, you have tracked: {{buffer}}")
</repl>

As another example, when the context isn't that long (e.g. >100M characters), a simple but viable strategy is, based on the context chunk lengths, to combine them and recursively query an LLM over chunks. For example, if the context is a List[str], we ask the same query over each chunk sequentially:
<repl>
query = "A man became famous for his book "The Great Gatsby". How many jobs did he have?"
# Suppose our context is ~1M chars, and we want each sub-LLM query to be ~0.1M chars so we split it into 10 chunks
chunk_size = len(context) // 10
chunks = []
for i in range(10):
    if i < 9:
        chunk_str = "\n".join(context[i*chunk_size:(i+1)*chunk_size])
    else:
        chunk_str = "\n".join(context[i*chunk_size:])
    chunks.append(chunk_str)

# Query each chunk sequentially
answers = []
for chunk in chunks:
    answer = llm_query(f"Try to answer the following query: {{query}}. Here are the documents:\n{{chunk}}. Only answer if you are confident in your answer based on the evidence.")
    answers.append(answer)
for i, answer in enumerate(answers):
    print(f"I got the answer from chunk {{i}}: {{answer}}")
final_answer = llm_query(f"Aggregating all the answers per chunk, answer the original query about total number of jobs: {{query}}\\n\\nAnswers:\\n" + "\\n".join(answers))
</repl>

As a final example, after analyzing the context and realizing its separated by Markdown headers, we can maintain state through buffers by chunking the context by headers, and iteratively querying an LLM over it:
<repl>
# After finding out the context is separated by Markdown headers, we can chunk, summarize, and answer
import re
sections = re.split(r'### (.+)', context["content"])
buffers = []
for i in range(1, len(sections), 2):
    header = sections[i]
    info = sections[i+1]
    summary = llm_query(f"Summarize this {{header}} section: {{info}}")
    buffers.append(f"{{header}}: {{summary}}")
final_answer = llm_query(f"Based on these summaries, answer the original query: {{query}}\\n\\nSummaries:\\n" + "\\n".join(buffers))
</repl>
In the next step, we can return FINAL_VAR(final_answer).

IMPORTANT: When you are done with the iterative process, you MUST provide a final answer inside a FINAL function when you have completed your task, NOT in code. Do not use these tags unless you have completed your task. You have two options:
1. Use FINAL(your final answer here) to provide the answer directly
2. Use FINAL_VAR(variable_name) to return a variable you have created in the REPL environment as your final output

Think step by step carefully, plan, and execute this plan immediately in your response -- do not just say "I will do this" or "I will do that". Output to the REPL environment and recursive LLMs as much as possible. Remember to explicitly answer the original query in your final answer.
"""


## Add allowed imports here, for now only allow imports that are needed for the RLMExecutorToolGroup

_ALLOWED_IMPORTS = {
    "collections",
    "datetime",
    "functools",
    "itertools",
    "json",
    "math",
    "random",
    "re",
    "string",
}


def safe_import(name: str, globals=None, locals=None, fromlist=(), level: int = 0):
    """
    Allow safe imports for only sub-lm calls, not top level
    """
    if level != 0:
        raise ImportError("Relative imports are disabled in this environment")
    if not isinstance(name, str) or not name:
        raise ImportError("Invalid module name")

    top_level = name.split(".", 1)[0]
    if top_level not in _ALLOWED_IMPORTS:
        raise ImportError(f"Import of '{top_level}' is not allowed")

    return __import__(name, globals, locals, fromlist, level)


SAFE_BUILTINS = {
    # Core types and functions
    "print": print,
    "len": len,
    "str": str,
    "int": int,
    "float": float,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "bool": bool,
    "type": type,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "reversed": reversed,
    "range": range,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "any": any,
    "all": all,
    "pow": pow,
    "divmod": divmod,
    "chr": chr,
    "ord": ord,
    "hex": hex,
    "bin": bin,
    "oct": oct,
    "repr": repr,
    "ascii": ascii,
    "format": format,
    "hash": hash,
    "id": id,
    "iter": iter,
    "next": next,
    "slice": slice,
    "callable": callable,
    "hasattr": hasattr,
    "getattr": getattr,
    "setattr": setattr,
    "delattr": delattr,
    "dir": dir,
    "vars": vars,
    "bytes": bytes,
    "bytearray": bytearray,
    "memoryview": memoryview,
    "complex": complex,
    "object": object,
    "super": super,
    "property": property,
    "staticmethod": staticmethod,
    "classmethod": classmethod,
    "__import__": safe_import,
    # Exceptions
    "Exception": Exception,
    "BaseException": BaseException,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "FileNotFoundError": FileNotFoundError,
    "OSError": OSError,
    "IOError": IOError,
    "RuntimeError": RuntimeError,
    "NameError": NameError,
    "ImportError": ImportError,
    "StopIteration": StopIteration,
    "AssertionError": AssertionError,
    "NotImplementedError": NotImplementedError,
    "ArithmeticError": ArithmeticError,
    "LookupError": LookupError,
    "Warning": Warning,
    # Blocked / restricted
    "input": None,
    "eval": None,
    "exec": None,
    "compile": None,
    "globals": None,
    "locals": None,
    # see RLMExecutorToolGroup._safe_open
    "open": None,
}


def truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    """
    Truncate any prints or stdio outputs to max characters to prevent excessive context size increase.
    """
    if max_chars <= 0:
        return "", True
    if text is None:
        return "", False
    if len(text) <= max_chars:
        return text, False

    head = max(0, max_chars - 200)
    tail = min(200, max_chars // 5)
    if head + tail > max_chars:
        tail = max(0, max_chars - head)

    truncated = (
        text[:head] + f"\n... <truncated {len(text) - (head + tail)} chars> ...\n" + (text[-tail:] if tail else "")
    )
    return truncated, True


# Colored terminal box for debugging and logging (this was vibecoded so we can remove and add to logger instead)
def print_box(title: str, content: str, *, color: str = "cyan", width: int = 96) -> None:
    """
    Print a simple colored box to the terminal (no extra deps).
    """
    if content is None:
        content = ""

    colors = {
        "reset": "\033[0m",
        "cyan": "\033[36m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "red": "\033[31m",
        "magenta": "\033[35m",
        "blue": "\033[34m",
        "gray": "\033[90m",
    }
    c = colors.get(color, colors["cyan"])
    reset = colors["reset"]

    # Clamp width so small terminals don't look terrible.
    width = max(60, min(int(width), 140))
    inner_w = width - 4

    def _wrap_lines(s: str) -> list[str]:
        if not s:
            return [""]
        out: list[str] = []
        for line in s.splitlines():
            if line.strip() == "":
                out.append("")
                continue
            out.extend(
                textwrap.wrap(
                    line,
                    width=inner_w,
                    replace_whitespace=False,
                    drop_whitespace=False,
                )
            )
        return out

    title_line = f" {title} "
    top = "┌" + "─" * (width - 2) + "┐"
    mid = "├" + "─" * (width - 2) + "┤"
    bot = "└" + "─" * (width - 2) + "┘"

    print(c + top + reset)
    print(c + "│" + reset + title_line[:inner_w].ljust(inner_w) + c + "│" + reset)
    print(c + mid + reset)
    for line in _wrap_lines(content):
        print(c + "│" + reset + line[:inner_w].ljust(inner_w) + c + "│" + reset)
    print(c + bot + reset)


# =========================================================================
#   Core RLM Implementation
# =========================================================================


class RLMExecutorToolGroup(ToolGroup):
    """
    A persistent, sandboxed Python REPL + iterative RLM loop.
    - One ToolGroup instance == one persistent REPL namespace (globals/locals) for a "run"
    - Subsequent tool calls share variables unless explicitly reset/closed
    - Expose `llm_query(...)` inside the REPL so code can make recursive LM calls

    By treating the RLM as a tool call, we can easily integrate it into any environment that supports tool calling.
    This allows us to pass in any context or query from any given environment, either as a preprocessing or postprocessing step.
    The hope is that this RLM environment is persistent until it returns a valid context or reaches a maximum number of turns.

    TODO(devpatel): Benchmark and test on more complex tasks and larger contexts.

    """

    # =========================================================================
    #   Setup and Initialization
    # =========================================================================

    def __init__(
        self,
        max_turns: int = 10,
        max_tokens: int = 1024,
        max_repl_output_chars: int = 4000,
    ):
        self.temp_dir = tempfile.mkdtemp(
            prefix=f"repl_env_{uuid.uuid4()}_"
        )  # Create a temporary directory for the REPL environment
        self._lock = threading.Lock()  # Create lock for every code execution (cell in notebook)
        self.original_cwd = os.getcwd()

        self.globals_dict: dict[str, Any] = {}
        self.locals_dict: dict[str, Any] = {}

        self.max_turns = int(max_turns)
        self.max_tokens = int(max_tokens)
        self.max_repl_output_chars = int(max_repl_output_chars)

        self.lm_client: OpenAI | None = None  # Currently, we support vLLM and OpenAI as inference engines.
        self.model: str | None = None
        self.init_prompt: str | None = None  # This is set to the DEFAULT_INIT_PROMPT.

        self.turns = 0
        self._reset_repl_namespace()  # Initialize the REPL namespace with the default builtins and custom builtins.

        super().__init__(name="RLMExecutorToolGroup")

    def _reset_repl_namespace(self) -> None:
        """
        Initialize the REPL namespace with the default builtins and custom builtins.
        """
        safe_builtins = SAFE_BUILTINS.copy()

        ## Register custom builtins here
        safe_builtins["open"] = self._safe_open

        self.globals_dict = {
            "__builtins__": safe_builtins,
            "__name__": "__main__",
            "llm_query": self.llm_query,
        }

    def reset(self, *, keep_context: bool = True) -> None:
        """
        Reset REPL variables (locals), but keep configuration and temp dir.
        """
        context = self.locals_dict.get("context") if keep_context else None
        context_path = self.locals_dict.get("context_path") if keep_context else None
        self.locals_dict = {}
        if keep_context and context is not None:
            self.locals_dict["context"] = context
        if keep_context and context_path is not None:
            self.locals_dict["context_path"] = context_path
        self.turns = 0
        self._reset_repl_namespace()

    def _ensure_setup(self) -> None:
        """Ensure the recursive LM client is initialized."""
        if self.lm_client is not None and self.model is not None and self.init_prompt is not None:
            return
        else:
            raise ValueError(
                "Inference engine not setup; call engine_setup(base_url, model, init_prompt, api_key) first."
            )

    def _safe_open(self, file: str, mode: str = "r", *args, **kwargs):
        """
        Restricted open():
        - only allows paths under this tool's temp_dir (we create a new temp dir for every RLM tool call)
        - disallows write/append/update modes by default
        """
        if not isinstance(file, str) or not file:
            raise ValueError("Invalid file path")
        if not isinstance(mode, str) or not mode:
            raise ValueError("Invalid open mode")

        abs_path = os.path.abspath(file)
        temp_root = os.path.abspath(self.temp_dir)
        if not (abs_path == temp_root or abs_path.startswith(temp_root + os.sep)):
            raise PermissionError("File access outside temp_dir is not allowed")

        if any(flag in mode for flag in ("w", "a", "+")):
            raise PermissionError("Write access is not allowed in this environment")

        return open(abs_path, mode, *args, **kwargs)

    # =========================================================================
    #   String and Context Helper Methods
    # =========================================================================

    def _context_metadata_prompt(self) -> str:
        """
        Default context metadata with previews for RLM context inspection.

        TODO(dev): Add custom context metadata templates here w. jinja2.
        """
        ctx = self.locals_dict.get("context", None)
        ctx_type = type(ctx).__name__
        # Preview the length and type of the context, not really too important since subsequent lm calls will have access to the full context.
        if isinstance(ctx, str):
            total = len(ctx)
            preview = [min(2000, total)]
        elif isinstance(ctx, (list, tuple)):
            total = sum(len(str(x)) for x in ctx)
            preview = [len(str(x)) for x in ctx[:10]]
        elif isinstance(ctx, dict):
            total = sum(len(str(k)) + len(str(v)) for k, v in itertools.islice(ctx.items(), 1000))
            preview = [len(ctx)]
        else:
            total = len(str(ctx)) if ctx is not None else 0
            preview = []

        return f"Your context is a {ctx_type} with {total} total characters. " f"(Preview lens: {preview})"

    def _extract_final_answer(self, response: str) -> Optional[str]:
        """
        Supports FINAL_VAR and FINAL directives.
        """
        final_var_match = re.search(
            r"^\s*FINAL_VAR\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)\s*$",
            response,
            re.IGNORECASE | re.MULTILINE,
        )
        if final_var_match:
            variable_name = final_var_match.group(1)
            if variable_name in self.locals_dict:
                value = self.locals_dict[variable_name]
                return str(value)
            return f"Error: Variable '{variable_name}' not found in environment"

        final_direct_match = re.search(
            r"^\s*FINAL\s*\(\s*(.+?)\s*\)\s*$",
            response,
            re.IGNORECASE | re.DOTALL | re.MULTILINE,
        )
        if final_direct_match:
            return final_direct_match.group(1)

        return None

    # =========================================================================
    #   Code Execution and REPL Environment Management
    # =========================================================================

    def _execute_code(self, code: str) -> dict:
        """
        Execute Python in the persistent REPL namespace.
        Use lock
        """

        start_time = time.perf_counter()

        with self._lock:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            stdout_buf, stderr_buf = io.StringIO(), io.StringIO()  # Capture print and error outputs to strings
            try:
                sys.stdout, sys.stderr = stdout_buf, stderr_buf
                old_cwd = os.getcwd()
                try:
                    os.chdir(self.temp_dir)  # Create temp dir before executing anyc ode
                    combined = {
                        **self.globals_dict,
                        **self.locals_dict,
                    }  # Combine global and local variables for execution.
                    exec(code, combined, combined)  # Execute the code in the REPL namespace.
                finally:
                    os.chdir(old_cwd)  # Don't execute code unless we're in the temp dir.

                # Update local variables with any new variables created across calls, including deletions.
                self.locals_dict = {
                    key: value
                    for key, value in combined.items()
                    if key not in self.globals_dict and not key.startswith("_")
                }

                stdout = stdout_buf.getvalue()
                stderr = stderr_buf.getvalue()
            except Exception as e:
                stdout = stdout_buf.getvalue()
                stderr = stderr_buf.getvalue() + f"\n{type(e).__name__}: {e}"
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr

        execution_time = time.perf_counter() - start_time
        return {
            "stdout": stdout,
            "stderr": stderr,
            "locals": self.locals_dict.copy(),
            "execution_time": execution_time,
        }

    def close(self) -> None:
        """
        Close the REPL environment and clear the temp directory.
        """
        if getattr(self, "temp_dir", None) and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.temp_dir = None
        self.globals_dict.clear()
        self.locals_dict.clear()

    def __enter__(self):
        """
        Enter the REPL environment. Required for with-statement context management.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the REPL environment. Required for with-statement context management.
        """
        self.close()
        return False

    def __del__(self):
        """
        Delete the REPL environment. Required for garbage collection.
        """
        try:
            self.close()
        except Exception:
            pass

    # =========================================================================
    #   LM Query Method Definitions
    # =========================================================================
    def _load_context(self, context: str) -> str:  # 1. Load Context into the REPL.
        """
        Load context into the REPL (internal method).
        """
        context_payload: Any = context
        if isinstance(context, str):
            s = context.strip()
            looks_like_json = s and ((s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")))
            if looks_like_json:
                try:
                    context_payload = json.loads(s)
                except json.JSONDecodeError:
                    # Keep original string (including whitespace) if parsing fails.
                    context_payload = context

        self.locals_dict["context"] = context_payload

        # Write to a file inside temp_dir for convenience/debugging.
        is_text = isinstance(context_payload, str)
        context_path = os.path.join(self.temp_dir, "context.txt" if is_text else "context.json")
        with open(context_path, "w", encoding="utf-8") as f:
            if is_text:
                f.write(context_payload)
            else:
                json.dump(context_payload, f, ensure_ascii=False)

        self.locals_dict["context_path"] = context_path

        return json.dumps(
            {"status": 200, "message": "Context loaded successfully", "context_type": type(context_payload).__name__}
        )

    def engine_setup(
        self,
        base_url: str | None = None,
        model: str = "gpt-4o-mini",
        init_prompt: str = "",
        openai_api_key: str | None = None,
    ) -> str:
        """
        Set up environment for recursive LM calls.
        """
        # TODO(dev): Find better fix for this weird httpx error.
        if not base_url:
            base_url = "https://api.openai.com/v1"
        parsed = urlparse(base_url)
        if not parsed.scheme:
            base_url = f"https://{base_url}"
            parsed = urlparse(base_url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Invalid base_url scheme for OpenAI client: {base_url!r}")

        if openai_api_key is None or openai_api_key == "":
            openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key is None or openai_api_key == "":
            openai_api_key = "EMPTY"

        hostname = parsed.hostname or ""
        if hostname == "api.openai.com" and openai_api_key == "EMPTY":
            raise ValueError(
                "`OPENAI_API_KEY` must be set when using the official OpenAI endpoint (https://api.openai.com)."
            )
        self.lm_client = OpenAI(base_url=base_url, api_key=openai_api_key)
        if not init_prompt:
            init_prompt = DEFAULT_INIT_PROMPT

        self.model = model
        self.init_prompt = init_prompt
        self._reset_repl_namespace()
        return json.dumps({"status": "ok"})

    def llm_query(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Recursive LM call available inside the REPL.
        """

        assert self.lm_client is not None, "Inference engine not setup; call _rlm_setup(...) first."
        used_model = model or self.model
        assert used_model is not None, "Model not configured; call rlm_setup(...) first."

        msgs: list[dict[str, str]] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": prompt})

        return (
            self.lm_client.chat.completions.create(
                model=used_model,
                messages=msgs,
                max_tokens=self.max_tokens,
            )
            .choices[0]
            .message.content
        )

    # =========================================================================
    #   Tool Call API (Add all Tool Methods Here)
    # =========================================================================

    @tool  # 2. RLM Tool Call - Main entry point for RLM tool calling.
    def rlm(self, query: str, context: str) -> str:

        self._ensure_setup()
        self._load_context(context)

        assert self.lm_client is not None
        assert self.model is not None
        assert self.init_prompt is not None

        # Reset per-call counters
        self.turns = 0

        first_turn_safeguard = (
            "You have not inspected `context` yet. Your first step should be to use a <repl> block "
            "to peek at the structure of `context` (e.g., type, length, small preview). "
            "Do NOT provide a final answer yet.\n\n"
        )
        messages = [
            {"role": "system", "content": self.init_prompt},
            {"role": "assistant", "content": self._context_metadata_prompt()},
            {"role": "user", "content": first_turn_safeguard + query},
        ]

        for i in range(self.max_turns):
            self.turns += 1

            response = self.lm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
            )
            response_text = response.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": response_text})

            has_final_answer = bool(re.search(r"\bFINAL_VAR\b|\bFINAL\s*\(", response_text, re.IGNORECASE))

            # Find ALL REPL blocks in the response. Support <repl>...</repl>

            code_matches = re.findall(r"<repl>(.*?)</repl>", response_text, re.DOTALL | re.IGNORECASE)

            print_box(
                f"MODEL OUTPUT (turn {self.turns}/{self.max_turns})",
                response_text,
                color="cyan",
            )
            if not code_matches and not has_final_answer:
                print_box(
                    "NO REPL BLOCKS DETECTED",
                    "Model did not emit any REPL blocks this turn.",
                    color="yellow",
                )

            all_results: list[str] = []
            last_execution_result: dict | None = None
            for idx, code in enumerate(code_matches, start=1):
                code = code.strip()
                print_box(f"EXECUTING REPL CODE (block {idx})", code, color="magenta")
                execution_result = self._execute_code(code=code)
                last_execution_result = execution_result

                stdout = execution_result.get("stdout", "") or ""
                stderr = execution_result.get("stderr", "") or ""
                stdout, stdout_trunc = truncate_text(stdout, self.max_repl_output_chars)
                stderr, stderr_trunc = truncate_text(stderr, self.max_repl_output_chars)

                result_message: list[str] = []
                if stdout:
                    suffix = "\n[stdout truncated]" if stdout_trunc else ""
                    result_message.append(f"Output:\n{stdout}{suffix}")
                if stderr:
                    suffix = "\n[stderr truncated]" if stderr_trunc else ""
                    result_message.append(f"Error:\n{stderr}{suffix}")
                if not execution_result.get("stdout") and not execution_result.get("stderr"):
                    result_message.append("Code executed successfully (no output)")

                result_text = "\n".join(result_message)
                all_results.append(result_text)

                print_box(
                    f"REPL RESULT (block {idx})",
                    result_text,
                    color="green" if not execution_result.get("stderr") else "red",
                )

            observation_content = "\n\n".join(all_results)
            observation_content, obs_trunc = truncate_text(observation_content, self.max_repl_output_chars * 2)
            if obs_trunc:
                observation_content += "\n[observation truncated]"

            if not observation_content and not has_final_answer:
                observation_content = (
                    "No REPL blocks were found in your last message. "
                    "If you need to run Python, wrap it in <repl>...</repl> blocks. "
                    'If you are done, respond with FINAL("...") or FINAL_VAR(variable_name).'
                )
            messages.append({"role": "user", "content": observation_content})

            if i == 0:
                messages.append(
                    {"role": "user", "content": "Continue. Use <repl> blocks as needed, or finish with FINAL(...)."}
                )

            done = self.turns >= self.max_turns or has_final_answer
            final_answer_value = None
            if has_final_answer:
                final_answer_value = self._extract_final_answer(response_text)

            if done:
                break

        last_execution_result = last_execution_result or {}
        metadata: dict[str, Any] = {
            "execution_time": last_execution_result.get("execution_time", 0),
            "has_error": bool(last_execution_result.get("stderr", "")),
            "has_final_answer": bool(has_final_answer),
            "turns": self.turns,
            "total_messages": len(messages),
        }
        if final_answer_value is not None:
            metadata["final_answer"] = final_answer_value

        return json.dumps(
            {
                "final_answer": final_answer_value,
                "has_final_answer": bool(has_final_answer),
                "metadata": metadata,
                "tail_messages": messages[-3:] if len(messages) >= 3 else messages,
            }
        )

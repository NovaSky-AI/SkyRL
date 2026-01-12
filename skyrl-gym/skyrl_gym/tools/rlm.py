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
from typing import Any, Optional
from omegaconf import DictConfig
from openai import OpenAI

from skyrl_gym.tools.core import ToolGroup, tool


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
    "statistics",
    "string",
}


def safe_import(name: str, globals=None, locals=None, fromlist=(), level: int = 0):

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


class RLMExecutorToolGroup(ToolGroup):
    """
    A persistent, sandboxed Python REPL + iterative RLM loop.
    - One ToolGroup instance == one persistent REPL namespace (globals/locals) for a "run"
    - Subsequent tool calls share variables unless explicitly reset/closed
    - Expose `llm_query(...)` inside the REPL so code can make recursive LM calls

    TODO(dev): pass query endpoint to explicit inference engine, currently we use the OpenAI API directly
    """

    def __init__(
        self,
        max_turns: int = 10,
        max_tokens: int = 1024,
        max_repl_output_chars: int = 4000,
    ):
        self.temp_dir = tempfile.mkdtemp(prefix=f"repl_env_{uuid.uuid4()}_")
        self._lock = threading.Lock()
        self.original_cwd = os.getcwd()

        self.globals_dict: dict[str, Any] = {}
        self.locals_dict: dict[str, Any] = {}

        self.max_turns = int(max_turns)
        self.max_tokens = int(max_tokens)
        self.max_repl_output_chars = int(max_repl_output_chars)

        self.lm_client: OpenAI | None = None
        self.model: str | None = None
        self.init_prompt: str | None = None

        self.turns = 0
        self._reset_repl_namespace()

        super().__init__(name="RLMExecutorToolGroup")

    def _safe_open(self, file: str, mode: str = "r", *args, **kwargs):
        """
        Restricted open():
        - only allows paths under this tool's temp_dir
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

    def _reset_repl_namespace(self) -> None:
        safe_builtins = SAFE_BUILTINS.copy()

        ## Register custom builtins here
        safe_builtins["open"] = self._safe_open

        self.globals_dict = {
            "__builtins__": safe_builtins,
            "__name__": "__main__",
            "llm_query": self.llm_query,
            "llm_query_batched": self.llm_query_batched,
            "FINAL_VAR": self.final_var,
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

    def _context_metadata_prompt(self) -> str:
        """
        Default context metadata with previews for RLM context inspection.

        TODO(dev): Add custom context metadata templates here w. jinja2.
        """
        ctx = self.locals_dict.get("context", None)
        ctx_type = type(ctx).__name__
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

    def engine_setup(self, env_cfg: DictConfig) -> None:
        """Setup the inference engine (OpenAI-compatible)."""
        openai_api_key = getattr(env_cfg, "openai_api_key", None) or os.getenv("OPENAI_API_KEY")
        if openai_api_key is None:
            raise ValueError("`OPENAI_API_KEY` must be set (param/env_cfg/env var)")

        base_url = getattr(env_cfg, "base_url", None)
        model = getattr(env_cfg, "model", None)
        init_prompt = getattr(env_cfg, "init_prompt", None)
        if not base_url or not model or not init_prompt:
            raise ValueError("env_cfg must include base_url, model, and init_prompt")

        self.lm_client = OpenAI(base_url=base_url, api_key=openai_api_key)
        self.model = str(model)
        self.init_prompt = str(init_prompt)

        # Refresh helpers inside the REPL
        self._reset_repl_namespace()

    def _extract_final_answer(self, response: str) -> Optional[str]:
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

        final_answer_match = re.search(
            r'^\s*FINAL\s*\(\s*["\'](.+?)["\']\s*\)\s*$',
            response,
            re.IGNORECASE | re.DOTALL | re.MULTILINE,
        )
        if final_answer_match:
            return final_answer_match.group(1)

        final_direct_match = re.search(
            r"^\s*FINAL\s*\(\s*(.+?)\s*\)\s*$",
            response,
            re.IGNORECASE | re.DOTALL | re.MULTILINE,
        )
        if final_direct_match:
            return final_direct_match.group(1)

        return None

    def _execute_code(self, code: str) -> dict:
        """Execute Python in the persistent REPL namespace."""
        import sys

        start_time = time.perf_counter()

        with self._lock:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
            try:
                sys.stdout, sys.stderr = stdout_buf, stderr_buf
                old_cwd = os.getcwd()
                try:
                    os.chdir(self.temp_dir)
                    combined = {**self.globals_dict, **self.locals_dict}
                    exec(code, combined, combined)
                finally:
                    os.chdir(old_cwd)

                for key, value in combined.items():
                    if key not in self.globals_dict and not key.startswith("_"):
                        self.locals_dict[key] = value

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
        if getattr(self, "temp_dir", None) and os.path.exists(self.temp_dir):
            import shutil

            shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.temp_dir = None
        self.globals_dict.clear()
        self.locals_dict.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def llm_query(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Recursive LM call available inside the REPL."""
        assert self.lm_client is not None, "Inference engine not setup; call _engine_setup(...) first."
        used_model = model or self.model
        assert used_model is not None, "Model not configured; call _engine_setup(...) first."

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

    def final_var(self, variable_name: str) -> str:
        variable_name = variable_name.strip().strip("\"'").strip()
        if variable_name in self.locals_dict:
            return str(self.locals_dict[variable_name])
        return f"Error: Variable '{variable_name}' not found in environment"

    @tool
    def rlm_setup(
        self,
        base_url: str,
        model: str,
        init_prompt: str,
        openai_api_key: Optional[str] = None,
    ) -> str:
        """
        Set up environment for recursive LM calls.
        """
        if openai_api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key is None:
            raise ValueError("`OPENAI_API_KEY` must be set")
        self.lm_client = OpenAI(base_url=base_url, api_key=openai_api_key)
        self.model = model
        self.init_prompt = init_prompt
        self._reset_repl_namespace()
        return json.dumps({"status": "ok"})

    @tool
    def rlm_load_context(self, context: str) -> str:
        """
        Load context into the REPL. Call this before calling `rlm` to set the context.
        """
        context_payload: Any = context
        if isinstance(context, str):
            s = context.strip()
            looks_like_json = s and ((s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")))
            if looks_like_json:
                try:
                    context_payload = json.loads(s)
                except Exception:
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

    @tool
    def rlm_close(self) -> str:
        self.close()
        return json.dumps({"status": "closed"})

    @tool
    def rlm(self, query: str) -> str:

        assert self.lm_client is not None, "Inference engine not setup; call _engine_setup(...) first."
        assert self.model is not None, "Model not configured; call _engine_setup(...) first."
        assert self.init_prompt is not None, "System prompt not configured; call _engine_setup(...) first."

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

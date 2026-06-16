import io
import shutil
import signal
import tempfile
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Safe builtins — blocks eval/exec/compile/input/globals/locals
# ---------------------------------------------------------------------------

_SAFE_BUILTINS = {
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
    # Blocked to prevent sandbox escape. To expose specific libraries to the RLM,
    # pre-bind them into self.globals in setup() (e.g. self.globals["math"] = math)
    # so the model can use them without needing __import__ or open.
    "__import__": None,
    "open": None,
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
    # Blocked (None = raises NameError on access)
    "input": None,
    "eval": None,
    "exec": None,
    "compile": None,
    "globals": None,
    "locals": None,
}

# Names that are always restored after every execution so model overwrites don't persist.
RESERVED_TOOL_NAMES: frozenset = frozenset(
    {
        "FINAL_VAR",
        "SHOW_VARS",
        "context",
        "llm_query",
        "llm_query_batched",
        "rlm_query",
        "rlm_query_batched",
    }
)


@dataclass
class REPLResult:
    stdout: str
    stderr: str
    locals: Dict[str, Any]  # snapshot of self.locals after execution
    final_answer: Optional[str]  # set if FINAL_VAR() was called during execution


def _can_use_sigalrm() -> bool:
    """SIGALRM is only usable from the main thread on Unix."""
    return hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()


class PersistentREPL:
    """
    A persistent Python REPL that maintains state (variables) across
    multiple execute() calls. Used by RLMEnv to give the model a stateful
    programming environment it can interact with across turns.

    Globals hold builtins and scaffold functions (FINAL_VAR, SHOW_VARS).
    Locals hold user-created variables and the context payload.
    After every execution scaffold names are restored so the model cannot
    permanently overwrite them.
    """

    def __init__(
        self,
        timeout: float = 15.0,
        custom_tools: Optional[Dict[str, Any]] = None,
        lm_callback: Optional[Callable[[List[str]], List[str]]] = None,
        subcall_fn: Optional[Callable[[str], str]] = None,
    ):
        self.timeout = timeout
        self.custom_tools: Dict[str, Any] = custom_tools or {}
        self.lm_callback = lm_callback
        self.subcall_fn = subcall_fn
        self.temp_dir = tempfile.mkdtemp(prefix="skyrl_repl_")
        self._validate_custom_tools()
        self.setup()

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------

    def setup(self):
        self.globals: Dict[str, Any] = {
            "__builtins__": _SAFE_BUILTINS.copy(),
            "__name__": "__main__",
        }
        self.locals: Dict[str, Any] = {}
        self._last_final_answer: Optional[str] = None

        self._exec_combined: Optional[Dict[str, Any]] = None  # live combined dict during exec
        self.globals["FINAL_VAR"] = self._final_var
        self.globals["SHOW_VARS"] = self._show_vars

        if self.lm_callback is not None:
            self.globals["llm_query"] = self._llm_query
            self.globals["llm_query_batched"] = self._llm_query_batched
            self.globals["rlm_query"] = self._rlm_query
            self.globals["rlm_query_batched"] = self._rlm_query_batched

        for name, value, _ in _iter_tool_entries(self.custom_tools):
            if callable(value):
                self.globals[name] = value
            else:
                self.locals[name] = value

    def cleanup(self):
        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass
        if hasattr(self, "globals"):
            self.globals.clear()
        if hasattr(self, "locals"):
            self.locals.clear()

    def __del__(self):
        self.cleanup()

    # ------------------------------------------------------------------
    # Context loading
    # ------------------------------------------------------------------

    def add_context(self, context_payload, context_index: int = 0):
        """Bind the context payload directly into the REPL namespace.

        Assigning to self.locals avoids depending on `open`/`__import__` inside
        the sandbox (both are blocked) and skips an unnecessary temp-file
        round-trip.
        """
        var_name = f"context_{context_index}"
        self.locals[var_name] = context_payload
        if context_index == 0:
            self.locals["context"] = context_payload

    # ------------------------------------------------------------------
    # Scaffold functions injected into the REPL namespace
    # ------------------------------------------------------------------

    def _final_var(self, variable_name) -> str:
        """Return the value of a variable as the final answer, or stringify a direct value."""
        if not isinstance(variable_name, str):
            answer = str(variable_name)
            self._last_final_answer = answer
            return answer
        variable_name = variable_name.strip().strip("\"'")
        # Look in the live combined dict first (set during exec), then fall back to self.locals
        lookup = self._exec_combined if self._exec_combined is not None else self.locals
        if variable_name in lookup:
            answer = str(lookup[variable_name])
            self._last_final_answer = answer
            return answer
        available = [k for k in lookup if not k.startswith("_") and k not in self.globals]
        if available:
            msg = (
                f"Error: Variable '{variable_name}' not found. "
                f"Available variables: {available}. "
                f"You must create and assign a variable BEFORE calling FINAL_VAR on it."
            )
        else:
            msg = (
                f"Error: Variable '{variable_name}' not found. "
                f"No variables have been created yet. "
                f"You must create and assign a variable in a ```repl``` block BEFORE calling FINAL_VAR on it."
            )
        print(msg)
        return msg

    def _show_vars(self) -> str:
        """Show all user-created variables in the REPL."""
        lookup = self._exec_combined if self._exec_combined is not None else self.locals
        available = {k: type(v).__name__ for k, v in lookup.items() if not k.startswith("_") and k not in self.globals}
        if not available:
            return "No variables created yet. Use ```repl``` blocks to create variables."
        return f"Available variables: {available}"

    # ------------------------------------------------------------------
    # LM query functions (registered only when lm_callback is provided)
    # ------------------------------------------------------------------

    def _llm_query(self, prompt: str, model: Optional[str] = None) -> str:
        """Make a direct LLM call. Returns the response as a string."""
        try:
            return self.lm_callback([prompt])[0]
        except Exception as e:
            return f"Error: LM query failed - {e}"

    def _llm_query_batched(self, prompts: List[str], model: Optional[str] = None) -> List[str]:
        """Make batched LLM calls. Returns list of responses in the same order as prompts."""
        try:
            return self.lm_callback(prompts)
        except Exception as e:
            return [f"Error: LM query failed - {e}"] * len(prompts)

    @staticmethod
    def _parse_child_result(result: str) -> Any:
        """Try to parse a child's string result back into a Python object.

        Child agents return their final answer as a string (via tokenizer.decode).
        If the string is a valid Python literal (e.g. a list), parse it so the
        parent can work with it as a native object.
        """
        import ast

        try:
            return ast.literal_eval(result)
        except (ValueError, SyntaxError):
            return result

    def _rlm_query(self, prompt: str, model: Optional[str] = None, context: Any = None) -> Any:
        """Spawn a child RLM agent with its own REPL for deeper reasoning on a subtask.

        Falls back to a plain llm_query if no subcall_fn is configured.

        Args:
            context: If provided, overrides the child's REPL ``context`` variable
                (e.g. a single paper string instead of the parent's full dict).
        """
        if self.subcall_fn is not None:
            try:
                result = self.subcall_fn(prompt, context=context)
                return self._parse_child_result(result)
            except Exception as e:
                return f"Error: RLM query failed - {e}"
        return self._llm_query(prompt, model)

    def _rlm_query_batched(
        self, prompts: List[str], model: Optional[str] = None, context_list: Optional[List[Any]] = None
    ) -> List[Any]:
        """Spawn child RLM agents for multiple prompts in parallel.

        Results are returned in the same order as input prompts.
        Falls back to llm_query_batched if no subcall_fn is configured.

        Args:
            context_list: If provided, must be the same length as *prompts*.
                Each element overrides the child's REPL ``context`` variable
                (e.g. a single paper string instead of the parent's full dict).
        """
        if self.subcall_fn is not None:
            contexts = context_list or [None] * len(prompts)
            if len(prompts) <= 1:
                return [self._rlm_query(p, model, context=c) for p, c in zip(prompts, contexts)]

            results: List[Any] = [""] * len(prompts)
            lock = threading.Lock()
            completions: List[tuple] = []

            def _run(index: int, prompt: str, context: Any) -> None:
                try:
                    result = self.subcall_fn(prompt, context=context)
                    parsed = self._parse_child_result(result)
                    with lock:
                        completions.append((index, parsed))
                    results[index] = parsed
                except Exception as e:
                    results[index] = f"Error: RLM query failed - {e}"

            with ThreadPoolExecutor(max_workers=min(2, len(prompts))) as executor:
                futures = [executor.submit(_run, i, p, c) for i, (p, c) in enumerate(zip(prompts, contexts))]
                for f in as_completed(futures):
                    f.result()

            return results

        return self._llm_query_batched(prompts, model)

    def _restore_scaffold(self):
        """Restore reserved names after execution so model overwrites don't persist."""
        for name in RESERVED_TOOL_NAMES:
            if name == "FINAL_VAR":
                self.globals["FINAL_VAR"] = self._final_var
            elif name == "SHOW_VARS":
                self.globals["SHOW_VARS"] = self._show_vars
            elif name == "context" and "context_0" in self.locals:
                self.locals["context"] = self.locals["context_0"]
            elif name == "llm_query" and self.lm_callback is not None:
                self.globals["llm_query"] = self._llm_query
            elif name == "llm_query_batched" and self.lm_callback is not None:
                self.globals["llm_query_batched"] = self._llm_query_batched
            elif name == "rlm_query" and self.lm_callback is not None:
                self.globals["rlm_query"] = self._rlm_query
            elif name == "rlm_query_batched" and self.lm_callback is not None:
                self.globals["rlm_query_batched"] = self._rlm_query_batched

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, code: str) -> REPLResult:
        # SIGALRM is preferred because the OS delivers the signal directly into
        # the executing frame, cleanly interrupting even infinite loops or
        # blocking I/O.  The thread-based fallback can only *abandon* a stuck
        # thread (Python has no way to kill a thread), so runaway code keeps
        # burning CPU as a leaked daemon thread.
        #
        # However, SIGALRM is only available on Unix *and* only from the main
        # thread (Python raises ValueError if you call signal.signal() from a
        # non-main thread).  In practice this means Ray workers — which run env
        # logic on spawned threads — must use the thread fallback.
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        error_str: Optional[str] = None

        if _can_use_sigalrm():
            error_str = self._execute_with_sigalrm(code, stdout_buf, stderr_buf)
        else:
            error_str = self._execute_with_thread_timeout(code, stdout_buf, stderr_buf)

        final_answer = self._last_final_answer
        self._last_final_answer = None

        return REPLResult(
            stdout=stdout_buf.getvalue(),
            stderr=stderr_buf.getvalue() + (error_str or ""),
            locals=self.locals.copy(),
            final_answer=final_answer,
        )

    def _execute_with_sigalrm(self, code: str, stdout_buf: io.StringIO, stderr_buf: io.StringIO) -> Optional[str]:
        def _raise_timeout(*_):
            raise TimeoutError("Code execution timed out")

        old_alarm = None
        try:
            old_alarm = signal.signal(signal.SIGALRM, _raise_timeout)
            signal.alarm(int(self.timeout))
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                combined = {**self.globals, **self.locals}
                self._exec_combined = combined
                exec(code, combined, combined)
                self._exec_combined = None
                for key, value in combined.items():
                    if key not in self.globals and not key.startswith("_"):
                        self.locals[key] = value
                self._restore_scaffold()
        except TimeoutError:
            self._exec_combined = None
            return f"Timeout after {int(self.timeout)} seconds\n"
        except Exception:
            self._exec_combined = None
            return traceback.format_exc()
        finally:
            signal.alarm(0)
            if old_alarm is not None:
                signal.signal(signal.SIGALRM, old_alarm)
        return None

    def _execute_with_thread_timeout(
        self, code: str, stdout_buf: io.StringIO, stderr_buf: io.StringIO
    ) -> Optional[str]:
        """Fallback when SIGALRM is unavailable (e.g. non-main thread in Ray workers)."""
        result: dict = {"error": None}

        def _run():
            try:
                with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                    combined = {**self.globals, **self.locals}
                    self._exec_combined = combined
                    exec(code, combined, combined)
                    self._exec_combined = None
                    for key, value in combined.items():
                        if key not in self.globals and not key.startswith("_"):
                            self.locals[key] = value
                    self._restore_scaffold()
            except Exception:
                self._exec_combined = None
                result["error"] = traceback.format_exc()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=self.timeout)
        if t.is_alive():
            return f"Timeout after {int(self.timeout)} seconds\n"
        return result["error"]

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_custom_tools(self):
        conflicts = set(self.custom_tools.keys()) & RESERVED_TOOL_NAMES
        if conflicts:
            raise ValueError(
                f"Custom tools cannot override reserved REPL names: {sorted(conflicts)}. "
                f"Reserved: {sorted(RESERVED_TOOL_NAMES)}"
            )


def _iter_tool_entries(custom_tools: Optional[Dict[str, Any]]):
    """Yield (name, value, description) for each custom tool.

    Supports two declaration formats:
    1. Plain:        {"name": callable_or_value}
    2. With desc:    {"name": {"tool": callable_or_value, "description": "..."}}
    """
    if not custom_tools:
        return
    for name, entry in custom_tools.items():
        if isinstance(entry, dict) and "tool" in entry:
            desc = entry.get("description")
            yield name, entry["tool"], desc if isinstance(desc, str) else None
        else:
            yield name, entry, None

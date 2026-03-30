import io
import json
import os
import shutil
import signal
import tempfile
import threading
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Safe builtins — blocks eval/exec/compile/input/globals/locals
# ---------------------------------------------------------------------------

_SAFE_BUILTINS = {
    "print": print, "len": len, "str": str, "int": int, "float": float,
    "list": list, "dict": dict, "set": set, "tuple": tuple, "bool": bool,
    "type": type, "isinstance": isinstance, "issubclass": issubclass,
    "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
    "sorted": sorted, "reversed": reversed, "range": range,
    "min": min, "max": max, "sum": sum, "abs": abs, "round": round,
    "any": any, "all": all, "pow": pow, "divmod": divmod,
    "chr": chr, "ord": ord, "hex": hex, "bin": bin, "oct": oct,
    "repr": repr, "ascii": ascii, "format": format, "hash": hash, "id": id,
    "iter": iter, "next": next, "slice": slice, "callable": callable,
    "hasattr": hasattr, "getattr": getattr, "setattr": setattr,
    "delattr": delattr, "dir": dir, "vars": vars,
    "bytes": bytes, "bytearray": bytearray, "memoryview": memoryview,
    "complex": complex, "object": object, "super": super,
    "property": property, "staticmethod": staticmethod, "classmethod": classmethod,
    "__import__": __import__, "open": open,
    # Exceptions
    "Exception": Exception, "BaseException": BaseException,
    "ValueError": ValueError, "TypeError": TypeError, "KeyError": KeyError,
    "IndexError": IndexError, "AttributeError": AttributeError,
    "FileNotFoundError": FileNotFoundError, "OSError": OSError, "IOError": IOError,
    "RuntimeError": RuntimeError, "NameError": NameError, "ImportError": ImportError,
    "StopIteration": StopIteration, "AssertionError": AssertionError,
    "NotImplementedError": NotImplementedError, "ArithmeticError": ArithmeticError,
    "LookupError": LookupError, "Warning": Warning,
    # Blocked (None = raises NameError on access)
    "input": None, "eval": None, "exec": None, "compile": None,
    "globals": None, "locals": None,
}

# Names that are always restored after every execution so model overwrites don't persist.
RESERVED_TOOL_NAMES: frozenset = frozenset({
    "FINAL_VAR", "SHOW_VARS", "context",
})


@dataclass
class REPLResult:
    stdout: str
    stderr: str
    locals: Dict[str, Any]       # snapshot of self.locals after execution
    final_answer: Optional[str]  # set if FINAL_VAR() was called during execution


class _REPLTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _REPLTimeout("Code execution timed out")


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
    ):
        self.timeout = timeout
        self.custom_tools: Dict[str, Any] = custom_tools or {}
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

        for name, entry in self.custom_tools.items():
            value = _extract_tool_value(entry)
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
        """Write context to a temp file and load it into self.locals via execute()."""
        var_name = f"context_{context_index}"
        if isinstance(context_payload, str):
            context_path = os.path.join(self.temp_dir, f"context_{context_index}.txt")
            with open(context_path, "w") as f:
                f.write(context_payload)
            self.execute(f"with open(r'{context_path}', 'r') as f:\n    {var_name} = f.read()")
        else:
            context_path = os.path.join(self.temp_dir, f"context_{context_index}.json")
            with open(context_path, "w") as f:
                json.dump(context_payload, f)
            self.execute(f"import json\nwith open(r'{context_path}', 'r') as f:\n    {var_name} = json.load(f)")
        if context_index == 0:
            self.execute(f"context = {var_name}")

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
        available = {
            k: type(v).__name__
            for k, v in lookup.items()
            if not k.startswith("_") and k not in self.globals
        }
        if not available:
            return "No variables created yet. Use ```repl``` blocks to create variables."
        return f"Available variables: {available}"

    def _restore_scaffold(self):
        """Restore reserved names after execution so model overwrites don't persist."""
        for name in RESERVED_TOOL_NAMES:
            if name == "FINAL_VAR":
                self.globals["FINAL_VAR"] = self._final_var
            elif name == "SHOW_VARS":
                self.globals["SHOW_VARS"] = self._show_vars
            elif name == "context" and "context_0" in self.locals:
                self.locals["context"] = self.locals["context_0"]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, code: str) -> REPLResult:
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

    def _execute_with_sigalrm(
        self, code: str, stdout_buf: io.StringIO, stderr_buf: io.StringIO
    ) -> Optional[str]:
        old_alarm = None
        try:
            old_alarm = signal.signal(signal.SIGALRM, _timeout_handler)
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
        except _REPLTimeout:
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


# ---------------------------------------------------------------------------
# Custom tool helpers (ported from rlm/rlm/environments/base_env.py)
# ---------------------------------------------------------------------------

@dataclass
class ToolInfo:
    """Parsed information about a custom tool."""
    name: str
    value: Any
    description: Optional[str] = None

    @property
    def is_callable(self) -> bool:
        return callable(self.value)


def _extract_tool_value(entry: Any) -> Any:
    """Extract the callable/value from a plain entry or {'tool': ..., 'description': ...} dict."""
    if isinstance(entry, dict) and "tool" in entry:
        return entry["tool"]
    return entry


def _parse_tool_entry(name: str, entry: Any) -> ToolInfo:
    """Parse a custom tool entry into a ToolInfo.

    Supports two formats:
    1. Plain value:          {"name": callable_or_value}
    2. With description:     {"name": {"tool": callable_or_value, "description": "..."}}
    """
    if isinstance(entry, dict) and "tool" in entry:
        value = entry["tool"]
        description = entry.get("description")
        return ToolInfo(name=name, value=value, description=description if isinstance(description, str) else None)
    return ToolInfo(name=name, value=entry, description=None)


def _parse_custom_tools(custom_tools: Optional[Dict[str, Any]]) -> List[ToolInfo]:
    """Parse all custom tools into ToolInfo objects."""
    if not custom_tools:
        return []
    return [_parse_tool_entry(name, entry) for name, entry in custom_tools.items()]


def format_tools_for_prompt(custom_tools: Optional[Dict[str, Any]]) -> Optional[str]:
    """Format custom tools for inclusion in the system prompt.

    Returns a formatted string describing available tools, or None if no tools.
    Matches the format used by rlm/rlm/environments/base_env.py.
    """
    tool_infos = _parse_custom_tools(custom_tools)
    if not tool_infos:
        return None

    lines = []
    for tool in tool_infos:
        if tool.is_callable:
            lines.append(f"- `{tool.name}`: {tool.description}" if tool.description else f"- `{tool.name}`: A custom function")
        else:
            lines.append(f"- `{tool.name}`: {tool.description}" if tool.description else f"- `{tool.name}`: A custom {type(tool.value).__name__} value")
    return "\n".join(lines)

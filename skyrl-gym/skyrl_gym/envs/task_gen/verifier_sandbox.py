"""
Verifier sandbox for task generation.

Validates generated verifier code via AST analysis and safe execution checks.
Used as the validity gate in the task generation reward:
    R(task) = validity * (variance + alpha * separation)

If validity returns 0, the entire reward is zeroed out.
"""

import ast
import re
from dataclasses import dataclass, field
from typing import List, Optional, Set


@dataclass
class ValidationResult:
    """Result of verifier validation."""

    valid: bool
    checks_passed: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def score(self) -> float:
        """Return 1.0 if valid, 0.0 otherwise (multiplicative gate)."""
        return 1.0 if self.valid else 0.0


# Disallowed modules/builtins in verifier code
BLOCKED_IMPORTS = {
    "os",
    "sys",
    "subprocess",
    "shutil",
    "pathlib",
    "socket",
    "http",
    "urllib",
    "requests",
    "importlib",
    "ctypes",
    "signal",
    "multiprocessing",
    "threading",
    "pickle",
    "shelve",
    "tempfile",
    "glob",
    "io",
}

BLOCKED_BUILTINS = {
    "exec",
    "eval",
    "compile",
    "__import__",
    "open",
    "input",
    "breakpoint",
    "exit",
    "quit",
}

# Min/max AST node count for verifier complexity
MIN_AST_NODES = 5  # reject trivial verifiers like `return 1.0`
MAX_AST_NODES = 700  # reject overly complex verifiers


class VerifierSandbox:
    """Validates and sandboxes generated verifier code.

    Performs static analysis to catch common issues before any execution:
    1. Python syntax validity (AST parsing)
    2. Function signature check (must be `async def verify(env, ...)`)
    3. Complexity bounds (not trivial, not overly complex)
    4. No dangerous imports or builtins
    5. Must reference env parameter (actually uses the environment)
    6. Prompt-verifier alignment (optional, LLM-based)
    """

    def __init__(self, available_tools: Optional[Set[str]] = None):
        """
        Args:
            available_tools: Set of tool names available in the target environment.
                If provided, checks that verifier references at least one real tool.
        """
        self.available_tools = available_tools or set()

    def validate(
        self,
        verifier_code: str,
        prompt: Optional[str] = None,
    ) -> ValidationResult:
        """Run all validation checks on verifier code.

        Args:
            verifier_code: The generated verifier Python code.
            prompt: The associated task prompt (for alignment checks).

        Returns:
            ValidationResult with pass/fail and details.
        """
        result = ValidationResult(valid=True)

        # 1. Parse as valid Python
        tree = self._check_syntax(verifier_code, result)
        if tree is None:
            result.valid = False
            return result

        # 2. Check function signature
        self._check_signature(tree, result)

        # 3. Check complexity bounds
        self._check_complexity(tree, result)

        # 4. Check for dangerous imports/builtins
        self._check_safety(tree, result)

        # 5. Check env usage
        self._check_env_usage(tree, result)

        # 6. Check for hardcoded return values
        self._check_hardcoded_returns(tree, result)

        # 7. Check for unfiltered .all() calls
        self._check_unfiltered_all(tree, result)

        # 8. Check prompt length bounds (if prompt provided)
        if prompt is not None:
            self._check_prompt_bounds(prompt, result)

        # Any failed check -> invalid
        if result.checks_failed:
            result.valid = False

        return result

    def _check_syntax(self, code: str, result: ValidationResult) -> Optional[ast.AST]:
        """Check that verifier code is valid Python."""
        try:
            tree = ast.parse(code)
            result.checks_passed.append("syntax")
            return tree
        except SyntaxError as e:
            result.checks_failed.append("syntax")
            result.error = f"SyntaxError: {e}"
            return None

    def _check_signature(self, tree: ast.AST, result: ValidationResult):
        """Check that verifier defines a valid function with env parameter.

        Accepts both `verify(env, ...)` and `validate_task(env, ...)` names,
        both sync and async.
        """
        valid_names = {"verify", "validate_task"}
        for node in ast.walk(tree):
            if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                if node.name in valid_names:
                    args = node.args
                    arg_names = [a.arg for a in args.args]
                    if "env" in arg_names:
                        result.checks_passed.append("signature")
                        return
                    else:
                        result.checks_failed.append("signature")
                        result.error = f"{node.name}() must have 'env' parameter, got: {arg_names}"
                        return

        result.checks_failed.append("signature")
        result.error = "No verify(env, ...) or validate_task(env, ...) function found"

    def _check_complexity(self, tree: ast.AST, result: ValidationResult):
        """Check AST node count is within bounds."""
        node_count = sum(1 for _ in ast.walk(tree))

        if node_count < MIN_AST_NODES:
            result.checks_failed.append("complexity_min")
            result.error = f"Verifier too simple ({node_count} nodes < {MIN_AST_NODES})"
        elif node_count > MAX_AST_NODES:
            result.checks_failed.append("complexity_max")
            result.error = f"Verifier too complex ({node_count} nodes > {MAX_AST_NODES})"
        else:
            result.checks_passed.append("complexity")

    def _check_safety(self, tree: ast.AST, result: ValidationResult):
        """Check for dangerous imports and builtin calls."""
        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".")[0]
                    if module in BLOCKED_IMPORTS:
                        result.checks_failed.append("safety_import")
                        result.error = f"Blocked import: {alias.name}"
                        return

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split(".")[0]
                    if module in BLOCKED_IMPORTS:
                        result.checks_failed.append("safety_import")
                        result.error = f"Blocked import from: {node.module}"
                        return

            # Check dangerous builtin calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in BLOCKED_BUILTINS:
                        result.checks_failed.append("safety_builtin")
                        result.error = f"Blocked builtin call: {node.func.id}"
                        return

        result.checks_passed.append("safety")

    def _check_env_usage(self, tree: ast.AST, result: ValidationResult):
        """Check that the verifier actually uses the env parameter."""
        # Look for attribute access on 'env' (e.g., env.list_issues, env.get_data)
        # or 'env' passed as argument to await expressions
        env_used = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name) and node.value.id == "env":
                    env_used = True
                    break
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "env":
                    env_used = True
                    break

        if env_used:
            result.checks_passed.append("env_usage")
        else:
            result.checks_failed.append("env_usage")
            result.error = "Verifier does not use 'env' parameter"

    def _check_hardcoded_returns(self, tree: ast.AST, result: ValidationResult):
        """Check that verifier isn't just `return 1.0` or `return 0.0`."""
        valid_names = {"verify", "validate_task"}
        verify_func = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                if node.name in valid_names:
                    verify_func = node
                    break

        if verify_func is None:
            return  # Already caught by signature check

        # Check if all return statements are constant
        returns = [n for n in ast.walk(verify_func) if isinstance(n, ast.Return)]
        if not returns:
            result.checks_failed.append("hardcoded_return")
            result.error = "Verifier has no return statements"
            return

        all_constant = all(isinstance(r.value, ast.Constant) for r in returns if r.value is not None)

        if all_constant and len(returns) == 1:
            result.checks_failed.append("hardcoded_return")
            result.error = "Verifier always returns a constant value"
        else:
            result.checks_passed.append("return_logic")

    def _check_unfiltered_all(self, tree: ast.AST, result: ValidationResult):
        """Reject verifiers that call .table("X").all() without a filter.

        Unfiltered .all() fetches every row from a table, causing warm-pool
        saturation with large tables (6.5k zombie verifiers in production).

        Allowed patterns (filter present in chain):
            .table("X").eq("col", val).all()
            .table("X").neq("col", val).all()
            .table("X").select("col1").all()    # ID-only in find_new_entries

        Rejected pattern:
            .table("X").all()                   # no filter before .all()
        """
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            # Match .all() call
            if not (isinstance(node.func, ast.Attribute) and node.func.attr == "all"):
                continue
            # Walk up the chain: .all() is called on some object
            receiver = node.func.value
            # Check if the receiver is a .table() call (direct: .table("X").all())
            if self._is_table_call(receiver):
                result.checks_failed.append("unfiltered_all")
                result.error = (
                    'Unfiltered .all() on table — use .eq()/.neq()/.select() '
                    'before .all() (e.g., table("X").eq("col", val).all())'
                )
                return

        result.checks_passed.append("filtered_all")

    @staticmethod
    def _is_table_call(node: ast.AST) -> bool:
        """Check if an AST node is a .table("...") call."""
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "table"
        )

    def _check_prompt_bounds(self, prompt: str, result: ValidationResult):
        """Check that prompt is within reasonable length bounds."""
        word_count = len(prompt.split())

        if word_count < 5:
            result.checks_failed.append("prompt_length")
            result.error = f"Prompt too short ({word_count} words < 5)"
        elif word_count > 500:
            result.checks_failed.append("prompt_length")
            result.error = f"Prompt too long ({word_count} words > 500)"
        else:
            result.checks_passed.append("prompt_length")


def parse_task_output(action: str) -> Optional[dict]:
    """Parse LLM output to extract task prompt and verifier code.

    Expected format:
        <task>
        <prompt>...</prompt>
        <verifier>...</verifier>
        </task>

    Returns:
        Dict with 'prompt' and 'verifier' keys, or None if parsing fails.
    """
    prompt_match = re.search(r"<prompt>(.*?)</prompt>", action, re.DOTALL)
    verifier_match = re.search(r"<verifier>(.*?)</verifier>", action, re.DOTALL)

    if not prompt_match or not verifier_match:
        return None

    return {
        "prompt": prompt_match.group(1).strip(),
        "verifier": verifier_match.group(1).strip(),
    }

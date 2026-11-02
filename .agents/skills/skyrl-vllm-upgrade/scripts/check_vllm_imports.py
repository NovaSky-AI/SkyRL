"""Statically resolve every ``vllm`` import in SkyRL against an unpacked vLLM wheel.

Catches moved/removed modules and names (e.g. 0.33 moving ``vllm.entrypoints.launchers.cli_args``
to ``vllm.entrypoints.common.cli_args``) without a GPU box or even a Linux one: nothing is imported,
the vLLM tree is only parsed.

Usage (from the repo root):
    python3 .agents/skills/skyrl-vllm-upgrade/scripts/check_vllm_imports.py <unpacked-wheel-dir> [paths...]

``<unpacked-wheel-dir>`` contains the ``vllm/`` package (``unzip vllm-X.whl 'vllm/*' -d DIR``).
``paths`` default to ``skyrl tests examples``. Exit code 1 if anything does not resolve.

A name counts as resolved if the target module defines, assigns, imports or re-exports it, has a
module-level ``__getattr__``, or the name is itself a submodule. Imports inside ``try:`` blocks
are reported separately: they are usually deliberate version probes.
"""

import ast
import sys
from functools import lru_cache
from pathlib import Path

WHEEL = Path(sys.argv[1]).resolve()
PATHS = sys.argv[2:] or ["skyrl", "tests", "examples"]


def module_file(mod: str) -> Path | None:
    base = WHEEL.joinpath(*mod.split("."))
    for cand in (base.with_suffix(".py"), base / "__init__.py"):
        if cand.exists():
            return cand
    return None


def is_package_or_ext(mod: str) -> bool:
    base = WHEEL.joinpath(*mod.split("."))
    return base.is_dir() or any(base.parent.glob(base.name + ".*.so"))


def _resolve_relative(mod: str, node: ast.ImportFrom, is_pkg: bool) -> str:
    if node.level == 0:
        return node.module or ""
    parts = mod.split(".")
    base = parts if is_pkg else parts[:-1]
    base = base[: len(base) - (node.level - 1)]
    return ".".join(base + ([node.module] if node.module else []))


@lru_cache(maxsize=None)
def defined_names(mod: str) -> tuple[frozenset, bool]:
    """(names bound at module level, has module __getattr__). Follows `from x import *`."""
    f = module_file(mod)
    if f is None:
        return frozenset(), False
    tree = ast.parse(f.read_text(), filename=str(f))
    is_pkg = f.name == "__init__.py"
    names: set[str] = set()
    has_getattr = False

    def visit(body):
        nonlocal has_getattr
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
                has_getattr |= node.name == "__getattr__"
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    for n in ast.walk(t):
                        if isinstance(n, ast.Name):
                            names.add(n.id)
            elif isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
                src = _resolve_relative(mod, node, is_pkg)
                if src and src != mod:
                    star, star_getattr = defined_names(src)
                    names.update(star)
                    has_getattr |= star_getattr
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    names.add((a.asname or a.name).split(".")[0])
            elif isinstance(node, (ast.If, ast.Try, ast.With)):
                for field in ("body", "orelse", "finalbody", "handlers"):
                    for sub in getattr(node, field, []) or []:
                        visit(sub.body if isinstance(sub, ast.ExceptHandler) else [sub])

    visit(tree.body)
    return frozenset(names), has_getattr


def check(mod: str, name: str | None) -> bool:
    if name is None:
        return module_file(mod) is not None or is_package_or_ext(mod)
    if module_file(f"{mod}.{name}") is not None or is_package_or_ext(f"{mod}.{name}"):
        return True
    if module_file(mod) is None:
        return is_package_or_ext(mod) and not WHEEL.joinpath(*mod.split(".")).is_dir()
    names, has_getattr = defined_names(mod)
    return name in names or has_getattr or name == "*"


def main() -> int:
    misses, probes = [], []
    for root in PATHS:
        for py in sorted(Path(root).rglob("*.py")):
            try:
                tree = ast.parse(py.read_text(), filename=str(py))
            except SyntaxError:
                continue
            in_try = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Try):
                    for sub in node.body:
                        for n in ast.walk(sub):
                            in_try.add(id(n))
            for node in ast.walk(tree):
                items = []
                if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    if node.module == "vllm" or node.module.startswith("vllm."):
                        items = [(node.module, a.name) for a in node.names]
                elif isinstance(node, ast.Import):
                    items = [(a.name, None) for a in node.names if a.name == "vllm" or a.name.startswith("vllm.")]
                for mod, name in items:
                    if not check(mod, name):
                        where = (
                            f"{py}:{node.lineno}: from {mod} import {name}"
                            if name
                            else f"{py}:{node.lineno}: import {mod}"
                        )
                        (probes if id(node) in in_try else misses).append(where)
    for m in misses:
        print("UNRESOLVED", m)
    for p in probes:
        print("unresolved (inside try:, likely a version probe)", p)
    print(f"{len(misses)} unresolved, {len(probes)} in try blocks")
    return 1 if misses else 0


if __name__ == "__main__":
    sys.exit(main())

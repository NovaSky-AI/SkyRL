"""Path rules: which paths of a trajectory's graph train, and which model nodes on each.

A rule is a function of the ``MessageGraph`` that returns ``Row``s: a path of
node ids from a root down to any node, and the model-authored nodes on it that
train. ``skycap.samples`` turns the rows into samples (tokens, loss mask,
logprobs, routes, sampling mask), so a rule never deals with tokens.

Two rules are built in:

* ``all`` -- a row per root-to-leaf path, in leaf creation order. Each model
  node trains in the first row that contains it, so a shared prefix trains once.
* ``final`` -- one row: the path to the reply of the last model call to
  return, the conversation as the harness left it, with every model node on it
  a target. Branches off it, such as a reply the harness discarded and asked
  again for, don't train. It assumes the harness's last call is its main loop's:
  one whose side calls (a subagent, a summarizer) can return after that needs a
  custom rule.

Any other function of that shape is a custom rule. A server accepts the rules
it was built with, by name (``rule_registry``): a ``finish`` request names one and
never imports code. A custom rule given as ``"pkg.module:function"`` is
imported when the server is built.

Rows are checked before samples are built from them: a path must follow parent
links down from a root, and each target must be a model node on its row's path.
A node may be a target in at most one row, as in ``all``: trained in two, it
would count twice in the loss.
"""

from __future__ import annotations

import importlib
import operator
from collections.abc import Callable, Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import NamedTuple

from skycap.graph import MessageGraph


class Row(NamedTuple):
    """One training row: a path from a root, as node ids, and the model nodes on it that train."""

    path: Sequence[int]
    targets: Sequence[int]


#: A function of the sealed graph to the rows that train. A plain ``(path, targets)`` tuple is a row too.
PathRule = Callable[[MessageGraph], Iterable[Row]]

#: The ``code`` of a ``finish`` error body when the path rule raised, so a client can tell it from a server fault.
PATH_RULE_FAILED = "path_rule_failed"


def all_paths(graph: MessageGraph) -> list[Row]:
    """A row per root-to-leaf path; each model node trains in the first path that contains it."""
    trained: set[int] = set()
    rows: list[Row] = []
    for path in graph.paths():
        targets = [node for node in path if graph.nodes[node].author == "model" and node not in trained]
        trained.update(targets)
        rows.append(Row(path, targets))
    return rows


def final_path(graph: MessageGraph) -> list[Row]:
    """The path to the last model call's reply, with every model node on it a target."""
    # By the call rather than the last node made: a reply equal to an earlier one is found, not made, so the
    # last node made can be on a branch the harness discarded. Ties go to the node made later.
    ends = [(call.t_end, node.id) for node in graph for call in node.calls]
    if not ends:
        return []
    path = graph.path_to(max(ends)[1])
    return [Row(path, [node for node in path if graph.nodes[node].author == "model"])]


#: The rules every server accepts.
BUILTIN_RULES: Mapping[str, PathRule] = MappingProxyType({"all": all_paths, "final": final_path})


def import_rule(spec: str) -> PathRule:
    """The function a ``"pkg.module:function"`` import path names."""
    module_name, _, attribute = spec.partition(":")
    if not module_name or not attribute:
        raise ValueError(f"a path rule is one of {sorted(BUILTIN_RULES)} or 'pkg.module:function', not {spec!r}")
    rule = getattr(importlib.import_module(module_name), attribute)
    if not callable(rule):
        raise TypeError(f"path rule {spec!r} is not callable")
    return rule


def load_rule(spec: str) -> PathRule:
    """A built-in rule by name, or a custom one by import path."""
    return BUILTIN_RULES[spec] if spec in BUILTIN_RULES else import_rule(spec)


def rule_registry(custom: Mapping[str, PathRule | str] | None = None) -> dict[str, PathRule]:
    """The built-in rules plus ``custom``, by name. A custom rule given as a string is imported."""
    rules = dict(BUILTIN_RULES)
    for name, rule in (custom or {}).items():
        if name in BUILTIN_RULES:
            raise ValueError(f"{name!r} is a built-in path rule")
        rules[name] = import_rule(rule) if isinstance(rule, str) else rule
    return rules


def check_rows(graph: MessageGraph, rows: Iterable[Row]) -> list[Row]:
    """``rows`` with plain-int lists, or a ``ValueError`` naming the first one that isn't a valid row."""
    checked: list[Row] = []
    trained: set[int] = set()
    for index, (path, targets) in enumerate(rows):
        path, targets = [operator.index(node) for node in path], [operator.index(node) for node in targets]
        problem = _problem(graph, path, targets, trained)
        if problem is not None:
            raise ValueError(f"path rule row {index}: {problem}")
        checked.append(Row(path, targets))
    return checked


def _problem(graph: MessageGraph, path: list[int], targets: list[int], trained: set[int]) -> str | None:
    """What is wrong with one row, or None. Adds its targets to ``trained``."""
    if not path:
        return "the path is empty"
    if not all(0 <= node < len(graph) for node in path):
        return f"the path {path} names nodes the graph doesn't have"
    if graph.nodes[path[0]].parent is not None:
        return f"the path starts at node {path[0]}, which is not a root"
    for parent, child in zip(path, path[1:]):
        if graph.nodes[child].parent != parent:
            return f"node {child} is not a child of node {parent}"
    on_path = set(path)
    for target in targets:
        if target not in on_path:
            return f"target {target} is not on the path"
        if graph.nodes[target].author != "model":
            return f"target {target} is not model-authored"
        if target in trained:
            return f"target {target} is already trained by this or an earlier row"
        trained.add(target)
    return None

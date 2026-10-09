"""`skycap check`: whether a harness keeps a history that should be linear on one path.

A harness that only ever appends to its history produces one root-to-leaf path
per trajectory. Every fork is a place where a request's history stopped
matching what the graph already had, and comparing the new branch's first node
with the sibling it should have matched says why:

* ``resample``: the model replied again to a history it had already answered
  (a retry, ``n > 1``, or a reply to a history the harness cut short).
* ``edited reply``: the harness sent back a model reply with fields changed,
  such as stripped reasoning or a repaired tool call.
* ``re-rendered``: token mode only. The same message, but its re-render didn't
  reproduce the tokens the model sampled, so it couldn't continue from them.
* ``different message``: a different message after the same history, such as
  a compaction, a subagent or a rewritten turn.
* ``tools or model``: the same message, sent with another tool set or model.
  A match covers both, so this starts a new root.

A second root is compared with the earlier roots the same way.

This reads only the record documents, never the token sidecars, so it is cheap
to run over a whole record directory.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import islice
from pathlib import Path
from typing import Any, Literal

import zstandard

from skycap import record
from skycap.graph import MessageGraph, Node
from skycap.hashing import canonical_bytes, canonical_message, rendered_fields

Kind = Literal["resample", "edited reply", "re-rendered", "different message", "tools or model"]


@dataclass(frozen=True, slots=True)
class Fork:
    """Where a trajectory's history left the path it was on, and why."""

    #: The node the fork is under; ``None`` for a second root.
    parent: int | None
    #: The first node of the new branch.
    node: int
    #: The earlier sibling it was compared with: the one its history should have matched.
    sibling: int
    kind: Kind
    #: What differs, in words.
    detail: str


@dataclass(frozen=True, slots=True)
class Report:
    """One trajectory's paths and forks, or why its record couldn't be read."""

    trajectory_id: str
    status: str
    paths: int = 0
    forks: tuple[Fork, ...] = ()
    #: Token mode: calls whose prompt was rendered from the messages instead of
    #: extending the previous call's tokens (``CallInfo.bridged is False``).
    unbridged_calls: int = 0
    #: Set when the record couldn't be read; nothing else is then known.
    error: str | None = None

    @property
    def linear(self) -> bool:
        """One path, or none for a trajectory that made no calls."""
        return self.error is None and self.paths <= 1


#: The errors a malformed or foreign record raises while it is read.
_READ_ERRORS = (OSError, ValueError, KeyError, TypeError, AssertionError, zstandard.ZstdError)


def check_document(document: dict[str, Any]) -> Report:
    """Check one record document, as ``record.read_document`` returns it."""
    if not isinstance(document, dict):
        return Report("?", "unreadable", error=f"the document is a JSON {type(document).__name__}, not an object")
    version = document.get("format_version")
    if version != record.FORMAT_VERSION:
        return Report(
            document.get("id", "?"),
            "unreadable",
            error=f"format_version {version}, this skycap reads {record.FORMAT_VERSION}",
        )
    graph = MessageGraph()
    record.add_nodes(graph, document)
    tokens = (document.get("capture") or {}).get("mode") == "tokens"
    normalize = _token_message if tokens else canonical_message
    # Each message as the graph's matching sees it, normalized once.
    messages = {node.id: normalize(node.message) for node in graph}
    forks = [fork for parent in [None, *graph.branch_points()] for fork in _forks_under(graph, parent, messages)]
    return Report(
        trajectory_id=document["id"],
        status=document["status"],
        paths=len(graph.leaves()),
        forks=tuple(sorted(forks, key=lambda fork: fork.node)),
        unbridged_calls=graph.unbridged_calls(),
    )


def check_dir(record_dir: Path, trajectory_ids: Sequence[str] = ()) -> list[Report]:
    """Check the given trajectories, or every one in ``record_dir``.

    A record that can't be read becomes a report with ``error`` set rather
    than stopping the rest.
    """
    reports = []
    for trajectory_id in trajectory_ids or list(record.list_ids(record_dir)):
        try:
            report = check_document(record.read_document(record_dir, trajectory_id))
            # A document too malformed to name itself is named by its file.
            reports.append(report if report.error is None else replace(report, trajectory_id=trajectory_id))
        except _READ_ERRORS as error:
            reports.append(Report(trajectory_id, "unreadable", error=f"{type(error).__name__}: {error}"))
    return reports


def _token_message(message: Mapping[str, Any]) -> dict[str, Any]:
    """A token-mode message as its match hash sees it: the rendered fields (``MatchKey.fields``)."""
    return canonical_message(rendered_fields(message))


def _forks_under(graph: MessageGraph, parent: int | None, messages: dict[int, dict[str, Any]]) -> Iterator[Fork]:
    """One fork per child after the first: each started a path its earlier siblings don't share."""
    children = [graph.nodes[i] for i in graph.children(parent)]
    earlier = _Siblings(messages)
    for index, node in enumerate(children):
        if index:
            sibling = earlier.closest(node, islice(children, index))
            kind, detail = _cause(sibling, node, messages)
            yield Fork(parent=parent, node=node.id, sibling=sibling.id, kind=kind, detail=detail)
        earlier.add(node)


class _Siblings:
    """The siblings seen so far under one parent, indexed for ``closest``."""

    def __init__(self, messages: dict[int, dict[str, Any]]) -> None:
        self.messages = messages
        self.latest: Node | None = None
        self.by_match: dict[str, list[Node]] = {}
        self.by_message: dict[bytes, Node] = {}
        #: The latest sibling of each role, and the latest model reply of each role.
        self.by_role: dict[str | None, Node] = {}
        self.model_by_role: dict[str | None, Node] = {}

    def add(self, node: Node) -> None:
        self.latest = node
        self.by_match.setdefault(node.match_hash, []).append(node)
        self.by_message[canonical_bytes(self.messages[node.id])] = node
        self.by_role[node.role] = node
        if node.author == "model":
            self.model_by_role[node.role] = node

    def closest(self, node: Node, earlier: Iterable[Node]) -> Node:
        """The earlier sibling ``node``'s history should have matched.

        One with the same match hash first (only the tokens or the sampling
        differ), preferring a model reply, then one with the same message (only
        the tools or model differ). Otherwise the same-role sibling, preferring
        a model reply: for a model node the latest, since every model sibling is
        a resample of it; for a client node the one with the fewest differing
        fields, the only case that compares against every earlier sibling.
        """
        same_match = self.by_match.get(node.match_hash)
        if same_match:
            return ([s for s in same_match if s.author == "model"] or same_match)[-1]
        same_message = self.by_message.get(canonical_bytes(self.messages[node.id]))
        if same_message is not None:
            return same_message
        if node.role not in self.by_role:
            assert self.latest is not None
            return self.latest
        if node.author == "model":
            return self.model_by_role.get(node.role) or self.by_role[node.role]
        message = self.messages[node.id]
        return min(
            (s for s in earlier if s.role == node.role),
            key=lambda s: (_count_changes(self.messages[s.id], message), s.author != "model", -s.id),
        )


def _cause(sibling: Node, node: Node, messages: dict[int, dict[str, Any]]) -> tuple[Kind, str]:
    before, after = messages[sibling.id], messages[node.id]
    if node.author == "model":
        if node.match_hash == sibling.match_hash:
            if sibling.author == "model":
                return "resample", "the same reply again, under other sampling parameters or tokenized differently"
            return "resample", "the model sampled a message the harness had already written"
        if sibling.role == node.role:
            return "resample", "a second reply to the same history"
        return "resample", f"a reply to a history the other path continues with {_a(sibling.role)} message"
    if node.match_hash == sibling.match_hash:
        return "re-rendered", "the same message, rendered to different tokens"
    if before == after:
        return "tools or model", "the same message, sent with a different tool set or model"
    if sibling.role != node.role:
        return "different message", f"{_a(node.role)} message where the other path has {_a(sibling.role)} message"
    changes = ", ".join(_changes(before, after))
    if sibling.author == "model":
        return "edited reply", f"the harness sent back the model's reply with {changes}"
    return "different message", f"a different {node.role} message: {changes}"


def _count_changes(before: dict[str, Any], after: dict[str, Any]) -> int:
    """How many top-level fields differ (``len(_changes(...))`` without building the words)."""
    return len(before.keys() ^ after.keys()) + sum(before[k] != after[k] for k in before.keys() & after.keys())


def _changes(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """Which top-level fields differ, e.g. ``["content changed", "reasoning_content dropped"]``."""
    changes = []
    for key in sorted(before.keys() | after.keys()):
        if key not in after:
            changes.append(f"{key} dropped")
        elif key not in before:
            changes.append(f"{key} added")
        elif before[key] != after[key]:
            changes.append(f"{key} changed")
    return changes


def _a(role: str | None) -> str:
    word = role or "unnamed"
    vowel = word[0].lower() in "aeio" or word.lower().startswith("un")
    return f"{'an' if vowel else 'a'} {word}"


def format_reports(reports: Sequence[Report]) -> str:
    """One line per trajectory, one per fork under it, and a summary line."""
    lines = []
    for report in reports:
        if report.error is not None:
            lines.append(f"{report.trajectory_id}  unreadable: {report.error}")
            continue
        shape = "empty" if report.paths == 0 else "linear" if report.linear else f"{report.paths} paths"
        lines.append(f"{report.trajectory_id}  {report.status}  {shape}")
        for fork in report.forks:
            where = "new root" if fork.parent is None else f"under node {fork.parent}"
            lines.append(f"  node {fork.node} ({where}, vs node {fork.sibling}): {fork.kind}: {fork.detail}")
        if report.unbridged_calls:
            lines.append(f"  {report.unbridged_calls} unbridged call(s): prompt re-rendered, not extended")
    readable = [report for report in reports if report.error is None]
    summary = f"{sum(report.linear for report in readable)} of {len(readable)} trajectories linear"
    kinds = Counter(fork.kind for report in readable for fork in report.forks)
    if kinds:
        summary += "; forks: " + ", ".join(f"{count} {kind}" for kind, count in kinds.most_common())
    if len(readable) < len(reports):
        summary += f"; {len(reports) - len(readable)} unreadable"
    lines.append(summary)
    return "\n".join(lines)

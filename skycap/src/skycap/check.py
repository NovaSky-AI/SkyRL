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
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
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
    forks = [fork for parent in [None, *(node.id for node in graph)] for fork in _forks_under(graph, parent, messages)]
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
            reports.append(check_document(record.read_document(record_dir, trajectory_id)))
        except _READ_ERRORS as error:
            reports.append(Report(trajectory_id, "unreadable", error=f"{type(error).__name__}: {error}"))
    return reports


def _token_message(message: Mapping[str, Any]) -> dict[str, Any]:
    """A token-mode message as its match hash sees it: the rendered fields (``MatchKey.fields``)."""
    return canonical_message(rendered_fields(message))


def _forks_under(graph: MessageGraph, parent: int | None, messages: dict[int, dict[str, Any]]) -> Iterator[Fork]:
    """One fork per child after the first: each started a path its earlier siblings don't share."""
    children = [graph.nodes[i] for i in graph.children(parent)]
    by_match: dict[str, list[Node]] = {}
    by_message: dict[bytes, Node] = {}
    for index, node in enumerate(children):
        if index:
            sibling = _sibling_to_compare(children, index, by_match, by_message, messages)
            kind, detail = _cause(sibling, node, messages)
            yield Fork(parent=parent, node=node.id, sibling=sibling.id, kind=kind, detail=detail)
        by_match.setdefault(node.match_hash, []).append(node)
        by_message[canonical_bytes(messages[node.id])] = node


def _sibling_to_compare(
    children: list[Node],
    index: int,
    by_match: dict[str, list[Node]],
    by_message: dict[bytes, Node],
    messages: dict[int, dict[str, Any]],
) -> Node:
    """The earlier sibling ``children[index]``'s history should have matched: the closest one.

    One with the same match hash first (only the tokens or the sampling
    differ), preferring a model reply, then one with the same message (only the
    tools or model differ). Otherwise the same-role sibling, preferring a model
    reply: for a model node the latest, since every model sibling is a resample
    of it; for a client node the one with the fewest differing fields.
    """
    node, earlier = children[index], children[:index]
    same_match = by_match.get(node.match_hash)
    if same_match:
        return ([s for s in same_match if s.author == "model"] or same_match)[-1]
    same_message = by_message.get(canonical_bytes(messages[node.id]))
    if same_message is not None:
        return same_message
    same_role = [s for s in earlier if s.role == node.role]
    if not same_role:
        return earlier[-1]
    if node.author == "model":
        return ([s for s in same_role if s.author == "model"] or same_role)[-1]
    return min(
        same_role,
        key=lambda s: (len(_changes(messages[s.id], messages[node.id])), s.author != "model", -s.id),
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
    return f"{'an' if word[0] in 'aeio' else 'a'} {word}"


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

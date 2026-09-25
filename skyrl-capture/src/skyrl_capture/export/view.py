"""A fully materialized trajectory, loaded once per export.

Every exporter needs the same things: the trajectory record, its exchanges in
observed order, the message graph with node payloads, and the exchange-level
DAG implied by prefix matching. Loading that once and sharing it keeps exports
deterministic and avoids each format re-deriving the same structure
differently.

The nodes are the graph's own `GraphNode`s, not copies: the state holds one
representation of a node and the exporters read it. What this adds is the
exchange DAG, which is derived rather than stored, and the child index.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from skyrl_capture.domain import timing
from skyrl_capture.domain.graph import ConversationGraph, GraphNode
from skyrl_capture.domain.records import ActiveTrajectory, TrajectoryRecord


@dataclass
class TrajectoryView:
    trajectory: Any
    exchanges: list[Any]
    nodes: dict[str, GraphNode]
    node_order: list[str]
    children: dict[str | None, list[str]] = field(default_factory=dict)
    # Exchange DAG.
    exchange_by_id: dict[str, Any] = field(default_factory=dict)
    exchange_by_output_node: dict[str, str] = field(default_factory=dict)
    exchange_parent: dict[str, str | None] = field(default_factory=dict)
    exchange_children: dict[str, list[str]] = field(default_factory=dict)
    exchange_roots: list[str] = field(default_factory=list)
    # Exchanges whose delivery to the client was never confirmed, and the model
    # nodes they produced. Every exporter has to answer for these, and each one
    # answers differently: the graph shows them flagged, replay leaves them out,
    # and training exports refuse to train on them.
    uncertain_exchange_ids: set[str] = field(default_factory=set)
    uncertain_node_ids: set[str] = field(default_factory=set)
    # The nodes as a graph, for the questions the graph already answers --
    # leaves, branch points, paths. Rebuilt for a committed record rather than
    # re-derived, so a fork on screen and a fork in an export are one answer.
    graph: ConversationGraph = field(default_factory=ConversationGraph)

    # -- graph helpers -----------------------------------------------------
    def path_to(self, node_id: str) -> list[GraphNode]:
        """Root-to-node path."""
        path: list[GraphNode] = []
        current: str | None = node_id
        while current is not None:
            node = self.nodes.get(current)
            if node is None:
                break
            path.append(node)
            current = node.parent_id
        path.reverse()
        return path

    def leaves(self) -> list[str]:
        parents = {node.parent_id for node in self.nodes.values() if node.parent_id}
        return [node_id for node_id in self.node_order if node_id not in parents]

    def exchange_path_messages(self, exchange: Any) -> list[dict[str, Any]]:
        """Every message this exchange sent, in order."""
        leaf = exchange["input_leaf_node_id"]
        if not leaf:
            return []
        return [node.message for node in self.path_to(leaf) if node.message is not None]

    def exchange_tail_messages(self, exchange: Any) -> list[dict[str, Any]]:
        """Messages this exchange added beyond its parent exchange's context.

        Derived from the graph path rather than from ``input_node_ids`` so that
        a repeated continuation -- whose tail nodes already existed and were
        therefore not newly committed -- still exports its tail correctly.
        """
        leaf = exchange["input_leaf_node_id"]
        parent_output = exchange["parent_output_node_id"]
        if not leaf:
            return []
        path = self.path_to(leaf)
        if not parent_output:
            return [node.message for node in path if node.message is not None]
        for index, node in enumerate(path):
            if node.id == parent_output:
                return [item.message for item in path[index + 1 :] if item.message is not None]
        return [node.message for node in path if node.message is not None]


def view_of(record: TrajectoryRecord) -> TrajectoryView:
    """The exporters' view of a finished trajectory.

    Nothing is derived that was not already compiled: the nodes and their order
    come out of the record as they went in, and the only computation is the
    exchange DAG and the timing columns, which are answers about a set of
    exchanges rather than properties of one.
    """
    return _view(
        trajectory=record.trajectory,
        rows=[exchange.row for exchange in record.exchanges],
        nodes=list(record.nodes),
        uncertain={
            exchange.id for exchange in record.exchanges if exchange.delivery_uncertain
        },
    )


def view_of_active(active: ActiveTrajectory) -> TrajectoryView:
    """The same view, for a trajectory that has not finished yet.

    One shape for both, so an exporter, the `paths` route and the viewer cannot
    tell a trajectory mid-run from one an hour old. What differs is only where
    the nodes came from.
    """
    return _view(
        trajectory=active.document(),
        rows=active.exchange_rows(),
        nodes=active.graph.ordered(),
        uncertain={
            exchange.id for exchange in active.exchanges if exchange.delivery_uncertain
        },
    )


def _view(
    *,
    trajectory: Any,
    rows: list[dict[str, Any]],
    nodes: list[GraphNode],
    uncertain: set[str] | None = None,
) -> TrajectoryView:
    by_id = {node.id: node for node in nodes}
    children: dict[str | None, list[str]] = {}
    for node in nodes:
        children.setdefault(node.parent_id, []).append(node.id)
    uncertain = uncertain or set()
    graph = ConversationGraph()
    for node in nodes:
        graph.nodes[node.id] = node
        graph.order.append(node.id)
    view = TrajectoryView(
        trajectory=trajectory,
        exchanges=timing.with_derived(rows),
        nodes=by_id,
        node_order=[node.id for node in nodes],
        children=children,
        uncertain_exchange_ids=set(uncertain),
        uncertain_node_ids={
            row["output_node_id"]
            for row in rows
            if row["id"] in uncertain and row.get("output_node_id")
        },
        graph=graph,
    )
    _build_exchange_dag(view)
    return view


def _build_exchange_dag(view: TrajectoryView) -> None:
    """Derive the exchange-level DAG from graph attribution.

    A child exchange is one whose matched prefix ended exactly at another
    exchange's ``model``-authored node. That is the only relationship with an
    exact-evidence basis, and the only one AIPerf may export as a fork.
    """
    for exchange in view.exchanges:
        view.exchange_by_id[exchange["id"]] = exchange
        if exchange["output_node_id"]:
            view.exchange_by_output_node[exchange["output_node_id"]] = exchange["id"]

    for exchange in view.exchanges:
        identifier = exchange["id"]
        parent_output = exchange["parent_output_node_id"]
        parent = view.exchange_by_output_node.get(parent_output) if parent_output else None
        if parent == identifier:
            parent = None
        view.exchange_parent[identifier] = parent
        if parent is None:
            view.exchange_roots.append(identifier)
        else:
            view.exchange_children.setdefault(parent, []).append(identifier)

    for children in view.exchange_children.values():
        children.sort(key=lambda item: (view.exchange_by_id[item]["sequence"], item))

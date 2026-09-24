"""Training samples: one row per root-to-leaf path.

Each model-authored node is a training target in exactly one row, the first
path (in leaf creation order) that contains it. A shared prefix appears in
every row that shares it and trains once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from skycap.graph import MessageGraph


@dataclass(slots=True)
class Sample:
    leaf: int
    path: list[int]
    messages: list[dict[str, Any]]
    #: Model nodes this row trains on.
    targets: list[int] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {"leaf": self.leaf, "path": self.path, "messages": self.messages, "targets": self.targets}


def build_samples(graph: MessageGraph) -> list[Sample]:
    trained: set[int] = set()
    samples: list[Sample] = []
    for path in graph.paths():
        targets = [node for node in path if graph.nodes[node].author == "model" and node not in trained]
        trained.update(targets)
        samples.append(
            Sample(
                leaf=path[-1],
                path=path,
                messages=[graph.nodes[node].message for node in path],
                targets=targets,
            )
        )
    return samples

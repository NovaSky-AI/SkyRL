"""Training samples: one row per root-to-leaf path.

Each model-authored node is a training target in exactly one row, the first
path (in leaf creation order) that contains it. A shared prefix appears in
every row that shares it and trains once.

In token mode a row also carries the concatenated tokens of its path, aligned
arrays for training: a loss mask over the sampled tokens of its targets, the
rollout logprobs, the routed experts (when every node on the path has them) and
the sampling mask (when every target has one).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from skycap.graph import MessageGraph, Node
from skycap.tokens.engine import pack


@dataclass(slots=True)
class Sample:
    leaf: int
    path: list[int]
    messages: list[dict[str, Any]]
    #: Model nodes this row trains on.
    targets: list[int] = field(default_factory=list)
    input_ids: list[int] | None = None
    loss_mask: list[int] | None = None
    logprobs: list[float] | None = None
    #: ``[len(input_ids), layers, k]``.
    routed_experts: np.ndarray | None = None
    #: Per position, the ids the sampler could have drawn; empty where ``loss_mask`` is 0.
    sampling_mask: list[list[int]] | None = None

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "leaf": self.leaf,
            "path": self.path,
            "messages": self.messages,
            "targets": self.targets,
        }
        if self.input_ids is not None:
            out.update(input_ids=self.input_ids, loss_mask=self.loss_mask, logprobs=self.logprobs)
            out["routed_experts"] = pack(self.routed_experts) if self.routed_experts is not None else None
            out["sampling_mask"] = _csr(self.sampling_mask) if self.sampling_mask is not None else None
        return out


def build_samples(graph: MessageGraph) -> list[Sample]:
    trained: set[int] = set()
    samples: list[Sample] = []
    for path in graph.paths():
        targets = [node for node in path if graph.nodes[node].author == "model" and node not in trained]
        trained.update(targets)
        sample = Sample(
            leaf=path[-1], path=path, messages=[graph.nodes[node].message for node in path], targets=targets
        )
        nodes = [graph.nodes[node] for node in path]
        if all(node.tokens is not None for node in nodes):
            _fill_tokens(sample, nodes, set(targets))
        samples.append(sample)
    return samples


def _fill_tokens(sample: Sample, nodes: list[Node], targets: set[int]) -> None:
    input_ids: list[int] = []
    loss_mask: list[int] = []
    logprobs: list[float] = []
    mask_rows: list[list[int]] | None = []
    saw_mask = False
    routed: list[np.ndarray] | None = []
    for node in nodes:
        tokens = node.tokens
        assert tokens is not None
        length = len(tokens.token_ids)
        input_ids.extend(tokens.token_ids)
        logprobs.extend(tokens.logprobs if tokens.logprobs is not None else [0.0] * length)
        trains = node.id in targets and tokens.sampled_start is not None
        start = tokens.sampled_start if trains else length
        assert start is not None
        loss_mask.extend([0] * start + [1] * (length - start))
        if mask_rows is not None:
            if not trains:
                mask_rows.extend([] for _ in range(length))
            elif tokens.sampling_mask is None:
                mask_rows = None
            else:
                saw_mask = True
                mask_rows.extend([] for _ in range(start))
                mask_rows.extend(tokens.sampling_mask)
        if routed is not None:
            if tokens.routed_experts is None or len(tokens.routed_experts) != length:
                routed = None
            else:
                routed.append(np.asarray(tokens.routed_experts))
    sample.input_ids, sample.loss_mask, sample.logprobs = input_ids, loss_mask, logprobs
    sample.sampling_mask = mask_rows if saw_mask else None
    if routed:
        sample.routed_experts = np.concatenate(routed)


def _csr(rows: list[list[int]]) -> dict[str, list[int]]:
    offsets = [0]
    ids: list[int] = []
    for row in rows:
        ids.extend(row)
        offsets.append(len(ids))
    return {"ids": ids, "offsets": offsets}

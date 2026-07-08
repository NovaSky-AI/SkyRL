"""Stateful samplers for :class:`~skyrl.train.sft_trainer.SFTTrainer`.

These samplers plug into ``torchdata.stateful_dataloader.StatefulDataLoader``
so the sampling position is captured in the dataloader's ``state_dict`` and
restored on resume. A sampler exposes ``state_dict``/``load_state_dict``; the
``StatefulDataLoader`` fast-forwards to the saved position when iteration
resumes after a checkpoint load.

Core ships :class:`StatefulSequentialSampler` (backing the ``sampler="sequential"``
config option). The ``sampler="custom"`` path loads a user-supplied sampler from
``SFTConfig.sampler_class_path`` via :func:`import_sampler_class`, instantiating
it as ``ClassName(tokenized, **sampler_kwargs)``. See
``examples/train/sft/curriculum_sampler.py`` for a reference custom sampler.
"""

from __future__ import annotations

import importlib
from typing import Iterator, Sequence

import torch

__all__ = [
    "StatefulSequentialSampler",
    "import_sampler_class",
]


def import_sampler_class(class_path: str) -> type:
    """Import a sampler class from a ``module.path.ClassName`` string.

    Args:
        class_path: Fully-qualified import path, e.g.
            ``"examples.train.sft.curriculum_sampler.CurriculumLearningSampler"``.
            The import runs inside a Ray task (which does not inherit the
            driver's ``PYTHONPATH``), so the module must be importable from the
            worker's ``sys.path`` -- e.g. a dotted path from the repo root when
            launching from it.

    Returns:
        The resolved class object.

    Raises:
        ValueError: If ``class_path`` is not a dotted ``module.ClassName`` path.
    """
    module_path, _, class_name = class_path.rpartition(".")
    if not module_path:
        raise ValueError(
            f"Invalid sampler_class_path '{class_path}'; expected a dotted path like " f"'my_module.MySampler'."
        )
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class StatefulSequentialSampler(torch.utils.data.Sampler[int]):
    """Yield indices ``0..len-1`` in order, resumable across checkpoints.

    Unlike ``torch.utils.data.SequentialSampler``, this tracks an internal
    ``position`` cursor and exposes ``state_dict``/``load_state_dict`` so that
    a ``StatefulDataLoader`` can resume mid-epoch from the exact next index.
    The cursor resets to ``0`` once an epoch is exhausted, so a fresh iterator
    starts over from the beginning.
    """

    def __init__(self, data_source: Sequence):
        self.data_source = data_source
        self.position = 0

    def __iter__(self) -> Iterator[int]:
        while self.position < len(self.data_source):
            idx = self.position
            self.position += 1
            yield idx
        # Reset so the next epoch (a fresh ``iter()``) starts from the top.
        self.position = 0

    def __len__(self) -> int:
        return len(self.data_source)

    def state_dict(self) -> dict:
        return {"position": self.position}

    def load_state_dict(self, state: dict) -> None:
        self.position = state["position"]

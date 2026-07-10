"""Deprecated location: ``DataMixingSampler`` is now part of core SkyRL.

Weighted multi-dataset mixing no longer requires a custom sampler -- configure it
natively via the list-based dataset fields, e.g.::

    bash examples/train/sft/run_sft_megatron.sh \
        'train_datasets=[allenai/tulu-3-sft-mixture,yahma/alpaca-cleaned]' \
        'train_dataset_splits=[train[:50000],train[:10000]]' \
        'train_dataset_weights=[0.8,0.2]'

With multiple ``train_datasets`` and the default ``sampler=random``, the trainer
uses :class:`skyrl.train.dataset.samplers.DataMixingSampler` automatically,
configured with the tokenized per-dataset lengths and ``train_dataset_weights``.

This module re-exports the core class so existing configs pointing
``sampler_class_path`` here keep working. Note a behavior change from the old
example: the core sampler draws a *fresh* weighted plan every epoch (the example
replayed one fixed plan), and its ``state_dict`` now includes the generator
state so mid-epoch checkpoint resume is exact. Old position-only checkpoints
load with a warning.
"""

from skyrl.train.dataset.samplers import DataMixingSampler

__all__ = ["DataMixingSampler"]

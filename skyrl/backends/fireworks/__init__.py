"""Fireworks Training API integration helpers.

The hosted runtime is intentionally kept separate from SkyRL's Ray worker
dispatch.  This package starts with the provider-neutral GRPO datum conversion;
the service, training, and inference adapters build on that contract.
"""

from skyrl.backends.fireworks.grpo import (
    GRPODatumSpec,
    build_tinker_grpo_datums,
    training_batch_to_grpo_datum_specs,
)

__all__ = [
    "GRPODatumSpec",
    "build_tinker_grpo_datums",
    "training_batch_to_grpo_datum_specs",
]

"""Runtime backport of Megatron-Bridge PR #6184 for the pinned Bridge revision.

Bridge's ``PackedPerExpertLinear.sharded_state_dict`` (the per-expert side of
``experts_shared_outer_loras=True`` adapters) builds the packed weight's sharded tensor
without the required ``pg_collection`` kwarg, so dist-checkpoint saving raises
``TypeError``. This supplies the same MPU-backed collection the other grouped-expert
adapters resolve via ``_get_pg_collection``.
"""

from __future__ import annotations

import inspect


def apply_packed_per_expert_sharded_state_dict_patch() -> None:
    """Install #6184 before saving or loading a shared-outer LoRA checkpoint."""
    from megatron.bridge.peft import utils

    cls = getattr(utils, "PackedPerExpertLinear", None)
    if cls is None or getattr(cls, "_skyrl_sharded_state_dict_patch", False):
        return
    # Skip the backport when Bridge includes #6184.
    if "pg_collection" in inspect.getsource(cls.sharded_state_dict):
        return

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        key = f"{prefix}weight"
        return {
            key: utils._make_grouped_expert_sharded_tensor(
                self.weight.data,
                key,
                tp_axis=None,
                sharded_offsets=sharded_offsets,
                pg_collection=utils._get_pg_collection(required_pgs=["ep", "expt_tp", "expt_dp"]),
            )
        }

    cls.sharded_state_dict = sharded_state_dict
    cls._skyrl_sharded_state_dict_patch = True

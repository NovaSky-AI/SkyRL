"""Don't copy TE FusedAdam's state in ``DistributedOptimizer.load_state_dict``.

Upstream reads the inner optimizer's ``state_dict()`` for its param groups and reuses the state as
placeholders that ``load_parameter_state_from_*`` overwrites. TE's ``FusedAdam`` returns fp32 copies
of all state from ``state_dict()`` and re-creates it in ``load_state_dict``, so a resume holds an
extra fp32 copy of the optimizer state. When the FusedAdam state is already allocated, only the
param groups (hyperparameters and ``step``) are restored; other optimizers take upstream's path.
"""

import torch
from loguru import logger

_APPLIED = False


def _restore_param_groups(optimizer, state_dict) -> None:
    saved_groups = state_dict["param_groups"]
    if len(saved_groups) != len(optimizer.param_groups):
        raise ValueError("loaded state dict has a different number of parameter groups")
    for group, saved in zip(optimizer.param_groups, saved_groups):
        if len(group["params"]) != len(saved["params"]):
            raise ValueError(
                "loaded state dict contains a parameter group that doesn't match the size of optimizer's group"
            )
        for key, value in saved.items():
            if key != "params":
                group[key] = value


def apply_dist_optim_te_load_state_dict_patch() -> None:
    global _APPLIED
    if _APPLIED:
        return
    try:
        from transformer_engine.pytorch.optimizers import FusedAdam
    except ImportError:
        return
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    original = DistributedOptimizer.load_state_dict

    def load_state_dict(self, state_dict):
        inner = self.optimizer
        if not isinstance(inner, FusedAdam) or len(inner.state) == 0:
            return original(self, state_dict)
        inner.state_dict = lambda: {**torch.optim.Optimizer.state_dict(inner), "state": {}}
        inner.load_state_dict = lambda sd: _restore_param_groups(inner, sd)
        try:
            return original(self, state_dict)
        finally:
            del inner.state_dict, inner.load_state_dict

    load_state_dict.__wrapped__ = original
    DistributedOptimizer.load_state_dict = load_state_dict
    _APPLIED = True
    logger.info("DistributedOptimizer.load_state_dict: TE FusedAdam state is not copied during checkpoint load")

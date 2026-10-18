"""Which Megatron parameters a merged-LoRA weight sync has to send.

With ``merge_lora=True`` every sync exports ``base + B @ A`` for the LoRA-wrapped linears. All
other base weights are frozen: they never change, and vLLM already holds the copies it loaded
from the same checkpoint. Re-sending them (embedding, LM head, norms, hyper-connections, ...)
costs transfer time and, worse, receive-side memory for the largest of them. This module picks
the conversion tasks a sync must keep, as a ``param_filter`` for ``MegatronWeightSource``.

**What changes.** A task changes when its Megatron parameter is

- the base of a LoRA wrapper (``<module>.to_wrap.<param>``): the bridge merges the adapter into
  ``...to_wrap.weight``, and the wrapper's other params (bias, a fused ``layer_norm_weight``) ride
  along so the vLLM linear holding weight and bias is reloaded whole;
- any other trainable parameter (``requires_grad``), e.g. a module a user unfroze;
- any persistent buffer (e.g. a router's ``expert_bias``, which the load-balancing update moves
  even with a frozen router).

**What vLLM needs whole.** vLLM's layerwise reload (``initialize_layerwise_reload`` /
``finalize_layerwise_reload``) treats every module separately. A module none of whose tensors
arrive is put back as it was. A module that receives *some* of its tensors is rebuilt from
scratch and its unsent tensors come back uninitialized. vLLM fuses checkpoint tensors across
Megatron modules (``FusedMoE`` holds every expert's ``fc1`` *and* ``fc2``; MLA's
``fused_qkv_a_proj`` holds ``q_down`` and ``kv_down``), so "only the LoRA-wrapped module" is not a
safe unit. The unit here is the decoder sub-block a changed task lives in, ``...layers.<i>.<block>.``
(``self_attention``, ``mlp``, a hyper-connection site, ...): vLLM fuses within those, never across.
Top-level modules (``embedding.word_embeddings``, ``output_layer``) are their own unit.

**Correctness precondition**: vLLM's unsent weights must still be the checkpoint's. It boots with
them unless ``load_format="dummy"`` (FP8 serialized sync), and level-2 sleep discards them; both
are excluded by config validation and the forced level-1 sleep (``_uses_partial_weight_sync``).
With full fine-tuning every weight is trainable, so nothing would be dropped; the filter is only
built for LoRA.

Import-light (torch only) so it is testable without vLLM.
"""

import re
from typing import Any, Iterable, Optional, Sequence, Set

import torch
from loguru import logger

__all__ = ["MergedLoraSyncFilter", "select_sync_names", "sync_unit"]

LORA_BASE_MARKER = ".to_wrap."
_LAYER_BLOCK_RE = re.compile(r"^(.*?(?:^|\.)layers\.\d+\.[^.]+\.)")


def sync_unit(name: str) -> str:
    """The prefix of the parameter group vLLM may fuse ``name`` with (the unit sent or skipped whole)."""
    match = _LAYER_BLOCK_RE.match(name)
    if match:
        return match.group(1)
    if LORA_BASE_MARKER in name:
        return name.partition(LORA_BASE_MARKER)[0] + "."
    return name.rsplit(".", 1)[0] + "."


def select_sync_names(names: Sequence[str], changed: Iterable[str]) -> Set[str]:
    """Names in ``names`` that share a :func:`sync_unit` with any name in ``changed``."""
    units = {sync_unit(name) for name in changed}
    return {name for name in names if sync_unit(name) in units}


def _locally_changed(tasks: Iterable[Any]) -> Set[str]:
    """Global names of the tasks this rank holds that a merged-LoRA sync changes."""
    changed = set()
    for task in tasks:
        name = task.global_param_name
        if LORA_BASE_MARKER in name:
            changed.add(name)
            continue
        weight = task.param_weight
        if weight is None:  # held by another pipeline stage
            continue
        if not isinstance(weight, torch.nn.Parameter) or weight.requires_grad:
            changed.add(name)
    return changed


def _all_ranks_union(local: Set[str], group: Optional[Any] = None) -> Set[str]:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return set(local)
    gathered = [None] * torch.distributed.get_world_size(group)
    torch.distributed.all_gather_object(gathered, sorted(local), group=group)
    return set().union(*(set(names) for names in gathered))


class MergedLoraSyncFilter:
    """``param_filter`` for ``MegatronWeightSource`` under ``merge_lora=True``.

    Resolved on the first call, from the bridge's conversion tasks. That first call is the
    source's first export (``metadata()``), which every trainer rank makes at the same point, so
    the one collective here (the union of each rank's trainable names and buffers: under PP and
    EP no rank holds every parameter) cannot desynchronize them. Every rank then filters the same
    global name list, as ``MegatronWeightSource`` requires.
    """

    def __init__(self, bridge: Any, module: Any, group: Optional[Any] = None) -> None:
        self._bridge = bridge
        self._module = module
        self._group = group
        self._keep: Optional[Set[str]] = None

    def resolve(self) -> Set[str]:
        if self._keep is None:
            tasks = self._bridge.get_conversion_tasks(self._module)
            names = [task.global_param_name for task in tasks]
            changed = _all_ranks_union(_locally_changed(tasks), self._group)
            if not changed:
                raise ValueError(
                    "sync_lora_targets_only: no LoRA-wrapped (`.to_wrap.`) or trainable parameter in the "
                    "Megatron model, so a merged-LoRA sync would send nothing."
                )
            keep = select_sync_names(names, changed)
            skipped = sorted({sync_unit(name) for name in names} - {sync_unit(name) for name in keep})
            logger.info(
                "Merged-LoRA sync sends {} of {} conversion tasks ({} units skipped, e.g. {})",
                len(keep),
                len(names),
                len(skipped),
                skipped[:6],
            )
            self._keep = keep
        return self._keep

    def __call__(self, global_param_name: str) -> bool:
        return global_param_name in self.resolve()

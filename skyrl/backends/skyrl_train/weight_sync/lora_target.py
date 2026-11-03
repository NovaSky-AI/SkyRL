"""LoRA adapter as a weight-sync *target*.

Weight sync has two independent axes: the **transport** that moves tensors (an
engine pair from ``weight_senders`` / ``weight_receivers``: NCCL broadcast, CUDA
IPC, ...) and the **target** that says what the tensors are and how the receiver
applies them. The default target is the base model (``model.load_weights``).
This module defines the LoRA adapter target: the trainer ships the PEFT adapter
tensors through whichever transport is configured, and the vLLM worker builds a
``LoRAModel`` from the received GPU tensors instead of reading
``adapter_model.safetensors``.

The target reaches the receiver once per sync, before ``send_weights()``, as the
argument of ``skyrl_set_lora_receive_target`` on the worker extension::

    {
        "kind": "lora",
        "lora_name": "<vLLM adapter name>",
        "adapter_config": {...},          # adapter_config.json contents
        "aliases": {public_key: sent_key},  # keys not on the wire; share a sent tensor
    }

It rides ``/collective_rpc`` rather than the transfer itself because the
transfer is vLLM's: ``TrainerWeightTransferEngine.send_weights()`` owns the
round trip and its per-round payload is a fixed dataclass of names, dtypes and
shapes. Adding a channel to it would mean subclassing the update info of every
transport; the worker extension already exists for exactly this class of
problem (see ``new_inference_worker_wrap``).

``aliases`` is how duplicated adapters stay cheap. With
``share_expert_adapters=true`` one adapter serves every expert an EP rank owns,
and the PEFT export replicates it under one key per expert, so the public
adapter is many times larger than its unique bytes (30.77 GB vs 0.62 GB on
GLM-5.3 at rank 32). Only unique tensors are sent; the receiver re-attaches the
public names to the same GPU storage before handing the dict to vLLM.

Imports torch only: the vLLM worker patch and the trainer-side source both read
it, and it must stay importable in a process without the wheel.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, Mapping, Optional, Tuple

import torch

LORA_RECEIVE_TARGET_KIND = "lora"

# ``LoRARequest.lora_path`` value that tells the patched vLLM worker LoRA
# manager to build the adapter from tensors staged in the worker process
# rather than from a directory. See ``patches/vllm/patch_lora_in_memory.py``.
IN_MEMORY_LORA_PATH_PREFIX = "skyrl-memory://"

_EXPERT_KEY = re.compile(r"^(?P<prefix>.*\.experts\.)(?P<index>\d+)(?P<suffix>\..*)$")


def in_memory_lora_path(lora_name: str) -> str:
    """The ``lora_path`` marker for an adapter staged in the vLLM worker."""
    return f"{IN_MEMORY_LORA_PATH_PREFIX}{lora_name}"


def lora_name_from_in_memory_path(lora_path: str) -> Optional[str]:
    """Inverse of :func:`in_memory_lora_path`; ``None`` for ordinary paths."""
    if not lora_path.startswith(IN_MEMORY_LORA_PATH_PREFIX):
        return None
    return lora_path[len(IN_MEMORY_LORA_PATH_PREFIX) :]


def build_lora_receive_target(
    lora_name: str,
    adapter_config: Mapping[str, object],
    aliases: Mapping[str, str],
) -> Dict[str, object]:
    if not lora_name:
        raise ValueError("lora_name cannot be empty")
    return {
        "kind": LORA_RECEIVE_TARGET_KIND,
        "lora_name": lora_name,
        "adapter_config": dict(adapter_config),
        "aliases": dict(aliases),
    }


def is_lora_receive_target(receive_target: Optional[Mapping[str, object]]) -> bool:
    return bool(receive_target) and receive_target.get("kind") == LORA_RECEIVE_TARGET_KIND


def expand_lora_aliases(tensors: Dict[str, torch.Tensor], aliases: Mapping[str, str]) -> Dict[str, torch.Tensor]:
    """Re-attach alias keys to their sent tensor. Shares storage, copies nothing."""
    expanded = dict(tensors)
    for public_key, sent_key in aliases.items():
        if sent_key not in tensors:
            raise KeyError(f"alias {public_key!r} refers to {sent_key!r}, which was not received")
        expanded[public_key] = tensors[sent_key]
    return expanded


def _expert_alias_group(key: str, experts_per_shared_adapter: int) -> Optional[Tuple[str, int]]:
    """Group id for a per-expert adapter key, or ``None`` for non-expert keys.

    ``experts_per_shared_adapter`` is how many consecutive expert indices are
    expected to carry the same LoRA tensor: experts
    ``[g*n, (g+1)*n)`` form group ``g``. With ``share_expert_adapters`` that is
    the experts one EP rank owns, since they train through one adapter, but
    the value is a sharing span, not a parallelism fact.
    """
    match = _EXPERT_KEY.match(key)
    if match is None:
        return None
    group = int(match.group("index")) // experts_per_shared_adapter
    return (f"{match.group('prefix')}*{match.group('suffix')}", group)


def dedupe_shared_expert_adapters(
    adapter_state: "Mapping[str, torch.Tensor] | Iterable[Tuple[str, torch.Tensor]]",
    experts_per_shared_adapter: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    """Split the adapter into tensors to send and aliases onto them.

    ``experts_per_shared_adapter`` is the number of consecutive expert keys that
    may share one identical adapter tensor (``1`` disables aliasing). Members of
    a group are checked with ``torch.equal`` against the group's first key and a
    member that differs is sent on its own. The grouping is therefore only a
    hint: a wrong value costs bandwidth, never correctness.

    ``adapter_state`` may be a lazy iterable of ``(key, tensor)`` pairs. Only
    the retained tensors are referenced, so a duplicate that the exporter
    materialized is freed as soon as it has been compared: peak memory is the
    unique adapter plus one tensor, not the public adapter.

    Returns ``(to_send, aliases)`` where ``to_send`` preserves the input order
    and ``aliases`` maps every omitted key to the key that carries its value.
    """
    if experts_per_shared_adapter <= 0:
        raise ValueError(f"experts_per_shared_adapter must be positive, got {experts_per_shared_adapter}")
    to_send: Dict[str, torch.Tensor] = {}
    aliases: Dict[str, str] = {}
    canonical_by_group: Dict[Tuple[str, int], str] = {}
    pairs = adapter_state.items() if isinstance(adapter_state, Mapping) else adapter_state
    for key, tensor in pairs:
        group = _expert_alias_group(key, experts_per_shared_adapter)
        if group is None:
            to_send[key] = tensor
            continue
        canonical_key = canonical_by_group.get(group)
        if canonical_key is None:
            canonical_by_group[group] = key
            to_send[key] = tensor
            continue
        canonical = to_send[canonical_key]
        if (
            canonical.dtype == tensor.dtype
            and canonical.shape == tensor.shape
            and (canonical.data_ptr() == tensor.data_ptr() or torch.equal(canonical, tensor))
        ):
            aliases[key] = canonical_key
        else:
            to_send[key] = tensor
    return to_send, aliases

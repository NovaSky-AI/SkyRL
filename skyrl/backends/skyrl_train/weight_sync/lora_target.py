"""LoRA adapter as a weight-sync *target*.

Weight sync has two independent axes: the **transport** that moves tensors
(``WeightTransferStrategy``: NCCL broadcast, CUDA IPC, ...) and the **target**
that says what the tensors are and how the receiver applies them. The default
target is the base model (``model.load_weights``). This module defines the LoRA
adapter target: the trainer ships the PEFT adapter tensors through whichever
transport is configured, and the vLLM worker builds a ``LoRAModel`` from the
received GPU tensors instead of reading ``adapter_model.safetensors``.

The target travels to the receiver once per sync as the ``receive_target``
argument of ``skyrl_start_weight_update``::

    {
        "kind": "lora",
        "lora_name": "<vLLM adapter name>",
        "adapter_config": {...},          # adapter_config.json contents
        "aliases": {public_key: sent_key},  # keys not on the wire; share a sent tensor
    }

``aliases`` is how duplicated adapters stay cheap. With
``share_expert_adapters=true`` one adapter serves every expert an EP rank owns,
and the PEFT export replicates it under one key per expert, so the public
adapter is many times larger than its unique bytes (30.77 GB vs 0.62 GB on
GLM-5.3 at rank 32). Only unique tensors are sent; the receiver re-attaches the
public names to the same GPU storage before handing the dict to vLLM.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

import torch

from skyrl.backends.skyrl_train.weight_sync.base import WeightChunk, torch_dtype_name
from skyrl.backends.skyrl_train.weight_sync.weight_extractor import WeightExtractor

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


def chunk_adapter_state(
    adapter_state: Mapping[str, torch.Tensor],
    dtype: torch.dtype,
    bucket_size_bytes: int,
) -> Iterator[WeightChunk]:
    """Yield ``WeightChunk``s of at most ``bucket_size_bytes`` in the wire dtype.

    Casting happens here, so the tensors the receiver hands to vLLM are already
    in ``lora_dtype`` and vLLM's per-key ``.to()`` is a no-op that keeps aliases
    sharing storage. A cast to a different dtype would materialize every alias.
    """
    if bucket_size_bytes <= 0:
        raise ValueError("bucket_size_bytes must be positive")
    names: List[str] = []
    tensors: List[torch.Tensor] = []
    size = 0

    def flush() -> Optional[WeightChunk]:
        if not names:
            return None
        chunk = WeightChunk(
            names=list(names),
            dtypes=[str(t.dtype) for t in tensors],
            shapes=[list(t.shape) for t in tensors],
            tensors=list(tensors),
        )
        names.clear()
        tensors.clear()
        return chunk

    for name, tensor in adapter_state.items():
        tensor = tensor.detach().to(dtype=dtype).contiguous()
        nbytes = tensor.numel() * tensor.element_size()
        if names and size + nbytes > bucket_size_bytes:
            yield flush()
            size = 0
        names.append(name)
        tensors.append(tensor)
        size += nbytes
    chunk = flush()
    if chunk is not None:
        yield chunk


class LoraAdapterExtractor(WeightExtractor):
    """``WeightExtractor`` whose stream is a PEFT adapter rather than the model.

    Subclasses implement :meth:`export_adapter_stream` (a collective on the
    trainer) yielding the public adapter's ``(key, tensor)`` pairs on GPU, and
    :meth:`finalize_adapter`, which turns the retained unique tensors into the
    layout vLLM loads plus the ``adapter_config``. This class dedupes shared
    expert adapters as the stream arrives, publishes the ``receive_target`` the
    transport forwards to the receiver, and chunks the unique tensors.

    Dedupe happens on the stream, before finalize, because the exporter may
    materialize every public key: on GLM-5.3 that is 30.77 GB per rank against
    0.62 GB unique. ``finalize_adapter`` therefore only ever sees canonical
    keys. Alias groups are per (module, expert-group, lora_A|lora_B), so the
    ``lora_A`` sibling of a canonical ``lora_B`` is itself canonical and any
    per-key transform that pairs A with B still has both.

    ``receive_target`` is read by ``WeightTransferSender.send`` *before*
    ``extract_weights``, so the export runs lazily on first access of either
    and is cached until the chunk stream has been consumed.
    """

    def __init__(self, *, experts_per_shared_adapter: int, bucket_size_threshold_GB: float = 1.0):
        self._experts_per_shared_adapter = experts_per_shared_adapter
        self._bucket_size_bytes = int(bucket_size_threshold_GB * 1024**3)
        self._lora_name: Optional[str] = None
        self._prepared: Optional[Tuple[Dict[str, torch.Tensor], Dict[str, str], Dict[str, object]]] = None

    def set_lora_name(self, lora_name: str) -> None:
        """Name vLLM registers the adapter under; set per sync (multi-tenant)."""
        self._lora_name = lora_name

    def export_adapter_stream(self) -> Iterable[Tuple[str, torch.Tensor]]:
        """Yield the public adapter's ``(key, tensor)`` pairs. Collective."""
        raise NotImplementedError

    def finalize_adapter(
        self, adapter_state: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
        """Map the unique tensors to what vLLM loads; return them with ``adapter_config``.

        Must keep every key it is given (values may change) so the alias map
        built on the stream stays valid; it may rename only keys that can never
        be aliased. The default is the identity with an empty config.
        """
        return adapter_state, {}

    def _prepare(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, str], Dict[str, object]]:
        if self._prepared is None:
            to_send, aliases = dedupe_shared_expert_adapters(
                self.export_adapter_stream(), self._experts_per_shared_adapter
            )
            to_send, adapter_config = self.finalize_adapter(to_send)
            missing = set(aliases.values()) - set(to_send)
            if missing:
                raise RuntimeError(f"finalize_adapter dropped aliased keys: {sorted(missing)[:5]}")
            self._prepared = (to_send, aliases, adapter_config)
        return self._prepared

    @property
    def derives_metadata_from_chunks(self) -> bool:
        # Names and shapes come off the chunk stream; there is no second export.
        return True

    def get_weight_metadata(self, dtype: torch.dtype) -> Dict[str, List]:
        to_send, _, _ = self._prepare()
        dtype_name = torch_dtype_name(dtype)
        return {
            "names": list(to_send),
            "dtype_names": [dtype_name] * len(to_send),
            "shapes": [list(t.shape) for t in to_send.values()],
        }

    @property
    def receive_target(self) -> Dict[str, object]:
        if self._lora_name is None:
            raise RuntimeError("set_lora_name must be called before publishing a LoRA adapter")
        _, aliases, adapter_config = self._prepare()
        return build_lora_receive_target(self._lora_name, adapter_config, aliases)

    def extract_weights(self, dtype: torch.dtype) -> Iterator[WeightChunk]:
        to_send, _, _ = self._prepare()
        try:
            yield from chunk_adapter_state(to_send, dtype, self._bucket_size_bytes)
        finally:
            # The next sync re-exports; nothing from this one is worth keeping.
            self._prepared = None

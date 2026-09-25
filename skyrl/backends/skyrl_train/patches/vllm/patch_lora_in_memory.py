"""Runtime patch: let vLLM's worker LoRA manager load an adapter from tensors.

vLLM 0.28 can only load a LoRA adapter from a directory: ``LoRARequest`` carries
a path, and ``WorkerLoRAManager._load_adapter`` reads ``adapter_config.json`` and
``adapter_model.safetensors`` from it, materializes every tensor on pinned CPU
memory and copies it into the GPU LoRA slots. ``LoRAModel.from_lora_tensors``
exists but nothing public reaches it (vllm#4068 asked for this in 2024 and was
closed as stale). For RL weight sync that means every adapter update is a
file write, a file read, and a CPU round trip of the full public adapter.

This patch adds the missing entry point without a fork:

* :func:`stage_in_memory_adapter` records ``(tensors, peft_config)`` for an
  adapter name in the worker process. The tensors are the received GPU
  tensors; nothing is copied.
* ``WorkerLoRAManager._load_adapter`` is wrapped. A request whose ``lora_path``
  is ``skyrl-memory://<name>`` is served from the staged tensors through
  ``LoRAModel.from_lora_tensors`` with the same validation the directory path
  applies (``PEFTHelper.validate_legal``, expected-module check, non-local
  expert filtering under EP, ``hf_to_vllm_mapper``, ``lora_skip_prefixes``).
  Every other request goes to the original method untouched.

The adapter is built on the GPU (``device=self.device``) rather than on CPU:
the tensors already live there and, because the sender casts to ``lora_dtype``
first, ``from_lora_tensors``'s per-key ``.to()`` returns the same tensor, so
aliases created by :func:`expand_lora_aliases` keep sharing storage. Loading on
CPU would pin a private copy per key and re-inflate a deduplicated adapter to
its public size.

``add_lora`` still flows API server -> engine core -> every worker, so the
adapter name in the OpenAI-compatible model registry stays correct and
``load_inplace=True`` replaces the previous generation exactly as the
directory path does. Staged tensors stay registered after the load: the
``LoRAModel`` shares their storage (same device and dtype), so this costs
nothing, and it lets vLLM rebuild the adapter after an LRU eviction the way it
would re-read a directory. The next sync for the same name replaces them;
``discard_in_memory_adapter`` frees them on unload.

Sharing storage has one consequence: vLLM applies ``lora_alpha / r`` by
multiplying ``lora_b`` *in place* on every load (``LoRALayerWeights.optimize``),
so a non-unit scale would compound on each LRU rebuild. Staged adapters must
therefore arrive with the scale already folded into ``lora_B`` and
``lora_alpha == r`` in their config; the loader rejects anything else.

Remove this module once vLLM ships a tensor-backed LoRA request.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import torch
from loguru import logger

from skyrl.backends.skyrl_train.weight_sync.lora_target import (
    lora_name_from_in_memory_path,
)

_PATCHED = False
_ORIGINAL_LOAD_ADAPTER = None


@dataclass
class StagedLoRAAdapter:
    tensors: Dict[str, torch.Tensor]
    peft_config: Dict[str, Any]


# Per worker process. Keyed by the vLLM adapter name, which is what the
# ``LoRARequest`` carries; the ``lora_int_id`` is only allocated by the API
# server at load time.
_STAGED: Dict[str, StagedLoRAAdapter] = {}


def stage_in_memory_adapter(
    lora_name: str, tensors: Mapping[str, torch.Tensor], peft_config: Mapping[str, Any]
) -> None:
    """Make ``tensors`` the adapter the next ``skyrl-memory://<lora_name>`` load builds."""
    if not lora_name:
        raise ValueError("lora_name cannot be empty")
    if not tensors:
        raise ValueError(f"no tensors staged for LoRA adapter {lora_name!r}")
    _STAGED[lora_name] = StagedLoRAAdapter(tensors=dict(tensors), peft_config=dict(peft_config))


def staged_adapter_names() -> list[str]:
    return list(_STAGED)


def discard_in_memory_adapter(lora_name: str) -> bool:
    return _STAGED.pop(lora_name, None) is not None


def _expected_lora_modules(adapter_manager) -> set[str]:
    # Mirrors the expansion at the top of WorkerLoRAManager._load_adapter.
    expected: list[str] = []
    packed = adapter_manager.packed_modules_mapping
    for module in adapter_manager.supported_lora_modules:
        if module in packed:
            expected.extend(packed[module])
        else:
            expected.append(module)
        if module == "experts":
            expected.append(module)
    return set(expected)


def _check_unexpected_modules(
    tensor_names,
    expected_lora_modules: set[str],
    weights_mapper,
    skip_prefixes,
    lora_name: str,
) -> None:
    # Same rules as LoRAModel.from_local_checkpoint.check_unexpected_modules.
    from vllm.lora.lora_model import LoRAModel, is_base_embedding_weights
    from vllm.lora.utils import parse_fine_tuned_lora_name

    unexpected = []
    for tensor_name in tensor_names:
        if is_base_embedding_weights(tensor_name) or "base_layer" in tensor_name:
            continue
        if skip_prefixes and LoRAModel._should_skip_module(tensor_name, skip_prefixes):
            continue
        module_name, _ = parse_fine_tuned_lora_name(tensor_name, weights_mapper)
        if ".experts" in module_name:
            expert_suffix = module_name[module_name.find(".experts") + 1 :]
            if expert_suffix not in expected_lora_modules:
                unexpected.append(module_name)
        elif module_name.rsplit(".", 1)[-1] not in expected_lora_modules:
            unexpected.append(module_name)
    if unexpected:
        raise ValueError(
            f"While loading in-memory LoRA adapter {lora_name!r}, expected target modules in "
            f"{expected_lora_modules} but received {unexpected}."
        )


def _load_in_memory_adapter(manager, lora_request, staged: StagedLoRAAdapter):
    from vllm.lora.lora_model import _is_remote_expert_key
    from vllm.lora.peft_helper import PEFTHelper

    adapter_manager = manager._adapter_manager
    # vLLM applies lora_alpha / r by multiplying lora_b in place on every load
    # (LoRALayerWeights.optimize), and the staged tensors are handed over by
    # reference and reused to rebuild an LRU-evicted adapter. A non-unit scale
    # would therefore compound on each reload. The trainer folds the scale into
    # lora_B and publishes lora_alpha == r; refuse anything else.
    r = staged.peft_config.get("r")
    lora_alpha = staged.peft_config.get("lora_alpha", r)
    if lora_alpha != r:
        raise ValueError(
            f"in-memory LoRA adapter {lora_request.lora_name!r} has lora_alpha={lora_alpha} != r={r}. "
            "Staged adapters must carry alpha / r folded into lora_B and publish lora_alpha == r "
            "(fold_lora_alpha_for_vllm), because vLLM scales lora_b in place on every load."
        )
    peft_helper = PEFTHelper.from_dict(staged.peft_config)
    peft_helper.validate_legal(manager.lora_config)

    model = adapter_manager.model
    weights_mapper = getattr(model, "hf_to_vllm_mapper", None)
    if weights_mapper is not None:
        weights_mapper = weights_mapper.get_rename_mapper()
    skip_prefixes = getattr(model, "lora_skip_prefixes", None)

    _check_unexpected_modules(
        staged.tensors.keys(),
        _expected_lora_modules(adapter_manager),
        weights_mapper,
        skip_prefixes,
        lora_request.lora_name,
    )

    tensors = staged.tensors
    moe_ep_spec = getattr(adapter_manager, "moe_ep_load_spec", None)
    if moe_ep_spec is not None:
        tensors = {k: v for k, v in tensors.items() if not _is_remote_expert_key(k, moe_ep_spec)}

    lora = manager._lora_model_cls.from_lora_tensors(
        lora_model_id=lora_request.lora_int_id,
        tensors=tensors,
        peft_helper=peft_helper,
        device=manager.device,
        dtype=manager.lora_config.lora_dtype,
        model_vocab_size=manager.vocab_size,
        weights_mapper=weights_mapper,
        skip_prefixes=skip_prefixes,
    )
    lora.is_3d_lora_weight = lora_request.is_3d_lora_weight
    return lora


def _patched_load_adapter(self, lora_request):
    lora_name = lora_name_from_in_memory_path(lora_request.lora_path)
    if lora_name is None:
        return _ORIGINAL_LOAD_ADAPTER(self, lora_request)
    staged = _STAGED.get(lora_name)
    if staged is None:
        raise RuntimeError(
            f"LoRA adapter {lora_name!r} was requested from memory ({lora_request.lora_path}) but no "
            "tensors are staged in this worker: either no weight update with a LoRA receive_target has "
            "run for it yet, or it was unloaded. A resync stages it again."
        )
    return _load_in_memory_adapter(self, lora_request, staged)


def apply_lora_in_memory_patch() -> None:
    """Install the wrapper on ``WorkerLoRAManager`` (idempotent, worker-process local)."""
    global _PATCHED, _ORIGINAL_LOAD_ADAPTER
    if _PATCHED:
        return
    try:
        from vllm.lora.worker_manager import WorkerLoRAManager
    except ModuleNotFoundError:
        # Trainer processes import the worker-extension module too.
        return
    _ORIGINAL_LOAD_ADAPTER = WorkerLoRAManager._load_adapter
    WorkerLoRAManager._load_adapter = _patched_load_adapter
    _PATCHED = True
    logger.debug("in-memory LoRA patch installed on vllm WorkerLoRAManager._load_adapter")


def is_patched() -> bool:
    return _PATCHED

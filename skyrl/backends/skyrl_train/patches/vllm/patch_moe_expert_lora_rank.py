"""Runtime patch: size vLLM's MoE expert LoRA slots at the expert adapter rank.

vLLM allocates every LoRA slot at ``max_lora_rank``, MoE expert slots included,
and its fused MoE-LoRA kernels take their rank from the slot shape
(``lora_a_stacked[0].shape[-2]``). Expert adapters with a smaller rank (e.g.
``rank // top_k`` under ``normalize_moe_lora``) are zero-padded into the slots,
and the padded rows and columns are still loaded and multiplied.

This patch gives MoE expert layers their own slot rank, read from
``additional_config["max_expert_lora_rank"]`` (``additional_config`` is part of
vLLM's config hash). Without the key, nothing changes. With it:

* ``LoRAModelManager._create_lora_modules`` builds the ``MoERunner`` wrappers
  with a copy of the LoRA config whose ``max_lora_rank`` is the expert rank.
  Dense and attention wrappers keep ``max_lora_rank``.
* ``LoRAModelManager._create_merged_loras_inplace`` rejects an adapter whose
  expert tensors have a rank above the expert slot rank, before it is
  registered. Smaller ranks are zero-padded by ``set_lora`` as before.
* ``LoRAModelManager.get_dummy_lora_warmup_rank`` caps the warmup adapter's
  rank at the expert slot rank.
* With ``fully_sharded_loras``, the expert rank must be divisible by the MoE
  tensor parallel size (the w13 lora_A rank dim is sharded).

TODO: remove once vLLM supports a separate expert LoRA slot rank.
"""

import copy
import functools
import inspect
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

ADDITIONAL_CONFIG_KEY = "max_expert_lora_rank"

_PATCHED = False


def _expert_lora_rank(manager) -> Optional[int]:
    return getattr(manager, "_skyrl_expert_lora_rank", None)


def _read_expert_lora_rank(lora_config, vllm_config) -> Optional[int]:
    additional_config = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional_config, dict) or additional_config.get(ADDITIONAL_CONFIG_KEY) is None:
        return None
    rank = int(additional_config[ADDITIONAL_CONFIG_KEY])
    if not 0 < rank <= lora_config.max_lora_rank:
        raise ValueError(
            f"additional_config.{ADDITIONAL_CONFIG_KEY} must be in [1, max_lora_rank="
            f"{lora_config.max_lora_rank}], got {rank}."
        )
    return None if rank == lora_config.max_lora_rank else rank


def _with_rank(lora_config, rank: int):
    """Copy of ``lora_config`` with ``max_lora_rank=rank``.

    ``dataclasses.replace`` would re-validate ``max_lora_rank`` against vLLM's ``MaxLoRARanks``
    literal, which has no 2 or 4.
    """
    expert_config = copy.copy(lora_config)
    object.__setattr__(expert_config, "max_lora_rank", rank)
    return expert_config


def _module_ranks(weights) -> list[int]:
    """Ranks of a packed MoE LoRA: lora_A is ``(..., rank, in)``, lora_B ``(..., out, rank)``."""
    lora_a = weights.lora_a if isinstance(weights.lora_a, list) else [weights.lora_a]
    lora_b = weights.lora_b if isinstance(weights.lora_b, list) else [weights.lora_b]
    return [a.shape[-2] for a in lora_a if a is not None] + [b.shape[-1] for b in lora_b if b is not None]


def validate_expert_adapter_ranks(manager, lora_model) -> None:
    """Raise if an expert module of ``lora_model`` has a rank above the expert slot rank."""
    from vllm.lora.layers import FusedMoE3DWithLoRA, FusedMoEWithLoRA

    slot_rank = _expert_lora_rank(manager)
    if slot_rank is None:
        return
    for module_name, module in manager.modules.items():
        if not isinstance(module, (FusedMoEWithLoRA, FusedMoE3DWithLoRA)):
            continue
        weights = lora_model.loras.get(module_name)
        if weights is None:
            continue
        ranks = _module_ranks(weights)
        if ranks and max(ranks) > slot_rank:
            raise ValueError(
                f"LoRA adapter {lora_model.id}: {module_name} has rank {max(ranks)}, above the MoE expert "
                f"LoRA slot rank {slot_rank} (additional_config.{ADDITIONAL_CONFIG_KEY})."
            )


def apply_moe_expert_lora_rank_patch() -> None:
    """Install the wrappers on ``LoRAModelManager`` (idempotent, worker-process local)."""
    global _PATCHED
    if _PATCHED:
        return
    try:
        import vllm.lora.model_manager as model_manager
    except ModuleNotFoundError:
        # Trainer processes import the worker-extension module too.
        return

    manager_cls = model_manager.LoRAModelManager
    original_init = manager_cls.__init__
    original_create_lora_modules = manager_cls._create_lora_modules
    original_create_merged = manager_cls._create_merged_loras_inplace
    original_warmup_rank = manager_cls.get_dummy_lora_warmup_rank
    init_signature = inspect.signature(original_init)

    @functools.wraps(original_init)
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        bound = init_signature.bind(self, *args, **kwargs).arguments
        # Read before the original __init__, which allocates the slots.
        self._skyrl_expert_lora_rank = _read_expert_lora_rank(bound["lora_config"], bound["vllm_config"])
        if (
            self._skyrl_expert_lora_rank is not None
            and "from_layer" not in original_create_lora_modules.__code__.co_names
        ):
            raise RuntimeError(
                f"additional_config.{ADDITIONAL_CONFIG_KEY} is not supported by this vLLM version: "
                "LoRAModelManager._create_lora_modules no longer calls from_layer."
            )
        original_init(self, *args, **kwargs)

    @functools.wraps(original_create_lora_modules)
    def _create_lora_modules(self) -> None:
        rank = _expert_lora_rank(self)
        if rank is None:
            return original_create_lora_modules(self)
        from vllm.model_executor.layers.fused_moe import MoERunner

        expert_config = _with_rank(self.lora_config, rank)
        original_from_layer = model_manager.from_layer

        def from_layer(layer, max_loras, lora_config, *args, **kwargs):
            if isinstance(layer, MoERunner):
                tp_size = layer.moe_config.moe_parallel_config.tp_size
                if lora_config.fully_sharded_loras and rank % tp_size != 0:
                    raise ValueError(
                        f"fully_sharded_loras requires the MoE expert LoRA slot rank ({rank}) to be "
                        f"divisible by the MoE tensor parallel size ({tp_size})."
                    )
                lora_config = expert_config
            return original_from_layer(layer, max_loras, lora_config, *args, **kwargs)

        # _create_lora_modules resolves `from_layer` from the module globals at call time.
        model_manager.from_layer = from_layer
        try:
            original_create_lora_modules(self)
        finally:
            model_manager.from_layer = original_from_layer
        logger.info("LoRA slot ranks: %d (dense/attention), %d (MoE experts).", self.lora_config.max_lora_rank, rank)

    @functools.wraps(original_create_merged)
    def _create_merged_loras_inplace(self, lora_model) -> None:
        original_create_merged(self, lora_model)
        validate_expert_adapter_ranks(self, lora_model)

    @functools.wraps(original_warmup_rank)
    def get_dummy_lora_warmup_rank(self, default_rank: int) -> int:
        rank = _expert_lora_rank(self)
        if rank is None:
            return original_warmup_rank(self, default_rank)
        warmup_rank = original_warmup_rank(self, min(default_rank, rank))
        if warmup_rank > rank:
            raise ValueError(
                f"Unable to choose a dummy LoRA warmup rank: fully sharded modules need {warmup_rank}, "
                f"above the MoE expert LoRA slot rank {rank}."
            )
        return warmup_rank

    manager_cls.__init__ = __init__
    manager_cls._create_lora_modules = _create_lora_modules
    manager_cls._create_merged_loras_inplace = _create_merged_loras_inplace
    manager_cls.get_dummy_lora_warmup_rank = get_dummy_lora_warmup_rank
    _PATCHED = True

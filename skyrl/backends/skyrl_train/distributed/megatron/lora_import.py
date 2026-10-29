"""Load a HuggingFace PEFT LoRA adapter into a Megatron-Bridge LoRA model.

Megatron fuses several HF projections into one linear (``linear_qkv`` <- q/k/v,
GatedDeltaNet ``in_proj`` <- in_proj_qkv/z/b/a, ``linear_fc1`` <- gate/up) and gives the
fused module a single adapter. A PEFT adapter trained elsewhere (Tinker, HF ``peft``) has
one adapter per HF projection with its own ``lora_A``, so the fused module is built at
the sum of the constituents' ranks and its ``lora_B`` is block-diagonal
(:func:`stack_lora_pieces`); the function it computes is exactly the sum of the
separate adapters. The scale ratio between the adapter's ``alpha / r`` and the fused
module's ``alpha / fused_rank`` is folded into ``lora_B``.

Placement reuses the bridge's own machinery: :meth:`build_adapter_conversion_tasks`
pairs every Megatron adapter with the HF names of its base weight(s) and a mapping whose
``hf_to_megatron`` scatters a full tensor to this rank's TP shard, and
:meth:`_get_fused_adapter_linear_out_slices` run on an index tensor yields the rows each
HF projection occupies in the fused (TP-interleaved) ``lora_B`` layout.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch
from loguru import logger
from megatron.bridge.models.conversion.param_mapping import (
    ColumnParallelMapping,
    RowParallelMapping,
)
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.peft.utils import is_expert_linear
from megatron.core import parallel_state as mpu
from megatron.core.utils import unwrap_model

from skyrl.backends.skyrl_train.distributed.lora_import import (
    LoraPiece,
    PeftLoraAdapter,
    canonical_module_name,
    hf_name_variants,
    stack_lora_pieces,
)

_WEIGHT_SUFFIX_RE = re.compile(r"\.weight\d*$")


@dataclass
class SkyRLLoRA(LoRA):
    """Megatron-Bridge ``LoRA`` with a per-module rank override.

    ``module_dims`` maps a module's full name (``decoder.layers.3.self_attention.linear_qkv``)
    or its leaf name (``linear_qkv``) to the rank to build it at; everything else uses
    ``dim``. ``alpha`` is shared, so an overridden module scales by ``alpha / module_dim``
    and the importer folds the difference into its ``lora_B``.
    """

    module_dims: Dict[str, int] = field(default_factory=dict)

    def dim_for(self, full_name: Optional[str]) -> int:
        if not full_name:
            return self.dim
        if full_name in self.module_dims:
            return self.module_dims[full_name]
        return self.module_dims.get(full_name.rsplit(".", 1)[-1], self.dim)

    def transform(self, module, name=None, prefix=None):
        full_name = f"{prefix}.{name}" if prefix else name
        dim = self.dim_for(full_name)
        if dim == self.dim:
            return super().transform(module, name, prefix)
        saved_dim = self.dim
        self.dim = dim
        try:
            return super().transform(module, name, prefix)
        finally:
            self.dim = saved_dim


@dataclass
class MegatronLoraImportPlan:
    target_modules: List[str]
    """Megatron leaf names the adapter trains (``linear_qkv``, ``output_layer``, ...)."""
    module_dims: Dict[str, int]
    """Rank overrides for :class:`SkyRLLoRA`, keyed by leaf or full module name."""
    groups: Dict[str, List[LoraPiece]]
    """Adapter pieces per Megatron module (global name, e.g. ``decoder.layers.0.mlp.linear_fc1``)."""
    constituents: Dict[str, List[str]]
    """HF base weight names each Megatron module fuses, in the bridge's order."""
    max_rank: int


def _mapping_registry(bridge):
    # Bridges whose `mapping_registry` probes the checkpoint read `hf_pretrained`;
    # `_model_bridge` only gets it from the AutoBridge (see gdn_in_proj_lora_is_safe).
    model_bridge = bridge._model_bridge
    model_bridge.hf_pretrained = bridge.hf_pretrained
    return model_bridge, model_bridge.mapping_registry()


def _lookup_megatron_param(registry, module: str):
    for variant in hf_name_variants(module):
        mapping = registry.hf_to_megatron_lookup(f"{variant}.weight")
        if mapping is not None:
            return mapping
    return None


def _hf_param_names(mapping) -> List[str]:
    hf_param = mapping.hf_param
    return [hf_param] if isinstance(hf_param, str) else list(hf_param.values())


def fused_rank(pieces: Sequence[LoraPiece], tp_size: int) -> int:
    """Rank of the Megatron adapter that holds ``pieces`` (padded to a multiple of ``tp_size``)."""
    return stack_lora_pieces(pieces, target_scale=1.0, pad_to_multiple=tp_size).rank


def plan_megatron_lora_import(
    bridge,
    adapter: PeftLoraAdapter,
    *,
    tp_size: int,
    pp_size: int,
    target_modules: Optional[List[str]] = None,
) -> MegatronLoraImportPlan:
    """Map every adapter piece to its Megatron module and size the fused adapters.

    ``target_modules`` is the user's explicit Megatron target list, or ``None`` for
    ``"all-linear"``, in which case the targets are exactly the modules the adapter
    trains. An explicit list must cover every adapted module; modules in the list that
    the adapter does not cover start as fresh zero adapters.
    """
    _, registry = _mapping_registry(bridge)

    groups: Dict[str, List[LoraPiece]] = {}
    constituents: Dict[str, List[str]] = {}
    unmatched: List[str] = []
    for module, piece in adapter.pieces.items():
        mapping = _lookup_megatron_param(registry, module)
        if mapping is None:
            unmatched.append(module)
            continue
        megatron_module = _WEIGHT_SUFFIX_RE.sub("", mapping.megatron_param)
        if is_expert_linear(megatron_module):
            raise NotImplementedError(
                f"init_adapter_path: {module} targets MoE expert module {megatron_module}; importing "
                "expert LoRA adapters into Megatron is not supported"
            )
        groups.setdefault(megatron_module, []).append(piece)
        constituents.setdefault(megatron_module, _hf_param_names(mapping))
    if unmatched:
        raise ValueError(
            f"init_adapter_path: {len(unmatched)} adapted module(s) have no Megatron counterpart in this "
            f"model's bridge mappings: {unmatched[:8]}"
        )

    # Order each group by the bridge's constituent order so ranks stack deterministically.
    for megatron_module, pieces in groups.items():
        order = {
            canonical_module_name(_WEIGHT_SUFFIX_RE.sub("", n)): i for i, n in enumerate(constituents[megatron_module])
        }
        pieces.sort(key=lambda p: order.get(canonical_module_name(p.module), len(order)))

    dims_by_module = {m: fused_rank(pieces, tp_size) for m, pieces in groups.items()}
    dims_by_leaf: Dict[str, set] = {}
    for megatron_module, dim in dims_by_module.items():
        dims_by_leaf.setdefault(megatron_module.rsplit(".", 1)[-1], set()).add(dim)
    module_dims: Dict[str, int] = {}
    for leaf, dims in dims_by_leaf.items():
        if len(dims) == 1:
            module_dims[leaf] = next(iter(dims))
        else:
            if pp_size > 1:
                raise NotImplementedError(
                    f"init_adapter_path: {leaf} adapters differ in rank across layers ({sorted(dims)}); "
                    "per-layer ranks are only supported with pipeline_model_parallel_size=1"
                )
            for megatron_module, dim in dims_by_module.items():
                if megatron_module.rsplit(".", 1)[-1] == leaf:
                    module_dims[megatron_module] = dim

    adapted_leaves = sorted(dims_by_leaf)
    if target_modules is None:
        targets = adapted_leaves
    else:
        missing = [leaf for leaf in adapted_leaves if leaf not in target_modules]
        if missing:
            raise ValueError(
                f"init_adapter_path: the adapter trains {missing} but trainer.policy.model.lora.target_modules="
                f'{list(target_modules)} does not include them; add them or use target_modules="all-linear"'
            )
        targets = list(target_modules)

    return MegatronLoraImportPlan(
        target_modules=targets,
        module_dims=module_dims,
        groups=groups,
        constituents=constituents,
        max_rank=max(dims_by_module.values()),
    )


def _global_out_features(task, tp_size: int) -> int:
    local = int(task.param_weight.shape[0])
    return local * tp_size if isinstance(task.mapping, ColumnParallelMapping) else local


def _global_rank(task, tp_size: int) -> int:
    local = int(task.param_weight.shape[0])
    if isinstance(task.mapping, ColumnParallelMapping):
        return local * tp_size
    if isinstance(task.mapping, RowParallelMapping):
        return local
    return local


@torch.no_grad()
def load_peft_adapter_into_megatron(
    bridge,
    model_chunks: List[torch.nn.Module],
    plan: MegatronLoraImportPlan,
    lora_cls: LoRA,
) -> Dict[str, Any]:
    """Copy the planned adapter tensors into the model's LoRA parameters.

    Runs on every rank (the bridge mappings are TP collectives). Returns counts for
    logging: modules loaded, modules whose adapter was widened past ``lora_cls.dim``,
    and modules that stay at their fresh initialization.
    """
    model_bridge, registry = _mapping_registry(bridge)
    unwrapped = unwrap_model(model_chunks)
    tasks_by_base = model_bridge.build_adapter_conversion_tasks(unwrapped)
    tp_size = mpu.get_tensor_model_parallel_world_size()

    missing_groups = [name for name in plan.groups if name not in tasks_by_base]
    if missing_groups:
        raise ValueError(
            f"init_adapter_path: no LoRA adapter was built for {missing_groups[:8]}; the target_modules "
            "do not match the adapter"
        )

    loaded = widened = fresh = 0
    for base_prefix, tasks in tasks_by_base.items():
        pieces = plan.groups.get(base_prefix)
        if pieces is None:
            fresh += 1
            continue
        for task in tasks:
            if task.adapter_key is not None:
                raise NotImplementedError("init_adapter_path requires lora_type='lora' (canonical_lora unsupported)")
            if task.requires_expert_splits or is_expert_linear(base_prefix):
                raise NotImplementedError(f"init_adapter_path: expert adapter {base_prefix} is not supported")
            if task.linear_in_task.megatron_module is None:
                continue  # lives on another pipeline stage
            chunk = unwrapped[task.linear_in_task.vp_stage or 0]
            base_hf_names = model_bridge._get_base_hf_param_names_for_adapter(registry, base_prefix, None, ".weight")
            rank = _global_rank(task.linear_in_task, tp_size)
            target_scale = float(lora_cls.alpha) / rank
            stacked = stack_lora_pieces(pieces, target_scale=target_scale, pad_to_multiple=tp_size)
            if stacked.rank != rank:
                raise ValueError(
                    f"init_adapter_path: {base_prefix} was built at rank {rank} but the adapter needs "
                    f"{stacked.rank}; the LoRA plan and model disagree"
                )

            out_features = _global_out_features(task.linear_out_task, tp_size)
            lora_b_full = _place_lora_b(
                model_bridge, unwrapped, base_prefix, base_hf_names, pieces, stacked.lora_bs, out_features, rank
            )

            local_a = task.linear_in_task.mapping.hf_to_megatron(stacked.lora_a, chunk)
            task.linear_in_task.param_weight.data.copy_(local_a)
            local_b = task.linear_out_task.mapping.hf_to_megatron(lora_b_full, chunk)
            task.linear_out_task.param_weight.data.copy_(local_b)
            loaded += 1
            if rank != lora_cls.dim:
                widened += 1

    stats = {"loaded": loaded, "widened": widened, "fresh": fresh, "max_rank": plan.max_rank}
    if torch.distributed.get_rank() == 0:
        logger.info(
            "init_adapter_path: loaded {} Megatron LoRA modules ({} widened past rank {}, {} left at fresh init)",
            loaded,
            widened,
            lora_cls.dim,
            fresh,
        )
    return stats


def _place_lora_b(
    model_bridge,
    unwrapped,
    base_prefix: str,
    base_hf_names: List[str],
    pieces: Sequence[LoraPiece],
    lora_bs: Sequence[torch.Tensor],
    out_features: int,
    rank: int,
) -> torch.Tensor:
    """Assemble the fused ``lora_B`` (``out_features`` x ``rank``) in Megatron's row layout."""
    by_canonical: Dict[str, torch.Tensor] = {}
    for piece, lora_b in zip(pieces, lora_bs):
        by_canonical[canonical_module_name(piece.module)] = lora_b

    index = torch.arange(out_features, dtype=torch.int64).unsqueeze(1)
    slices = model_bridge._get_fused_adapter_linear_out_slices(unwrapped, base_hf_names, index, is_expert=False)
    lora_b_full = torch.zeros(out_features, rank, dtype=torch.float32)
    if slices is None:
        if len(base_hf_names) != 1 or len(lora_bs) != 1:
            raise ValueError(f"init_adapter_path: {base_prefix} fuses {base_hf_names} but the bridge cannot split it")
        lora_b = lora_bs[0]
        if lora_b.shape[0] > out_features:
            raise ValueError(f"init_adapter_path: {base_prefix} lora_B has {lora_b.shape[0]} rows > {out_features}")
        # Vocab-parallel output layers pad the vocabulary at the end; the HF adapter is unpadded.
        lora_b_full[: lora_b.shape[0]] = lora_b
        return lora_b_full

    placed = set()
    for hf_name, rows in slices.items():
        if not torch.is_tensor(rows):
            continue  # projection absent from the HF model (ABSENT_PROJECTION)
        canonical = canonical_module_name(_WEIGHT_SUFFIX_RE.sub("", hf_name))
        lora_b = by_canonical.get(canonical)
        if lora_b is None:
            continue  # constituent not adapted: its rows stay zero
        rows = rows.squeeze(1)
        if rows.numel() != lora_b.shape[0]:
            raise ValueError(
                f"init_adapter_path: {hf_name} spans {rows.numel()} rows of {base_prefix} but lora_B has "
                f"{lora_b.shape[0]} rows"
            )
        lora_b_full[rows] = lora_b
        placed.add(canonical)
    unplaced = sorted(set(by_canonical) - placed)
    if unplaced:
        raise ValueError(f"init_adapter_path: {unplaced} are not constituents of {base_prefix} ({base_hf_names})")
    return lora_b_full

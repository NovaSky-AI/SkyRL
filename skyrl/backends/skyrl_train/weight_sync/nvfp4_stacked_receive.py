"""vLLM-side loader for the stacked NVFP4 expert wire format (``nvfp4_stacked.py``).

Runs inside the vLLM worker, under the receive engine's layerwise reload (``start_weight_update`` called
``initialize_layerwise_reload``). For each MoE layer it collects the 8 stacked tensors and then does what vLLM's
``layerwise._layerwise_process`` does for that layer -- materialize, load, ``process_weights_after_loading``,
copy into the kernel tensors -- except that the "load" is one ``copy_`` per parameter (the local experts' rows of
the stack) instead of buffering ~1.5k per-expert ``FusedMoE.weight_loader`` calls through
``online_process_loader`` and replaying them. The layer's processing (the TRT-LLM NVFP4 re-layout) is vLLM's own
code, run once, so the final parameters equal the per-expert path's bytes as long as the copies write what the
per-expert loader writes. That holds for ModelOpt NVFP4 with MoE TP = 1 (expert parallel), no padded TP shards,
no redundant experts, and parameters whose shapes equal the stack's local slice -- every one of which is checked;
anything else falls back to :func:`unstack_to_per_expert` through ``model.load_weights``, i.e. exactly today's
path.

Pinned to vLLM's layerwise reload internals (``vllm/model_executor/model_loader/reload/``; written for 0.32,
re-checked unchanged in 0.33). If they cannot be imported, every layer takes the fallback.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from skyrl.backends.skyrl_train.weight_sync.nvfp4_stacked import (
    STACKED_LEAVES,
    local_to_global_expert_ids,
    parse_stacked_wire_name,
    unstack_to_per_expert,
)

logger = logging.getLogger(__name__)

# Logging-only phase timing on the receiver (SKYRL_NVFP4_SYNC_TIMING=1).
_TIMING = os.environ.get("SKYRL_NVFP4_SYNC_TIMING", "0") == "1"


def _map_hf_name(model: torch.nn.Module, name: str) -> str:
    mapper = getattr(model, "hf_to_vllm_mapper", None)
    if mapper is None:
        return name
    mapped = mapper.apply_list([name])
    if len(mapped) != 1:
        raise ValueError(f"Unable to map NVFP4 stacked layer name {name!r}")
    return mapped[0]


def find_expert_module(model: torch.nn.Module, layer_prefix: str) -> Tuple[torch.nn.Module, str]:
    """The module that owns ``w13_weight`` for an HF MoE layer prefix (``model.layers.<i>.mlp.experts``)."""
    mapped = _map_hf_name(model, f"{layer_prefix}.w13_weight").rpartition(".")[0]
    for candidate in (mapped, f"{mapped}.routed_experts"):
        try:
            module = model.get_submodule(candidate)
        except AttributeError:
            continue
        if "w13_weight" in module._parameters:
            return module, candidate
    raise ValueError(f"No FusedMoE expert module with w13_weight for NVFP4 stacked layer {layer_prefix!r}")


def _expected_local_shapes(stacks: Dict[str, torch.Tensor], num_local: int, num_global: int, use_global_sf: bool):
    def local(t: torch.Tensor) -> Tuple[int, ...]:
        return (num_local, *t.shape[1:])

    shapes = {leaf: local(stacks[leaf]) for leaf in STACKED_LEAVES}
    if use_global_sf:
        shapes["w13_input_scale"] = (num_global, *stacks["w13_input_scale"].shape[1:])
        shapes["w2_input_scale"] = (num_global, *stacks["w2_input_scale"].shape[1:])
    return shapes


def fast_path_plan(module: torch.nn.Module, stacks: Dict[str, torch.Tensor]) -> Tuple[Optional[Any], str]:
    """``(plan, "")`` when the direct-copy load is provably equivalent to the per-expert loader, else ``(None, why)``.

    The plan is ``(local_to_global index tensor or None, {leaf: "local" | "global"})``.
    """
    quant_method = getattr(module, "quant_method", None)
    if quant_method is None or "ModelOptNvFp4" not in type(quant_method).__name__:
        return None, f"quant method {type(quant_method).__name__} is not ModelOpt NVFP4"
    moe_config = getattr(module, "moe_config", None)
    if moe_config is None:
        return None, "no moe_config"
    if getattr(moe_config.moe_parallel_config, "tp_size", None) != 1:
        return None, "MoE tensor parallel (tp_size != 1)"
    if getattr(moe_config, "tp_shard_with_padding", False):
        return None, "padded TP shards"
    if not getattr(moe_config, "is_act_and_mul", True):
        return None, "not act-and-mul"
    num_global = int(getattr(module, "global_num_experts", -1))
    num_local = int(getattr(module, "local_num_experts", -1))
    stacked_experts = stacks["w13_weight"].shape[0]
    if num_global != stacked_experts:
        return None, f"global_num_experts {num_global} != stacked experts {stacked_experts} (redundant experts?)"
    manager = getattr(module, "expert_map_manager", None)
    if getattr(manager, "num_fused_shared_experts", 0):
        return None, "fused shared experts"
    try:
        g2l = [int(module._map_global_expert_id_to_local_expert_id(g)) for g in range(num_global)]
        l2g = local_to_global_expert_ids(g2l, num_local)
    except Exception as exc:  # noqa: BLE001 - any irregular map takes the fallback
        return None, f"expert map: {exc}"
    use_global_sf = bool(getattr(quant_method, "use_global_sf", False))
    expected = _expected_local_shapes(stacks, num_local, num_global, use_global_sf)
    for leaf in STACKED_LEAVES:
        param = module._parameters.get(leaf)
        if param is None:
            return None, f"missing parameter {leaf}"
        if tuple(param.shape) != expected[leaf] or param.dtype != stacks[leaf].dtype:
            return (
                None,
                f"{leaf}: param {tuple(param.shape)} {param.dtype} vs expected {expected[leaf]} {stacks[leaf].dtype}",
            )
    from vllm.model_executor.model_loader.reload.meta import SKIP_LOAD_TENSORS
    from vllm.model_executor.model_loader.reload.utils import get_layer_tensors

    loadable = {n for n in get_layer_tensors(module) if n not in SKIP_LOAD_TENSORS}
    if loadable != set(STACKED_LEAVES):
        return None, f"layer has other loadable tensors: {sorted(loadable - set(STACKED_LEAVES))}"
    identity = l2g == list(range(num_local)) and num_local == num_global
    index = None if identity else torch.tensor(l2g, dtype=torch.long)
    sources = {
        leaf: ("global" if use_global_sf and leaf.endswith("input_scale") else "local") for leaf in STACKED_LEAVES
    }
    return (index, sources), ""


def copy_stacks_into(module: torch.nn.Module, stacks: Dict[str, torch.Tensor], plan: Any) -> None:
    """Write each stacked tensor's local rows into the (materialized) parameter, as the per-expert loader would."""
    index, sources = plan
    for leaf in STACKED_LEAVES:
        dst = getattr(module, leaf).data
        src = stacks[leaf]
        if sources[leaf] == "local" and index is not None:
            src = src.index_select(0, index.to(src.device))
        dst.copy_(src)


class Nvfp4StackedReceiver:
    """Collects one receive round's stacked tensors per layer and loads each layer once it is complete."""

    def __init__(self, model: torch.nn.Module, fallback_load: Callable[[List[Tuple[str, torch.Tensor]]], Any]):
        self.model = model
        self.fallback_load = fallback_load
        self.pending: Dict[str, Dict[str, torch.Tensor]] = {}
        self.layers_fast = 0
        self.layers_fallback = 0
        self.seconds = 0.0
        self._warned = False

    def accept(self, wire_name: str, tensor: torch.Tensor) -> None:
        parsed = parse_stacked_wire_name(wire_name)
        assert parsed is not None
        layer_prefix, leaf = parsed
        layer = self.pending.setdefault(layer_prefix, {})
        if leaf in layer:
            raise ValueError(f"NVFP4 stacked tensor {wire_name!r} arrived twice in one round")
        layer[leaf] = tensor
        if len(layer) == len(STACKED_LEAVES):
            del self.pending[layer_prefix]
            self._load_layer(layer_prefix, layer)

    def end_of_batch(self) -> None:
        """Own any partial layer's tensors: the transport may reuse its receive buffer for the next batch."""
        for layer in self.pending.values():
            for leaf, tensor in layer.items():
                layer[leaf] = tensor.clone()

    def finish(self) -> None:
        if self.pending:
            raise RuntimeError(
                "NVFP4 stacked sync: incomplete layers at the end of the round: "
                + str({k: sorted(v) for k, v in list(self.pending.items())[:4]})
            )
        if _TIMING or self.layers_fallback:
            logger.warning(
                "NVFP4 stacked receive: %d layers direct, %d layers per-expert fallback, %.3fs in layer loads",
                self.layers_fast,
                self.layers_fallback,
                self.seconds,
            )

    def _fallback(self, layer_prefix: str, stacks: Dict[str, torch.Tensor], why: str) -> None:
        if not self._warned:
            logger.warning("NVFP4 stacked receive: per-expert fallback for %s (%s)", layer_prefix, why)
            self._warned = True
        self.layers_fallback += 1
        self.fallback_load(list(unstack_to_per_expert(layer_prefix, stacks)))

    def _load_layer(self, layer_prefix: str, stacks: Dict[str, torch.Tensor]) -> None:
        t0 = time.perf_counter()
        try:
            try:
                from vllm.model_executor.model_loader.reload import layerwise
            except ImportError as exc:
                self._fallback(layer_prefix, stacks, f"vLLM layerwise reload unavailable: {exc}")
                return
            module, _ = find_expert_module(self.model, layer_prefix)
            info = layerwise.get_layerwise_info(module)
            if not info.can_load():
                self._fallback(layer_prefix, stacks, "layer is not under layerwise reload")
                return
            if info.loaded_weights or info.load_numel:
                self._fallback(layer_prefix, stacks, "layer already has buffered per-tensor loads")
                return
            plan, why = fast_path_plan(module, stacks)
            if plan is None:
                self._fallback(layer_prefix, stacks, why)
                return
            process_layer_with_direct_copy(module, info, lambda: copy_stacks_into(module, stacks, plan))
            self.layers_fast += 1
        finally:
            self.seconds += time.perf_counter() - t0


def process_layer_with_direct_copy(module: torch.nn.Module, info: Any, load: Callable[[], None]) -> None:
    """vLLM's (0.32/0.33) ``layerwise._layerwise_process`` with its buffered-loader replay replaced by ``load()``.

    Kept line for line with upstream (materialize; reset the online-quant flag; unwrap the layerwise loaders;
    *load*; ``process_weights_after_loading`` under the same timing context; ``update_param_tp_status``; copy into
    the kernel tensors; reset) so the layer ends in exactly the state the per-expert path leaves it in.
    """
    from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
    from vllm.model_executor.model_loader.reload import layerwise
    from vllm.model_executor.model_loader.reload.meta import materialize_layer
    from vllm.model_executor.model_loader.reload.utils import get_layer_tensors

    materialize_layer(module, info)
    if hasattr(module, "_already_called_process_weights_after_loading"):
        delattr(module, "_already_called_process_weights_after_loading")
    for param in get_layer_tensors(module).values():
        param.weight_loader = layerwise._get_original_loader(param)

    load()

    quant_method = getattr(module, "quant_method", None)
    if isinstance(quant_method, QuantizeMethodBase):
        from vllm.model_executor.layers.quantization.online.fp8 import OnlineLinearBase
        from vllm.model_executor.layers.quantization.online.moe_base import (
            OnlineMoEMethodBase,
        )
        from vllm.model_executor.model_loader.utils import (
            online_quantization_timing_context,
        )

        timing_context = (
            online_quantization_timing_context(module, quant_method)
            if isinstance(quant_method, (OnlineLinearBase, OnlineMoEMethodBase))
            else nullcontext()
        )
        with timing_context:
            quant_method.process_weights_after_loading(module)
        if hasattr(module, "update_param_tp_status"):
            module.update_param_tp_status()

    if info.kernel_tensors is not None:
        layerwise._copy_and_restore_kernel_tensors(module, info)
    info.reset()
    layerwise.LOADING_LAYERS.discard(module)

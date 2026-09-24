"""``WeightSource`` implementations over SkyRL's training backends.

A ``WeightSource`` is vLLM's trainer-side weight contract
(``vllm.distributed.weight_transfer.base``): ``metadata()`` and ``__iter__``,
which must agree element for element. Chunking is vLLM's: the packed producers
cut bounded-memory chunks out of a fixed buffer and consume the source lazily,
so a source's only obligation is to be a lazy generator that does not retain.

Sharded RDT needs two more channels (per-rank ownership under PP/EP, and a group
index) that vLLM has no concept of; they live in
``sharded_rdt/sharded_rdt_base.GroupedWeightSource``.

``LoraAdapterWeightSource`` streams a PEFT adapter instead of the model, for
adapter-only sync (Megatron ``merge_lora=false``) with
``lora.sync_mode=memory``. It is the one source that must be prepared before the
send: see its docstring.

Imports vLLM at module scope, so import this lazily from anything that must work
without the wheel.
"""

from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import torch
from loguru import logger
from vllm.distributed.weight_transfer.base import (
    ParamMeta,
    WeightSource,
    materialize_full_tensor,
)

__all__ = [
    "FsdpWeightSource",
    "LoraAdapterWeightSource",
    "MegatronLoraAdapterSource",
    "MegatronWeightSource",
    "ParamMeta",
    "SerializedFp8WeightSource",
    "WeightSource",
    "materialize_full_tensor",
]


class FsdpWeightSource(WeightSource):
    """``WeightSource`` over an FSDP2-sharded HF model.

    ``metadata()`` reads ``state_dict()`` shapes only: an FSDP2 ``DTensor``'s
    ``.shape`` is already the global shape, so declaring the stream costs no
    collective. Iteration all-gathers each parameter (``full_tensor()``), which
    IS a collective, so every trainer rank must iterate the same source in the
    same order.

    Args:
        model: the inner HF module (``self.model.model`` on the worker), whose
            ``state_dict()`` keys are the names vLLM expects.
        dtype: inference dtype. Both channels use it.
        weight_prefix: prepended to every name (``"language_model."`` when
            syncing a CausalLM backbone into a vLLM multimodal namespace).
    """

    def __init__(self, model: torch.nn.Module, dtype: torch.dtype, weight_prefix: str = "") -> None:
        self._model = model
        self._dtype = dtype
        self._prefix = weight_prefix or ""

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @property
    def weight_prefix(self) -> str:
        return self._prefix

    def metadata(self) -> List[ParamMeta]:
        sd = self._model.state_dict()
        return [ParamMeta(f"{self._prefix}{key}", self._dtype, tuple(param.shape)) for key, param in sd.items()]

    def __iter__(self) -> Iterator[Tuple[str, torch.Tensor]]:
        # The caller selects this rank's CUDA device before iterating; a worker
        # thread does not inherit it (see Worker._weight_sync_thread).
        device = torch.cuda.current_device() if torch.cuda.is_available() else None
        for key, param in self._model.state_dict().items():
            if device is not None:
                param = param.to(device, non_blocking=True)
            full = materialize_full_tensor(param).to(self._dtype).detach().contiguous()
            yield f"{self._prefix}{key}", full


class MegatronWeightSource(WeightSource):
    """``WeightSource`` over a Megatron model, via Megatron-Bridge.

    ``export_hf_weights(conversion_tasks=None)`` streams the whole model in
    HF-canonical order, gathering each parameter across TP / PP / EP internally
    and giving all ranks full tensors. It is a generator, so its collectives run
    as the consumer pulls and a lazy consumer never holds more than the tensors
    it is working on.

    Exporting in ONE call is required: the bridge's ``_accumulate_grouped_export``
    needs every task sharing a ``group_key`` present in the same call, or the
    expert weights are silently never yielded.

    There is no shape-only export, so ``metadata()`` runs a dry export and
    caches. Engines call it every round; the cost is one-time.
    """

    def __init__(self, bridge: Any, module: Any, dtype: torch.dtype) -> None:
        self._bridge = bridge
        self._module = module
        self._dtype = dtype
        self._meta: Optional[List[ParamMeta]] = None

    def _export(self) -> Iterator[Tuple[str, torch.Tensor]]:
        return self._bridge.export_hf_weights(self._module, show_progress=False, conversion_tasks=None)

    def metadata(self) -> List[ParamMeta]:
        if self._meta is None:
            meta: List[ParamMeta] = []
            for name, tensor in self._export():
                meta.append(ParamMeta(name, self._dtype, tuple(tensor.shape)))
                del tensor
            self._meta = meta
        return self._meta

    def __iter__(self) -> Iterator[Tuple[str, torch.Tensor]]:
        # See FsdpWeightSource.__iter__ on the device.
        device = torch.cuda.current_device() if torch.cuda.is_available() else None
        for name, tensor in self._export():
            full = tensor.to(device=device, dtype=self._dtype, non_blocking=True).detach().contiguous()
            yield name, full


class SerializedFp8WeightSource(WeightSource):
    """Serialize a dense source into blockwise-FP8 checkpoint tensors.

    The vLLM trainer engines require metadata and iteration to expose the same
    expanded stream. The first metadata call therefore runs one dry conversion
    and caches the result; subsequent weight syncs stream the wrapped source
    lazily through vLLM's packed buffer.
    """

    def __init__(self, source: WeightSource, config: Any) -> None:
        self._source = source
        self._config = config
        self._meta: Optional[List[ParamMeta]] = None

    def _serialized(self) -> Iterator[Tuple[str, torch.Tensor]]:
        from skyrl.backends.skyrl_train.weight_sync.fp8 import (
            iter_serialized_fp8_tensors,
        )

        for name, tensor in self._source:
            for serialized_name, serialized_tensor in iter_serialized_fp8_tensors(
                name,
                tensor,
                tensor.dtype,
                self._config,
            ):
                yield serialized_name, serialized_tensor.detach().contiguous()

    def metadata(self) -> List[ParamMeta]:
        if self._meta is None:
            self._meta = [ParamMeta(name, tensor.dtype, tuple(tensor.shape)) for name, tensor in self._serialized()]
        return self._meta

    def __iter__(self) -> Iterator[Tuple[str, torch.Tensor]]:
        yield from self._serialized()


class LoraAdapterWeightSource(WeightSource):
    """``WeightSource`` whose stream is a PEFT adapter rather than the model.

    Subclasses implement :meth:`export_adapter_stream` (a collective on the
    trainer) yielding the public adapter's ``(key, tensor)`` pairs on GPU, and
    :meth:`finalize_adapter`, which turns the retained unique tensors into the
    layout vLLM loads plus the ``adapter_config``. This class dedupes shared
    expert adapters as the stream arrives and publishes the ``receive_target``
    the worker extension forwards to the receivers.

    Dedupe happens on the stream, before finalize, because the exporter may
    materialize every public key: on GLM-5.3 that is 30.77 GB per rank against
    0.62 GB unique. ``finalize_adapter`` therefore only ever sees canonical
    keys. Alias groups are per (module, expert-group, lora_A|lora_B), so the
    ``lora_A`` sibling of a canonical ``lora_B`` is itself canonical and any
    per-key transform that pairs A with B still has both.

    Unlike the model sources, the export cannot be re-run per channel: the
    dedupe is what decides which keys exist, so ``metadata()`` and iteration
    must see one export. :meth:`prepare` runs it once, on every rank, and the
    caller must invoke it before ``send_weights()`` -- the ``receive_target``
    is only known afterwards and has to reach the workers first. Iteration
    drops the cache, so the next round re-exports.

    Retaining the unique adapter between the two channels is what the model
    sources avoid, and is affordable for exactly the same reason the dedupe
    exists: it is the deduplicated adapter, not the model.
    """

    def __init__(self, *, dtype: torch.dtype, experts_per_shared_adapter: int) -> None:
        self._dtype = dtype
        self._experts_per_shared_adapter = experts_per_shared_adapter
        self._lora_name: Optional[str] = None
        self._prepared: Optional[Tuple[Dict[str, torch.Tensor], Dict[str, str], Dict[str, Any]]] = None

    def set_lora_name(self, lora_name: str) -> None:
        """Name vLLM registers the adapter under; set per sync (multi-tenant)."""
        self._lora_name = lora_name

    def export_adapter_stream(self) -> Iterable[Tuple[str, torch.Tensor]]:
        """Yield the public adapter's ``(key, tensor)`` pairs. Collective."""
        raise NotImplementedError

    def finalize_adapter(
        self, adapter_state: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Map the unique tensors to what vLLM loads; return them with ``adapter_config``.

        Must keep every key it is given (values may change) so the alias map
        built on the stream stays valid; it may rename only keys that can never
        be aliased. The default is the identity with an empty config.
        """
        return adapter_state, {}

    def prepare(self) -> None:
        """Run the export, dedupe and finalize. Collective; call on every rank.

        Idempotent within a round. Iteration drops the result, so the next round
        needs its own call.
        """
        from skyrl.backends.skyrl_train.weight_sync.lora_target import (
            dedupe_shared_expert_adapters,
        )

        if self._prepared is not None:
            return
        to_send, aliases = dedupe_shared_expert_adapters(self.export_adapter_stream(), self._experts_per_shared_adapter)
        to_send, adapter_config = self.finalize_adapter(to_send)
        missing = set(aliases.values()) - set(to_send)
        if missing:
            raise RuntimeError(f"finalize_adapter dropped aliased keys: {sorted(missing)[:5]}")
        self._prepared = (to_send, aliases, adapter_config)

    def _prepared_state(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, str], Dict[str, Any]]:
        """The current round's export. Raises rather than exporting implicitly.

        An implicit export here would be a *collective* run by whichever rank
        happened to ask -- so a stray read after the stream has been consumed
        (the round is over and the cache is dropped) would hang the job in an
        EP all-gather that the other ranks never join, for the full NCCL
        watchdog timeout. Failing immediately, by name, is the whole point.
        """
        if self._prepared is None:
            raise RuntimeError(
                f"{type(self).__name__} has no prepared adapter: call prepare() on every rank "
                "before reading the source, and do not read it again after the stream has been "
                "consumed (each round re-exports)."
            )
        return self._prepared

    @property
    def receive_target(self) -> Dict[str, Any]:
        """What the receivers apply this round's stream to. Needs :meth:`prepare`."""
        from skyrl.backends.skyrl_train.weight_sync.lora_target import (
            build_lora_receive_target,
        )

        if self._lora_name is None:
            raise RuntimeError("set_lora_name must be called before publishing a LoRA adapter")
        _, aliases, adapter_config = self._prepared_state()
        return build_lora_receive_target(self._lora_name, adapter_config, aliases)

    def metadata(self) -> List[ParamMeta]:
        to_send, _, _ = self._prepared_state()
        return [ParamMeta(name, self._dtype, tuple(t.shape)) for name, t in to_send.items()]

    def __iter__(self) -> Iterator[Tuple[str, torch.Tensor]]:
        to_send, _, _ = self._prepared_state()
        # See FsdpWeightSource.__iter__ on the device.
        device = torch.cuda.current_device() if torch.cuda.is_available() else None
        try:
            for name, tensor in to_send.items():
                # Cast here, so what the receiver stages is already in the wire
                # dtype and vLLM's per-key `.to()` is a no-op that keeps expanded
                # aliases sharing one storage.
                yield name, tensor.to(device=device, dtype=self._dtype, non_blocking=True).detach().contiguous()
        finally:
            # The next sync re-exports; nothing from this one is worth keeping.
            self._prepared = None


class MegatronLoraAdapterSource(LoraAdapterWeightSource):
    """LoRA adapter source for the Megatron backend.

    The same pipeline ``_save_lora_adapters_and_sync`` runs before writing files
    (bridge adapter export, fused-expert 3D -> flat PEFT rewrite,
    ``adapter_config``), minus the file: tensors stay on GPU and go to the
    transport. The export is collective on every rank; the base class dedupes it
    as it streams.

    Unlike the disk path, the rewrite and the config are computed on every rank
    rather than only rank 0, because every rank iterates the source and the two
    channels must agree across ranks as well as with each other.

    Megatron imports are local to the methods: this module is imported on the
    FSDP path too.
    """

    def __init__(
        self,
        *,
        dtype: torch.dtype,
        experts_per_shared_adapter: int,
        bridge: Any,
        actor_module: Any,
        lora_cls: Any,
        base_model_name_or_path: str,
    ) -> None:
        super().__init__(dtype=dtype, experts_per_shared_adapter=experts_per_shared_adapter)
        self._bridge = bridge
        self._actor_module = actor_module
        self._lora_cls = lora_cls
        self._base_model_name_or_path = base_model_name_or_path

    def export_adapter_stream(self) -> Iterator[Tuple[str, torch.Tensor]]:
        for name, tensor in self._bridge.export_adapter_weights(self._actor_module, cpu=False, show_progress=False):
            yield f"base_model.model.{name}", tensor

    def finalize_adapter(
        self, adapter_state: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        from megatron.bridge.models.conversion.peft_bridge import (
            build_adapter_config_dict,
            infer_target_modules_from_adapter_weights,
        )

        from skyrl.backends.skyrl_train.distributed.megatron.lora_export import (
            fold_lora_alpha_for_vllm,
            fold_lora_rank_scale_for_vllm,
            mark_alpha_folded,
        )
        from skyrl.backends.skyrl_train.distributed.megatron.megatron_utils import (
            _convert_moe_experts_lora_to_vllm,
        )

        # The same two rewrites ``_save_lora_adapters_and_sync`` applies before
        # writing the files, in the same order. Both return new tensors: the
        # bridge's exports may alias live adapter parameters, which must not be
        # modified. The 3D -> flat rewrite renames only fused
        # ``experts.gate_up_proj`` / ``experts.down_proj`` keys, which carry no
        # expert index and so are never aliased.
        config_rank = self._lora_cls.dim
        adapter_state, rescaled = fold_lora_rank_scale_for_vllm(adapter_state, config_rank=config_rank)
        if rescaled and torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logger.info(
                "LoRA sync (memory): folded rank scale into lora_B for vLLM (config r={}): {}",
                config_rank,
                ", ".join(f"{n} tensors at rank {r} x{config_rank / r:g}" for r, n in sorted(rescaled.items())),
            )
        # vLLM multiplies lora_B by lora_alpha / r in place on every load, and the
        # staged tensors are handed to it by reference and reused for LRU reloads.
        # Fold that scale here and publish lora_alpha == r so vLLM's scale is 1.
        adapter_state = fold_lora_alpha_for_vllm(adapter_state, config_rank=config_rank, alpha=self._lora_cls.alpha)
        adapter_state = _convert_moe_experts_lora_to_vllm(adapter_state)
        target_modules = sorted(set(infer_target_modules_from_adapter_weights(adapter_state.keys())) - {"base_layer"})
        adapter_config = mark_alpha_folded(
            build_adapter_config_dict(
                self._lora_cls,
                target_modules=target_modules,
                base_model_name_or_path=self._base_model_name_or_path,
            )
        )
        return adapter_state, adapter_config

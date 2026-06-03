"""Per-worker store of LoRA adapter weights and optimizer state.

Holds one CPU-pinned snapshot per registered model_id plus a single pristine
slot used to seed newly-created adapters. At any moment exactly one adapter is
"live" in the worker's `actor_module` + `DistributedOptimizer`; swap_to() moves
LoRA bucket params and DistributedOptimizer fp32-main / Adam state between live
GPU storage and the per-adapter CPU slot via tensor.copy_().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.optimizer import ChainedOptimizer


def iter_opts(opt) -> List[Any]:
    """Yield underlying Megatron optimizers, unwrapping ChainedOptimizer."""
    if isinstance(opt, ChainedOptimizer):
        return list(opt.chained_optimizers)
    return [opt]


def _iter_buffers(model_chunks) -> Iterable[Tuple[int, int, Any]]:
    """Yield (mc_idx, buf_idx, buffer) for every LoRA-trainable DDP buffer."""
    for mc_idx, mc in enumerate(model_chunks):
        if not isinstance(mc, DDP):
            continue
        bufs = list(mc.buffers) + list(mc.expert_parallel_buffers)
        for buf_idx, buf in enumerate(bufs):
            yield mc_idx, buf_idx, buf


def _new_pinned_like(t: torch.Tensor) -> torch.Tensor:
    """Allocate a pinned-CPU tensor with the same shape/dtype as t."""
    return torch.empty_like(t, device="cpu").pin_memory()


def _expected_lora_param_check(model_chunks) -> None:
    """Sanity-check: every trainable param under DDP buffers is a LoRA adapter param.

    Megatron's DDP filters out requires_grad=False params before bucket
    construction. With the LoRA pre-wrap hook freezing base params, only
    LoRA A/B params should remain. If a future change breaks this invariant
    (e.g. an unfrozen bias or new trainable head), we want to fail loudly
    rather than silently swap the wrong tensors.
    """
    for mc_idx, _buf_idx, buf in _iter_buffers(model_chunks):
        for param in getattr(buf, "params", []):
            mc = model_chunks[mc_idx]
            name = next(
                (n for n, p in mc.named_parameters() if p is param),
                None,
            )
            if name is None:
                continue
            if "adapter" not in name:
                raise RuntimeError(
                    f"AdapterStore: trainable non-adapter param '{name}' found in "
                    f"DDP buffer {mc_idx}/{_buf_idx}; multi-LoRA swap would "
                    f"corrupt this param. Refusing to register."
                )


@dataclass(frozen=True)
class LoraSignature:
    """Immutable identity of a LoRA configuration. All registered adapters
    must share the same signature; otherwise tensor shapes won't match across
    swaps."""

    rank: int
    alpha: int
    target_modules: Tuple[str, ...]
    lora_type: str
    tp_size: int
    pp_size: int
    ep_size: int

    @classmethod
    def from_lora_config(cls, lora_config, lora_type: str = "lora") -> "LoraSignature":
        targets = lora_config.target_modules
        if isinstance(targets, str):
            targets_tuple = (targets,)
        else:
            targets_tuple = tuple(targets)
        return cls(
            rank=int(lora_config.rank),
            alpha=int(lora_config.alpha),
            target_modules=targets_tuple,
            lora_type=lora_type,
            tp_size=mpu.get_tensor_model_parallel_world_size(),
            pp_size=mpu.get_pipeline_model_parallel_world_size(),
            ep_size=(
                mpu.get_expert_model_parallel_world_size()
                if hasattr(mpu, "get_expert_model_parallel_world_size")
                else 1
            ),
        )


@dataclass
class AdapterSlot:
    """Per-adapter pinned-CPU storage mirroring the live GPU LoRA state.

    Layout:
      cpu_param_data[mc_idx] -> list[Tensor], one per buffer in
          (mc.buffers + mc.expert_parallel_buffers).
      cpu_grad_data[mc_idx]  -> same shape as cpu_param_data; mirrors
          buffer.grad_data so that grads accumulated by an interrupted
          forward_backward aren't lost when another tenant runs in the
          gap before this adapter's optim_step.
      cpu_main_param[opt_idx][g] -> list[Tensor], shapes matching
          opt.shard_fp32_from_float16_groups[g].
      cpu_opt_state[opt_idx][g][i] -> dict[str, Tensor], mirroring
          opt.optimizer.state[main_param] for every tensor-valued entry
          (exp_avg, exp_avg_sq, step, ...).
    """

    cpu_param_data: List[List[torch.Tensor]] = field(default_factory=list)
    cpu_grad_data: List[List[torch.Tensor]] = field(default_factory=list)
    cpu_main_param: List[List[List[torch.Tensor]]] = field(default_factory=list)
    cpu_opt_state: List[List[List[dict]]] = field(default_factory=list)
    # Per-param-group state from optimizer.param_groups[g]. TE FusedAdam (used
    # by Megatron's DistributedOptimizer) tracks `step` here at the group
    # level, not per-param. Without snapshotting this, the step counter
    # advances globally across adapters and breaks Adam bias correction.
    cpu_param_group_state: List[List[dict]] = field(default_factory=list)


class AdapterStore:
    """Per-worker registry of LoRA adapter slots.

    One AdapterStore lives on each Megatron PolicyWorker. It owns CPU storage
    for every registered adapter plus a pristine template; the live GPU model
    + optimizer always reflect the slot identified by `current_id`.

    Operations are local: snapshot/restore is a series of tensor.copy_()s that
    issue no collectives. Callers are responsible for the surrounding
    dist.barrier() (we recommend before and after the swap; see swap_to docs).
    """

    def __init__(self) -> None:
        self._slots: dict[str, AdapterSlot] = {}
        self._pristine: Optional[AdapterSlot] = None
        self._current_id: Optional[str] = None
        self._signature: Optional[LoraSignature] = None

    @property
    def current_id(self) -> Optional[str]:
        return self._current_id

    @property
    def signature(self) -> Optional[LoraSignature]:
        return self._signature

    def has(self, model_id: str) -> bool:
        return model_id in self._slots

    def num_adapters(self) -> int:
        return len(self._slots)

    def registered_ids(self) -> List[str]:
        """List the model_ids of every registered adapter (excluding pristine)."""
        return list(self._slots.keys())

    # ------------------------------------------------------------------
    # Slot allocation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_scalar_state(v: Any) -> bool:
        """True for scalar values we want to round-trip in optimizer state.
        Skips the 'params' list and any other non-scalar object."""
        return isinstance(v, (int, float, bool)) or (isinstance(v, torch.Tensor) and v.numel() <= 1)

    def _allocate_empty_slot(self, model_chunks, optimizer) -> AdapterSlot:
        slot = AdapterSlot()
        # Param data + grad data: one pinned bf16 tensor each per (mc, buffer).
        # Grads must travel with the slot — otherwise an interleaved tenant's
        # forward_backward will clobber unconsumed grads via zero_grad_buffer
        # at the top of forward_backward. See docs/.../multi_lora_design.mdx.
        for mc_idx, _buf_idx, buf in _iter_buffers(model_chunks):
            while len(slot.cpu_param_data) <= mc_idx:
                slot.cpu_param_data.append([])
                slot.cpu_grad_data.append([])
            slot.cpu_param_data[mc_idx].append(_new_pinned_like(buf.param_data))
            slot.cpu_grad_data[mc_idx].append(_new_pinned_like(buf.grad_data))
        # Main params + optimizer state: per (opt_idx, group, param_idx).
        for _opt in iter_opts(optimizer):
            opt_main: List[List[torch.Tensor]] = []
            opt_state: List[List[dict]] = []
            groups = getattr(_opt, "shard_fp32_from_float16_groups", None) or []
            for g, group in enumerate(groups):
                main_g: List[torch.Tensor] = []
                state_g: List[dict] = []
                for main_param in group:
                    main_g.append(_new_pinned_like(main_param))
                    state = _opt.optimizer.state.get(main_param, {})
                    # Tensor entries get pinned-CPU mirrors; non-tensor scalar
                    # entries (e.g. PyTorch Adam's `state['step']` Python int)
                    # are stored by value and re-applied on restore. Without
                    # this, the global Adam step counter would leak across
                    # adapters and break bias correction.
                    state_g.append(
                        {k: _new_pinned_like(v) if isinstance(v, torch.Tensor) else v for k, v in state.items()}
                    )
                opt_main.append(main_g)
                opt_state.append(state_g)
            slot.cpu_main_param.append(opt_main)
            slot.cpu_opt_state.append(opt_state)
            # Per-param-group scalar state (notably TE FusedAdam's `step`).
            # We don't snapshot the `params` list or any non-scalar field;
            # static config (lr/betas/eps/weight_decay) is left to whatever
            # the live optimizer carries. Only dynamic counters round-trip.
            group_state: List[dict] = []
            for pg in _opt.optimizer.param_groups:
                group_state.append({k: v for k, v in pg.items() if self._is_scalar_state(v)})
            slot.cpu_param_group_state.append(group_state)
        return slot

    @torch.no_grad()
    def _snapshot(self, slot: AdapterSlot, model_chunks, optimizer) -> None:
        """Copy live GPU state into `slot` (CPU)."""
        for mc_idx, buf_idx, buf in _iter_buffers(model_chunks):
            slot.cpu_param_data[mc_idx][buf_idx].copy_(buf.param_data, non_blocking=True)
            slot.cpu_grad_data[mc_idx][buf_idx].copy_(buf.grad_data, non_blocking=True)
        for opt_idx, _opt in enumerate(iter_opts(optimizer)):
            groups = getattr(_opt, "shard_fp32_from_float16_groups", None) or []
            for g, group in enumerate(groups):
                for i, main_param in enumerate(group):
                    slot.cpu_main_param[opt_idx][g][i].copy_(main_param, non_blocking=True)
                    state = _opt.optimizer.state.get(main_param, {})
                    cpu_state = slot.cpu_opt_state[opt_idx][g][i]
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            if k in cpu_state and isinstance(cpu_state[k], torch.Tensor):
                                cpu_state[k].copy_(v, non_blocking=True)
                        else:
                            # Python scalar (e.g. Adam's int `step`); copy by value.
                            cpu_state[k] = v
            # Per-param-group scalar state (TE FusedAdam tracks `step` here).
            for pg_idx, pg in enumerate(_opt.optimizer.param_groups):
                slot_pg = slot.cpu_param_group_state[opt_idx][pg_idx]
                for k, v in pg.items():
                    if self._is_scalar_state(v):
                        slot_pg[k] = v

    @torch.no_grad()
    def _restore(self, slot: AdapterSlot, model_chunks, optimizer) -> None:
        """Copy `slot` (CPU) into live GPU state."""
        for mc_idx, buf_idx, buf in _iter_buffers(model_chunks):
            buf.param_data.copy_(slot.cpu_param_data[mc_idx][buf_idx], non_blocking=True)
            buf.grad_data.copy_(slot.cpu_grad_data[mc_idx][buf_idx], non_blocking=True)
        for opt_idx, _opt in enumerate(iter_opts(optimizer)):
            groups = getattr(_opt, "shard_fp32_from_float16_groups", None) or []
            for g, group in enumerate(groups):
                for i, main_param in enumerate(group):
                    main_param.copy_(slot.cpu_main_param[opt_idx][g][i], non_blocking=True)
                    state = _opt.optimizer.state.get(main_param, {})
                    cpu_state = slot.cpu_opt_state[opt_idx][g][i]
                    # Restore both tensor and non-tensor entries: tensors get
                    # copy_() into existing GPU storage; Python scalars (e.g.
                    # Adam's int `step`) get assigned back into the live
                    # optimizer.state dict so the next .step() bias-corrects
                    # using this adapter's own step count, not a global one.
                    for k, slot_v in cpu_state.items():
                        if isinstance(slot_v, torch.Tensor):
                            live_v = state.get(k)
                            if isinstance(live_v, torch.Tensor):
                                live_v.copy_(slot_v, non_blocking=True)
                        else:
                            state[k] = slot_v
            # Restore per-param-group scalars (TE FusedAdam's `step`, etc.).
            for pg_idx, pg in enumerate(_opt.optimizer.param_groups):
                slot_pg = slot.cpu_param_group_state[opt_idx][pg_idx]
                for k, slot_v in slot_pg.items():
                    if isinstance(slot_v, torch.Tensor):
                        live_v = pg.get(k)
                        if isinstance(live_v, torch.Tensor):
                            live_v.copy_(slot_v, non_blocking=True)
                        else:
                            pg[k] = slot_v.clone()
                    else:
                        pg[k] = slot_v

    # ------------------------------------------------------------------
    # Public API used by the worker
    # ------------------------------------------------------------------

    def register_pristine(self, model_chunks, optimizer, signature: LoraSignature) -> None:
        """Capture the freshly-initialised LoRA state as the pristine template.

        Must be called once per worker, after the optimizer state has been
        materialised (e.g. via DistributedOptimizer._init_optimizer_states_with_dummy_values).
        Subsequent registrations will copy this slot to seed new adapters.
        """
        if self._pristine is not None:
            raise RuntimeError("AdapterStore.register_pristine called twice")
        _expected_lora_param_check(model_chunks)
        self._signature = signature
        self._pristine = self._allocate_empty_slot(model_chunks, optimizer)
        self._snapshot(self._pristine, model_chunks, optimizer)

    @torch.no_grad()
    def create(self, model_id: str, model_chunks, optimizer, signature: LoraSignature) -> None:
        """Register a new adapter slot.

        - First registration: this is also the live adapter; allocate a slot
          but skip the pristine→slot copy because the live state already
          equals pristine. `current_id` becomes `model_id`.
        - Subsequent registrations: allocate slot and copy pristine → slot.
          Live state is unchanged (no swap). The new adapter only becomes
          live when the next `swap_to(model_id)` is issued.
        """
        if self._signature is None:
            raise RuntimeError("AdapterStore.create called before register_pristine")
        if signature != self._signature:
            raise ValueError(
                f"AdapterStore: lora signature mismatch for '{model_id}'. "
                f"Pristine={self._signature}, requested={signature}. "
                f"Multi-LoRA requires identical (rank, alpha, target_modules, "
                f"lora_type, tp/pp/ep sizes) across all adapters."
            )
        if model_id in self._slots:
            raise ValueError(f"AdapterStore: adapter '{model_id}' already registered")

        slot = self._allocate_empty_slot(model_chunks, optimizer)
        if self._current_id is None:
            # First adapter: live state IS pristine; slot will be filled on
            # the next snapshot (i.e. swap-away). Treat live as authoritative.
            self._current_id = model_id
        else:
            # Seed the new slot from pristine.
            self._copy_slot(self._pristine, slot)
        self._slots[model_id] = slot

    @torch.no_grad()
    def _copy_slot(self, src: AdapterSlot, dst: AdapterSlot) -> None:
        """CPU→CPU copy used to seed a new slot from the pristine template."""
        for mc_idx, mc_buffers in enumerate(src.cpu_param_data):
            for buf_idx, t in enumerate(mc_buffers):
                dst.cpu_param_data[mc_idx][buf_idx].copy_(t)
        for mc_idx, mc_grads in enumerate(src.cpu_grad_data):
            for buf_idx, t in enumerate(mc_grads):
                dst.cpu_grad_data[mc_idx][buf_idx].copy_(t)
        for opt_idx, opt_groups in enumerate(src.cpu_main_param):
            for g, group in enumerate(opt_groups):
                for i, t in enumerate(group):
                    dst.cpu_main_param[opt_idx][g][i].copy_(t)
        for opt_idx, opt_groups in enumerate(src.cpu_opt_state):
            for g, group in enumerate(opt_groups):
                for i, state in enumerate(group):
                    dst_state = dst.cpu_opt_state[opt_idx][g][i]
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            if k in dst_state and isinstance(dst_state[k], torch.Tensor):
                                dst_state[k].copy_(v)
                        else:
                            dst_state[k] = v
        # Per-param-group scalar state (TE FusedAdam's step lives here).
        for opt_idx, src_groups in enumerate(src.cpu_param_group_state):
            for pg_idx, src_pg in enumerate(src_groups):
                dst_pg = dst.cpu_param_group_state[opt_idx][pg_idx]
                for k, v in src_pg.items():
                    if isinstance(v, torch.Tensor):
                        dst_pg[k] = v.clone()
                    else:
                        dst_pg[k] = v

    @torch.no_grad()
    def delete(self, model_id: str) -> None:
        """Drop the slot for `model_id`.

        If `model_id` was the current adapter, `current_id` is cleared. The
        live GPU state is left untouched (it now mirrors a deleted adapter);
        the next `swap_to` will overwrite it.
        """
        if model_id not in self._slots:
            raise KeyError(f"AdapterStore: unknown adapter '{model_id}'")
        del self._slots[model_id]
        if self._current_id == model_id:
            self._current_id = None

    @torch.no_grad()
    def swap_to(self, model_id: str, model_chunks, optimizer) -> None:
        """Make `model_id` the live adapter on this worker.

        Algorithm (all under torch.no_grad):
            1. dist.barrier(dp_group)
            2. snapshot live → current's slot (skipped if current_id is None)
            3. cuda stream sync (D2H done)
            4. restore target's slot → live
            5. cuda stream sync (H2D done)
            6. dist.barrier(dp_group)

        Caller responsibility: the trailing barrier guarantees all DP ranks
        agree on the live adapter before the next collective. TP/PP/EP groups
        do not need barriers because the swap is identical-shape on all
        ranks within those groups (LoRA signature is fixed).
        """
        if model_id not in self._slots:
            raise KeyError(f"AdapterStore: unknown adapter '{model_id}'")
        if self._current_id == model_id:
            return  # no-op fast path

        dp_group = mpu.get_data_parallel_group()
        if dist.is_available() and dist.is_initialized():
            dist.barrier(group=dp_group)

        if self._current_id is not None:
            current_slot = self._slots[self._current_id]
            self._snapshot(current_slot, model_chunks, optimizer)
            torch.cuda.current_stream().synchronize()

        target_slot = self._slots[model_id]
        self._restore(target_slot, model_chunks, optimizer)
        torch.cuda.current_stream().synchronize()

        self._current_id = model_id

        if dist.is_available() and dist.is_initialized():
            dist.barrier(group=dp_group)

"""Isolate the MTP / draft head into its OWN grad buffer + ``DistributedOptimizer``.

Even though the draft loss is autograd-clean (it only touches ``.mtp.*`` params), letting the head
share the policy's single Megatron DDP grad buffer changes that buffer's layout and perturbs the
floating-point result of the policy gradient's distributed reduction.
"""

from __future__ import annotations

import contextlib
from typing import List, Optional, Union

import torch
from megatron.core import parallel_state as mpu
from megatron.core.distributed import DistributedDataParallel
from megatron.core.distributed import (
    finalize_model_grads as _mcore_finalize_model_grads,
)
from megatron.core.distributed.distributed_data_parallel_config import (
    DistributedDataParallelConfig,
)
from megatron.core.utils import get_model_config

from skyrl.train.config.config import get_config_as_dict


def is_mtp_param_name(name: str) -> bool:
    """True for parameter names that belong to the MTP / draft head."""
    return ".mtp." in name or name.startswith("mtp.")


def _resolve_mtp_module(policy_module):
    """Return the ``host.mtp`` submodule (or None) for a (possibly DDP/Float16-wrapped) policy chunk."""
    from skyrl.backends.skyrl_train.mtp.hidden_capture import (
        _resolve_mtp_host,
        _unwrap_model,
    )

    host = _resolve_mtp_host(_unwrap_model(policy_module))
    return getattr(host, "mtp", None)


def freeze_mtp_params_pre_wrap(model_or_models: Union[torch.nn.Module, List[torch.nn.Module]]):
    """Provider pre-wrap hook: set the MTP/draft-head params ``requires_grad=False`` so Megatron's DDP
    excludes them from the POLICY grad buffer (the only Megatron lever for buffer exclusion is
    ``requires_grad`` at construction time). They are re-enabled and given their own buffer + optimizer
    by :class:`MTPOptimizer` after the policy is wrapped. Result: the policy grad buffer and its
    distributed reduction are byte-identical to a model built with no MTP head.

    Modifies in place AND returns the model unchanged. (The bridge's ``pre_wrap_hook`` property
    composes hooks via ``model = hook(model)`` with NO None-guard, so a hook that returns None would
    break the chain for any subsequent hook — we must return the model.)
    """
    models = model_or_models if isinstance(model_or_models, list) else [model_or_models]
    num_frozen = 0
    for model in models:
        for name, param in model.named_parameters():
            if is_mtp_param_name(name):
                param.requires_grad = False
                num_frozen += 1
    # Only registered when C-full is on, so zero matches means the `.mtp.` naming drifted -- the head
    # would silently train in the policy grad buffer and defeat the isolation. Fail loud instead.
    if num_frozen == 0:
        raise RuntimeError(
            "freeze_mtp_params_pre_wrap matched no parameters via is_mtp_param_name (expected '.mtp.' "
            "or 'mtp.' in the name). The MTP submodule naming likely changed -- C-full grad isolation "
            "would silently break. Update is_mtp_param_name in mtp/mtp_optim.py."
        )
    return model_or_models


def make_policy_finalize_excluding_mtp(mtp_params: List[torch.nn.Parameter]):
    """Wrap Megatron's ``finalize_model_grads`` so the MTP head is hidden during the POLICY finalize.

    The policy finalize iterates every ``requires_grad`` param of the GPTModel (the head lives inside it)
    and coalesces their sequence-parallel/layernorm grads into ONE tensor-parallel all-reduce. If the
    head's grads were included, that coalesced reduction would differ from the no-MTP build -> the exact
    perturbation we are eliminating. Backward is already complete when finalize runs, so transiently
    clearing the head's ``requires_grad`` is a pure read-time filter (no effect on the head's grads,
    which already live in the separate MTP buffer and are reduced by :meth:`MTPOptimizer.step`).
    """

    def finalize(model, *args, **kwargs):
        saved = [(p, p.requires_grad) for p in mtp_params]
        for p in mtp_params:
            p.requires_grad = False
        try:
            return _mcore_finalize_model_grads(model, *args, **kwargs)
        finally:
            for p, rg in saved:
                p.requires_grad = rg

    return finalize


def _build_mtp_ddp_config(ddp_config) -> DistributedDataParallelConfig:
    """Build the head's ``DistributedDataParallelConfig`` from the policy's, but with overlap disabled.

    ``overlap_grad_reduce=False`` so the head's grads simply accumulate into its buffer across
    micro-batches and are reduced once in :meth:`MTPOptimizer.finalize_grads` — there is no
    per-microbatch sync to coordinate with the policy's pipeline schedule (the head is not in the chunk
    list passed to ``forward_backward_func``). ``overlap_param_gather=False`` for the same reason.
    """
    cfg = DistributedDataParallelConfig()
    cfg.use_distributed_optimizer = True
    if ddp_config is not None:
        for k, v in get_config_as_dict(ddp_config).items():
            setattr(cfg, k, v)
    cfg.overlap_grad_reduce = False
    cfg.overlap_param_gather = False
    return cfg


class MTPOptimizer:
    """Owns the MTP/draft head's isolated grad buffer + ``DistributedOptimizer`` (C-full).

    The head stays inside the policy ``GPTModel`` (so capture / replay / weight-export are unchanged) but
    its params are excluded from the policy DDP buffer (via :func:`freeze_mtp_params_pre_wrap`) and wrapped
    here in a SEPARATE Megatron DDP, so the policy's distributed grad reduction is byte-identical to a
    no-MTP model. The head is co-trained at full strength under this optimizer.
    """

    def __init__(self, policy_module, ddp_config, optim_config, scheduler_config, num_training_steps: int):
        from skyrl.backends.skyrl_train.distributed.megatron.optimizer import (
            get_megatron_optimizer,
            get_megatron_optimizer_param_scheduler,
        )

        self.mtp_module = _resolve_mtp_module(policy_module)
        if self.mtp_module is None:
            raise RuntimeError(
                "MTPOptimizer: no `.mtp` head found on the policy model. Enable MTP "
                "(trainer.mtp.enabled / mtp_num_layers) or disable mtp_separate_optimizer."
            )

        # Re-enable grads on the head (frozen during the policy wrap by the pre-wrap hook) BEFORE
        # building its DDP + optimizer. Setting requires_grad here — before this fresh wrap and its
        # optimizer init — avoids the post-wrap requires_grad change that deadlocks dist-optimizer init.
        for p in self.mtp_module.parameters():
            p.requires_grad = True
        self.mtp_params: List[torch.nn.Parameter] = list(self.mtp_module.parameters())

        self._model_config = get_model_config(policy_module)
        self.mtp_ddp = DistributedDataParallel(
            config=self._model_config,
            ddp_config=_build_mtp_ddp_config(ddp_config),
            module=self.mtp_module,
        )
        self.optimizer = get_megatron_optimizer([self.mtp_ddp], optim_config)
        self.scheduler = get_megatron_optimizer_param_scheduler(
            optimizer=self.optimizer,
            config=scheduler_config,
            num_training_steps=num_training_steps,
        )

    def zero_grad_buffer(self) -> None:
        """Zero the head's grad buffer at the start of a forward-backward (mirrors the policy chunks)."""
        self.mtp_ddp.zero_grad_buffer()

    @contextlib.contextmanager
    def hidden(self):
        """Exclude the head from the POLICY optimizer's grad-norm + clip for the duration of the
        policy step, then restore so the separate MTP optimizer (stepped right after) reduces/steps
        the real grads.

        IMPORTANT: Megatron's ``get_main_grads_for_grad_norm()`` collects a param's grad whenever
        ``grad is not None`` — it does NOT check ``requires_grad``. So flipping ``requires_grad`` alone
        (the original implementation) does NOT remove the head from the policy grad-norm/clip: the head's
        grad is already populated by backward, so it still gets counted and inflates ``policy/grad_norm``,
        over-clipping the policy (a head→policy coupling through the shared clip). We therefore stash and
        clear the grad-bearing attributes Megatron may read (``grad`` / ``main_grad`` / ``decoupled_grad``)
        so the head is genuinely invisible to the policy norm, and restore them on exit (the underlying
        grad-buffer data is untouched — we only detach the attribute references). ``requires_grad`` is
        also cleared for any path that does honor it."""
        _GRAD_ATTRS = ("grad", "main_grad", "decoupled_grad")
        saved = []
        for p in self.mtp_params:
            grads = {a: getattr(p, a, None) for a in _GRAD_ATTRS}
            saved.append((p, p.requires_grad, grads))
            p.requires_grad = False
            for a in _GRAD_ATTRS:
                if getattr(p, a, None) is not None:
                    setattr(p, a, None)
        try:
            yield
        finally:
            for p, rg, grads in saved:
                p.requires_grad = rg
                for a, g in grads.items():
                    if g is not None:
                        setattr(p, a, g)

    def finalize_grads(self) -> None:
        """Reduce the head's accumulated grads: DP reduce-scatter (own buffer) + TP all-reduce of the
        head's sequence-parallel / layernorm grads. We call the targeted primitives rather than the full
        ``finalize_model_grads`` so we never touch the policy buffer and avoid the embedding/PP-cross-stage
        logic that assumes a complete GPTModel chunk (it would break for tied-embedding models)."""
        from megatron.core.distributed.finalize_model_grads import (
            _allreduce_non_tensor_model_parallel_grads,
        )

        self.mtp_ddp.finish_grad_sync()
        _allreduce_non_tensor_model_parallel_grads(
            [self.mtp_ddp], self._model_config, mpu.get_tensor_model_parallel_group()
        )

    def step(self) -> Optional[float]:
        """Finalize the head's grads, step its optimizer + scheduler, and zero its grads.

        Returns the head's grad norm (after the optimizer's own clip), or None if unavailable.
        """
        self.finalize_grads()
        _, grad_norm, _ = self.optimizer.step()
        self.scheduler.step(1)
        self.optimizer.zero_grad()
        if grad_norm is not None and hasattr(grad_norm, "item"):
            grad_norm = grad_norm.item()
        return grad_norm

    # -- checkpointing -------------------------------------------------------------------------------
    def state_dict(self):
        return {"optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict()}

    def load_state_dict(self, state_dict) -> None:
        if not state_dict:
            return
        if state_dict.get("optimizer") is not None:
            self.optimizer.load_state_dict(state_dict["optimizer"])
        if state_dict.get("scheduler") is not None:
            self.scheduler.load_state_dict(state_dict["scheduler"])

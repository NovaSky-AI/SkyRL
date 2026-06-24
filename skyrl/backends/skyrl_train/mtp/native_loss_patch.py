"""Disable Megatron's native in-forward MTP loss (``process_mtp_loss``).

SkyRL trains the MTP/draft head with its OWN decoupled soft-CE distillation loss (see
``mtp/soft_ce.py`` and ``MegatronModelWrapper``), NOT Megatron's pretraining-style MTP auxiliary
loss. But ``GPTModel.forward`` / ``HybridModel.forward`` call ``process_mtp_loss`` unconditionally
whenever the model is built with MTP heads (``mtp_num_layers`` set) and run in training/eval. That
native loss is a hard cross-entropy whose gradient — unless ``mtp_detach_heads`` is set — flows
straight into the shared trunk and output embedding, corrupting the RL policy (inflated grad-norm
and entropy collapse), with a magnitude set by Megatron's own scaling (so it is independent of
SkyRL's ``mtp_loss_weight``).

Megatron only short-circuits the native loss when BOTH ``labels`` and ``input_ids`` are None. A
megatron-core update added "derive labels from ``input_ids`` (e.g. RL training)", so passing no
labels no longer disables it. Rather than depend on that label / forward-gating behaviour (which is
exactly what silently broke), we replace ``process_mtp_loss`` at its call sites with a no-op that
returns the main-model hidden chunk and computes no loss. SkyRL's decoupled head training is
unaffected — the head still trains via SkyRL's explicit loss.

Robustness: this patch is loud, not silent. It raises if a target module no longer exposes
``process_mtp_loss`` (a Megatron rename). Its one blind spot is Megatron *inlining* the loss into
the forward instead of calling the function — which the grad-isolation acceptance test (the policy
gradient must match a no-MTP build, see ``tests/.../megatron/test_mtp_grad_coupling.py``) is there
to catch.
"""

from __future__ import annotations

import importlib

import torch

# Modules whose ``forward`` calls ``process_mtp_loss``. Each does ``from ...multi_token_prediction
# import process_mtp_loss``, binding the name in its OWN namespace, so we must patch each call
# site's module (patching the definition module would not rebind the already-imported names).
_PATCH_TARGETS = (
    "megatron.core.models.gpt.gpt_model",
    "megatron.core.models.hybrid.hybrid_model",
)
_ATTR = "process_mtp_loss"
_SENTINEL = "_skyrl_native_mtp_loss_disabled"


def _skyrl_skip_native_mtp_loss(hidden_states, *args, **kwargs):
    """No-op replacement for Megatron's ``process_mtp_loss``.

    Megatron splits ``hidden_states`` into ``1 + mtp_num_layers`` chunks along dim 0 and returns the
    first (the main-model hidden states) after applying the MTP loss via ``MTPLossAutoScaler``. We
    return that same first chunk WITHOUT computing or applying any loss, so no native-MTP gradient
    reaches the model. Both known call sites pass ``hidden_states`` and ``config`` as kwargs; the
    positional fallback is defensive only.
    """
    if hidden_states is None:
        hidden_states = kwargs.get("hidden_states")
    config = kwargs.get("config")
    if config is None:
        config = next((a for a in args if hasattr(a, "mtp_num_layers")), None)
    num_layers = getattr(config, "mtp_num_layers", None) if config is not None else None
    if not num_layers:
        return hidden_states
    return torch.chunk(hidden_states, 1 + num_layers, dim=0)[0]


def disable_native_mtp_loss() -> None:
    """Replace ``process_mtp_loss`` at every known call site with a no-op (idempotent).

    Raises ``RuntimeError`` if a target module exists but no longer exposes ``process_mtp_loss``
    (Megatron renamed/removed it) or if no target module can be imported at all — so the breakage
    is loud instead of silently re-enabling the native loss.
    """
    patched_any = False
    for mod_name in _PATCH_TARGETS:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue  # not present in this Megatron build (e.g. no hybrid models)
        if getattr(mod, _SENTINEL, False):
            patched_any = True
            continue
        if not hasattr(mod, _ATTR):
            raise RuntimeError(
                f"Cannot disable native MTP loss: '{mod_name}' no longer exposes '{_ATTR}'. "
                "Megatron's MTP-loss API changed — update mtp/native_loss_patch.py."
            )
        setattr(mod, _ATTR, _skyrl_skip_native_mtp_loss)
        setattr(mod, _SENTINEL, True)
        patched_any = True
    if not patched_any:
        raise RuntimeError(
            f"Cannot disable native MTP loss: none of {_PATCH_TARGETS} could be imported. "
            "Megatron's MTP module layout changed — update mtp/native_loss_patch.py."
        )

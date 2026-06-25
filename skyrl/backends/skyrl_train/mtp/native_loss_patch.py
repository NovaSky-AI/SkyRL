"""Disable Megatron's native in-forward MTP loss (``process_mtp_loss``).

SkyRL trains the MTP head with its own decoupled soft-CE loss, but ``GPTModel``/``HybridModel`` call
``process_mtp_loss`` unconditionally when MTP heads exist; its hard-CE gradient flows into the shared
trunk and corrupts the RL policy (inflated grad-norm, entropy collapse). SkyRL used to short-circuit
it by passing no labels, but a megatron-core update derives labels from ``input_ids`` -- so we now
replace ``process_mtp_loss`` at its call sites with a no-op instead.

Loud, not silent: raises if a target module no longer exposes ``process_mtp_loss`` (a Megatron
rename). Blind spot: Megatron inlining the loss into the forward (no test currently covers this).
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

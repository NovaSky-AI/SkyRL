"""Disable Megatron's native in-forward MTP loss (``process_mtp_loss``).

SkyRL trains the MTP head with its own decoupled soft-CE loss, but ``GPTModel``/``HybridModel`` call
``process_mtp_loss`` unconditionally when MTP heads exist; its hard-CE gradient flows into the shared
trunk and corrupts the RL policy.
"""

from __future__ import annotations

import sys

import torch


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


def _forbid_native_mtp_loss_autoscaler(*args, **kwargs):
    raise RuntimeError("Megatron's native MTP loss reached MTPLossAutoScaler.apply")


def disable_native_mtp_loss() -> None:
    """Replace megatron's ``process_mtp_loss`` with a no-op (idempotent)"""
    try:
        from megatron.core.transformer import multi_token_prediction as mtp_mod
    except ImportError as e:
        raise RuntimeError(
            "Cannot disable native MTP loss: megatron.core.transformer.multi_token_prediction " "is not importable."
        ) from e

    original = getattr(mtp_mod, "process_mtp_loss", None)
    if original is None:
        raise RuntimeError(
            "Cannot disable native MTP loss: megatron's multi_token_prediction module no longer "
            "exposes 'process_mtp_loss'."
        )
    autoscaler = getattr(mtp_mod, "MTPLossAutoScaler", None)
    if autoscaler is None:
        raise RuntimeError(
            "Cannot disable native MTP loss: megatron's multi_token_prediction module no longer "
            "exposes 'MTPLossAutoScaler'"
        )

    # Rebind every existing `from ... import process_mtp_loss` binding, then the definition
    # module itself so modules imported after this call also get the no-op.
    for mod in list(sys.modules.values()):
        try:
            if getattr(mod, "process_mtp_loss", None) is original:
                mod.process_mtp_loss = _skyrl_skip_native_mtp_loss
        except Exception:
            continue  # lazy modules may raise on attribute access
    mtp_mod.process_mtp_loss = _skyrl_skip_native_mtp_loss
    autoscaler.apply = _forbid_native_mtp_loss_autoscaler

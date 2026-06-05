# Decoupled capture of the trunk hidden states that feed the MTP/draft head.
#
# This is the SkyRL analogue of NeMo-RL's ``HiddenStateCapture``
# (https://github.com/NVIDIA-NeMo/RL/blob/main/nemo_rl/models/megatron/draft/hidden_capture.py).
#
# Megatron's ``GPTModel`` runs its native MTP block (``self.mtp``) inside the forward whenever
# ``self.mtp_process`` is set. Crucially, ``MultiTokenPredictionBlock.forward`` passes the trunk
# hidden states through as the *first* chunk of its output, and ``process_mtp_loss`` returns that
# chunk as the hidden states for the *main* logits. So we must NOT tamper with the in-forward call:
# the main policy logits have to stay connected to the trunk for the policy loss to train it.
#
# Instead we:
#   1. register a forward pre-hook on ``self.mtp`` that simply *records* the keyword arguments
#      Megatron built for it (input_ids, position_ids, the live trunk hidden states, rotary
#      embeddings, attention mask, packed-seq params, the shared embedding, ...). The pre-hook does
#      not modify the call, so the in-forward MTP run proceeds normally and the main logits stay
#      coupled to the trunk.
#   2. after the forward, re-invoke ``self.mtp`` with the recorded arguments but with the trunk
#      hidden states *detached* (when ``detach_trunk``). This second pass is the decoupled draft
#      forward: its gradient reaches the MTP-head (and shared embedding/output) parameters but never
#      the policy backbone. Re-using the recorded kwargs means we never have to reconstruct rotary
#      embeddings or attention masks ourselves.
#
# The in-forward MTP layers are recomputed (their output is discarded by ``process_mtp_loss`` when
# no labels are passed), so there is a small extra-compute cost — one MTP block forward, typically a
# single transformer layer. This is a deliberate trade for robustness over reaching into GPTModel's
# rotary/mask construction.

from __future__ import annotations

from contextlib import contextmanager
from typing import List, Optional


def _unwrap_gpt_model(model):
    """Return the underlying GPTModel from a (possibly DDP/Float16) wrapper."""
    from megatron.core.utils import unwrap_model

    return unwrap_model(model)


class MTPHiddenCapture:
    """Record the MTP block's inputs during the forward, then replay it decoupled.

    Use as a context manager around the policy ``model(...)`` call. After the forward,
    :meth:`compute_student_hidden_states` returns one hidden-state tensor per MTP depth
    (``[seq, batch, hidden]`` in Megatron's internal layout), ready to be projected by the shared
    output layer.
    """

    def __init__(self, model, detach_trunk: bool = True):
        self.gpt_model = _unwrap_gpt_model(model)
        self.detach_trunk = detach_trunk
        self._args = None
        self._kwargs = None
        self._handles: list = []

    @property
    def mtp_num_layers(self) -> int:
        return int(getattr(self.gpt_model.config, "mtp_num_layers", 0) or 0)

    def _pre_hook(self, _module, args, kwargs):
        # Record (do not modify) the arguments Megatron built for the MTP block.
        self._args = args
        self._kwargs = dict(kwargs)
        return None

    @contextmanager
    def capture(self):
        mtp = getattr(self.gpt_model, "mtp", None)
        if mtp is None:
            # Model has no MTP heads on this rank/stage; nothing to capture.
            yield self
            return
        self._args = None
        self._kwargs = None
        self._handles.append(mtp.register_forward_pre_hook(self._pre_hook, with_kwargs=True))
        try:
            yield self
        finally:
            for h in self._handles:
                h.remove()
            self._handles.clear()

    def compute_student_hidden_states(self) -> Optional[List]:
        """Replay the MTP block on detached trunk hidden states and split per depth.

        Returns ``None`` if the block was never called (e.g. a non-post-process pipeline stage).
        """
        if self._kwargs is None:
            return None
        import torch

        kwargs = dict(self._kwargs)
        hidden = kwargs.get("hidden_states")
        if hidden is not None and self.detach_trunk:
            kwargs["hidden_states"] = hidden.detach()

        # MultiTokenPredictionBlock returns the concatenated hidden states
        # [trunk; mtp_0; mtp_1; ...] along the sequence (dim 0).
        captured = self.gpt_model.mtp(*self._args, **kwargs)
        num_layers = self.mtp_num_layers
        chunks = list(torch.chunk(captured, 1 + num_layers, dim=0))
        # Drop the (passthrough) trunk chunk; keep the per-MTP-depth chunks.
        return chunks[1:]


@contextmanager
def maybe_capture_mtp_hidden(model, enabled: bool, detach_trunk: bool = True):
    """Context manager returning an ``MTPHiddenCapture`` when ``enabled``, else ``None``."""
    if not enabled:
        yield None
        return
    capture = MTPHiddenCapture(model, detach_trunk=detach_trunk)
    with capture.capture():
        yield capture

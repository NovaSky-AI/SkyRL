# Decoupled capture of the trunk hidden states feeding the MTP/draft head (inspired by NeMo-RL,
# adapted for SkyRL's Megatron backend). Works for any Megatron model exposing ``self.mtp``.
# A forward pre-hook records the kwargs Megatron builds for ``self.mtp``; after the forward we
# re-invoke it with the trunk hidden states (and the shared re-embedding) detached, so the draft
# gradient reaches the head but no policy parameter. Reusing the captured kwargs avoids rebuilding
# rotary embeddings / masks.

from __future__ import annotations

from contextlib import contextmanager
from typing import List, Optional


def _unwrap_model(model):
    """Return the underlying Megatron model (GPTModel / MambaModel / ...) from a wrapper."""
    from megatron.core.utils import unwrap_model

    return unwrap_model(model)


def _resolve_mtp_host(model):
    """Return the submodule that owns the native MTP block (``.mtp``)."""
    cur, seen = model, set()
    while cur is not None and id(cur) not in seen:
        if getattr(cur, "mtp", None) is not None:
            return cur
        seen.add(id(cur))
        cur = getattr(cur, "language_model", None)
    return model


def _mtp_layer_offset(mtp_block) -> int:
    """Number of passthrough chunks ahead of this stage's MTP-depth chunks.

    ``MultiTokenPredictionBlock.forward`` chunks its input into ``1 + offset`` passthrough chunks
    (the trunk hidden states plus any MTP outputs forwarded from earlier pipeline/VP stages),
    appends one new chunk per MTP depth, and concatenates everything along dim 0. ``offset`` is 0 in
    the common single-stage case; we resolve it via megatron-core so the replay split below is also
    correct under PP/VPP. Fails loud if the helper is missing (megatron-core moved it) -- a silent 0
    would mis-split the hidden states under PP/VPP."""
    try:
        from megatron.core.transformer.multi_token_prediction import (
            get_mtp_layer_offset,
        )
    except ImportError as e:
        raise RuntimeError(
            "megatron-core no longer exposes get_mtp_layer_offset; the MTP replay chunk split "
            "cannot be resolved under PP/VPP. Update mtp/hidden_capture.py."
        ) from e
    return int(get_mtp_layer_offset(mtp_block.config, getattr(mtp_block, "vp_stage", None)))


class MTPHiddenCapture:
    """Record the MTP block's inputs during the forward, then replay it decoupled.

    Use as a context manager around the policy ``model(...)`` call. After the forward,
    :meth:`compute_student_hidden_states` returns one hidden-state tensor per MTP depth
    (``[seq, batch, hidden]`` in Megatron's internal layout), ready to be projected by the shared
    output layer.
    """

    def __init__(self, model):
        # The module that owns the MTP block is also what the caller projects through (output_layer /
        # shared weight), so they must match.
        self.model = _resolve_mtp_host(_unwrap_model(model))
        self._args = None
        self._kwargs = None
        self._handles: list = []
        self._prev_training = None

    @property
    def mtp_num_layers(self) -> int:
        return int(getattr(self.model.config, "mtp_num_layers", 0) or 0)

    def _pre_hook(self, _module, args, kwargs):
        # Record (do not modify) the arguments Megatron built for the MTP block.
        self._args = args
        self._kwargs = dict(kwargs)
        return None

    @contextmanager
    def capture(self):
        mtp = getattr(self.model, "mtp", None)
        if mtp is None:
            # Model has no MTP heads on this rank/stage; nothing to capture.
            yield self
            return
        self._args = None
        self._kwargs = None
        # Run the MTP block in eval mode (forward + replay): Megatron's full-recompute path routes
        # through CheckpointFunction, which can't save the non-tensor PackedSeqParams for backward.
        # Eval skips it (gradients still flow) and MTP is one tiny layer. Correct ONLY while the
        # head has no dropout -- eval would silently drop it from training otherwise.
        mtp_config = getattr(mtp, "config", None)
        for attr in ("hidden_dropout", "attention_dropout"):
            dropout = getattr(mtp_config, attr, 0) or 0
            assert dropout == 0, (
                f"MTP capture runs the MTP block in eval mode (recompute/PackedSeqParams "
                f"workaround), which would silently disable {attr}={dropout}. Set it to 0 or "
                "rework the capture."
            )
        self._prev_training = mtp.training
        mtp.eval()
        self._handles.append(mtp.register_forward_pre_hook(self._pre_hook, with_kwargs=True))
        try:
            yield self
        finally:
            for h in self._handles:
                h.remove()
            self._handles.clear()
            if self._prev_training:
                mtp.train()

    def compute_student_hidden_states(self) -> Optional[List]:
        """Replay the MTP block on detached trunk hidden states and split per depth.

        Returns ``None`` if the block was never called (e.g. a non-post-process pipeline stage).
        """
        if self._kwargs is None:
            return None
        import torch

        kwargs = dict(self._kwargs)
        hidden = kwargs.get("hidden_states")
        # We only patch the keyword arg. Megatron passes hidden_states as a kwarg today; if a
        # future version passes it positionally the detach would silently no-op (re-coupling the
        # trunk), so fail loudly instead.
        assert hidden is not None, (
            "MTP capture: 'hidden_states' was not passed to the MTP block as a keyword argument "
            "(megatron-core call convention changed?); cannot detach the trunk for decoupled "
            "draft training."
        )
        kwargs["hidden_states"] = hidden.detach()

        # The block re-embeds the rolled ids: a second gradient path into the embedding weight (== the
        # lm_head on tied models) besides the output projection. Detach it too.
        if kwargs.get("embedding") is not None:
            orig_embedding = kwargs["embedding"]

            def _detached_embedding(*emb_args, **emb_kwargs):
                return orig_embedding(*emb_args, **emb_kwargs).detach()

            kwargs["embedding"] = _detached_embedding

        mtp = self.model.mtp
        # MultiTokenPredictionBlock concatenates, along dim 0:
        #   [<1 + offset> passthrough chunks (trunk + earlier-stage MTP outputs)]
        #   + [<mtp_num_layers> new MTP-depth chunks produced on this stage].
        # We want this stage's new MTP-depth chunks (the last `num_layers`).
        captured = mtp(*self._args, **kwargs)
        num_layers = self.mtp_num_layers
        total_chunks = 1 + _mtp_layer_offset(mtp) + num_layers
        chunks = list(torch.chunk(captured, total_chunks, dim=0))
        return chunks[-num_layers:]


@contextmanager
def maybe_capture_mtp_hidden(model, enabled: bool):
    """Context manager returning an ``MTPHiddenCapture`` when ``enabled``, else ``None``."""
    if not enabled:
        yield None
        return
    capture = MTPHiddenCapture(model)
    with capture.capture():
        yield capture

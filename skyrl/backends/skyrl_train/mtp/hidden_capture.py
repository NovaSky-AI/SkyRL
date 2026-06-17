# Decoupled capture of the trunk hidden states that feed the MTP/draft head. Inspired by NeMo-RL's
# draft-model training (https://github.com/NVIDIA-NeMo/RL), adapted for SkyRL's Megatron backend.
#
# Works for any Megatron model with native MTP heads exposing ``self.mtp`` (the shared
# ``MultiTokenPredictionBlock``): ``GPTModel`` (DeepSeek-V3 / GLM / Qwen3-Next) and ``MambaModel``
# (Qwen3.5 / NemotronH). The in-forward ``self.mtp(...)`` passes the trunk hidden states through to
# ``process_mtp_loss`` as the main logits, so we must not tamper with it. Instead:
#   1. a forward pre-hook records the kwargs Megatron built for ``self.mtp`` (without modifying them);
#   2. after the forward, we re-invoke ``self.mtp`` with those kwargs but the trunk hidden states
#      detached -- the decoupled draft pass, whose gradient reaches the MTP head but never the policy
#      backbone. Reusing the kwargs avoids reconstructing rotary embeddings / attention masks.
# The replay costs one extra MTP-block forward (typically a single layer) -- a deliberate trade for
# robustness over reaching into GPTModel's rotary/mask construction.

from __future__ import annotations

from contextlib import contextmanager
from typing import List, Optional


def _unwrap_model(model):
    """Return the underlying Megatron model (GPTModel / MambaModel / ...) from a wrapper."""
    from megatron.core.utils import unwrap_model

    return unwrap_model(model)


def _resolve_mtp_host(model):
    """Return the submodule that owns the native MTP block (``.mtp``).

    Usually the model itself, but for VL wrappers (e.g. Qwen3.5-VL) the MTP head is nested at
    ``model.language_model``. Descend the ``.language_model`` chain until we find a module with ``.mtp``;
    if none, return the model unchanged (capture then yields ``None``)."""
    seen = set()
    cur = model
    for _ in range(4):  # bounded descent guard against pathological nesting / cycles
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        if getattr(cur, "mtp", None) is not None:
            return cur
        cur = getattr(cur, "language_model", None)
    return model


def _mtp_layer_offset(mtp_block) -> int:
    """Number of passthrough chunks ahead of this stage's MTP-depth chunks.

    ``MultiTokenPredictionBlock.forward`` chunks its input into ``1 + offset`` passthrough chunks
    (the trunk hidden states plus any MTP outputs forwarded from earlier pipeline/VP stages),
    appends one new chunk per MTP depth, and concatenates everything along dim 0. ``offset`` is 0 in
    the common single-stage case; we resolve it via megatron-core so the replay split below is also
    correct under PP/VPP. Falls back to 0 if the helper is unavailable (older megatron-core)."""
    try:
        from megatron.core.transformer.multi_token_prediction import (
            get_mtp_layer_offset,
        )

        return int(get_mtp_layer_offset(mtp_block.config, getattr(mtp_block, "vp_stage", None)))
    except Exception:
        return 0


class MTPHiddenCapture:
    """Record the MTP block's inputs during the forward, then replay it decoupled.

    Use as a context manager around the policy ``model(...)`` call. After the forward,
    :meth:`compute_student_hidden_states` returns one hidden-state tensor per MTP depth
    (``[seq, batch, hidden]`` in Megatron's internal layout), ready to be projected by the shared
    output layer.
    """

    def __init__(self, model, detach_trunk: bool = True, detach_shared_embedding: bool = False):
        # The module that owns the MTP block is also what the caller projects through (output_layer /
        # shared weight), so they must match.
        self.model = _resolve_mtp_host(_unwrap_model(model))
        self.detach_trunk = detach_trunk
        # The MTP block re-embeds the rolled input ids through the shared embedding -- a second
        # gradient path into the embedding weight besides the output projection. detach_shared_embedding
        # cuts it too, so tied-embedding models don't keep training the lm_head through the lookup.
        self.detach_shared_embedding = detach_shared_embedding
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
        # Eval skips it (dropout is 0 here; gradients still flow) and MTP is one tiny layer.
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
        if self.detach_trunk:
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
        if self.detach_shared_embedding and kwargs.get("embedding") is not None:
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
def maybe_capture_mtp_hidden(model, enabled: bool, detach_trunk: bool = True, detach_shared_embedding: bool = False):
    """Context manager returning an ``MTPHiddenCapture`` when ``enabled``, else ``None``."""
    if not enabled:
        yield None
        return
    capture = MTPHiddenCapture(model, detach_trunk=detach_trunk, detach_shared_embedding=detach_shared_embedding)
    with capture.capture():
        yield capture

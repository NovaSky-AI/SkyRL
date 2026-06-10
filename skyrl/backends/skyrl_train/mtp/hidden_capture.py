# Decoupled capture of the trunk hidden states that feed the MTP/draft head.
#
# This is the SkyRL analogue of NeMo-RL's ``HiddenStateCapture``
# (https://github.com/NVIDIA-NeMo/RL/blob/main/nemo_rl/models/megatron/draft/hidden_capture.py).
#
# Works for any Megatron model that builds native MTP heads and exposes ``self.mtp`` (the shared
# ``MultiTokenPredictionBlock``): ``GPTModel`` (DeepSeek-V3 / GLM / Qwen3-Next, ...) and
# ``MambaModel`` (Qwen3.5 / NemotronH, ...). Both run ``self.mtp(...)`` + ``process_mtp_loss(...)``
# inside ``forward`` and surface the same output layout, so the capture below is model-agnostic.
#
# Megatron runs its native MTP block (``self.mtp``) inside the forward whenever ``self.mtp_process``
# is set. Crucially, ``MultiTokenPredictionBlock.forward`` passes the trunk hidden states through as
# a passthrough chunk of its output, and ``process_mtp_loss`` returns that chunk as the hidden states
# for the *main* logits. So we must NOT tamper with the in-forward call: the main policy logits have
# to stay connected to the trunk for the policy loss to train it.
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


def _unwrap_model(model):
    """Return the underlying Megatron model (GPTModel / MambaModel / ...) from a wrapper."""
    from megatron.core.utils import unwrap_model

    return unwrap_model(model)


def _resolve_mtp_host(model):
    """Return the submodule that actually owns the native MTP block (``.mtp``).

    For a plain ``GPTModel`` / ``MambaModel`` that is the model itself. For vision-language wrappers
    (e.g. Qwen3.5-VL's ``Qwen3VLModel``) the text backbone *and* its MTP head are nested one level
    down at ``model.language_model`` (the bridge maps the heads to ``language_model.mtp.*``), so the
    top-level model has no ``.mtp`` and the capture would otherwise silently find nothing. We descend
    through the common ``.language_model`` nesting until we find a module exposing ``.mtp``; if none
    does, return the top-level model unchanged (capture then yields ``None``, as before)."""
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
        # Resolve the module that owns the MTP block: the unwrapped model itself for plain
        # GPT/Mamba models, or the nested ``.language_model`` for VL wrappers (e.g. Qwen3.5-VL).
        # capture.model is also what the caller projects through (output_layer / shared weight),
        # so it must be the same module that holds the heads.
        self.model = _resolve_mtp_host(_unwrap_model(model))
        self.detach_trunk = detach_trunk
        # The MTP block re-embeds the rolled input ids through the model's shared embedding
        # (``embedding=self.embedding`` in its kwargs) — a second gradient path into the shared
        # embedding weight, separate from the output projection. When the caller wants the shared
        # weights fully isolated from the draft loss (``mtp_detach_shared_output``), the replay must
        # detach this path too, else tied-embedding models still train the lm_head through the
        # lookup (slime's MTP-RL patch detaches the same two paths).
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
        # Run the MTP block in eval mode for both the in-forward pass and the replay. Megatron's
        # full-activation-recompute path (recompute_granularity='full' and module.training) routes
        # through CheckpointFunction, which cannot save the non-tensor PackedSeqParams for backward.
        # Eval skips that path (dropout is 0 in these configs; gradients still flow), and MTP is a
        # single tiny layer so keeping its activations is negligible.
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
            # The detach below only patches the *keyword* argument. Megatron currently calls
            # ``self.mtp(...)`` with kwargs only (GPTModel + MambaModel); if a future version passes
            # hidden_states positionally, the detach would silently no-op and the draft gradient
            # would couple back into the policy trunk — fail loudly instead.
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

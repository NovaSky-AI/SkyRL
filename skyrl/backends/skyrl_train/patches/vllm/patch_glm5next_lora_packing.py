"""vLLM LoRA support for GLM-5.3-Flash (``merge_lora=false``), mostly a backport of
vllm-project/vllm#56327 (open upstream; not in vLLM 0.30).

Applied from ``new_inference_worker_wrap`` so it lands in every worker before model init:

1. ``Glm5NextForConditionalGeneration.packed_modules_mapping`` (#56327). Without it the class
   inherits ``Glm4vForConditionalGeneration``'s mapping, which names none of this model's fused
   runtime projections, so the adapter's separately-stored HF submodules have nothing to
   assemble onto.
2. ``MergedColumnParallelLinearWithLoRA.output_ids`` honoring ``replicated_shard_ids`` (#56327).
   KDA's ``in_proj_qkvbfg_a`` declares ``replicated_shard_ids=(4, 5)``: ``f_a_proj``/``g_a_proj``
   are duplicated across TP ranks (``parallel_mode="duplicated"`` on the Megatron side), so their
   LoRA-B must stay whole on every rank.
3. A ``.contiguous()`` guard in ``PunicaWrapperGPU.add_shrink`` (not in #56327): KDA splits its
   fused projection into non-contiguous views, which trips ``assert inputs.is_contiguous()`` in
   the triton ``lora_shrink``. ``f_b_proj``/``g_b_proj`` are still left out of the target lists;
   re-adding them with this guard in place is untested.
4. The ``kv_b_proj`` adapter on MLA's absorbed decode path (#56327 commit ed6aaff3). Decode never
   runs the ``kv_b_proj`` module, so without this the adapter is dropped for every decode token.

Known gap, not patched: MLA *prefill* also skips the ``kv_b_proj`` adapter. The attention impl
keeps a plain reference to the original ``kv_b_proj`` that LoRA wrapping never replaces, and
sparse MLA prefills up to ``index_topk`` tokens (and prefix-cached context chunks) up-project
K/V through it. vllm-project/vllm#56718 fixes this upstream.

TODO: remove once #56327 (and #56718) land in the pinned vLLM.
"""

import logging

logger = logging.getLogger(__name__)

_PATCHED = False

# Verbatim from the PR's Glm5NextForConditionalGeneration.packed_modules_mapping.
_GLM5NEXT_PACKED_MODULES_MAPPING = {
    "gate_up_proj": ["gate_proj", "up_proj"],
    "in_proj_qkvbfg_a": [
        "q_proj",
        "k_proj",
        "v_proj",
        "b_proj",
        "f_a_proj",
        "g_a_proj",
    ],
    "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
    "wk_weights_proj": ["wk", "weights_proj"],
    "qkv": ["qkv"],
}


def _patch_packed_modules_mapping() -> list[str]:
    """Install the mapping on whichever GLM-5.3-Flash entrypoints this build exposes.

    ``language_model_only`` selects the ``ForCausalLM`` variant, so set both when present.
    """
    import vllm.models.glm5next as glm5next

    patched = []
    for cls_name in ("Glm5NextForConditionalGeneration", "Glm5NextForCausalLM"):
        cls = getattr(glm5next, cls_name, None)
        if cls is None:
            continue
        # Only override the inherited (wrong) mapping, never a build that already has the fix.
        if cls.__dict__.get("packed_modules_mapping") == _GLM5NEXT_PACKED_MODULES_MAPPING:
            continue
        cls.packed_modules_mapping = dict(_GLM5NEXT_PACKED_MODULES_MAPPING)
        patched.append(cls_name)
    return patched


def _patch_replicated_shard_ids() -> bool:
    """Keep LoRA-B whole for shards the base layer marks as replicated."""
    from vllm.lora.layers.column_parallel_linear import (
        MergedColumnParallelLinearWithLoRA,
    )

    original_init = MergedColumnParallelLinearWithLoRA.__init__

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        replicated_shard_ids = getattr(self.base_layer, "replicated_shard_ids", ())
        if not replicated_shard_ids:
            return
        self.output_ids = tuple(0 if i in replicated_shard_ids else self.tp_rank for i in range(self.n_slices))

    MergedColumnParallelLinearWithLoRA.__init__ = __init__
    return True


def _patch_lora_shrink_contiguity() -> bool:
    """Make the LoRA shrink GEMM accept a non-contiguous input.

    Not part of vllm#56327 -- that PR leaves this untouched. KDA runs one fused
    ``in_proj_qkvbfg_a`` GEMM and ``.split()``s it, so ``f_a``/``g_a`` are views whose
    ``stride(0)`` is the fused width: the ``.view(-1, shape[-1])`` inside ``add_shrink``
    still succeeds (the last dim is unit-stride) but ``lora_shrink`` then trips
    ``assert inputs.is_contiguous()``. Without this, ``f_b_proj``/``g_b_proj`` cannot be
    LoRA targets at all.

    The copy is only taken when the input is actually non-contiguous, and for KDA that is
    a ``[num_tokens, head_dim]`` bf16 tensor (head_dim=128 on GLM-5.3-Flash), so the cost
    is negligible next to the GEMM it feeds.
    """
    from vllm.lora.punica_wrapper.punica_gpu import PunicaWrapperGPU

    original_add_shrink = PunicaWrapperGPU.add_shrink

    def add_shrink(self, y, x, lora_a_stacked, scale, **kwargs):
        if not x.is_contiguous():
            x = x.contiguous()
        return original_add_shrink(self, y, x, lora_a_stacked, scale, **kwargs)

    PunicaWrapperGPU.add_shrink = add_shrink
    return True


def apply_glm5next_lora_packing_patch() -> None:
    """Apply all four pieces once per process; a build without GLM-5.3-Flash is a no-op."""
    global _PATCHED
    if _PATCHED:
        return
    try:
        patched_classes = _patch_packed_modules_mapping()
        _patch_replicated_shard_ids()
        _patch_lora_shrink_contiguity()
        _patch_mla_kv_b_proj_lora()
    except (ModuleNotFoundError, ImportError) as e:
        logger.info(f"Skipping GLM-5.3-Flash LoRA packing patch: {e}")
        return
    _PATCHED = True
    logger.info(
        "Patched vLLM for GLM-5.3-Flash LoRA (vllm#56327): packed_modules_mapping on "
        f"{patched_classes or '<none>'}, replicated_shard_ids honored in merged LoRA-B "
        "loading, non-contiguous LoRA shrink inputs accepted, kv_b_proj adapter applied "
        "to the absorbed MLA projections"
    )


# --------------------------------------------------------------------------------------
# vllm#56327, commit ed6aaff3 ("Fix missing LoRA updates in MLA and DSA indexer")
# --------------------------------------------------------------------------------------
#
# MLA never runs its `kv_b_proj` module on the decode path. `process_weights_after_loading`
# splits the weight into the absorbed `W_UK_T` / `W_UV` and decode does
# `torch.bmm(mqa_q_nope, W_UK_T)` / `torch.bmm(x, W_UV)` directly, so a LoRA adapter on
# `kv_b_proj` -- which only ever applies inside the wrapped module's forward -- reaches
# prefill and is silently dropped in decode. Since a response is almost all decode tokens,
# the adapter we train on `linear_kv_up_proj` was very nearly inert at generation time.
#
# The upstream fix computes the adapter delta directly against the absorbed, head-major
# layout and adds it to the bmm output, so nothing has to rebuild W_UK_T/W_UV.
#
# NOT vendored: the same commit's `glm5next/common/attention.py` hunk, which fixes the DSA
# indexer's `wk_weights_proj` fp32 weight cache bypassing its LoRA wrapper. That module is
# absent from our `lora_target_modules`, so it is never wrapped and the hunk is a no-op
# here. Re-check if `wk_weights_proj` is ever added to the target list.
#
# TODO: remove together with the rest of this module once vllm#56327 lands in the pinned vLLM.

# Inserted verbatim-in-spirit from the PR. vLLM 0.30 ends the query-projection branch chain
# with the unpadded bmm writing straight into a (B, N, L) buffer; every branch leaves
# `mqa_ql_nope` as (B, N, L) by the fp8-quant check, so the delta is added there.
_FORWARD_IMPL_ANCHOR = """                    torch.bmm(mqa_q_nope, W_UK_T, out=mqa_ql_nope.transpose(0, 1))

            if fp8_attention and self.impl.supports_quant_query_input:"""

_FORWARD_IMPL_REPLACEMENT = """                    torch.bmm(mqa_q_nope, W_UK_T, out=mqa_ql_nope.transpose(0, 1))

            self._apply_lora_projection(mqa_q_nope, mqa_ql_nope, is_query=True)

            if fp8_attention and self.impl.supports_quant_query_input:"""


def _apply_lora_projection(self, x, out, *, is_query: bool) -> None:
    """Add kv_b_proj adapters to the absorbed, head-major MLA projection.

    ``x`` is the bmm input and ``out`` the bmm output, both head-major:
    (N, B, qk_nope_head_dim) -> (B, N, kv_lora_rank) for the query projection, and
    (N, B, kv_lora_rank) -> (B, N, v_head_dim) for the value projection.

    ``kv_b_proj`` maps kv_lora_rank -> num_heads * (qk_nope_head_dim + v_head_dim), so its
    ``lora_b`` carries both halves; split at ``qk_nope_head_dim`` to get the K half (query
    path) and the V half (value path). Associativity lets us fold as ``(x @ B) @ A`` rather
    than materializing the full ``B @ A`` weight delta.
    """
    import torch
    from vllm.distributed import get_dcp_group, get_tp_group
    from vllm.lora.layers.column_parallel_linear import ColumnParallelLinearWithLoRA

    layer = self.kv_b_proj
    # Only wrapped when LoRA is enabled AND kv_b_proj is in lora_target_modules.
    if not isinstance(layer, ColumnParallelLinearWithLoRA):
        return
    lora_a = layer.lora_a_stacked[0]
    lora_b = layer.lora_b_stacked[0]
    if layer.lora_config.fully_sharded_loras and layer.tp_size > 1:
        lora_a = get_tp_group().all_gather(lora_a, dim=2)
    if is_query and self.dcp_q_replicate:
        lora_b = get_dcp_group().all_gather(lora_b, dim=2)
    # MQA may consume only the decode prefix of the mapped batch.
    indices = layer.punica_wrapper.token_lora_indices[: x.shape[1]]
    for slot in range(lora_a.shape[0]):
        a = lora_a[slot, 0]
        b = lora_b[slot, 0].view(x.shape[0], self.qk_nope_head_dim + self.v_head_dim, -1)
        if is_query:
            delta = torch.matmul(x, b[:, : self.qk_nope_head_dim]) @ a
        else:
            delta = torch.matmul(x @ a.T, b[:, self.qk_nope_head_dim :].transpose(1, 2))
        out.add_(torch.where((indices == slot)[:, None, None], delta.transpose(0, 1), 0))


def _patch_mla_kv_b_proj_lora() -> bool:
    """Apply the kv_b_proj adapter at both absorbed projections."""
    import inspect
    import sys
    import textwrap

    from vllm.model_executor.layers.attention.mla_attention import MLAAttention

    if getattr(MLAAttention, "_skyrl_kv_b_proj_lora_patched", False):
        return True

    MLAAttention._apply_lora_projection = _apply_lora_projection

    # Value projection: a plain wrapper is enough -- recompute the head-major views the
    # original builds internally, let it run, then add the delta into the same storage.
    original_v_up_proj = MLAAttention._v_up_proj

    def _v_up_proj(self, x, out):
        lora_input = x.view(-1, self.num_heads, self.kv_lora_rank).transpose(0, 1)
        out_view = out.view(-1, self.num_heads, self.v_head_dim)
        original_v_up_proj(self, x, out)
        self._apply_lora_projection(lora_input, out_view, is_query=False)

    MLAAttention._v_up_proj = _v_up_proj

    # Query projection: the call site sits mid-``forward_impl`` with no wrappable seam, so
    # splice the one line into the method's source and recompile it in the module's own
    # globals. The anchor must match exactly once -- a vLLM bump that moves it fails loudly
    # here rather than silently leaving decode unpatched.
    module = sys.modules[MLAAttention.__module__]
    # Match against the source as it appears in the file (method body at its original
    # indent), then dedent the result so the `def` is compilable at module level.
    raw_src = inspect.getsource(MLAAttention.forward_impl)
    if raw_src.count(_FORWARD_IMPL_ANCHOR) != 1:
        raise RuntimeError(
            "GLM-5.3-Flash LoRA patch: expected exactly one absorbed-query-projection call "
            f"site in MLAAttention.forward_impl, found {raw_src.count(_FORWARD_IMPL_ANCHOR)}. "
            "The pinned vLLM has moved; re-derive the patch against vllm#56327 commit ed6aaff3."
        )
    patched_src = textwrap.dedent(raw_src.replace(_FORWARD_IMPL_ANCHOR, _FORWARD_IMPL_REPLACEMENT))
    namespace: dict = {}
    exec(compile(patched_src, f"<skyrl-patch:{module.__name__}.forward_impl>", "exec"), module.__dict__, namespace)
    MLAAttention.forward_impl = namespace["forward_impl"]

    MLAAttention._skyrl_kv_b_proj_lora_patched = True
    return True

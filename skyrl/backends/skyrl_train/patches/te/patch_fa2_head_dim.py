"""Backport of NVIDIA/TransformerEngine#3360 for transformer-engine 2.16.0.

TE 2.16.0 gates FlashAttention 2 for ``head_dim > 192`` behind a compute
capability allowlist of ``(8, 0), (9, 0), (10, 0), (12, 0)``. Upstream removed
that allowlist in #2836, #2629 accidentally reintroduced it, and #3360 removes
it again -- keeping only the real FA2 constraints (``head_dim <= 256`` and
``head_dim % 8 == 0``).

On an architecture outside the allowlist -- notably sm103 (B300/GB300) -- a
head_dim of 256 (Gemma 2/3, for example) makes TE reject FA2. If cuDNN fused
attention is also unavailable for the config, TE drops to unfused attention,
which materializes the quadratic attention matrix; upstream reported a 202 GiB
allocation and an OOM at 65k tokens.

This is applied as a source-level rebind rather than an edit to the installed
wheel: TE ships as a prebuilt wheel from Astral's CUDA index, and
``uv run --isolated`` builds a throwaway environment per invocation, so a
site-packages edit would not survive a run (nor reach other Ray nodes).
``get_attention_backend`` is a single ~1000 line function and the gate is an
inline boolean clause, so there is no sub-function seam to override -- we
recompile that one function with the clause removed and rebind it on its
module. Its only caller resolves it as a module attribute
(``dpa_utils.get_attention_backend``), so the rebind is picked up.

DELETE THIS PATCH once the transformer-engine pin moves to a release that
contains #3360. This is a temporary backport of an upstream fix, not a SkyRL
behavior change, and it has no reason to outlive the pin it works around.
It fails safe in the meantime: the match below is against the literal 2.16.0
source text, and TE expresses the same check against ``fa2_padded_head_dim``
after the padding refactor, so on a newer TE this logs and no-ops rather than
misfiring. That means a stale copy is quiet rather than harmful -- which also
means nothing will alert you to remove it. Check this file when bumping TE.
"""

import inspect

from loguru import logger

# The allowlist clause as it appears in TE 2.16.0. Newer TE expresses the same
# check against `fa2_padded_head_dim`; this patch deliberately does not match
# that form, so it no-ops instead of misfiring after a TE upgrade.
_SM_ALLOWLIST_CLAUSE = """        and (
            head_dim_qk > 256
            or head_dim_qk % 8 != 0
            or (
                head_dim_qk > 192
                and device_compute_capability not in ((8, 0), (9, 0), (10, 0), (12, 0))
            )
        )
"""

_REPLACEMENT_CLAUSE = """        and (head_dim_qk > 256 or head_dim_qk % 8 != 0)
"""

# Architectures TE 2.16.0 already allows; on these the gate is dead code.
_UNAFFECTED_COMPUTE_CAPABILITIES = ((8, 0), (9, 0), (10, 0), (12, 0))

_PATCHED_FLAG = "_skyrl_fa2_head_dim_patched"


def patch_fa2_head_dim_allowlist(force: bool = False) -> bool:
    """Remove TE's compute-capability gate on FlashAttention 2 for head_dim > 192.

    No-ops (returning False) when the current GPU is one TE already allows, when
    TE is not importable, or when TE's source does not match the 2.16.0 form.
    Pass ``force=True`` to patch regardless of the detected compute capability.

    Safe to call more than once; the second call is a no-op returning True.
    """
    try:
        from transformer_engine.pytorch.attention.dot_product_attention import (
            dot_product_attention as dpa,
        )
        from transformer_engine.pytorch.attention.dot_product_attention import (
            utils as dpa_utils,
        )
    except ImportError:
        logger.debug("transformer_engine not importable; skipping FA2 head_dim patch")
        return False

    target = getattr(dpa_utils, "get_attention_backend", None)
    if target is None:
        logger.warning("TE has no get_attention_backend; skipping FA2 head_dim patch")
        return False

    if getattr(target, _PATCHED_FLAG, False):
        return True

    if not force:
        compute_capability = dpa_utils.get_device_compute_capability()
        if compute_capability in _UNAFFECTED_COMPUTE_CAPABILITIES:
            logger.debug(
                "sm{} is already allowed by TE; skipping FA2 head_dim patch",
                ".".join(str(i) for i in compute_capability),
            )
            return False

    try:
        source = inspect.getsource(target)
    except (OSError, TypeError):
        logger.warning("Cannot read TE get_attention_backend source; skipping FA2 head_dim patch")
        return False

    if _SM_ALLOWLIST_CLAUSE not in source:
        # Most likely the TE pin moved past 2.16.0 and picked up #3360, in which
        # case this whole module should be deleted rather than left to no-op.
        logger.info(
            "TE get_attention_backend does not contain the sm allowlist for head_dim > 192 "
            "(likely fixed or refactored upstream); skipping FA2 head_dim patch. "
            "If TE now includes NVIDIA/TransformerEngine#3360, delete this patch module."
        )
        return False

    patched_source = source.replace(_SM_ALLOWLIST_CLAUSE, _REPLACEMENT_CLAUSE)
    # Execute in TE's own module namespace so the recompiled function keeps the
    # live module globals it depends on, and so the rebind lands on the module.
    exec(compile(patched_source, dpa_utils.__file__, "exec"), dpa_utils.__dict__)  # noqa: S102

    patched = dpa_utils.get_attention_backend
    if patched is target:
        logger.warning("Failed to rebind TE get_attention_backend; FA2 head_dim patch not applied")
        return False
    setattr(patched, _PATCHED_FLAG, True)

    # Backend selection is memoized per attention config; drop any entry chosen
    # by the unpatched function.
    dpa._attention_backends["attention_params"] = None
    dpa._attention_backends["backend_selection_requires_update"] = True

    logger.info("Applied TE FA2 head_dim patch (NVIDIA/TransformerEngine#3360 backport)")
    return True

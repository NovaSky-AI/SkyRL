"""Backport of NVIDIA/TransformerEngine#3360 for transformer-engine 2.19.0.

TE gates FlashAttention 2 for ``head_dim > 192`` behind a compute capability
allowlist of ``(8, 0), (9, 0), (10, 0), (12, 0)``. Upstream removed that
allowlist in #2836, #2629 accidentally reintroduced it, and #3360 removes it
again -- keeping only the real FA2 constraints (``head_dim <= 256`` and
``head_dim % 8 == 0``). **#3360 is still open and unmerged**, so the allowlist
is present in every release through at least 2.19.0 and this backport is still
required.

The clause survived a rename: 2.16.0 spelled it ``head_dim_qk``, and 2.17.0
onward spell it ``fa2_padded_head_dim = max(head_dim_qk, head_dim_v)``. Both
forms are matched below, because matching only one silently no-ops -- which is
exactly how the 2.16 -> 2.19 bump nearly shipped the gate back on.

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
actually contains #3360. This is a temporary backport of an upstream fix, not a
SkyRL behavior change, and it has no reason to outlive the pin it works around.

Verifying that is a source check, not a PR-number check: upstream has already
removed this clause once (#2836) and had it come back under a different variable
name, so "the fix merged" and "the gate is gone from the release you pinned" are
not the same claim. Grep the installed wheel::

    python -c "import inspect; \
from transformer_engine.pytorch.attention.dot_product_attention import utils as u; \
print('device_compute_capability not in' in inspect.getsource(u.get_attention_backend))"

If that prints False, delete this module. If it prints True, the gate is still
there whatever the changelog says. ``verify_fa2_head_dim.py`` does the same check
and then measures the effect on a real GPU.
"""

import inspect

from loguru import logger

# The allowlist clause, keyed by the variable TE uses to express it. 2.16.x
# spells it `head_dim_qk`; 2.17.0 through at least 2.19.0 spell it
# `fa2_padded_head_dim` (= max(head_dim_qk, head_dim_v)). The clause is otherwise
# identical, so both map onto the same rewrite: keep FA2's real limits (<= 256,
# % 8 == 0) and drop the compute-capability allowlist.
_CLAUSE_TEMPLATE = """        and (
            {v} > 256
            or {v} % 8 != 0
            or (
                {v} > 192
                and device_compute_capability not in ((8, 0), (9, 0), (10, 0), (12, 0))
            )
        )
"""

_REPLACEMENT_TEMPLATE = """        and ({v} > 256 or {v} % 8 != 0)
"""

# Ordered newest-spelling-first; only one will ever match a given TE.
_CLAUSE_VARIANTS = tuple(
    (_CLAUSE_TEMPLATE.format(v=v), _REPLACEMENT_TEMPLATE.format(v=v)) for v in ("fa2_padded_head_dim", "head_dim_qk")
)

# Architectures TE already allows; on these the gate is dead code.
_UNAFFECTED_COMPUTE_CAPABILITIES = ((8, 0), (9, 0), (10, 0), (12, 0))

_PATCHED_FLAG = "_skyrl_fa2_head_dim_patched"


def patch_fa2_head_dim_allowlist(force: bool = False) -> bool:
    """Remove TE's compute-capability gate on FlashAttention 2 for head_dim > 192.

    No-ops (returning False) when the current GPU is one TE already allows, when
    TE is not importable, or when TE's source matches none of the known forms.
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
        # Looked up defensively: this guard only narrows the blast radius, so if TE
        # ever renames it we fall through to the source match below, which is the
        # real version gate.
        get_compute_capability = getattr(dpa_utils, "get_device_compute_capability", None)
        if get_compute_capability is None:
            logger.debug("TE has no get_device_compute_capability; skipping the sm guard")
        else:
            compute_capability = get_compute_capability()
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

    match = next(((c, r) for c, r in _CLAUSE_VARIANTS if c in source), None)
    if match is None:
        # Either upstream finally removed the gate, or it was rewritten into a
        # third form. Those need opposite responses -- delete this module vs. add
        # a variant -- and only reading the source tells them apart, so say so
        # loudly instead of implying the former.
        still_gated = "device_compute_capability not in" in source
        logger.warning(
            "TE get_attention_backend matches none of the known head_dim allowlist forms "
            "(an sm allowlist {} present in the source). If it is gone, delete this patch "
            "module; if it is still there in a new spelling, add that spelling to "
            "_CLAUSE_VARIANTS -- do not assume the former.",
            "IS still" if still_gated else "is NOT",
        )
        return False

    clause, replacement = match
    patched_source = source.replace(clause, replacement)
    # Execute in TE's own module namespace so the recompiled function keeps the
    # live module globals it depends on, and so the rebind lands on the module.
    fn = getattr(dpa_utils, "__file__", None) or "<string>"
    exec(compile(patched_source, fn, "exec"), dpa_utils.__dict__)  # noqa: S102

    patched = dpa_utils.get_attention_backend
    if patched is target:
        logger.warning("Failed to rebind TE get_attention_backend; FA2 head_dim patch not applied")
        return False
    setattr(patched, _PATCHED_FLAG, True)

    # Backend selection is memoized per attention config; drop any entry chosen
    # by the unpatched function.
    backends = getattr(dpa, "_attention_backends", None)
    if isinstance(backends, dict):
        backends["attention_params"] = None
        backends["backend_selection_requires_update"] = True

    logger.info("Applied TE FA2 head_dim patch (NVIDIA/TransformerEngine#3360 backport)")
    return True

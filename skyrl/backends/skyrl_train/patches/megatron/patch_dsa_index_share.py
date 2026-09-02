"""Backport of NVIDIA/Megatron-LM#6793 for megatron-core 0.20.0.

DSA (the sparse-attention variant GLM 5 uses) lets a source layer compute top-k
indices that later layers reuse. The holder for that state is chosen per forward
by ``DSAttention._get_index_share_carrier``. On the packed path it is the
``PackedSeqParams`` object, which is genuinely per-forward. On the **non-packed**
path the fallback is ``attention_mask`` -- or ``self.config`` when there is no
mask -- and both outlive a single forward.

With activation recompute there can be several outstanding checkpointed forwards
at once (F1, F2, then B1, B2). A later forward overwrites the shared holder
before an earlier one is recomputed, so the earlier recompute consumes top-k
support belonging to the wrong forward -- silently wrong logits, no error.

Upstream fixes this by creating one private carrier per ``checkpointed_forward``
invocation and capturing it in each activation-checkpoint closure, so the
carrier is re-entered at recompute time. This module applies exactly that
change at runtime.

Scope: this only bites when ``packed_seq_params is None`` (SkyRL RL training
defaults to ``trainer.remove_microbatch_padding=True``, which packs), the model
is DSA, and ``dsa_indexer_topk_freq > 1``. It is a no-op everywhere else, which
is why it is safe to apply unconditionally from the Megatron worker.

The edit itself lives in ``dsa_index_share_recompute.patch`` next to this file,
byte-identical to the upstream commit so it can be refreshed and diffed
mechanically::

    git show <rev> -- megatron/core/recompute.py \\
        > skyrl/backends/skyrl_train/patches/megatron/dsa_index_share_recompute.patch

That diff is applied **in memory** to the function's source, not to the file on
disk. Under ``uv run --isolated`` megatron-core's files are hardlinked into the
shared uv cache, and the module is already imported by the time the Megatron
worker builds a model -- so an on-disk edit would need a reload, and would still
leave ``transformer_block``/``hybrid_block`` holding the old function. Applying
in memory avoids both, plus the file locking that concurrent Ray actors sharing
one environment would otherwise need.

``checkpointed_forward`` is imported *by name* into those two modules, so the
rebind below must cover them as well -- rebinding ``megatron.core.recompute``
alone would silently do nothing. It finds them by scanning ``sys.modules`` for
the old function object rather than by hardcoding names, so a new importer in a
future megatron-core is picked up too.

The diff is applied by GNU ``patch`` itself, to a throwaway copy of
``recompute.py`` under a temp directory -- so hunk context *and* line numbers are
validated by a battle-tested implementation rather than by hand-rolled parsing.
``--forward`` means an already-applied or mismatched patch is refused outright
instead of producing a mangled file. Only the patched ``checkpointed_forward``
is then compiled into the live module.

DELETE THIS PATCH once the megatron-core pin includes NVIDIA/Megatron-LM#6793.
It is a temporary backport of an upstream fix, not a SkyRL behavior change. It
fails safe in the meantime: if the patch does not apply cleanly this logs and
no-ops rather than misfiring, and it detects the real fix and steps aside. Check
this file when bumping the pin.
"""

import ast
import inspect
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator, Optional

from loguru import logger

_PATCHED_FLAG = "_skyrl_dsa_index_share_patched"

# Set by upstream once NVIDIA/Megatron-LM#6793 lands; its presence means the
# installed megatron-core already has the fix and this module should be deleted.
_UPSTREAM_SENTINEL = "_dsa_index_share_carrier_scope"

_PATCH_FILE = Path(__file__).with_name("dsa_index_share_recompute.patch")


# Path the patch's headers are relative to, recreated under a temp directory so
# `patch -p1` resolves `a/megatron/core/recompute.py` without touching the real
# install.
_PATCH_TARGET = Path("megatron") / "core" / "recompute.py"

_FUNCTION_NAME = "checkpointed_forward"


def _extract_function(module_source: str, name: str) -> Optional[str]:
    """Return the source of the top-level function ``name``, or None if absent."""
    try:
        tree = ast.parse(module_source)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(module_source, node)
    return None


def _apply_patch_file(module_source: str) -> Optional[str]:
    """Apply the shipped .patch to ``module_source`` with GNU patch, in a temp dir.

    Returns the patched module source, or None if the patch does not apply
    cleanly -- which is the fail-safe path: `patch --forward` refuses an
    already-applied or mismatched patch rather than producing a mangled file.
    """
    executable = shutil.which("patch")
    if executable is None:
        logger.warning("`patch` is not on PATH; skipping DSA index-share patch")
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / _PATCH_TARGET
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(module_source)
        result = subprocess.run(
            # --forward: refuse a reversed/already-applied patch instead of
            # "un-applying" it. -r -: never leave .rej files behind.
            [executable, "-p1", "--batch", "--forward", "--silent", "-r", "-", "-i", str(_PATCH_FILE)],
            cwd=tmpdir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.info(
                "{} does not apply to the installed megatron-core (patch exit {}: {}); "
                "skipping DSA index-share patch. If megatron-core now includes "
                "NVIDIA/Megatron-LM#6793, delete this patch module.",
                _PATCH_FILE.name,
                result.returncode,
                (result.stderr or result.stdout or "").strip().replace("\n", "; ") or "no output",
            )
            return None
        return target.read_text()


class _DSAIndexShareCarrier:
    """Per-forward DSA index-share carrier for non-packed activation recompute."""


_CURRENT_DSA_INDEX_SHARE_CARRIER: ContextVar[Optional[object]] = ContextVar(
    "current_dsa_index_share_carrier", default=None
)


@contextmanager
def _dsa_index_share_carrier_scope(carrier: object) -> Iterator[None]:
    """Expose one non-packed forward's index-share carrier to its checkpoint closures."""
    token = _CURRENT_DSA_INDEX_SHARE_CARRIER.set(carrier)
    try:
        yield
    finally:
        _CURRENT_DSA_INDEX_SHARE_CARRIER.reset(token)


def _make_index_share_carrier_getter(original):
    """Build the replacement ``DSAttention._get_index_share_carrier``.

    Identical to upstream: the packed carrier still wins, the per-forward carrier
    is consulted next, and the pre-existing ``attention_mask``/``config`` fallback
    is kept for forwards outside a checkpointed scope.
    """

    def _get_index_share_carrier(self, packed_seq_params, attention_mask):
        """Return the object that carries DSA top-k sharing state for this forward."""
        if packed_seq_params is not None:
            return packed_seq_params
        carrier = _CURRENT_DSA_INDEX_SHARE_CARRIER.get()
        if carrier is not None:
            return carrier
        return attention_mask if attention_mask is not None else self.config

    _get_index_share_carrier.__doc__ = original.__doc__ or _get_index_share_carrier.__doc__
    return _get_index_share_carrier


def patch_dsa_index_share(force: bool = False) -> bool:
    """Isolate the DSA index-share holder per checkpointed forward.

    No-ops (returning False) when megatron-core is not importable, when it has no
    DSA attention variant, when it already contains NVIDIA/Megatron-LM#6793, or
    when ``checkpointed_forward`` no longer matches the shipped patch's context.

    Pass ``force=True`` to patch even when the upstream fix is detected; this
    exists for tests and has no reason to be used in production.

    Safe to call more than once; the second call is a no-op returning True.
    """
    try:
        from megatron.core import recompute
    except ImportError:
        logger.debug("megatron.core not importable; skipping DSA index-share patch")
        return False

    target = getattr(recompute, "checkpointed_forward", None)
    if target is None:
        logger.warning("megatron.core.recompute has no checkpointed_forward; skipping DSA index-share patch")
        return False

    if getattr(target, _PATCHED_FLAG, False):
        return True

    try:
        from megatron.core.transformer.experimental_attention_variant import (
            dsa as dsa_module,
        )
    except ImportError:
        logger.debug("megatron-core has no DSA attention variant; skipping DSA index-share patch")
        return False

    if not force and hasattr(dsa_module, _UPSTREAM_SENTINEL):
        # The pin moved past NVIDIA/Megatron-LM#6793, so this module is dead weight.
        logger.info(
            "megatron-core already isolates the DSA index-share carrier "
            "(NVIDIA/Megatron-LM#6793); skipping DSA index-share patch. "
            "Delete skyrl/backends/skyrl_train/patches/megatron/patch_dsa_index_share.py."
        )
        return False

    attention_cls = getattr(dsa_module, "DSAttention", None)
    if attention_cls is None or not hasattr(attention_cls, "_get_index_share_carrier"):
        logger.warning("DSAttention._get_index_share_carrier is missing; skipping DSA index-share patch")
        return False

    if not _PATCH_FILE.is_file():
        logger.warning("{} is missing; skipping DSA index-share patch", _PATCH_FILE)
        return False

    try:
        module_source = inspect.getsource(recompute)
    except (OSError, TypeError):
        logger.warning("Cannot read megatron-core recompute.py source; skipping DSA index-share patch")
        return False

    # Resolve the whole edit before mutating anything, so a source drift cannot
    # leave the DSA module half-patched. The patch is applied to the whole file so
    # GNU patch validates hunk line numbers as well as context.
    patched_module = _apply_patch_file(module_source)
    if patched_module is None:
        return False

    patched_source = _extract_function(patched_module, _FUNCTION_NAME)
    if patched_source is None:
        logger.warning("Patched recompute.py has no {}; skipping DSA index-share patch", _FUNCTION_NAME)
        return False

    # Publish the carrier machinery on megatron-core's own DSA module: the
    # recompiled function imports these names from there, exactly as upstream does.
    dsa_module._DSAIndexShareCarrier = _DSAIndexShareCarrier
    dsa_module._CURRENT_DSA_INDEX_SHARE_CARRIER = _CURRENT_DSA_INDEX_SHARE_CARRIER
    dsa_module._dsa_index_share_carrier_scope = _dsa_index_share_carrier_scope
    attention_cls._get_index_share_carrier = _make_index_share_carrier_getter(attention_cls._get_index_share_carrier)

    # Execute in megatron-core's own module namespace so the recompiled function
    # keeps the live module globals it depends on (tensor_parallel, te_checkpoint,
    # get_fp8_context, ...), and so the rebind lands on the module.
    filename = getattr(recompute, "__file__", None) or "<string>"
    # Pad so the compiled function keeps its real first line, otherwise every
    # traceback frame inside it points at line 1 of recompute.py. Lines after the
    # patch's own insertion are still shifted by the inserted count.
    padding = "\n" * max(getattr(target, "__code__", None).co_firstlineno - 1, 0)
    exec(compile(padding + patched_source, filename, "exec"), recompute.__dict__)  # noqa: S102

    patched = recompute.checkpointed_forward
    if patched is target:
        logger.warning("Failed to rebind megatron-core checkpointed_forward; DSA index-share patch not applied")
        return False
    setattr(patched, _PATCHED_FLAG, True)

    # Modules that did `from megatron.core.recompute import checkpointed_forward`
    # hold their own reference, so rebinding `recompute` alone would leave them on
    # the unpatched function. Today that is transformer_block and hybrid_block, but
    # this scans by identity rather than by name so a new importer in a future
    # megatron-core is covered too. Anything not yet imported picks up the patched
    # function through `recompute` when it is.
    for module in list(sys.modules.values()):
        if getattr(module, "checkpointed_forward", None) is target:
            module.checkpointed_forward = patched

    logger.info("Applied megatron-core DSA index-share patch (NVIDIA/Megatron-LM#6793 backport)")
    return True

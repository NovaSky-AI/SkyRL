"""HybridModel.forward's ``output_processor`` hook, backported from NVIDIA/Megatron-LM#7054 (57103cf063).

The fused LM head scores log-probs through it; without it the head refuses HybridModel with MTP (e.g. GLM-5 under
megatron-bridge 0.8). Upstream's hunks are inlined below and inserted into the pinned forward's source at load time.
The result is swapped into megatron-core's own function object, under any ``functools.wraps`` wrappers other patches
installed first (DeepSeek-V4.1's per-forward state), so they stay installed and wrap the hooked forward, as they
would upstream's. Delete this module, its call in megatron_worker.py and its test once the pinned megatron-core
includes NVIDIA/Megatron-LM#7054 (57103cf063).
"""

import inspect
import shutil
import subprocess
import tempfile
import textwrap
import types
import typing
from pathlib import Path

from loguru import logger


class PatchDoesNotApply(RuntimeError):
    """The diff does not apply to the given source."""


def apply_patch(source: str, diff: str) -> str:
    """Return ``source`` with ``diff`` applied; raise ``PatchDoesNotApply`` if it does not apply.

    GNU patch does the work, on a throwaway copy (as in ``patch_dsa_index_share``): the real install is never
    touched. Naming the file explicitly means the diff's ``a/...`` headers are ignored, so a function's own source
    can be patched directly. ``--forward`` refuses an already-applied or reversed patch and ``-F 0`` allows no
    context fuzz, so anything but an exact context match fails.
    """
    executable = shutil.which("patch")
    if executable is None:
        raise PatchDoesNotApply("`patch` is not on PATH")
    with tempfile.TemporaryDirectory() as tmpdir:
        scratch = Path(tmpdir) / "source.py"
        scratch.write_text(source)
        result = subprocess.run(
            [executable, str(scratch), "--batch", "--forward", "--silent", "-F", "0", "-r", "-"],
            input=diff,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            output = (result.stderr or result.stdout or "").strip().replace("\n", "; ") or "no output"
            raise PatchDoesNotApply(f"patch exit {result.returncode}: {output}")
        return scratch.read_text()


# Verbatim from NVIDIA/Megatron-LM 57103cf063, megatron/core/models/hybrid/hybrid_model.py (PR #7054): its three
# HybridModel.forward hunks. Copyright NVIDIA, Apache-2.0.
_UPSTREAM_7054_HUNKS = '''--- a/forward.py
+++ b/forward.py
@@ -557,7 +557,9 @@ class HybridModel(LanguageModule, GraphableMegatronModule):
         padding_mask: Optional[Tensor] = None,
         compute_mtp_loss: bool = True,
         cp_batch: ContextParallelBatch | None = None,
-    ) -> Tensor:
+        output_processor: Optional[Callable[..., Any]] = None,
+        output_processor_context: Optional[Any] = None,
+    ) -> Any:
         """Forward function of the Hybrid model. This function passes the input tensors
         through the embedding layer, and then the decoder and finally into the post
         processing layer (optional).
@@ -571,6 +573,12 @@ class HybridModel(LanguageModule, GraphableMegatronModule):
                 ``labels`` still determine whether the model returns loss or logits.
                 Defaults to True.
             cp_batch: Input tensors and packed metadata keyed by CP layout.
+            output_processor (Callable, optional): Caller-provided post-decoder hook. When set,
+                it receives the decoder output and output-layer/loss helpers, and its return
+                value is returned directly. The default output projection, logits/loss
+                computation, and inference postprocessing are skipped.
+            output_processor_context (Any, optional): Opaque caller-owned context passed to
+                ``output_processor`` as ``context``.
         """
         # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
         # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.
@@ -842,6 +850,26 @@ class HybridModel(LanguageModule, GraphableMegatronModule):
                     ),
                     main_hidden_states=hidden_states,
                 )
+        # Give callers a post-decoder hook before the default output projection and loss path.
+        if output_processor is not None:
+            return output_processor(
+                hidden_states=hidden_states,
+                output_layer=self.output_layer,
+                output_weight=output_weight,
+                labels=labels,
+                loss_mask=loss_mask,
+                input_ids=input_ids,
+                position_ids=position_ids,
+                attention_mask=attention_mask,
+                decoder_input=decoder_input,
+                inference_context=inference_context,
+                packed_seq_params=packed_seq_params,
+                runtime_gather_output=runtime_gather_output,
+                context=output_processor_context,
+                compute_language_model_loss=self.compute_language_model_loss,
+                scale_logits=self._scale_logits,
+                config=self.config,
+            )
         sequence_parallel_override = False
         if (
             in_inference_mode
'''

# Patched forward -> its code, defaults, keyword defaults and annotations from before the patch.
_ORIGINALS: dict[types.FunctionType, tuple[types.CodeType, tuple | None, dict | None, dict]] = {}


def apply_hybrid_output_processor_patch() -> bool:
    """Insert upstream's hunks into ``HybridModel.forward``; True if the hook is installed after the call."""
    try:
        from megatron.core.models.hybrid import hybrid_model
    except ImportError:
        return False

    # megatron-core's own function, under any functools.wraps wrappers (importing model_bridges installs one).
    forward = inspect.unwrap(hybrid_model.HybridModel.forward)
    if forward in _ORIGINALS:
        return True
    if "output_processor" in inspect.signature(forward).parameters:
        logger.info("megatron-core includes NVIDIA/Megatron-LM#7054; delete patch_hybrid_output_processor.py")
        return False
    try:
        # The hunks carry the method's class indentation, so patch first and dedent after.
        source = textwrap.dedent(apply_patch(inspect.getsource(forward), _UPSTREAM_7054_HUNKS))
    except (PatchDoesNotApply, OSError, TypeError) as e:
        logger.warning(f"HybridModel output_processor patch does not apply to this megatron-core ({e}); skipping")
        return False

    # Upstream's signature annotates the new arguments with Any / Callable, which it imports at module level.
    hybrid_model.__dict__.setdefault("Any", typing.Any)
    hybrid_model.__dict__.setdefault("Callable", typing.Callable)
    # Compile in megatron-core's module namespace so forward keeps its live globals. Pad to the real first line so
    # tracebacks point into hybrid_model.py (lines after the hook are shifted by the inserted count).
    code = forward.__code__
    padding = "\n" * (code.co_firstlineno - 1)
    namespace: dict = {}
    exec(compile(padding + source, code.co_filename, "exec"), hybrid_model.__dict__, namespace)
    patched = namespace["forward"]
    original = (code, forward.__defaults__, forward.__kwdefaults__, dict(forward.__annotations__))
    # Swap the patched code into the existing function so wrappers that hold it call the hooked forward. Both closures
    # must match; megatron-core's forward has none (no super()).
    try:
        forward.__code__ = patched.__code__.replace(co_qualname=code.co_qualname)
    except ValueError as e:
        logger.warning(f"HybridModel output_processor patch cannot replace this forward's code ({e}); skipping")
        return False
    forward.__defaults__ = patched.__defaults__
    forward.__kwdefaults__ = patched.__kwdefaults__
    # In place: functools.wraps wrappers share this dict.
    forward.__annotations__.update(patched.__annotations__)
    _ORIGINALS[forward] = original
    logger.info("Applied the HybridModel output_processor patch (NVIDIA/Megatron-LM#7054 backport)")
    return True

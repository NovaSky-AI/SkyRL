"""Runtime patch: give vLLM's AOT compile artifact a per-device directory.

Problem
-------
vLLM keys its AOT compile artifact on everything *except* the GPU it was built
for, then stores it under a directory that only distinguishes rank and DP rank
(``vllm/compilation/decorators.py``)::

    factors = aot_compile_hash_factors(self.vllm_config)     # [env_hash, config_hash]
    factors.append(_model_hash_key(self.forward))
    hash_key = hashlib.sha256(str(factors).encode()).hexdigest()
    cache_dir = os.path.join(envs.VLLM_CACHE_ROOT, "torch_compile_cache",
                             "torch_aot_compile", hash_key)
    ...
    rank = self.vllm_config.parallel_config.rank
    dp_rank = self.vllm_config.parallel_config.data_parallel_index
    cache_dir = os.path.join(cache_dir, f"rank_{rank}_{dp_rank}")
    aot_compilation_path = os.path.join(cache_dir, "model")

Every ``TP=1`` engine on a node is ``rank_0_0``, so *all* of them share the one
path ``<hash>/rank_0_0/model`` even when they sit on different GPUs. Saves race
and the last one wins (``os.replace``); every later engine loads it, and any
engine that is not on the last saver's GPU dies::

    [decorators.py:311] Directly load AOT compilation from path .../rank_0_0/model
    RuntimeError: CUDA driver error: invalid argument
        static_triton_launcher._launch_kernel  <-  profile_run -> _dummy_run

Why the *path* and not the key
------------------------------
The artifact blob itself is device-agnostic -- measured, it is a plain protocol-4
pickle with zero ``get_raw_stream`` occurrences. What makes it device-specific is
that it references Inductor cache entries *by key*, and the generated wrapper
behind each key does bake the device in (``get_raw_stream(<index>)``). Loading
the blob on another GPU resolves those keys to the saver's wrappers.

That means the shared, hash-level ``inductor_cache/`` is **not** the problem and
must stay shared: it is already device-partitioned internally. Measured for one
TP=2 engine, a single ``inductor_cache/`` held

  * ``triton/0/...`` and ``triton/1/...``   -- Triton keys its own cache by device
    index already (164 each of ``.ptx`` / ``.cubin`` / ``.ttir`` / ``.ttgir`` /
    ``.llir`` / ``.source``, 328 ``.json``)
  * 45 generated ``.py`` wrappers baking ``get_raw_stream(0)`` **and** 45 baking
    ``get_raw_stream(1)`` -- Inductor's graph hash covers input tensor devices, so
    the two devices never collide
  * device-independent leftovers shared by both ranks: ``fxgraph/``,
    ``aotautograd/``, ``.best_config``, ``.kernel_perf``

So exactly one thing needs a device component: the per-rank artifact directory.
Suffixing it keeps the hash-level directory -- and all of the above sharing --
intact, which is both cheaper and tidier than moving the whole tree.

The sharing is sound because the node's GPUs are the same model: Inductor's
``get_system()`` records the device *name*, not the index or UUID, so two
different GPU models on one node would collide over ``.best_config`` and the
``triton/<idx>/`` cubins. That is an upstream torch/vLLM property, unchanged by
this patch, and it does not arise on SkyRL's homogeneous CI and training nodes.

Why the device *index* and not the UUID
---------------------------------------
The index is precisely the quantity that gets baked into the generated code, so
it is exactly the right equivalence class. Two workers that both run on their own
device 0 under ``CUDA_VISIBLE_DEVICES`` masking (vLLM's ``mp`` executor) emit
``get_raw_stream(0)`` and their artifacts *are* interchangeable -- keying on UUID
would split those needlessly. Two workers on true indices 1 and 2 (vLLM's ``ray``
executor, which sets ``RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES`` and then
"CUDA_VISIBLE_DEVICES is never modified" -- ``v1/executor/ray_executor_v2.py``)
emit different handles and must be split. Indexing gets both cases right.

Scope of the bug
----------------
Only the AOT tree is affected. The Dynamo tree (``torch_compile_cache/<10-hex>/``)
is keyed on ``code_hash``, which covers the traced files' absolute paths; under
``uv run --isolated`` every process gets a fresh ``~/.cache/uv/builds-v0/.tmpXXXX``
venv, so that tree never gets a hit at all (measured: a warm re-run minted a new
top-level directory). And TP>1 within a *single* engine was always safe -- ranks
differ, so the directories differ (measured: warm TP=2 loaded
``rank_0_0/model`` and ``rank_1_0/model`` respectively, zero errors).

Reproduced 2026-09-08 on 4x L40S (sm_89, matching the L4 CI box), vLLM 0.28.0,
torch 2.11.0+cu130, at SkyRL 0b286bac. It needs the *whole*
``tests/tinker/skyrl_train/`` directory: a single test file starts one engine, so
nothing is ever reused and the suite passes. Five engines, all ``rank_0_0``, one
byte-identical AOT key, spread over devices 1, 2, 1, 2, 3::

    12:23:47  dev 2  saved
    12:33:19  dev 1  saved      <- overwrote it
    12:36:19  dev 1  loaded  -> ok        (loader == last saver)
    12:40:46  dev 2  loaded  -> CUDA driver error
    12:42:00  dev 3  loaded  -> CUDA driver error

Same-device reuse is fine, which is why this presents as a flake: it turns purely
on whether an engine lands on the last saver's GPU.

Regressed by SkyRL #2167 (599ff343, "Re-enable vllm compile cache"), which stopped
force-setting ``VLLM_DISABLE_COMPILE_CACHE=1``. vLLM then enables AOT compile by
default on torch >= 2.10 (``envs.use_aot_compile``), which is what put the AOT tree
in play. That PR's own experiments did not surface it: exp2 was TP=2 (per-rank
directories differ) and exp1 was 8 TP=1 engines on 8 GPUs, one engine per GPU.

Upstream vllm#38962 fixed this the same way -- device index in the per-rank cache
path -- but it was reverted four hours later by vllm#53304 (it broke CPU startup by
querying ``torch.accelerator``), and the re-land vllm#53312 is still open. No
released vLLM, nor current main, has a device component here.

TODO: drop once vllm#53312 (or equivalent) ships.
"""

import os
from typing import Any

from loguru import logger

_PATCHED = False
_DEVICE_ATTR = "_skyrl_aot_device_path"
_WRAPPED_FLAG = "_skyrl_aot_save_wrapped"


def _device_tag() -> str | None:
    """Tag for the device this worker compiles on, or None when there is no GPU."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return f"dev{torch.cuda.current_device()}"
    except Exception as e:
        # Never let cache pathing break engine startup; without the tag we only
        # lose the isolation this patch adds.
        logger.debug(f"compile-cache device path: could not resolve device ({e})")
        return None


def _device_scoped_path(path: str) -> str:
    """``.../rank_0_0/model`` -> ``.../rank_0_0_dev1/model`` (unchanged on CPU)."""
    tag = _device_tag()
    if tag is None:
        return path
    parent, name = os.path.split(path)
    rank_dir = os.path.basename(parent)
    if not rank_dir.startswith("rank_"):
        # Layout moved under us; leave it alone rather than invent a directory.
        logger.warning(f"compile-cache device path: unexpected AOT layout {path!r}, not scoping")
        return path
    if rank_dir.endswith(f"_{tag}"):
        return path
    return os.path.join(os.path.dirname(parent), f"{rank_dir}_{tag}", name)


def _install_save_redirect(cls: type) -> None:
    """Point ``save_aot_compiled_function`` at the device-scoped path.

    ``support_torch_compile`` attaches this method per decorated class, so there
    is no module-level function to patch; we wrap it on first load instead, which
    is safe because vLLM always attempts the load before it compiles and saves.
    """
    # Flag the wrapper rather than the class: a subclass that merely *inherits*
    # an already-wrapped method resolves to it and is skipped, while a subclass
    # that `support_torch_compile` gave its own unwrapped method still gets
    # wrapped. A class-level flag would confuse those two cases.
    original_save = cls.save_aot_compiled_function
    if getattr(original_save, _WRAPPED_FLAG, False):
        return

    def save_aot_compiled_function(self: Any) -> None:
        scoped = getattr(self, _DEVICE_ATTR, None)
        if scoped is not None:
            self._aot_compilation_path = scoped
            self._aot_cache_dir = os.path.dirname(scoped)
        return original_save(self)

    setattr(save_aot_compiled_function, _WRAPPED_FLAG, True)
    cls.save_aot_compiled_function = save_aot_compiled_function


def apply_compile_cache_device_path_patch() -> None:
    """Scope vLLM's AOT compile artifact directory to the running GPU (idempotent).

    Wraps the module-level loader, which vLLM looks up by name at call time, so
    the redirect covers both halves of the round trip: the load reads the
    device-scoped directory, and the save is redirected to write it.
    """
    global _PATCHED
    if _PATCHED:
        return

    import vllm.compilation.decorators as decorators

    original_try_load = decorators._try_load_aot_compiled_fn

    def try_load_aot_compiled_fn(model: Any, path: str) -> Any:
        scoped = _device_scoped_path(path)
        # Stash before loading: on a miss vLLM compiles and then saves, and the
        # save must land in the same place the load looked.
        try:
            setattr(model, _DEVICE_ATTR, scoped)
            _install_save_redirect(type(model))
        except Exception as e:
            logger.debug(f"compile-cache device path: could not install save redirect ({e})")
            return original_try_load(model, path)
        return original_try_load(model, scoped)

    decorators._try_load_aot_compiled_fn = try_load_aot_compiled_fn
    _PATCHED = True
    logger.info("Patched vLLM AOT compile cache to use a per-device artifact directory")

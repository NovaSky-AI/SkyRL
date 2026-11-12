"""Inkling2 on SkyRL's vLLM: serve the model through :class:`model.Inkling2ForRL`.

The pinned vLLM (a build of the Inkling2 vLLM branch, see ``pyproject.toml``) ships the model
(``vllm.models.inkling2``), its config classes and its ``inkling2`` quantization config. This
plugin only swaps the architecture's class for ``model.Inkling2ForRL``, which adds what RL
training needs: BF16 (or MXFP8) routed experts that take BF16 weights, and a ``load_weights``
that accepts the partial weight streams of weight sync and loads them in place.

Loaded in every vLLM process through the ``vllm.general_plugins`` entry point; a no-op on a vLLM
without Inkling2.
"""

import importlib.util
import os

ARCH = "Inkling2ForConditionalGeneration"

_registered = False


def vllm_general_plugin() -> None:
    global _registered
    # vLLM's Ray workers are not children of the server actor, so they get Ray's default open-file
    # limit; NCCL weight sync over InfiniBand needs more.
    from skyrl.utils.rlimits import raise_open_file_limit

    raise_open_file_limit()
    if _registered or importlib.util.find_spec("vllm.models.inkling2") is None:
        return
    # vLLM's LoRA config rejects every "Inkling*" architecture unless this is 0, for Inkling v1's
    # fused all-reduce, which bypasses LoRA layers. Inkling2 does not have that path.
    os.environ.setdefault("INKLING_MULTIMEM_AR", "0")

    from vllm import ModelRegistry

    ModelRegistry.register_model(ARCH, "skyrl.backends.skyrl_train.patches.vllm.inkling2.model:Inkling2ForRL")
    _registered = True

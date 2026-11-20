"""Inkling2 on SkyRL's vLLM: serve the model through :class:`model.Inkling2ForRL`.

The pinned vLLM (a build of the Inkling2 vLLM branch, see ``pyproject.toml``) ships the model
(``vllm.models.inkling2``), its config classes and its ``inkling2`` quantization config. This
plugin only swaps the architectures' classes for ``model.Inkling2ForRL`` (and ``model.Inkling2LoopForRL`` for
looped checkpoints), which add what RL
training needs: BF16 (or MXFP8) routed experts that take BF16 weights, and a ``load_weights``
that accepts the partial weight streams of weight sync and loads them in place.

Loaded in every vLLM process through the ``vllm.general_plugins`` entry point; a no-op on a vLLM
without Inkling2.
"""

import importlib.util
import os

ARCHS = {
    "Inkling2ForConditionalGeneration": "Inkling2ForRL",
    "Inkling2LoopForConditionalGeneration": "Inkling2LoopForRL",
}

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

    for arch, cls in ARCHS.items():
        ModelRegistry.register_model(arch, f"skyrl.backends.skyrl_train.patches.vllm.inkling2.model:{cls}")
    _diff_loop_config_without_defaults()
    _registered = True


def _diff_loop_config_without_defaults() -> None:
    """Let Transformers print the build's looped config (``Inkling2LoopMMConfig``).

    ``PretrainedConfig.from_dict`` logs the config, and its ``to_diff_dict`` diffs it against a bare ``cls()``. The
    build's looped config requires a ``loop_config``, so the bare one raises and the checkpoint fails to load (under
    SkyRL's Transformers 5.16). ``has_no_defaults_at_init`` makes the diff skip that instance."""
    try:
        from vllm.models.inkling2.configs import Inkling2LoopMMConfig
    except ImportError:  # a build without looped checkpoints
        return
    Inkling2LoopMMConfig.has_no_defaults_at_init = True

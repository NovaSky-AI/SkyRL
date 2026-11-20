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
    _register_loop_model_type()
    _size_routed_experts_by_layer_runs()
    _registered = True


LOOP_MODEL_TYPE = "inkling2_loop"


def _register_loop_model_type() -> None:
    """Let vLLM read looped checkpoints' ``config.json`` (``model_type: inkling2_loop``, see ``loop_config``). No-op on
    builds that already register it."""
    from vllm.transformers_utils import config as config_utils

    if LOOP_MODEL_TYPE not in config_utils._CONFIG_REGISTRY:
        from skyrl.backends.skyrl_train.patches.vllm.inkling2.loop_config import (
            Inkling2LoopMMConfig,
        )

        config_utils._CONFIG_REGISTRY[LOOP_MODEL_TYPE] = Inkling2LoopMMConfig


def _size_routed_experts_by_layer_runs() -> None:
    """Routed-experts capture (R3) keeps one row per layer *run*: a looped checkpoint (``loop_config``) runs its core
    layers several times, routing each time, so its buffer holds ``num_layer_visits`` rows per token (in execution
    order, see ``model.Inkling2LoopModel``) instead of vLLM's ``num_hidden_layers``."""
    try:
        from vllm.model_executor.layers.fused_moe import (
            routed_experts_capturer as capturer,
        )
    except ImportError:
        return
    if getattr(capturer, "_skyrl_layer_runs", False):
        return
    shape = capturer._get_routed_experts_shape

    def _get_routed_experts_shape(vllm_config):
        num_layers, num_experts, topk = shape(vllm_config)
        text = vllm_config.model_config.hf_text_config
        loop = getattr(text, "loop_config", None)
        if loop is not None:
            num_layers = text.num_hidden_layers + (loop["num_passes"] - 1) * (loop["core_end"] - loop["core_start"])
        return num_layers, num_experts, topk

    capturer._get_routed_experts_shape = _get_routed_experts_shape
    capturer._skyrl_layer_runs = True

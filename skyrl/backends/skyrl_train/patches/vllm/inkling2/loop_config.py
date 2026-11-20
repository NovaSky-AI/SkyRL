"""vLLM config class of looped Inkling2 checkpoints (``model_type: inkling2_loop``) for vLLM builds without one.

The same fields as ``inkling2`` (``Inkling2MMConfig``), with ``text_config.loop_config``. A module-level class: vLLM
pickles the config into its engine processes.
"""

from vllm.models.inkling2.configs import Inkling2MMConfig

LOOP_MODEL_TYPE = "inkling2_loop"


class Inkling2LoopMMConfig(Inkling2MMConfig):
    model_type = LOOP_MODEL_TYPE

    # Explicit: Transformers applies a config class's own constructor specially (an inherited one leaves
    # ``text_config`` a dict).
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

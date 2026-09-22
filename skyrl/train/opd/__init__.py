"""On-policy distillation (OPD) for SkyRL.

The student samples its own rollouts; a frozen teacher served by an inference engine scores every
response token; the negative per-token reverse KL to the teacher becomes a dense advantage, on
its own (pure distillation) or on top of a task reward. Entry point:
``skyrl.train.entrypoints.main_opd``; run scripts under ``examples/train/on_policy_distillation``.

- ``teacher_client``: ``TeacherLogprobClient`` (abstract base) with the Fireworks and vLLM backends.
- ``trainer``: ``OPDTrainer``. Its ``generate`` runs one ``generator.generate`` call per prompt group
  concurrently and scores each group under the teacher as it finishes (any generator works); three
  more ``RayPPOTrainer`` overrides consume the teacher logprobs.
- ``config``: ``trainer.teacher`` and ``trainer.algorithm.opd`` blocks, ``OPDExpConfig``, ``validate_opd_cfg``.
- ``utils``: the pure helpers (batch splitting, padding, the advantage term).
"""

from skyrl.train.opd.config import (
    OPDAlgorithmConfig,
    OPDConfig,
    OPDExpConfig,
    OPDTrainerConfig,
    TeacherConfig,
    validate_opd_cfg,
)
from skyrl.train.opd.teacher_client import (
    FireworksTeacherClient,
    TeacherLogprobClient,
    VLLMTeacherClient,
)
from skyrl.train.opd.trainer import OPDTrainer
from skyrl.train.opd.utils import (
    TEACHER_LOGPROBS_KEY,
    apply_opd_to_advantages,
    pad_teacher_logprobs,
    split_generator_input,
)

__all__ = [
    "OPDAlgorithmConfig",
    "OPDConfig",
    "OPDExpConfig",
    "OPDTrainerConfig",
    "TeacherConfig",
    "validate_opd_cfg",
    "TEACHER_LOGPROBS_KEY",
    "FireworksTeacherClient",
    "TeacherLogprobClient",
    "VLLMTeacherClient",
    "OPDTrainer",
    "apply_opd_to_advantages",
    "pad_teacher_logprobs",
    "split_generator_input",
]

"""On-policy distillation (OPD) for SkyRL.

The student samples its own rollouts; a frozen teacher served by an inference engine scores every
response token; the negative per-token reverse KL to the teacher becomes a dense advantage, on
its own (pure distillation) or on top of a task reward. Entry point:
``skyrl.train.entrypoints.main_opd``; run scripts under ``examples/train/on_policy_distillation``.

- ``teacher_client``: ``TeacherLogprobClient`` (abstract base) with the launched (``SkyRLTeacherClient``
  over a ``RemoteInferenceClient``) and vLLM-URL backends.
- ``teacher_launch``: launching the teacher's vLLM deployment inside the job (``backend="skyrl"``).
- ``trainer``: ``OPDTrainer``. Its ``generate`` runs one ``generator.generate`` call per prompt group
  concurrently and scores each group under the teacher as it finishes (any generator works); three
  more ``RayPPOTrainer`` overrides consume the teacher logprobs.
- ``config``: ``trainer.teacher`` and ``trainer.algorithm.opd`` blocks, ``OPDExpConfig``, ``validate_opd_cfg``.
- ``utils``: the pure helpers (batch splitting, padding, the advantage term).
"""

from skyrl.train.opd.config import (
    TEACHER_BACKENDS,
    OPDAlgorithmConfig,
    OPDConfig,
    OPDExpConfig,
    OPDTrainerConfig,
    TeacherConfig,
    teacher_max_model_len,
    validate_opd_cfg,
)
from skyrl.train.opd.teacher_client import (
    SkyRLTeacherClient,
    TeacherLogprobClient,
    VLLMTeacherClient,
)
from skyrl.train.opd.teacher_launch import (
    launch_teacher,
    served_teacher_name,
    teacher_cli_args,
    teacher_start_port,
)
from skyrl.train.opd.trainer import OPDTrainer
from skyrl.train.opd.utils import (
    TEACHER_LOGPROBS_KEY,
    apply_opd_to_advantages,
    pad_teacher_logprobs,
    split_generator_input,
)

__all__ = [
    "TEACHER_BACKENDS",
    "OPDAlgorithmConfig",
    "OPDConfig",
    "OPDExpConfig",
    "OPDTrainerConfig",
    "TeacherConfig",
    "teacher_max_model_len",
    "validate_opd_cfg",
    "TEACHER_LOGPROBS_KEY",
    "SkyRLTeacherClient",
    "TeacherLogprobClient",
    "VLLMTeacherClient",
    "launch_teacher",
    "served_teacher_name",
    "teacher_cli_args",
    "teacher_start_port",
    "OPDTrainer",
    "apply_opd_to_advantages",
    "pad_teacher_logprobs",
    "split_generator_input",
]

"""On-policy distillation (OPD) for SkyRL.

Entry point: ``skyrl.train.entrypoints.main_opd``; read run scripts and documentation under
``examples/train/on_policy_distillation``.

- ``teacher_client``: ``TeacherLogprobClient`` (abstract base), which scores a ``[tokens, width]`` id matrix
  in one prefill and can return the teacher's own top-K, with the launched (``SkyRLTeacherClient`` over a
  ``RemoteInferenceClient``) and vLLM-URL backends.
- ``teacher_launch``: launching a teacher's vLLM deployment inside the job (``backend="skyrl"``).
- ``trainer``: ``OPDTrainer``. Its ``generate_on_input_batch`` routes each row to a teacher, runs one
  ``generator.generate`` call per row concurrently and scores each row under its teacher as it finishes (any
  generator works); three more ``RayPPOTrainer`` overrides consume the teacher logprobs.
- ``config``: ``trainer.teacher`` / ``trainer.teachers`` and ``trainer.algorithm.opd`` blocks, ``OPDExpConfig``,
  ``validate_opd_cfg``. A built config holds every teacher in ``trainer.teachers``.
- ``utils``: the pure helpers (batch splitting, padding, the advantage term).
- ``metrics``: the rollout diagnostics, and the top-K overlap metrics (``opd.topk_metrics``).
"""

from skyrl.train.opd.config import (
    DEFAULT_TEACHER,
    TEACHER_BACKENDS,
    OPDAlgorithmConfig,
    OPDConfig,
    OPDExpConfig,
    OPDTopKMetricsConfig,
    OPDTrainerConfig,
    TeacherConfig,
    teacher_logprob_width,
    teacher_path,
    teacher_topk,
    validate_opd_cfg,
)
from skyrl.train.opd.teacher_client import (
    CandidateScores,
    SkyRLTeacherClient,
    TeacherLogprobClient,
    VLLMTeacherClient,
)
from skyrl.train.opd.teacher_launch import launch_teacher, teacher_start_ports
from skyrl.train.opd.trainer import OPDTrainer
from skyrl.train.opd.utils import (
    TEACHER_INDEX_KEY,
    TEACHER_LOGPROBS_KEY,
    apply_opd_to_advantages,
    pad_teacher_logprobs,
    split_generator_input,
)

__all__ = [
    "DEFAULT_TEACHER",
    "TEACHER_BACKENDS",
    "OPDAlgorithmConfig",
    "OPDConfig",
    "OPDExpConfig",
    "OPDTopKMetricsConfig",
    "OPDTrainerConfig",
    "TeacherConfig",
    "teacher_logprob_width",
    "teacher_path",
    "teacher_topk",
    "validate_opd_cfg",
    "TEACHER_INDEX_KEY",
    "TEACHER_LOGPROBS_KEY",
    "CandidateScores",
    "SkyRLTeacherClient",
    "TeacherLogprobClient",
    "VLLMTeacherClient",
    "launch_teacher",
    "teacher_start_ports",
    "OPDTrainer",
    "apply_opd_to_advantages",
    "pad_teacher_logprobs",
    "split_generator_input",
]

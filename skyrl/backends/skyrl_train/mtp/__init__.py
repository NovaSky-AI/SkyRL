# Decoupled Multi-Token Prediction (MTP) draft-head training.
#
# This package mirrors the design used by NeMo-RL's draft-model training
# (https://github.com/NVIDIA-NeMo/RL, ``nemo_rl/models/megatron/draft`` and
# ``nemo_rl/algorithms/loss``): the trunk's hidden states are *detached* before
# the draft/MTP head runs (decoupling the draft gradient from the policy
# backbone), the head is supervised by an explicit, configurable loss
# (soft cross-entropy distillation against the policy's own next-token
# distribution, or hard next-token cross-entropy), and the two losses are
# combined legibly as ``policy_loss + w * draft_loss`` instead of being
# entangled inside Megatron's ``MTPLossAutoScaler``.
#
# The pieces here are backend-agnostic. Only the ``adapter`` wiring is
# backend-specific (Megatron today; FSDP is a planned follow-up).

from skyrl.backends.skyrl_train.mtp.draft_loss_wrapper import (
    DraftLossWrapper,
    combine_policy_and_draft_loss,
)
from skyrl.backends.skyrl_train.mtp.soft_ce import (
    build_teacher_logits,
    draft_hard_ce,
    draft_soft_ce,
)

__all__ = [
    "DraftLossWrapper",
    "combine_policy_and_draft_loss",
    "build_teacher_logits",
    "draft_hard_ce",
    "draft_soft_ce",
]

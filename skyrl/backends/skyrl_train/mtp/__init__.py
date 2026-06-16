# Decoupled Multi-Token Prediction (MTP) draft-head training.
#
# The trunk's hidden states are *detached* before the draft/MTP head runs
# (decoupling the draft gradient from the policy backbone), and the head is
# supervised by an explicit, configurable loss (soft cross-entropy distillation
# against the policy's own next-token distribution, or hard next-token
# cross-entropy) instead of being entangled inside Megatron's MTPLossAutoScaler.
#
# Consumers import directly from the submodules (``soft_ce``, ``hidden_capture``,
# ``adapter``); this package has no re-exports of its own.

# Decoupled Multi-Token Prediction (MTP) draft-head training.
#
# The trunk hidden states are detached before the MTP head runs, and the head is supervised by an
# explicit loss (soft-CE distillation against the policy's own next-token distribution, or hard CE)
# rather than Megatron's MTPLossAutoScaler. Consumers import directly from the submodules
# (``soft_ce``, ``hidden_capture``, ``adapter``); this package has no re-exports of its own.

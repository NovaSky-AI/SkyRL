"""RLM environment subclasses for the evidence-selection example task.

Importing this package registers ``evidence_rlm`` with skyrl_gym so it can be
selected via ``environment.env_class=evidence_rlm`` in training configs.
"""

from skyrl_gym.envs.registration import register

register(
    id="evidence_rlm",
    entry_point="examples.train.rlm.envs.evidence_rlm_env:EvidenceRLMEnv",
)

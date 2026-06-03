"""RLM environment subclasses for the evidence-selection example task.

Importing this package registers two env ids with skyrl_gym:
- ``evidence_rlm``: single-paper worker (always uses the worker prompt).
- ``multipaper_evidence_rlm``: parent/child orchestrator; the env reads
  ``extras["depth"]`` (stamped by the generator) to pick the parent prompt
  at depth 0 and the child prompt at depth >= 1.
"""

from skyrl_gym.envs.registration import register

register(
    id="evidence_rlm",
    entry_point="examples.train.rlm.multi_paper_env.evidence_rlm_env:EvidenceRLMEnv",
)
register(
    id="multipaper_evidence_rlm",
    entry_point="examples.train.rlm.multi_paper_env.evidence_rlm_env:MultipaperEvidenceRLMEnv",
)

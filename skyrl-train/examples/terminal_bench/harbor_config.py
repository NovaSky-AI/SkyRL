"""
Harbor configuration builder for SkyRL.

This module provides a pass-through configuration system that allows users to specify
Harbor-native configuration in YAML format, with SkyRL only injecting minimal runtime values.

Users can configure any Harbor option directly, and SkyRL stays "blind" to most Harbor internals.
"""

from copy import deepcopy
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from harbor.models.trial.config import TrialConfig


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Deep merge override into base dict. Override values take precedence.

    Args:
        base: Base dictionary
        override: Dictionary with values to merge in (takes precedence)

    Returns:
        New merged dictionary
    """
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


class HarborConfigBuilder:
    """
    Builds Harbor TrialConfig instances from user-provided configuration.

    The user provides a partial TrialConfig in YAML/DictConfig format, and this builder
    merges in the runtime-specific values that SkyRL needs to inject:

    Runtime-injected values (SkyRL provides these):
        - task.path: The task/prompt path
        - agent.model_name: Constructed as "hosted_vllm/{served_model_name}"
        - agent.kwargs.api_base: SkyRL's inference endpoint
        - agent.kwargs.session_id: Generated per-trial for sticky routing

    User can configure any Harbor-supported option in the YAML config.
    """

    def __init__(self, harbor_cfg: DictConfig):
        """
        Initialize the builder with user's Harbor configuration.

        Args:
            harbor_cfg: DictConfig containing the user's Harbor configuration.
                       This can be a partial TrialConfig directly, or contain a
                       "harbor" key with the partial TrialConfig.
        """
        # Convert to plain dict for easier manipulation
        self._harbor_template = OmegaConf.to_container(harbor_cfg, resolve=True)

    def build_trial_config(
        self,
        task_path: str | Path,
        model_name: str,
        api_base: str,
        session_id: str,
    ) -> TrialConfig:
        """
        Build a complete TrialConfig by merging user config with runtime values.

        Args:
            task_path: Path to the task (injected by SkyRL from prompt)
            model_name: Full model name (e.g., "hosted_vllm/model-alias")
            api_base: API base URL for the inference endpoint
            session_id: Session ID for sticky routing

        Returns:
            Complete TrialConfig ready for Harbor's Trial class
        """
        # Start with user's template
        config_dict = deepcopy(self._harbor_template)

        # Build the runtime overrides
        runtime_overrides = {
            "task": {
                "path": str(task_path),
            },
            "agent": {
                "model_name": model_name,
                "kwargs": {
                    "api_base": api_base,
                    "session_id": session_id,
                },
            },
        }

        # Deep merge: user config is base, runtime overrides take precedence
        merged = _deep_merge(config_dict, runtime_overrides)

        # Validate and create the TrialConfig via Pydantic
        return TrialConfig.model_validate(merged)

    @property
    def trials_dir(self) -> str:
        """Get the configured trials directory."""
        return self._harbor_template.get("trials_dir", "trials")

    @property
    def agent_name(self) -> str | None:
        """Get the configured agent name (for logging/debugging)."""
        agent_cfg = self._harbor_template.get("agent", {})
        return agent_cfg.get("name")

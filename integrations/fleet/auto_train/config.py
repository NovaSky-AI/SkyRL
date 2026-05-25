"""Auto-train configuration constants."""

from __future__ import annotations

SUPPORTED_MODALITIES: tuple[str, ...] = ("tool_use", "browser_use")

# S3
S3_DATASET_BUCKET = "fleet-internal-datasets"
S3_STATE_KEY = ".auto_train_state.json"
S3_DATASET_PATH_TEMPLATE = "{project_key}/openenv/all_{modality}.json"
AWS_DEFAULT_REGION = "us-east-1"

# Modality detection — the Supabase task_modality field is wrong for many
# "computer_use" entries. Real desktop computer_use envs all start with this
# prefix; anything else marked computer_use is actually browser_use.
COMPUTER_USE_ENV_PREFIX = "fos-"

# Excluded envs (from v6 pipeline: broken or unsuitable for training)
EXCLUDED_ENVS: frozenset[str] = frozenset({"google-maps", "carlisle"})

# Modality -> SkyPilot task YAML (relative to repo root)
MODALITY_YAML_MAP: dict[str, str] = {
    "tool_use": "tasks/openenv-fleet-grpo-qwen3_5-35b.yaml",
    "browser_use": "tasks/openenv-fleet-grpo-vl.yaml",
}

# Fields written to the OpenEnv JSON (must match what fleet-common-setup.sh
# downloads and what prepare_dataset.py reads).
OPENENV_FIELDS: tuple[str, ...] = (
    "task_key",
    "prompt",
    "env_key",
    "env_version",
    "data_key",
    "data_version",
    "verifier_code",
    "task_modality",
    "env_variables",
)

# Smoke test
SMOKE_SAMPLE_ENVS = 2
SMOKE_TTL_SECONDS = 300
SMOKE_RETRIES = 3

# Slack
SLACK_CHANNEL_DEFAULT = "#fleet-training-runs"

# Fleet Research team (used as default team filter for the auto-train trigger)
FLEET_RESEARCH_TEAM_ID = "042fd173-2a6d-4818-a025-df1d7df3abb6"

# Credential env vars required by fleet-launch.sh / fleet-preflight.sh
REQUIRED_LAUNCH_ENV_VARS: tuple[str, ...] = (
    "FLEET_API_KEY",
    "WANDB_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
)

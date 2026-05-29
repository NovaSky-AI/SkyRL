"""Auto-train configuration constants."""

from __future__ import annotations

import os

SUPPORTED_MODALITIES: tuple[str, ...] = ("tool_use", "browser_use", "computer_use")

# S3
S3_DATASET_BUCKET = "fleet-internal-datasets"
S3_STATE_KEY = ".auto_train_state.json"
S3_DATASET_PATH_TEMPLATE = "{dataset_key}/openenv/all_{modality}.json"
AWS_DEFAULT_REGION = "us-east-1"

# Modality detection — the Supabase task_modality field is wrong for many
# "computer_use" entries. Real desktop computer_use envs all start with this
# prefix; anything else marked computer_use is actually browser_use.
COMPUTER_USE_ENV_PREFIX = "fos-"

# Excluded envs (from v6 pipeline: broken or unsuitable for training)
EXCLUDED_ENVS: frozenset[str] = frozenset({"google-maps", "carlisle"})

# Modality -> SkyPilot task YAML (relative to repo root).
#
# fos-* computer_use envs use the SAME VL YAML as browser_use. Both are
# MCP-based, single-container, VL-policy. The only runtime difference is
# image_type: OpenEnv's FleetTaskEnv auto-sets image_type="mcp" when
# task_modality == "computer_use" (which exposes the `computer` tool on
# port 8081). gym-anything CUA is a separate two-VM pipeline and is NOT
# what fos-* tasks are.
MODALITY_YAML_MAP: dict[str, str] = {
    "tool_use": "tasks/openenv-fleet-grpo-qwen3_5-35b.yaml",
    "browser_use": "tasks/openenv-fleet-grpo-vl.yaml",
    "computer_use": "tasks/openenv-fleet-grpo-vl.yaml",
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

# Minimum task count for an auto-launch. Datasets below this fail with
# "dataset should be at least as large as train_batch_size" once training
# starts.
#
# Empirical floor (validated 2026-05-28 with airflow_synthetic_pipeline,
# 74 single-env tasks):
#   total_tasks
#     -> 0.80 * total                       (20% eval split, MAX_EVAL=20/env)
#     -> 0.20 * train                       (MAX_ENV_TRAIN_RATIO in
#                                            prepare_dataset.py: any single
#                                            env capped at 20% of train)
#     = 0.16 * total_tasks effective train
#   need >= train_batch_size (16) => total_tasks >= 100
#
# 100 is the minimum that survives the single-env case. 50 is enough for
# multi-env datasets where the per-env cap doesn't bind, but the trigger
# can't tell ahead of time. Pick the conservative bound.
MIN_TASKS_TO_LAUNCH = 100

# Maximum tasks per (dataset, modality) to export to S3 + train on. Larger
# datasets get deterministically downsampled (seeded by dataset_key) so each
# CI tick has bounded training time regardless of how big the source dataset
# is. Override per-run via AUTO_TRAIN_MAX_EXPORT_TASKS env var.
MAX_EXPORT_TASKS = int(os.environ.get("AUTO_TRAIN_MAX_EXPORT_TASKS", "200"))

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

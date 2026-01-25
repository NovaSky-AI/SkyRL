"""
Environment variable configuration for SkyRL.

All environment variables used by SkyRL should be defined here for discoverability.
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# Ray / Placement Group
# ─────────────────────────────────────────────────────────────────────────────

SKYRL_RAY_PG_TIMEOUT_IN_S = int(os.environ.get("SKYRL_RAY_PG_TIMEOUT_IN_S", 180))
"""
Timeout for allocating the placement group for different actors in SkyRL.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Worker / NCCL
# ─────────────────────────────────────────────────────────────────────────────

SKYRL_WORKER_NCCL_TIMEOUT_IN_S = int(os.environ.get("SKYRL_WORKER_NCCL_TIMEOUT_IN_S", 600))
"""
Timeout for initializing the NCCL process group for the worker, defaults to 10 minutes.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Inference Servers
# ─────────────────────────────────────────────────────────────────────────────

SKYRL_VLLM_DP_PORT_OFFSET = int(os.environ.get("SKYRL_VLLM_DP_PORT_OFFSET", 500))
"""
Offset for the data parallel port of the vLLM server.
"""
SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S = int(
    os.environ.get("SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S", 600)
)
"""
Timeout for waiting until the inference server is healthy.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Runtime Environment Exports
# ─────────────────────────────────────────────────────────────────────────────

SKYRL_LD_LIBRARY_PATH_EXPORT = str(os.environ.get("SKYRL_LD_LIBRARY_PATH_EXPORT", "False")).lower() in (
    "true",
    "1",
    "yes",
)
"""
Whether to export ``LD_LIBRARY_PATH`` environment variable from the driver to the workers with Ray's runtime env.

For example, if you are using RDMA, you may need to customize the ``LD_LIBRARY_PATH`` to include the RDMA libraries (Ex: EFA on AWS).
"""

SKYRL_PYTHONPATH_EXPORT = str(os.environ.get("SKYRL_PYTHONPATH_EXPORT", "False")).lower() in (
    "true",
    "1",
    "yes",
)
"""
Whether to export ``PYTHONPATH`` environment variable from the driver to the workers with Ray's runtime env.

See https://github.com/ray-project/ray/issues/56697 for details on why this is needed.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Feature Flags (Private)
# ─────────────────────────────────────────────────────────────────────────────

_SKYRL_USE_NEW_INFERENCE = str(os.environ.get("_SKYRL_USE_NEW_INFERENCE", "0")).lower() in (
    "true",
    "1",
    "yes",
)
"""
**Private feature flag** - Enables the new inference layer.

When enabled, uses `RemoteInferenceClient` with HTTP endpoints for inference
instead of the legacy `InferenceEngineClient` with Ray actors.

Default: False (uses legacy code path).
Set `_SKYRL_USE_NEW_INFERENCE=1` to enable the new inference layer.

This flag is intended for internal testing and will be removed once the new
inference layer is validated and made the default.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

SKYRL_LOG_LEVEL = os.environ.get("SKYRL_LOG_LEVEL", "INFO").upper()
"""
Log level for SkyRL. Controls log filtering and stdout verbosity.

- INFO (default): Training progress on stdout, infrastructure logs to file only
- DEBUG: All logs (including vLLM, Ray, workers) shown on stdout
- Also used by loguru for log level filtering (ERROR, WARNING, etc.)
"""

SKYRL_LOG_DIR = os.environ.get("SKYRL_LOG_DIR", "/tmp/skyrl-logs")
"""
Base directory for SkyRL log files (default: /tmp/skyrl-logs).

Infrastructure logs are written to: {SKYRL_LOG_DIR}/{run_name}/infra.log
"""

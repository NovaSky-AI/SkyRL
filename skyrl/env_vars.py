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

SKYRL_HTTP_CONNECTION_LIMIT = int(os.environ.get("SKYRL_HTTP_CONNECTION_LIMIT", 50_000))
"""
Maximum number of concurrent HTTP connections for the inference client, router,
and server.

This controls:
- aiohttp TCPConnector limit in `RemoteInferenceClient`
- connection pool limits in the router
- uvicorn TCP backlog in the router and vLLM server
"""

SKYRL_GENERATE_CONCURRENCY_PER_ENGINE = int(os.environ.get("SKYRL_GENERATE_CONCURRENCY_PER_ENGINE", 512))
"""
Maximum number of concurrent generate tasks per inference engine.

The effective concurrency limit is ``SKYRL_GENERATE_CONCURRENCY_PER_ENGINE * num_engines``.
Large batch sizes (e.g. 5120) can overwhelm the router's single-threaded
event loop and vLLM's accept queue when all requests fire simultaneously.
We ensure that at most this many tasks per engine are in-flight at
once; the rest queue locally and proceed as slots free up.

Set to 0 to disable throttling (all tasks fire immediately).
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

_SKYRL_USE_NEW_INFERENCE = str(os.environ.get("_SKYRL_USE_NEW_INFERENCE", "1")).lower() in (
    "true",
    "1",
    "yes",
)
"""
**Private feature flag** - Enables the new inference layer.

When enabled, uses `RemoteInferenceClient` with HTTP endpoints for inference
instead of the legacy `InferenceEngineClient` with Ray actors.

Default: True (uses new code path).
Set `_SKYRL_USE_NEW_INFERENCE=0` to disable the new inference layer.

This flag will be removed soon - the legacy path will be removed
"""

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

SKYRL_DUMP_INFRA_LOG_TO_STDOUT = str(os.environ.get("SKYRL_DUMP_INFRA_LOG_TO_STDOUT", "False")).lower() in (
    "true",
    "1",
    "yes",
)
"""
When enabled, infrastructure logs (vLLM, Ray, workers) are shown on stdout
instead of being redirected to the log file. Useful for debugging startup issues.

Default: False (infrastructure logs go to file only, stdout shows training progress).
Set ``SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1`` to show all logs on stdout.
"""

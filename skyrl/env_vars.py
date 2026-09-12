"""
Environment variable configuration for SkyRL.

All environment variables used by SkyRL should be defined here for discoverability.
"""

import os

_TRUTHY = ("true", "1", "yes")


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable, treating "true"/"1"/"yes" as True."""
    return os.environ.get(name, str(default)).strip().lower() in _TRUTHY


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

SKYRL_LD_LIBRARY_PATH_EXPORT = _env_flag("SKYRL_LD_LIBRARY_PATH_EXPORT")
"""
Whether to export ``LD_LIBRARY_PATH`` environment variable from the driver to the workers with Ray's runtime env.

For example, if you are using RDMA, you may need to customize the ``LD_LIBRARY_PATH`` to include the RDMA libraries (Ex: EFA on AWS).
"""

SKYRL_PYTHONPATH_EXPORT = _env_flag("SKYRL_PYTHONPATH_EXPORT")
"""
Whether to export ``PYTHONPATH`` environment variable from the driver to the workers with Ray's runtime env.

See https://github.com/ray-project/ray/issues/56697 for details on why this is needed.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Attention
# ─────────────────────────────────────────────────────────────────────────────

SKYRL_DISABLE_FA4 = str(os.environ.get("SKYRL_DISABLE_FA4", "False")).lower() in (
    "true",
    "1",
    "yes",
)
"""
Force Transformer Engine to ignore FlashAttention 4, falling back to FA2 (or
cuDNN fused attention). FA4 is opt-in via the ``fa4`` extra -- SkyRL ships FA4's
kernels inside the combined ``flash-attn`` wheel, but TE only enables them when
the metadata-only ``flash-attn-4`` companion is installed. This variable turns
FA4 back off for an environment that already has it, without re-resolving.

Useful for A/B-ing FA2 against FA4 without rebuilding the environment, and as an
escape hatch if an FA4 kernel misbehaves on a shape SkyRL exercises.

Default: False (use FA4 where supported).
"""

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

SKYRL_DUMP_INFRA_LOG_TO_STDOUT = _env_flag("SKYRL_DUMP_INFRA_LOG_TO_STDOUT")
"""
When enabled, infrastructure logs (vLLM, Ray, workers) are shown on stdout
instead of being redirected to the log file. Useful for debugging startup issues.

Default: False (infrastructure logs go to file only, stdout shows training progress).
Set ``SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1`` to show all logs on stdout.
"""

SKYRL_FORWARDING_INFERENCE_TIMEOUT_SEC = float(os.environ.get("SKYRL_FORWARDING_INFERENCE_TIMEOUT_SEC", 300))
"""
Read timeout in seconds for API-side requests forwarded to the SkyRL-Train-managed
inference engine. This is applicable for SkyRL's Tinker server in non-colocated setups.

The timeout must cover time spent queued behind other requests as well as generation time.
Equivalent to the ``--forwarding-inference-timeout-sec`` flag of the Tinker API server
(``EngineConfig.forwarding_inference_timeout_sec``); the flag takes precedence when both are set.
"""

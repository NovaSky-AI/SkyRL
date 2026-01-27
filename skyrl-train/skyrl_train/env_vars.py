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
# Inference Server
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

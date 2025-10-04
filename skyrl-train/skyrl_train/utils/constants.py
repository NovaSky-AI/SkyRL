import os

#
SKYRL_RAY_PG_TIMEOUT_IN_S = int(os.environ.get("SKYRL_RAY_PG_TIMEOUT_IN_S", 60))
"""
Timeout for allocating the placement group for different actors in SkyRL
"""

# For some reason the `LD_LIBRARY_PATH` is not exported to the worker with .env file.
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




import os


SKYRL_VLLM_DP_PORT_OFFSET = int(os.environ.get("SKYRL_VLLM_DP_PORT_OFFSET", 500))
"""
Offset for the data parallel port of the vLLM server.
"""
SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S = int(os.environ.get("SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S", 600))
"""
Timeout for waiting until the inference server is healthy.
"""
SKYRL_INCLUDE_PYTHONPATH_IN_RUNTIME_ENV = str(os.environ.get("SKYRL_INCLUDE_PYTHONPATH_IN_RUNTIME_ENV", "False")).lower() in (
    "true",
    "1",
    "yes",
)
"""
Whether to include the PYTHONPATH environment variable in the runtime 
environment. In case of using ray nightly, this will be needed to avoid 
dependencies issues by setting it to the local path where ray nightly is 
installed.
"""
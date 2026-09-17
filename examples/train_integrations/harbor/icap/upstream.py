"""SkyRL's fork of vLLM's token-in/token-out endpoint, as a capture target.

capture ships the wire itself, as `vllm`: `vllm serve` mounts
`/inference/v1/generate` for any generate-capable model, and the payloads are
vLLM's, not ours. What is ours is the path. `vllm_server_actor.py` serves the
same shape at `/skyrl/v1/generate`, and says why:

    We use a custom generate endpoint /skyrl/v1/generate because the native
    endpoint /inference/v1/generate does not support returning routed expert
    IDs. TODO: Migrate back to /inference/v1/generate once this is fixed on
    the vllm side.

vLLM's `GenerateResponseChoice` now carries `routed_experts`, so that TODO is
actionable — and when it happens this module goes away entirely and a target
becomes `{"type": "vllm", ...}`.

**Importing this module registers it.** In-process that is enough, because the
proxy runs here; `entrypoints/main_harbor_icap.py` imports it before creating a
target. For a separate `skyrl-capture serve`, name it on the command line:

    skyrl-capture serve --port 8080 \
        --upstream-module examples.train_integrations.harbor.icap.upstream
"""

from __future__ import annotations

from skyrl_capture.upstream.registry import register
from skyrl_capture.upstream.servers import VLLMTokensServer


class SkyRLTokensServer(VLLMTokensServer):
    """vLLM's wire on SkyRL's path. Everything else is inherited."""

    name = "skyrl"
    generate_path = "/skyrl/v1/generate"


register(SkyRLTokensServer())

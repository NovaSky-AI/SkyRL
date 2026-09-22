"""SkyRL's fork of vLLM's token-in/token-out endpoint, as a capture upstream.

capture ships the wire itself, as `vllm`: `vllm serve` mounts
`/inference/v1/generate` for any generate-capable model, and the payloads are
vLLM's, not ours. Three things here are ours.

**The path.** `vllm_server_actor.py` serves the same shape at
`/skyrl/v1/generate`, and says why:

    We use a custom generate endpoint /skyrl/v1/generate because the native
    endpoint /inference/v1/generate does not support returning routed expert
    IDs. TODO: Migrate back to /inference/v1/generate once this is fixed on
    the vllm side.

**`cache_salt`.** It keys the engine's prefix cache on the policy weights, so
it moves every training step: it can be neither startup configuration nor a
trajectory setting, and rides in the inference request that wants it. capture
does not know the field. It carries what it does not recognise -- see
`ChatRequest.extras` -- and leaves the meaning to the protocol for that
upstream, which is this one. So it is lifted out here, before the inherited
wire filters what is left against `SamplingParams` (which rejects a request
carrying a key it does not know), and sent as a top-level field -- exactly
where `RemoteInferenceClient._generate_single` puts it.

**Routed experts.** vLLM's own endpoint returns a base64 `.npy` covering
`num_tokens - 1` rows, which capture refuses rather than pad, so
`VLLMTitoProtocol` reports none. This endpoint returns a packed array covering
the sampled tokens, and `decode_packed_routed_experts` is the one decoder for
it -- imported rather than reimplemented, so the two sides cannot drift.

When vLLM's native endpoint gains the same routed-expert contract, this module
goes away and the upstream type becomes `vllm`.

**Importing this module registers it.** In-process that is enough, because the
proxy runs here. For a separate `skyrl-capture serve`, name it on the command
line, or put it in `Config.upstream_modules`:

    skyrl-capture serve --upstream-module \
        examples.train_integrations.harbor_capture.upstream
"""

from __future__ import annotations

from typing import Any, Sequence

from skyrl_capture.tito.types import TokenUpstreamError
from skyrl_capture.tito.upstream import VLLMTitoProtocol, register

from skyrl.backends.skyrl_train.inference_servers.generate_wire import (
    decode_packed_routed_experts,
)


class SkyRLTitoProtocol(VLLMTitoProtocol):
    """vLLM's wire on SkyRL's path, plus `cache_salt` and routed experts."""

    name = "skyrl"
    generate_path = "/skyrl/v1/generate"

    def request(
        self,
        *,
        prompt_token_ids: Sequence[int],
        sampling_params: dict[str, Any],
        model: str | None,
        session_id: str,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        # Lifted out *before* the base class filters what is left against
        # `SamplingParams`, which is what would otherwise drop it on the floor.
        remaining = dict(sampling_params)
        salt = remaining.pop("cache_salt", None)
        payload, headers = super().request(
            prompt_token_ids=prompt_token_ids,
            sampling_params=remaining,
            model=model,
            session_id=session_id,
        )
        if salt is not None:
            payload["cache_salt"] = str(salt)
        return payload, headers

    def response(self, body: Any) -> dict[str, Any]:
        result = super().response(body)
        packed = body["choices"][0].get("routed_experts")
        if packed is None:
            # Not requested, or an engine built without it. Absent is a
            # supported state downstream; wrong is not.
            return result
        if not isinstance(packed, dict):
            raise TokenUpstreamError(
                "routed experts must be the packed {data, shape, dtype} object "
                f"/skyrl/v1/generate sends, got {type(packed).__name__}"
            )
        try:
            indices = decode_packed_routed_experts(packed)
        except Exception as error:
            raise TokenUpstreamError(f"could not decode routed experts: {error}") from error
        rows = len(indices)
        if rows != len(result["completion_ids"]):
            # capture rejects a turn whose routed experts cover part of the
            # sequence rather than storing a mask it cannot align.
            raise TokenUpstreamError(
                f"routed experts cover {rows} tokens but the completion is "
                f"{len(result['completion_ids'])}"
            )
        result["routed_experts"] = indices.tolist() if hasattr(indices, "tolist") else indices
        return result


register(SkyRLTitoProtocol())

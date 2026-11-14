"""SkyRL's ``/skyrl/v1/generate``, as a skycap token-in/token-out engine.

It is vLLM's generate wire with three differences, all SkyRL's:

- the path, and a router that keeps per-session state, released at
  ``/finish_session`` when a trajectory ends;
- routed experts and sampler support come back as packed
  ``{data, shape, dtype}`` envelopes, decoded with the same functions
  ``RemoteInferenceClient`` uses, so the two readers can't drift;
- sampler support is asked for with ``return_sample_support`` and comes back
  as CSR rows of at most ``top_k`` IDs; with ``sample_support_logprobs``
  (score centering), each ID also carries the sampler's logprob.

skycap sends the trajectory id as ``X-Session-ID``, so the router's session and
the trajectory are the same one.
"""

from collections.abc import Mapping
from typing import Any

from skycap.tokens.engine import EngineError, EngineOutput, VLLMEngine

from skyrl.backends.skyrl_train.inference_servers.generate_wire import (
    PackedField,
    decode_packed_routed_experts,
    decode_packed_sample_support,
)


class SkyRLEngine(VLLMEngine):
    name = "skyrl"
    generate_path = "/skyrl/v1/generate"
    release_path = "/finish_session"

    def __init__(self, *, sample_support_logprobs: bool = False) -> None:
        self.sample_support_logprobs = sample_support_logprobs

    def request(self, *, sampling_mask: bool, **kwargs: Any) -> dict[str, Any]:
        body = super().request(sampling_mask=sampling_mask, **kwargs)
        if sampling_mask:
            body["return_sample_support"] = True
            if self.sample_support_logprobs:
                body["return_sample_support_logprobs"] = True
        return body

    def _side_channels(self, choice: Mapping[str, Any], output: EngineOutput) -> None:
        routed = choice.get(PackedField.ROUTED_EXPERTS)
        support = choice.get(PackedField.ROLLOUT_SAMPLE_SUPPORT)
        support_logprobs = choice.get(PackedField.ROLLOUT_SAMPLE_SUPPORT_LOGPROBS)
        try:
            if routed is not None:
                output.routed_experts = decode_packed_routed_experts(routed)
            rows = decode_packed_sample_support(support, support_logprobs) if support is not None else None
        except (TypeError, ValueError) as error:
            raise EngineError(f"unreadable side channel: {error}") from error
        if rows is not None:
            if len(rows) != len(output.completion_ids):
                raise EngineError(f"{len(rows)} sample-support rows for {len(output.completion_ids)} tokens")
            if self.sample_support_logprobs and rows.logprobs is None:
                raise EngineError("sample support came back without the sampler's member logprobs")
            spans = list(zip(rows.offsets[:-1], rows.offsets[1:]))
            output.sampling_mask = [rows.token_ids[a:b].tolist() for a, b in spans]
            if rows.logprobs is not None:
                output.sampling_mask_logprobs = [rows.logprobs[a:b].tolist() for a, b in spans]

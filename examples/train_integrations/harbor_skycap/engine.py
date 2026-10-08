"""SkyRL's ``/skyrl/v1/generate``, as a skycap token-in/token-out engine.

It is vLLM's generate wire with three differences, all SkyRL's:

- the path, and a router that keeps per-session state, released at
  ``/finish_session`` when a trajectory ends;
- routed experts and sampler support come back as packed
  ``{data, shape, dtype}`` envelopes, decoded with the same functions
  ``RemoteInferenceClient`` uses, so the two readers can't drift;
- sampler support is asked for with ``return_sample_support`` and is padded
  with ``-1`` to ``top_k``.

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
from skyrl.backends.skyrl_train.utils.sample_support import SAMPLE_SUPPORT_PADDING


class SkyRLEngine(VLLMEngine):
    name = "skyrl"
    generate_path = "/skyrl/v1/generate"
    release_path = "/finish_session"

    def request(self, *, sampling_mask: bool, **kwargs: Any) -> dict[str, Any]:
        body = super().request(sampling_mask=sampling_mask, **kwargs)
        if sampling_mask:
            body["return_sample_support"] = True
        return body

    def _side_channels(self, choice: Mapping[str, Any], output: EngineOutput) -> None:
        routed = choice.get(PackedField.ROUTED_EXPERTS)
        support = choice.get(PackedField.ROLLOUT_SAMPLE_SUPPORT)
        try:
            if routed is not None:
                output.routed_experts = decode_packed_routed_experts(routed)
            rows = decode_packed_sample_support(support) if support is not None else None
        except (TypeError, ValueError) as error:
            raise EngineError(f"unreadable side channel: {error}") from error
        if rows is not None:
            if rows.shape[0] != len(output.completion_ids):
                raise EngineError(f"{rows.shape[0]} sample-support rows for {len(output.completion_ids)} tokens")
            output.sampling_mask = [[int(t) for t in row if t != SAMPLE_SUPPORT_PADDING] for row in rows]

"""How one token-in/token-out engine is spelled on the wire.

Two wires ship. ``vllm`` is vLLM's own ``/inference/v1/generate``: routed
experts as a base64 ``.npy`` covering every token but the last, and a
``sampling_mask`` when the server runs with ``return_sampling_mask``.
``skyrl`` is SkyRL's ``/skyrl/v1/generate``: routed experts and the sampler's
top-k support as packed ``{data, shape, dtype}`` arrays, and a router with
``/finish_session`` to release a trajectory's session.
"""

from __future__ import annotations

import base64
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


class EngineError(Exception):
    """The engine did not answer in a way a turn can be built from."""


@dataclass(slots=True)
class EngineOutput:
    completion_ids: list[int]
    logprobs: list[float]
    finish_reason: str
    #: Rows cover sequence positions ``[routed_start, routed_start + len)``.
    routed_experts: np.ndarray | None = None
    routed_start: int = 0
    #: One support set per completion token.
    sampling_mask: list[list[int]] | None = None


class VLLMEngine:
    name = "vllm"
    generate_path = "/inference/v1/generate"
    release_path: str | None = None
    #: What vLLM's ``SamplingParams`` accepts; anything else is dropped rather than sent.
    sampling_keys = frozenset(
        {
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "seed",
            "stop",
            "stop_token_ids",
            "repetition_penalty",
            "frequency_penalty",
            "presence_penalty",
            "min_tokens",
            "ignore_eos",
        }
    )

    def request(
        self,
        *,
        prompt_ids: Sequence[int],
        sampling: Mapping[str, Any],
        model: str | None,
        cache_salt: str | None,
        sampling_mask: bool,
    ) -> dict[str, Any]:
        params = {key: value for key, value in sampling.items() if key in self.sampling_keys}
        params["logprobs"] = 0  # the sampled token's logprob only, which capture requires
        body: dict[str, Any] = {"token_ids": list(prompt_ids), "sampling_params": params}
        if model:
            body["model"] = model
        if cache_salt:
            body["cache_salt"] = cache_salt
        return body

    def parse(self, body: Any) -> EngineOutput:
        choice = _single_choice(body)
        completion = choice.get("token_ids")
        if not isinstance(completion, list) or not completion:
            raise EngineError("completion token ids are missing or empty")
        content = (choice.get("logprobs") or {}).get("content")
        if not isinstance(content, list):
            raise EngineError("the engine returned no sampled-token logprobs; token capture requires them")
        logprobs = [entry.get("logprob") if isinstance(entry, dict) else None for entry in content]
        if len(logprobs) != len(completion) or any(value is None for value in logprobs):
            raise EngineError(f"{len(completion)} completion tokens but {len(logprobs)} logprobs")
        output = EngineOutput(
            completion_ids=[int(t) for t in completion],
            logprobs=[float(v) for v in logprobs],
            finish_reason=str(choice.get("finish_reason") or "stop"),
        )
        self._side_channels(choice, output)
        return output

    def _side_channels(self, choice: Mapping[str, Any], output: EngineOutput) -> None:
        routed = choice.get("routed_experts")
        if isinstance(routed, str):
            output.routed_experts = np.load(io.BytesIO(base64.b64decode(routed)), allow_pickle=False)
        elif isinstance(routed, Mapping):
            output.routed_experts = unpack(routed)
            output.routed_start = int(routed.get("start") or 0)
        mask = choice.get("sampling_mask")
        if mask is not None:
            if len(mask) != len(output.completion_ids):
                raise EngineError(f"{len(mask)} sampling-mask rows for {len(output.completion_ids)} tokens")
            output.sampling_mask = [[int(t) for t in row] for row in mask]


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
        routed = choice.get("routed_experts")
        if routed is not None:
            output.routed_experts = unpack(routed)
        support = choice.get("rollout_sample_support")
        if support is not None:
            rows = unpack(support)
            if rows.shape[0] != len(output.completion_ids):
                raise EngineError(f"{rows.shape[0]} sample-support rows for {len(output.completion_ids)} tokens")
            output.sampling_mask = [[int(t) for t in row if t >= 0] for row in rows]


ENGINES: dict[str, type[VLLMEngine]] = {VLLMEngine.name: VLLMEngine, SkyRLEngine.name: SkyRLEngine}


def unpack(envelope: Mapping[str, Any]) -> np.ndarray:
    """A ``{data: base64, shape, dtype}`` array, as SkyRL packs them."""
    try:
        data = base64.b64decode(envelope["data"])
        return np.frombuffer(data, dtype=np.dtype(envelope["dtype"])).reshape(envelope["shape"]).copy()
    except (KeyError, TypeError, ValueError) as error:
        raise EngineError(f"invalid packed array: {error}") from error


def pack(array: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    return {
        "data": base64.b64encode(contiguous.tobytes()).decode("ascii"),
        "shape": list(contiguous.shape),
        "dtype": contiguous.dtype.name,
    }


def _single_choice(body: Any) -> Mapping[str, Any]:
    if not isinstance(body, Mapping):
        raise EngineError("engine response is not a JSON object")
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
        raise EngineError("expected exactly one choice")
    return choices[0]

"""How one token-in/token-out engine is spelled on the wire.

Token capture sends exact token ids and needs exact ones back. *How* that is
written down belongs to the engine, not to `TokenEngine`: vLLM's own
``/generate``, SkyRL's router and a hosted service disagree about batching,
field names, where session affinity lives, and whether logprobs are a boolean
or a count. Keeping each one behind a small protocol is what lets a new engine be a
class and a ``register()`` call.

This is the token-mode counterpart of `upstream/`, and the two are separate on
purpose: a text protocol is a *client-facing* API this proxy speaks, and this
is a *server-facing* one it calls. No base class spans both, so a plugin for
one never sees the other's types -- a text adapter cannot reach token
rendering, and this cannot reach HTTP forwarding.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from skyrl_capture.tito.types import TokenUpstreamError

logger = logging.getLogger(__name__)


class TitoProtocol:
    """The batched shape capture has always sent. Subclass to add an engine."""

    name = "tokens"

    def url(self, url: str) -> str:
        """The endpoint one generate call goes to, given the target's URL."""
        return url

    def request(
        self,
        *,
        prompt_token_ids: Sequence[int],
        sampling_params: dict[str, Any],
        model: str | None,
        session_id: str,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """Build one request. Returns ``(body, extra_headers)``."""
        payload: dict[str, Any] = {
            "prompt_token_ids": [list(prompt_token_ids)],
            "sampling_params": sampling_params,
            # Session affinity lets the router keep this trajectory's prefix
            # cache warm on one worker.
            "session_ids": [session_id],
            "session_id": session_id,
        }
        if model:
            payload["model"] = model
        return payload, {}

    def response(self, body: Any) -> dict[str, Any]:
        """Normalise one response into what a turn is built from."""
        if not isinstance(body, dict):
            raise TokenUpstreamError("token-in/token-out response must be a JSON object")
        response_ids = body.get("response_ids")
        if not isinstance(response_ids, list) or len(response_ids) != 1:
            raise TokenUpstreamError("expected exactly one response_ids entry")
        completion = response_ids[0]
        if not isinstance(completion, list) or not completion:
            raise TokenUpstreamError("completion token IDs are missing or empty")

        logprobs_batch = body.get("response_logprobs")
        if not isinstance(logprobs_batch, list) or len(logprobs_batch) != 1:
            raise TokenUpstreamError(
                "inference did not return selected-token logprobs; token capture requires them"
            )
        logprobs = logprobs_batch[0]
        if not isinstance(logprobs, list) or len(logprobs) != len(completion):
            raise TokenUpstreamError(
                f"completion IDs ({len(completion)}) and logprobs "
                f"({len(logprobs) if isinstance(logprobs, list) else 'none'}) have different lengths"
            )

        stop_reasons = body.get("stop_reasons")
        if not isinstance(stop_reasons, list) or len(stop_reasons) != 1:
            raise TokenUpstreamError("expected exactly one stop_reasons entry")

        routed = body.get("rollout_expert_indices")
        routed_entry = routed[0] if isinstance(routed, list) and len(routed) == 1 else None

        return {
            "completion_ids": [int(value) for value in completion],
            "completion_logprobs": [float(value) for value in logprobs],
            "stop_reason": str(stop_reasons[0]),
            "routed_experts": routed_entry,
        }

    def describe(self) -> dict[str, Any]:
        return {"name": self.name}


class VLLMTitoProtocol(TitoProtocol):
    """Token-in/token-out against vLLM's own endpoint.

    `vllm serve` mounts `/inference/v1/generate` for any generate-capable model
    (`api_server.py` calls `register_scale_out_api_routers`), so this needs no
    wrapper and no translating proxy in front of it.

    Five differences from capture's own shape:

    * the request is singular, not a batch of one;
    * the response is ``choices[0]``, not parallel arrays;
    * session affinity is the ``X-Session-ID`` header, not a body field. This
      one fails quietly -- the router still answers and only prefix-cache
      locality is lost, which is most of the reason to name a trajectory;
    * ``sampling_params`` must be filtered: the field is vLLM's own
      ``SamplingParams``, which rejects a request carrying keys it does not
      know;
    * ``logprobs`` changes type: a boolean in the OpenAI shape capture speaks,
      and *a count of top logprobs* here, where ``0`` means "the sampled token
      only" -- exactly what capture requires and refuses the turn for missing.
      Forwarded verbatim it is either a 400 or a silently empty list.

    A trainer that forks this endpoint inherits this wire and overrides the
    path; see ``generate_path``.
    """

    name = "vllm"
    #: Appended when the target URL is just the server's root, so a target can
    #: be `{"type": "vllm", "url": "http://engine:8000"}`. A fork that serves
    #: the same shape on another path overrides this and nothing else.
    generate_path = "/inference/v1/generate"
    #: What `SamplingParams` accepts and a chat client plausibly sets.
    #: Anything else is dropped rather than sent.
    sampling_keys = frozenset(
        {
            "max_tokens", "temperature", "top_p", "top_k", "min_p", "seed",
            "stop", "stop_token_ids", "repetition_penalty", "frequency_penalty",
            "presence_penalty", "logprobs", "n", "ignore_eos", "min_tokens",
            "skip_special_tokens", "spaces_between_special_tokens", "bad_words",
        }
    )

    def url(self, url: str) -> str:
        root = url.rstrip("/")
        return root if root.endswith(self.generate_path) else f"{root}{self.generate_path}"

    def _sampling(self, raw: dict[str, Any]) -> dict[str, Any]:
        params = {key: value for key, value in raw.items() if key in self.sampling_keys}
        dropped = set(raw) - set(params)
        if dropped:
            logger.debug("dropping sampling params vLLM does not accept: %s", sorted(dropped))
        if isinstance(params.get("logprobs"), bool):
            params["logprobs"] = 0 if params["logprobs"] else None
        if params.get("logprobs") is None:
            # Asking for none guarantees a turn capture will reject. Ask for the
            # sampled token only, which is the cheapest thing that works.
            params["logprobs"] = 0
        return params

    def request(
        self,
        *,
        prompt_token_ids: Sequence[int],
        sampling_params: dict[str, Any],
        model: str | None,
        session_id: str,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        payload: dict[str, Any] = {
            "token_ids": list(prompt_token_ids),
            "sampling_params": self._sampling(sampling_params),
        }
        if model:
            payload["model"] = model
        return payload, {"X-Session-ID": session_id}

    def response(self, body: Any) -> dict[str, Any]:
        if not isinstance(body, dict):
            raise TokenUpstreamError("token-in/token-out response must be a JSON object")
        choices = body.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise TokenUpstreamError(
                f"expected exactly one choice, got "
                f"{len(choices) if isinstance(choices, list) else 'none'}"
            )
        choice = choices[0]
        if not isinstance(choice, dict):
            raise TokenUpstreamError("choice must be a JSON object")

        completion = choice.get("token_ids")
        if not isinstance(completion, list) or not completion:
            raise TokenUpstreamError("completion token IDs are missing or empty")

        content = (choice.get("logprobs") or {}).get("content")
        if not isinstance(content, list):
            raise TokenUpstreamError(
                "inference did not return selected-token logprobs; token capture requires them"
            )
        logprobs = [entry.get("logprob") for entry in content if isinstance(entry, dict)]
        if len(logprobs) != len(completion) or any(value is None for value in logprobs):
            raise TokenUpstreamError(
                f"completion IDs ({len(completion)}) and logprobs ({len(logprobs)}) "
                "have different lengths"
            )

        return {
            "completion_ids": [int(value) for value in completion],
            "completion_logprobs": [float(value) for value in logprobs],
            "stop_reason": str(choice.get("finish_reason") or "stop"),
            # Routed experts come back as a base64 `.npy` covering
            # `num_tokens - 1` rows, since the last sampled token has not been
            # forwarded. capture rejects a turn whose routed experts cover only
            # part of the sequence, so they are omitted rather than padded into
            # a shape that would be wrong. `rollout_expert_indices` is optional
            # downstream.
            "routed_experts": None,
        }


class UnknownTitoProtocol(Exception):
    """No token-in/token-out protocol is registered under this name."""


_PROTOCOLS: dict[str, TitoProtocol] = {}


def register(protocol: TitoProtocol) -> TitoProtocol:
    _PROTOCOLS[protocol.name] = protocol
    return protocol


def get(name: str) -> TitoProtocol:
    try:
        return _PROTOCOLS[name]
    except KeyError:
        raise UnknownTitoProtocol(
            f"unsupported upstream type {name!r}; expected one of {', '.join(names())}"
        ) from None


def names() -> tuple[str, ...]:
    return tuple(sorted(_PROTOCOLS))


def is_registered(name: str) -> bool:
    return name in _PROTOCOLS


register(TitoProtocol())
register(VLLMTitoProtocol())

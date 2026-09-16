"""SkyRL's token wire, as an inference-capture upstream kind.

capture speaks protocols, not engines: `openai`, `anthropic`, and its own
batched `tokens` shape. SkyRL's router serves something else, so the wire is
ours to define -- carrying it in capture would mean that project tracking our
HTTP shape, and every other trainer's.

**Importing this module registers it.** In-process that is enough, because the
proxy runs here; `entrypoints/main_harbor_icap.py` imports it before creating a
target. For a separate `icap serve`, name it on the command line:

    icap serve --port 8080 \
        --upstream-module examples.train_integrations.harbor.icap.upstream

After that a target is `{"type": "skyrl", "url": "http://router:8000"}` and
nothing translating sits in between.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from inference_capture.tokens.types import TokenUpstreamError
from inference_capture.upstream.registry import register
from inference_capture.upstream.servers import TokensServer

logger = logging.getLogger(__name__)


class SkyRLTokensServer(TokensServer):
    """Token-in/token-out against SkyRL's inference router.

    Same mode and tokenizer requirement as ``tokens``; only the wire differs.
    Without this a deployment has to run a translating proxy in between, which
    is a component neither project owns.

    Five differences, each pinned by a test in ``tests/test_upstream_skyrl.py``:

    * the request is singular, not a batch of one;
    * the response is ``choices[0]``, not parallel arrays;
    * session affinity is the ``X-Session-ID`` header, not a body field. This
      one fails quietly -- the router still answers and only prefix-cache
      locality is lost, which is most of the reason to name a trajectory;
    * ``sampling_params`` must be filtered, because vLLM rejects a request
      carrying parameters it does not know;
    * ``logprobs`` changes type: a boolean in the OpenAI shape capture speaks,
      and *a count of top logprobs* in vLLM's, where ``0`` means "the sampled
      token only" -- exactly what capture requires and refuses the turn for
      missing. Forwarded verbatim it is either a 400 or a silently empty list.
    """

    name = "skyrl"
    #: Appended when the target URL is just the router's root, so a target can
    #: be `{"type": "skyrl", "url": "http://router:8000"}`.
    generate_path = "/skyrl/v1/generate"
    #: What vLLM's `SamplingParams` accepts and a chat client plausibly sets.
    #: Anything else is dropped rather than sent.
    sampling_keys = frozenset(
        {
            "max_tokens", "temperature", "top_p", "top_k", "min_p", "seed",
            "stop", "stop_token_ids", "repetition_penalty", "frequency_penalty",
            "presence_penalty", "logprobs", "n", "ignore_eos", "min_tokens",
            "skip_special_tokens", "spaces_between_special_tokens", "bad_words",
        }
    )

    def token_url(self, url: str) -> str:
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

    def token_request(
        self,
        *,
        prompt_token_ids: Sequence[int],
        sampling_params: dict[str, Any],
        model: str | None,
        session_id: str,
        cache_salt: str | None,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        payload: dict[str, Any] = {
            "token_ids": list(prompt_token_ids),
            "sampling_params": self._sampling(sampling_params),
        }
        if model:
            payload["model"] = model
        if cache_salt:
            payload["cache_salt"] = cache_salt
        return payload, {"X-Session-ID": session_id}

    def token_response(self, body: Any) -> dict[str, Any]:
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
            # SkyRL packs routed experts as a base64 numpy array under its own
            # codec. Decoding it here would mean tracking a format capture does
            # not own, and capture rejects a turn whose routed experts cover
            # only part of the sequence -- so they are omitted rather than
            # guessed at. `rollout_expert_indices` is optional downstream.
            "routed_experts": None,
        }


register(SkyRLTokensServer())

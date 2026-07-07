"""External Fireworks inference client (generation/eval only).

Fireworks' OpenAI-compatible ``/completions`` accepts a pre-tokenized integer-array ``prompt``
and, with ``return_token_ids=true``, returns the generated integer ``token_ids``. This gives
token-in/token-out against an external endpoint with no re-tokenization drift, so the stock
``SkyRLGymGenerator`` works unchanged.

Built on the Fireworks v1 SDK (``fireworks-ai``, installed via the ``fireworks`` uv extra):
``prompt`` accepts ``Iterable[Iterable[int]]``, ``return_token_ids`` is a first-class request
param, and the response ``Choice`` declares ``token_ids``/``prompt_token_ids``. The SDK carries
auth, retries with backoff, timeouts, and connection pooling. This module does not import vllm
and has no control plane: wake/sleep/etc. are no-ops, weight sync raises.

Sampling params arrive vLLM-shaped: the stock eval path emits them via
``get_sampling_params_for_backend(backend="vllm", ...)``, and :func:`_to_fireworks_sampling_params`
converts that dict to the subset Fireworks accepts at request time.

Part of the standalone Fireworks eval example (``examples/eval/fireworks``); constructed by
``FireworksEvalOnlyEntrypoint`` in ``main_eval_fireworks.py`` — core entrypoints never build it.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import httpx
from fireworks import AsyncFireworks
from loguru import logger

from skyrl.backends.skyrl_train.inference_servers.base import (
    InferenceEngineInput,
    InferenceEngineInterface,
    InferenceEngineOutput,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

_GEN_EVAL_ONLY = (
    "Fireworks is a generation/eval-only backend (external hosted endpoint with no weight sync). "
    "Use the vllm-backed entrypoints for training."
)

# Server root of the Fireworks data plane. The SDK appends `/v1/completions` to an overridden
# base_url, so this must NOT end in `/v1` (the constructor guards user-supplied values).
DEFAULT_FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference"

# vLLM-only sampling keys the Fireworks /completions API rejects. `skip_special_tokens` and
# `include_stop_str_in_output` are unnecessary: the client decodes responses locally from token
# ids, and Fireworks includes a matched stop string's tokens in `token_ids` (only its `text`
# field, which the client ignores, excludes it). `min_tokens` has no counterpart.
_FIREWORKS_UNSUPPORTED_KEYS = ("min_tokens", "skip_special_tokens", "include_stop_str_in_output")


def _to_fireworks_sampling_params(sampling_params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a vLLM-shaped sampling-params dict to the subset Fireworks accepts.

    Runs per request inside ``generate()``, and the vLLM emitter always includes the
    vLLM-only keys, so those are dropped silently (see ``_FIREWORKS_UNSUPPORTED_KEYS`` for why
    each is safe). Out-of-range ``top_k`` values are dropped with a warning (Fireworks accepts
    ``0..100``); the ``top_k=-1`` / ``min_p<=0`` disable-sentinels and ``None`` values are
    dropped silently (absence disables them; ``None`` would serialize as ``null`` via
    ``extra_body``).
    """
    params = dict(sampling_params)
    for key in _FIREWORKS_UNSUPPORTED_KEYS:
        params.pop(key, None)
    top_k = params.get("top_k")
    if top_k is not None and not (isinstance(top_k, int) and 0 <= top_k <= 100):
        if top_k != -1:  # -1 is the disable sentinel; absence disables top_k on Fireworks
            logger.warning(f"Dropping sampling param `top_k={top_k}`: Fireworks accepts 0..100.")
        del params["top_k"]
    if params.get("min_p") is not None and params["min_p"] <= 0:
        del params["min_p"]
    return {key: value for key, value in params.items() if value is not None}


class FireworksInferenceClient(InferenceEngineInterface):
    def __init__(
        self,
        model_name: str,
        tokenizer: "PreTrainedTokenizerBase",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        request_timeout: float = 600.0,
        *,
        _http_client: Optional[httpx.AsyncClient] = None,
    ):
        """Args:
        model_name: Fireworks model id used as the request ``model`` (e.g.
            ``accounts/fireworks/models/gpt-oss-20b``).
        tokenizer: The policy tokenizer; must be the served model's tokenizer since prompts
            are sent as raw token ids.
        base_url: Server root without ``/v1`` (defaults to the Fireworks data plane).
        api_key: API key sent as ``Authorization: Bearer``. Always passed explicitly so the
            SDK never falls back to the ``FIREWORKS_API_KEY`` env var; ``"EMPTY"`` placeholder
            keeps keyless self-hosted endpoints constructible.
        max_retries: SDK retry budget (backoff on 408/409/429/5xx and ``x-should-retry``).
        request_timeout: Per-request timeout in seconds. Overrides the SDK's 60s default,
            which is too short for long generations.
        _http_client: Internal-reserved injectable httpx client for offline testing with
            ``httpx.MockTransport``.
        """
        base_url = (base_url or DEFAULT_FIREWORKS_BASE_URL).rstrip("/")
        if base_url.endswith("/v1"):
            raise ValueError(
                "base_url is the server root; the fireworks SDK appends /v1/completions. "
                "Use e.g. https://api.fireworks.ai/inference, not .../inference/v1."
            )
        self._base_url = base_url
        self._model_name = model_name
        self._tokenizer = tokenizer
        self._client = AsyncFireworks(
            base_url=self._base_url,
            api_key=api_key or "EMPTY",
            max_retries=max_retries,
            timeout=request_timeout,
            http_client=_http_client,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    def get_endpoint_url(self) -> str:
        return self._base_url

    async def generate(
        self,
        input_batch: InferenceEngineInput,
        model: Optional[str] = None,
    ) -> InferenceEngineOutput:
        prompt_token_ids = input_batch.get("prompt_token_ids")
        if prompt_token_ids is None:
            raise ValueError("FireworksInferenceClient only accepts `prompt_token_ids`, not `prompts`.")
        if input_batch.get("mm_features"):
            raise NotImplementedError("FireworksInferenceClient does not support multi-modal features.")

        sampling_params = _to_fireworks_sampling_params(input_batch.get("sampling_params", {}))
        if sampling_params.get("n", 1) > 1:
            raise ValueError("n > 1 is not supported. Use `config.generator.n_samples_per_prompt` instead.")
        want_logprobs = sampling_params.get("logprobs") is not None

        # model/prompt/return_token_ids are typed SDK params; the sampling dict rides extra_body
        # (merged into the JSON request body) so additional_kwargs passthrough keeps working.
        completion = await self._client.completions.create(
            model=model or self._model_name,
            prompt=prompt_token_ids,
            return_token_ids=True,
            extra_body=sampling_params,
        )
        if not completion.choices:
            raise RuntimeError(f"Fireworks returned no choices: {completion!r}")
        choices = sorted(completion.choices, key=lambda choice: choice.index)

        response_ids: List[List[int]] = []
        responses: List[str] = []
        stop_reasons: List[str] = []
        response_logprobs: List[Optional[List[float]]] = []
        for choice in choices:
            token_ids = choice.token_ids
            # Re-encoding `choice.text` locally would silently reintroduce the retokenization
            # drift this backend exists to avoid, so a missing field is a hard error.
            assert token_ids is not None, (
                f"Fireworks response missing `token_ids` for choice {choice.index} despite " "return_token_ids=true."
            )
            response_ids.append(list(token_ids))
            # Decode locally to guarantee the InferenceEngineOutput invariant:
            # tokenizer.decode(response_ids[i], skip_special_tokens=True) == responses[i].
            responses.append(self._tokenizer.decode(token_ids, skip_special_tokens=True))
            stop_reasons.append(choice.finish_reason or "stop")
            if want_logprobs:
                logprobs = self._extract_logprobs(choice)
                # Silently emitting None here would surface far downstream as a confusing
                # length-validation failure on GeneratorOutput["rollout_logprobs"].
                if logprobs is None:
                    raise RuntimeError(
                        f"Sampling params requested logprobs but Fireworks returned none (or an "
                        f"unrecognized shape) for choice {choice.index}. Set "
                        f"generator.eval_sampling_params.logprobs=null (and "
                        f"generator.sampling_params.logprobs=null) if logprobs are not needed."
                    )
                response_logprobs.append(logprobs)

        return InferenceEngineOutput(
            responses=responses,
            response_ids=response_ids,
            stop_reasons=stop_reasons,
            response_logprobs=response_logprobs if want_logprobs else None,
            prompt_logprobs=None,
            rollout_expert_indices=None,
        )

    @staticmethod
    def _extract_logprobs(choice: Any) -> Optional[List[float]]:
        """Extract per-token logprobs from either Fireworks response shape.

        ``choice.logprobs`` is a union of the classic completions shape (``LogProbs``, carries
        ``token_logprobs``; what the live endpoint returns for integer ``logprobs``) and the
        OpenAI chat-style shape (``NewLogProbs``, carries ``content`` items with ``.logprob``).
        Null entries map to 0.0 rather than being dropped so the result stays aligned 1:1 with
        the generated token ids (downstream validation asserts equal lengths).
        """
        logprobs = choice.logprobs
        if logprobs is None:
            return None
        token_logprobs = getattr(logprobs, "token_logprobs", None)
        if token_logprobs is not None:
            return [logprob if logprob is not None else 0.0 for logprob in token_logprobs]
        content = getattr(logprobs, "content", None)
        if content is not None:
            return [item.logprob if item.logprob is not None else 0.0 for item in content]
        return None

    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        response = await self._client.completions.create(**request_payload.get("json", {}))
        return response.model_dump()

    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        response = await self._client.chat.completions.create(**request_payload.get("json", {}))
        return response.model_dump()

    async def render_chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("render_chat_completion is not supported for the Fireworks backend.")

    async def wake_up(self, *args: Any, **kwargs: Any):
        return {}

    async def sleep(self, *args: Any, **kwargs: Any):
        return {}

    async def reset_prefix_cache(self, reset_running_requests: bool = False):
        return {}

    async def pause_generation(self) -> None:
        return

    async def resume_generation(self) -> None:
        return

    async def finish_session(self, session_id: str) -> None:
        return

    async def teardown(self):
        await self._client.close()

    async def get_world_size(self) -> Tuple[int, int]:
        raise NotImplementedError(_GEN_EVAL_ONLY)

    async def init_weight_update_communicator(self, init_info):
        raise NotImplementedError(_GEN_EVAL_ONLY)

    async def update_named_weights(self, request):
        raise NotImplementedError(_GEN_EVAL_ONLY)

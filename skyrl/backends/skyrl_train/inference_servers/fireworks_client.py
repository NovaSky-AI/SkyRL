"""External Fireworks inference client (generation/eval only).

Fireworks' OpenAI-compatible ``/completions`` accepts a pre-tokenized integer-array ``prompt``
and, with ``return_token_ids=true``, returns the generated integer ``token_ids``. This gives
token-in/token-out against an external endpoint with no re-tokenization drift, so the stock
``SkyRLGymGenerator`` works unchanged.

Built on the Fireworks v1 SDK (``fireworks-ai``, installed via the ``fireworks`` uv extra):
``prompt`` accepts ``Iterable[Iterable[int]]``, ``return_token_ids`` is a first-class request
param, and the response ``Choice`` declares ``token_ids``/``prompt_token_ids``. The SDK carries
auth, retries with backoff, timeouts, and connection pooling. This module does not import vllm
and has no control plane: wake/sleep/etc. are no-ops, weight sync raises. Only the eval-only
entrypoint builds this client (training entrypoints reject ``backend='fireworks'``).
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import httpx
from fireworks import AsyncFireworks

from skyrl.backends.skyrl_train.inference_servers.base import (
    InferenceEngineInput,
    InferenceEngineInterface,
    InferenceEngineOutput,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

_GEN_EVAL_ONLY = (
    "Fireworks is a generation/eval-only backend (external hosted endpoint with no weight sync). "
    "Use backend='vllm' for training."
)

# Server root of the Fireworks data plane. The SDK appends `/v1/completions` to an overridden
# base_url, so this must NOT end in `/v1` (validate_inference_engine_cfg guards user-supplied
# values).
DEFAULT_FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference"


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
        _http_client: Internal-reserved injectable httpx client, used by tests with
            ``httpx.MockTransport``.
        """
        self._base_url = (base_url or DEFAULT_FIREWORKS_BASE_URL).rstrip("/")
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

        sampling_params = dict(input_batch.get("sampling_params") or {})
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
            response_logprobs.append(self._extract_logprobs(choice) if want_logprobs else None)

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
        # `choice.logprobs` is a union of the classic completions shape (carries
        # `token_logprobs`) and a content-based shape; only the classic shape is extracted.
        logprobs = choice.logprobs
        if logprobs is None:
            return None
        token_logprobs = getattr(logprobs, "token_logprobs", None)
        if not token_logprobs:
            return None
        return [logprob for logprob in token_logprobs if logprob is not None]

    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        response = await self._client.completions.create(**request_payload.get("json", {}))
        return response.model_dump()

    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        response = await self._client.chat.completions.create(**request_payload.get("json", {}))
        return response.model_dump()

    async def render_chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("render_chat_completion is not supported for the Fireworks backend.")

    async def wake_up(self, *args: Any, **kwargs: Any):
        # TODO: tokenizer handshake — probe with a text prompt + return_token_ids=true and
        # compare the returned prompt_token_ids against a local tokenizer.encode() of the same
        # string. A mispaired hf_tokenizer_name / served_model_name fails silently otherwise:
        # the server consumes raw token ids, so a wrong tokenizer yields degraded generations,
        # never an error.
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

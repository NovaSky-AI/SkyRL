"""Forwards EXTERNAL sample requests to the SkyRL-Train-managed vLLM.

Pair to :class:`ExternalInferenceClient`; resolves the target URL from
``EngineStateDB`` instead of from a user-supplied ``external_inference_url``.
"""

import asyncio
from time import monotonic

import httpx
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel.ext.asyncio.session import AsyncSession

from skyrl.backends.renderer import render_model_input
from skyrl.backends.utils import convert_vllm_prompt_logprobs
from skyrl.tinker import types
from skyrl.tinker.config import EngineConfig
from skyrl.tinker.db_models import EngineStateDB, RequestStatus
from skyrl.tinker.db_observability import database_pool_status
from skyrl.tinker.external_future_store import ExternalFutureStore
from skyrl.utils.log import logger


class InferenceForwardingError(RuntimeError):
    def __init__(self, status_code: int, response_text: str):
        self.status_code = status_code
        self.response_text = response_text
        super().__init__(f"vLLM /v1/completions returned {status_code}: {response_text}")


class InferenceForwardingTimeoutError(TimeoutError):
    """The absolute inference-forwarding operation deadline expired."""


def _safe_failure_message(error: Exception) -> str:
    """Describe a forwarding failure without persisting request or response data."""
    if isinstance(error, InferenceForwardingError):
        return f"{type(error).__name__}: HTTP {error.status_code}"
    return f"{type(error).__name__}: inference forwarding failed"


class SkyRLTrainInferenceForwardingClient:
    """Forwards EXTERNAL sample requests to the SkyRL-Train-managed vLLM."""

    _PROXY_URL_POLL_INTERVAL_SEC = 0.25
    _TRANSIENT_503_MAX_ATTEMPTS = 12
    _TRANSIENT_RETRY_INITIAL_DELAY_SEC = 1.0
    _TRANSIENT_RETRY_MAX_DELAY_SEC = 10.0

    def __init__(
        self,
        engine_config: EngineConfig,
        db_engine,
        external_future_store: ExternalFutureStore,
    ):
        self.engine_config = engine_config
        self.db_engine = db_engine
        self.external_future_store = external_future_store
        self._cached_proxy_url: str | None = None
        self._cache_lock = asyncio.Lock()
        # Backpressure layered: httpx pool -> vllm-router -> vLLM max_num_seqs.
        # Default `forwarding_inference_max_connections=None` is unlimited;
        # the only cost is file descriptors (raise `ulimit -n` accordingly).
        max_conn = engine_config.forwarding_inference_max_connections
        max_keepalive = max(max_conn // 4, 32) if max_conn is not None else None
        self._http_client: httpx.AsyncClient = httpx.AsyncClient(
            timeout=httpx.Timeout(engine_config.forwarding_inference_timeout_sec, connect=10.0),
            limits=httpx.Limits(
                max_connections=max_conn,
                max_keepalive_connections=max_keepalive,
            ),
        )

    async def aclose(self) -> None:
        """Close the persistent httpx client. Called from api.py lifespan shutdown."""
        await self._http_client.aclose()

    async def _read_proxy_url_from_db(self) -> str | None:
        async with AsyncSession(self.db_engine) as session:
            row = await session.get(EngineStateDB, 1)
            if row is None or row.inference_proxy_url is None:
                return None
            return row.inference_proxy_url

    async def _resolve_proxy_url(self, *, force_refresh: bool = False, deadline: float | None = None) -> str:
        # Skip the lock when the cache is warm so concurrent samples don't serialize.
        if not force_refresh and self._cached_proxy_url is not None:
            return self._cached_proxy_url
        async with self._cache_lock:
            if force_refresh or self._cached_proxy_url is None:
                if force_refresh:
                    self._cached_proxy_url = None
                loop = asyncio.get_running_loop()
                if deadline is None:
                    deadline = loop.time() + self.engine_config.forwarding_inference_timeout_sec
                while self._cached_proxy_url is None:
                    self._cached_proxy_url = await self._read_proxy_url_from_db()
                    if self._cached_proxy_url is not None:
                        break
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        raise RuntimeError(
                            "inference engine not ready: timed out waiting for a proxy URL in EngineStateDB"
                        )
                    await asyncio.sleep(min(self._PROXY_URL_POLL_INTERVAL_SEC, remaining))
            return self._cached_proxy_url

    async def call_and_store_result(
        self,
        request_id: int,
        sample_req,
        model_id: str,
        checkpoint_id: str,
        *,
        base_model: str | None = None,
    ):
        """Forward a sample request to vLLM and complete its external future."""
        forward_started = monotonic()
        prompt_tokens = sum(len(chunk.tokens) for chunk in sample_req.prompt.chunks if hasattr(chunk, "tokens"))
        try:
            result = await self._forward_with_retry(sample_req, model_id, base_model=base_model)
            status = RequestStatus.COMPLETED
        except asyncio.CancelledError:
            await self.external_future_store.complete(
                request_id,
                types.ErrorResponse(error="Forwarded inference cancelled during shutdown", status="failed"),
                RequestStatus.FAILED,
                cancellation_safe=False,
            )
            raise
        except SQLAlchemyError as e:
            logger.error(
                "Backend-forwarded sample failed failure_stage=proxy_database request_id=%s "
                "model_id=%s sampling_session_id=%s seq_id=%s prompt_tokens=%s max_tokens=%s "
                "elapsed_seconds=%.3f pool=%s error_type=%s",
                request_id,
                model_id,
                sample_req.sampling_session_id,
                sample_req.seq_id,
                prompt_tokens,
                sample_req.sampling_params.max_tokens,
                monotonic() - forward_started,
                database_pool_status(self.db_engine),
                type(e).__name__,
            )
            result = types.ErrorResponse(error=_safe_failure_message(e), status="failed")
            status = RequestStatus.FAILED
        except Exception as e:
            logger.error(
                "Backend-forwarded sample failed failure_stage=forward request_id=%s model_id=%s "
                "sampling_session_id=%s seq_id=%s prompt_tokens=%s max_tokens=%s "
                "elapsed_seconds=%.3f error_type=%s",
                request_id,
                model_id,
                sample_req.sampling_session_id,
                sample_req.seq_id,
                prompt_tokens,
                sample_req.sampling_params.max_tokens,
                monotonic() - forward_started,
                type(e).__name__,
            )
            result = types.ErrorResponse(error=_safe_failure_message(e), status="failed")
            status = RequestStatus.FAILED

        await self.external_future_store.complete(request_id, result, status)

    async def _forward_with_retry(self, sample_req, model_id: str, *, base_model: str | None) -> types.SampleOutput:
        loop = asyncio.get_running_loop()
        timeout_sec = self.engine_config.forwarding_inference_timeout_sec
        deadline = loop.time() + timeout_sec
        connect_attempt = 0
        no_worker_attempt = 0
        force_refresh = False

        try:
            async with asyncio.timeout_at(deadline):
                while True:
                    caught_error: Exception
                    try:
                        proxy_url = await self._resolve_proxy_url(force_refresh=force_refresh, deadline=deadline)
                        return await self._forward(proxy_url, sample_req, model_id, base_model=base_model)
                    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as error:
                        connect_attempt += 1
                        caught_error = error
                        max_attempts = 2
                        retry = connect_attempt < max_attempts
                        retry_attempt = connect_attempt
                    except InferenceForwardingError as error:
                        caught_error = error
                        retry = error.status_code == 503 and "No available workers" in error.response_text
                        if not retry:
                            raise
                        no_worker_attempt += 1
                        max_attempts = self._TRANSIENT_503_MAX_ATTEMPTS
                        retry = no_worker_attempt < max_attempts
                        retry_attempt = no_worker_attempt

                    if not retry:
                        raise caught_error
                    delay = min(
                        self._TRANSIENT_RETRY_INITIAL_DELAY_SEC * 2 ** (retry_attempt - 1),
                        self._TRANSIENT_RETRY_MAX_DELAY_SEC,
                        max(0.0, deadline - loop.time()),
                    )
                    logger.warning(
                        "Transient inference forwarding failure; refreshing proxy URL and retrying "
                        "attempt=%s max_attempts=%s retry_delay_seconds=%.1f error_type=%s",
                        retry_attempt,
                        max_attempts,
                        delay,
                        type(caught_error).__name__,
                    )
                    force_refresh = True
                    await asyncio.sleep(delay)
        except TimeoutError as error:
            raise InferenceForwardingTimeoutError(
                f"inference forwarding exceeded its {timeout_sec:g}-second operation timeout"
            ) from error

    async def _forward(
        self, proxy_url: str, sample_req, model_id: str, *, base_model: str | None
    ) -> types.SampleOutput:
        # model_id matches the LoRA name registered with vLLM during
        # save_weights_for_sampler; base_model is used for non-LoRA sampling.
        model_name = base_model if base_model else model_id

        model_input = sample_req.prompt.to_types()
        prompt_tokens = render_model_input([model_input])[0].prompt_ids

        sp = sample_req.sampling_params
        payload = {
            "model": model_name,
            "prompt": prompt_tokens,
            "n": sample_req.num_samples,
            "seed": sp.seed,
            "max_tokens": sp.max_tokens,
            "temperature": sp.temperature,
            "top_p": sp.top_p,
            "top_k": sp.top_k,
            # vllm-router rejects boolean; 1 = return the chosen token's logprob.
            "logprobs": 1,
            "stream": False,
            "return_token_ids": True,
        }
        # vLLM's `prompt_logprobs` is an int: 0 returns just the prompt tokens'
        # own logprobs, k>0 also returns the top-k per position.
        topk_prompt_logprobs = getattr(sample_req, "topk_prompt_logprobs", 0) or 0
        want_prompt_logprobs = bool(sample_req.prompt_logprobs) or topk_prompt_logprobs > 0
        if want_prompt_logprobs:
            payload["prompt_logprobs"] = topk_prompt_logprobs
        # SamplingParams.stop is polymorphic (list[str] | list[int]).
        stop = getattr(sp, "stop", None)
        if stop:
            if all(isinstance(s, int) for s in stop):
                payload["stop_token_ids"] = list(stop)
            elif all(isinstance(s, str) for s in stop):
                payload["stop"] = list(stop)

        # Pass X-Session-ID for deterministic routing
        headers = {}
        session_id = types.make_routing_session_id(sample_req.sampling_session_id, sample_req.seq_id)
        if session_id is not None:
            headers["X-Session-ID"] = session_id

        url = f"{proxy_url}/v1/completions"
        response = await self._http_client.post(url, json=payload, headers=headers)
        if response.status_code >= 400:
            raise InferenceForwardingError(response.status_code, response.text)
        try:
            result = response.json()
        except ValueError as e:
            # vllm-router can return HTML on transient errors even with 2xx status.
            raise RuntimeError(
                f"vLLM /v1/completions returned non-JSON ({response.status_code}, "
                f"content-type={response.headers.get('content-type')!r}): {response.text[:512]}"
            ) from e

        prompt_logprobs = None
        topk = None
        if want_prompt_logprobs:
            # All `n` choices share one prompt, so vLLM repeats the same prompt
            # logprobs on each choice; read them off the first.
            choices = result.get("choices") or []
            raw = choices[0].get("prompt_logprobs") if choices else None
            if raw is None:
                logger.warning("Requested prompt logprobs but vLLM /v1/completions returned none")
            prompt_logprobs, topk = convert_vllm_prompt_logprobs(prompt_tokens, raw, topk=topk_prompt_logprobs)

        sequences = []
        for choice in result.get("choices", []):
            tokens = choice.get("token_ids", [])
            lp = choice.get("logprobs") or {}
            logprobs = lp.get("token_logprobs") or []
            # vLLM occasionally returns None for logprobs under load; zero-fill so
            # RL advantage computation doesn't see a ragged shape.
            if not logprobs and tokens:
                logger.warning("No logprobs returned from vLLM — filling with zeros")
                logprobs = [0.0] * len(tokens)
            # Tinker's stop_reason is Literal["stop", "length"]; vLLM emits a wider set.
            finish_reason = choice.get("finish_reason")
            stop_reason = "stop" if finish_reason in ("stop", "stop_token") else "length"
            sequences.append(
                types.GeneratedSequence(
                    tokens=tokens,
                    logprobs=logprobs,
                    stop_reason=stop_reason,
                )
            )

        return types.SampleOutput(
            sequences=sequences,
            prompt_logprobs=prompt_logprobs,
            topk_prompt_logprobs=topk,
        )

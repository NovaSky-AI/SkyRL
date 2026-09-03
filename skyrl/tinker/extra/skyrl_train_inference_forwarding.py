"""Forwards EXTERNAL sample requests to the SkyRL-Train-managed vLLM.

Pair to :class:`ExternalInferenceClient`; resolves the target URL from
``EngineStateDB`` instead of from a user-supplied ``external_inference_url``.
"""

import asyncio
from datetime import datetime, timezone

import aiohttp
from sqlmodel.ext.asyncio.session import AsyncSession

from skyrl.backends.renderer import render_model_input
from skyrl.backends.utils import convert_vllm_prompt_logprobs
from skyrl.tinker import types
from skyrl.tinker.config import EngineConfig
from skyrl.tinker.db_models import EngineStateDB, FutureDB, RequestStatus
from skyrl.tinker.external_future_store import ExternalFutureStore, PreparedResult
from skyrl.tinker.extra.completion_decode import CompletionDecoder
from skyrl.tinker.proto_serialization import (
    sample_output_json_from_proto,
    serialize_sample_output,
)
from skyrl.utils.log import logger


class TransientInferenceError(RuntimeError):
    """A 5xx from vllm-router/vLLM: the request was rejected, not executed, so it is safe to retry."""


_ROUTER_CONNECT_TIMEOUT_SECONDS = 60.0


class SkyRLTrainInferenceForwardingClient:
    """Forwards EXTERNAL sample requests to the SkyRL-Train-managed vLLM."""

    # TODO: make `external_future_store` required and remove the FutureDB
    # write-back path in `call_and_store_result` — every production
    # construction (api.py lifespan) already passes a store.
    def __init__(
        self,
        engine_config: EngineConfig,
        db_engine,
        external_future_store: ExternalFutureStore | None = None,
    ):
        self.engine_config = engine_config
        self.db_engine = db_engine
        self.external_future_store = external_future_store
        self._cached_proxy_url: str | None = None
        self._cache_lock = asyncio.Lock()
        # Created on first use so it binds to the serving event loop.
        self._session: aiohttp.ClientSession | None = None
        self._decoder = CompletionDecoder()

    def _get_session(self) -> aiohttp.ClientSession:
        """Return the shared aiohttp session, creating it on first use.

        Backpressure is layered: connector limit -> vllm-router -> vLLM
        max_num_seqs. Default `forwarding_inference_max_connections=None` is
        unlimited; the only cost is file descriptors (raise `ulimit -n`
        accordingly). Requests beyond the limit wait in the connector's FIFO
        queue with no deadline, so a backlog of many thousands of samples
        drains at the engine's pace instead of failing.

        aiohttp rather than httpx: httpcore's pool rescans every connection
        for every request, so its per-request CPU grows with the number of
        in-flight samples (~28ms each at 512 in flight); aiohttp stays flat.
        """
        if self._session is None or self._session.closed:
            max_conn = self.engine_config.forwarding_inference_max_connections
            # keepalive_timeout must stay under the router's idle timeout so a
            # pooled connection is never reused after the server closed it.
            # Happy Eyeballs is off: a burst of connect timeouts cancels its
            # sock_connect calls mid-flight, and under uvloop the closed sockets'
            # descriptors get reused before the loop forgets them ("File
            # descriptor N is used by transport"), failing unrelated forwards.
            connector = aiohttp.TCPConnector(limit=max_conn or 0, keepalive_timeout=2, happy_eyeballs_delay=None)
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(
                    total=None,
                    # A saturated router can take tens of seconds to accept;
                    # that is queueing, not failure.
                    sock_connect=_ROUTER_CONNECT_TIMEOUT_SECONDS,
                    sock_read=self.engine_config.forwarding_inference_timeout_sec,
                ),
            )
        return self._session

    async def aclose(self) -> None:
        """Close the shared aiohttp session. Called from api.py lifespan shutdown."""
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def _read_proxy_url_from_db(self) -> str | None:
        async with AsyncSession(self.db_engine) as session:
            row = await session.get(EngineStateDB, 1)
            if row is None or row.inference_proxy_url is None:
                return None
            return row.inference_proxy_url

    async def _resolve_proxy_url(self, *, force_refresh: bool = False) -> str:
        # Skip the lock when the cache is warm so concurrent samples don't serialize.
        if not force_refresh and self._cached_proxy_url is not None:
            return self._cached_proxy_url
        async with self._cache_lock:
            if force_refresh or self._cached_proxy_url is None:
                url = await self._read_proxy_url_from_db()
                if url is None:
                    raise RuntimeError("inference engine not ready: no proxy URL published to EngineStateDB")
                self._cached_proxy_url = url
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
        """Forward a sample request to vLLM and resolve its future.

        With an ExternalFutureStore the result stays in memory; without one it
        is written back to the request's FutureDB row.
        """
        try:
            result = await self._forward_with_retry(sample_req, model_id, base_model=base_model)
            status = RequestStatus.COMPLETED
        except Exception as e:
            logger.exception("Backend-forwarded sample failed (request_id=%s)", request_id)
            result = types.ErrorResponse(error=str(e), status="failed")
            status = RequestStatus.FAILED

        if self.external_future_store is not None:
            await self.external_future_store.complete(request_id, result, status)
            return

        # TODO: remove this FutureDB write-back once `external_future_store`
        # is required (see __init__).
        async with AsyncSession(self.db_engine) as session:
            future = await session.get(FutureDB, request_id)
            if future is None:
                # Row was deleted between scheduling and completion (cancelled
                # request, stale-session GC). Nothing to write back.
                logger.warning("FutureDB row %s missing on completion write — skipping", request_id)
                return
            # `result_data` is a text column holding pre-serialized JSON.
            if isinstance(result, PreparedResult):
                future.result_data = result.json or sample_output_json_from_proto(result.proto)
            else:
                future.result_data = result.model_dump_json()
            future.status = status
            future.completed_at = datetime.now(timezone.utc)
            await session.commit()

    async def _forward_with_retry(self, sample_req, model_id: str, *, base_model: str | None) -> PreparedResult:
        # Retry only failures where the request demonstrably did not execute:
        # connect-phase errors and 5xx rejections from the router. Read and
        # write failures are ambiguous: vLLM may still be executing the
        # request, so retrying would duplicate generation load.
        try:
            try:
                proxy_url = await self._resolve_proxy_url()
                return await self._forward(proxy_url, sample_req, model_id, base_model=base_model)
            except (aiohttp.ClientConnectorError, aiohttp.ConnectionTimeoutError, TransientInferenceError) as e:
                logger.warning(
                    "Transient error talking to %s (%s: %s) — refreshing proxy URL and retrying once",
                    self._cached_proxy_url,
                    type(e).__name__,
                    e,
                )
                proxy_url = await self._resolve_proxy_url(force_refresh=True)
                return await self._forward(proxy_url, sample_req, model_id, base_model=base_model)
        except aiohttp.SocketTimeoutError as e:
            # Not retried (see above). Long-context requests routinely exceed the
            # default read deadline, so tell the caller how to raise it. The
            # message is stored in the FutureDB ErrorResponse and shown to clients.
            timeout_sec = self.engine_config.forwarding_inference_timeout_sec
            raise RuntimeError(
                f"Inference request to {self._cached_proxy_url} timed out after {timeout_sec:g}s waiting for "
                "a response (read timeout). The request was not retried because vLLM may still be "
                "executing it. If requests are expected to take this long (long prompts, large max_tokens, "
                "or queueing behind other requests), increase the deadline with "
                "`--forwarding-inference-timeout-sec` (EngineConfig.forwarding_inference_timeout_sec) or "
                "the SKYRL_FORWARDING_INFERENCE_TIMEOUT_SEC environment variable."
            ) from e

    async def _forward(self, proxy_url: str, sample_req, model_id: str, *, base_model: str | None) -> PreparedResult:
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
        async with self._get_session().post(url, json=payload, headers=headers) as response:
            body = await response.read()
            if response.status >= 500:
                raise TransientInferenceError(
                    f"vLLM /v1/completions returned {response.status}: {body.decode(errors='replace')}"
                )
            if response.status >= 400:
                raise RuntimeError(f"vLLM /v1/completions returned {response.status}: {body.decode(errors='replace')}")
            try:
                # Token ids and logprobs land directly in int32/float32 arrays;
                # no per-token Python objects are built (see completion_decode).
                choices = self._decoder.decode(body, want_prompt_logprobs=want_prompt_logprobs)
            except ValueError as e:
                # vllm-router can return HTML on transient errors even with 2xx status.
                raise RuntimeError(
                    f"vLLM /v1/completions returned non-JSON ({response.status}, "
                    f"content-type={response.headers.get('content-type')!r}): {body[:512].decode(errors='replace')}"
                ) from e

        prompt_logprobs = None
        topk = None
        if want_prompt_logprobs:
            # All `n` choices share one prompt, so vLLM repeats the same prompt
            # logprobs on each choice; read them off the first.
            raw = choices[0].prompt_logprobs if choices else None
            if raw is None:
                logger.warning("Requested prompt logprobs but vLLM /v1/completions returned none")
            prompt_logprobs, topk = convert_vllm_prompt_logprobs(prompt_tokens, raw, topk=topk_prompt_logprobs)

        # Tinker's stop_reason is Literal["stop", "length"]; vLLM emits a wider set.
        sequences = [
            ("stop" if choice.finish_reason in ("stop", "stop_token") else "length", choice.tokens, choice.logprobs)
            for choice in choices
        ]

        # Encode straight to the proto wire form the SDK retrieves; no pydantic
        # model or JSON text is built for the result (see PreparedResult).
        return PreparedResult(proto=serialize_sample_output(sequences, prompt_logprobs, topk))

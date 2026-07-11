"""Forwards EXTERNAL sample requests to the SkyRL-Train-managed vLLM.

Pair to :class:`ExternalInferenceClient`; resolves the target URL from
``EngineStateDB`` instead of from a user-supplied ``external_inference_url``.
"""

import asyncio
import base64
from datetime import datetime, timezone

import httpx
from sqlmodel.ext.asyncio.session import AsyncSession

from skyrl.backends.renderer import render_model_input
from skyrl.backends.utils import convert_vllm_prompt_logprobs
from skyrl.tinker import types
from skyrl.tinker.config import EngineConfig
from skyrl.tinker.db_models import EngineStateDB, FutureDB, RequestStatus
from skyrl.tinker.external_future_store import ExternalFutureStore
from skyrl.tinker.routed_experts_spool import RoutedExpertsSpool, resolve_spool_dir
from skyrl.utils.log import logger


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
        # Rollout Routing Replay (R3): the engine-managed vLLM attaches
        # per-sample rollout routing to /v1/completions choices when it was
        # launched with routing capture (which the backend enables in lockstep
        # with moe_enable_routing_replay). Since samples forwarded here never
        # touch the engine process, the routing is handed off through a spool
        # directory the training backend reads at forward_backward time. The
        # backend resolves the identical path from the same EngineConfig.
        self._routing_spool = RoutedExpertsSpool(
            resolve_spool_dir(engine_config.routed_experts_spool_dir, engine_config.database_url)
        )
        # Backpressure layered: httpx pool -> vllm-router -> vLLM max_num_seqs.
        # Default `forwarding_inference_max_connections=None` is unlimited;
        # the only cost is file descriptors (raise `ulimit -n` accordingly).
        max_conn = engine_config.forwarding_inference_max_connections
        max_keepalive = max(max_conn // 4, 32) if max_conn is not None else None
        self._http_client: httpx.AsyncClient = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,
                read=engine_config.forwarding_inference_timeout_sec,
                write=300.0,
                pool=300.0,
            ),
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
            future.result_data = result.model_dump_json()
            future.status = status
            future.completed_at = datetime.now(timezone.utc)
            await session.commit()

    async def _forward_with_retry(self, sample_req, model_id: str, *, base_model: str | None) -> types.SampleOutput:
        # Retry only failures that occur before a request can reach vLLM. Read
        # and write failures are ambiguous: vLLM may still be executing the
        # request, so retrying would duplicate generation load.
        try:
            try:
                proxy_url = await self._resolve_proxy_url()
                return await self._forward(proxy_url, sample_req, model_id, base_model=base_model)
            except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                logger.warning(
                    "Connection error talking to %s (%s: %s) — refreshing proxy URL and retrying once",
                    self._cached_proxy_url,
                    type(e).__name__,
                    e,
                )
                proxy_url = await self._resolve_proxy_url(force_refresh=True)
                return await self._forward(proxy_url, sample_req, model_id, base_model=base_model)
        except httpx.ReadTimeout as e:
            # Not retried (see above). Long-context requests routinely exceed the
            # default read deadline, so tell the caller how to raise it. The
            # message is stored in the FutureDB ErrorResponse and shown to clients.
            timeout_sec = self.engine_config.forwarding_inference_timeout_sec
            raise RuntimeError(
                f"Inference request to {self._cached_proxy_url} timed out after {timeout_sec:g}s waiting for "
                "a response (httpx.ReadTimeout). The request was not retried because vLLM may still be "
                "executing it. If requests are expected to take this long (long prompts, large max_tokens, "
                "or queueing behind other requests), increase the deadline with "
                "`--forwarding-inference-timeout-sec` (EngineConfig.forwarding_inference_timeout_sec) or "
                "the SKYRL_FORWARDING_INFERENCE_TIMEOUT_SEC environment variable."
            ) from e

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
            raise RuntimeError(f"vLLM /v1/completions returned {response.status_code}: {response.text}")
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
        routing_items: list[tuple[list[int], str]] = []
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
            # Rollout Routing Replay (R3): vLLM attaches routing (base64 .npy)
            # per choice iff the server was launched with routing capture on.
            routed_experts_b64 = choice.get("routed_experts")
            if routed_experts_b64:
                routing_items.append((tokens, routed_experts_b64))
            sequences.append(
                types.GeneratedSequence(
                    tokens=tokens,
                    logprobs=logprobs,
                    stop_reason=stop_reason,
                )
            )

        # Spool the routing before completing the future so the file exists by
        # the time the client submits forward_backward for these sequences.
        # Base-model samples (no model_id) can never be replayed — skip them.
        # File I/O runs off the event loop; failures must not fail the sample.
        if model_id and routing_items:
            await asyncio.to_thread(self._spool_routed_experts, model_id, prompt_tokens, routing_items)

        return types.SampleOutput(
            sequences=sequences,
            prompt_logprobs=prompt_logprobs,
            topk_prompt_logprobs=topk,
        )

    def _spool_routed_experts(
        self,
        model_id: str,
        prompt_tokens: list[int],
        routing_items: list[tuple[list[int], str]],
    ) -> None:
        """Write each choice's rollout routing to the R3 spool (best-effort).

        Keys files by ``(model_id, prompt + response tokens)`` — exactly the
        sequence the training backend reconstructs in forward_backward. The
        base64 payload is written as raw ``.npy`` bytes without parsing; the
        backend validates shape/dtype when it consumes the file.
        """
        for tokens, routed_experts_b64 in routing_items:
            try:
                npy_bytes = base64.b64decode(routed_experts_b64)
                self._routing_spool.write(model_id, list(prompt_tokens) + list(tokens), npy_bytes)
            except Exception as e:
                logger.warning("R3: failed to spool rollout routing for model %s: %s", model_id, e)

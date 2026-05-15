"""Forwards EXTERNAL sample requests to the SkyRL-Train-managed vLLM.

Pair to :class:`ExternalInferenceClient`; resolves the target URL from
``EngineStateDB`` instead of from a user-supplied ``external_inference_url``.
"""

import asyncio
from datetime import datetime, timezone

import httpx
from sqlmodel.ext.asyncio.session import AsyncSession

from skyrl.backends.renderer import render_model_input
from skyrl.tinker import types
from skyrl.tinker.config import EngineConfig
from skyrl.tinker.db_models import EngineStateDB, FutureDB, RequestStatus
from skyrl.utils.log import logger


class SkyRLTrainInferenceForwardingClient:
    """Forwards EXTERNAL sample requests to the SkyRL-Train-managed vLLM."""

    def __init__(self, engine_config: EngineConfig, db_engine):
        self.engine_config = engine_config
        self.db_engine = db_engine
        self._cached_proxy_url: str | None = None
        self._cache_lock = asyncio.Lock()
        # Backpressure layered: httpx pool -> vllm-router -> vLLM max_num_seqs.
        # Default `forwarding_inference_max_connections=None` is unlimited;
        # the only cost is file descriptors (raise `ulimit -n` accordingly).
        max_conn = engine_config.forwarding_inference_max_connections
        max_keepalive = max(max_conn // 4, 32) if max_conn is not None else None
        self._http_client: httpx.AsyncClient = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=10.0),
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
        """Forward a sample request to vLLM and write the result to FutureDB."""
        try:
            result = await self._forward_with_retry(sample_req, model_id, base_model=base_model)
            result_data = result.model_dump()
            status = RequestStatus.COMPLETED
        except Exception as e:
            logger.exception("Backend-forwarded sample failed (request_id=%s)", request_id)
            result_data = {"error": str(e), "status": "failed"}
            status = RequestStatus.FAILED

        async with AsyncSession(self.db_engine) as session:
            future = await session.get(FutureDB, request_id)
            if future is None:
                # Row was deleted between scheduling and completion (cancelled
                # request, stale-session GC). Nothing to write back.
                logger.warning("FutureDB row %s missing on completion write — skipping", request_id)
                return
            future.result_data = result_data
            future.status = status
            future.completed_at = datetime.now(timezone.utc)
            await session.commit()

    async def _forward_with_retry(self, sample_req, model_id: str, *, base_model: str | None) -> types.SampleOutput:
        # httpx.RequestError covers ConnectError, ReadError, TimeoutException, etc.
        # HTTP 4xx/5xx surfaces as RuntimeError below and is NOT retried.
        try:
            proxy_url = await self._resolve_proxy_url()
            return await self._forward(proxy_url, sample_req, model_id, base_model=base_model)
        except httpx.RequestError as e:
            logger.warning(
                "Network error talking to %s (%s: %s) — refreshing proxy URL and retrying once",
                self._cached_proxy_url,
                type(e).__name__,
                e,
            )
            proxy_url = await self._resolve_proxy_url(force_refresh=True)
            return await self._forward(proxy_url, sample_req, model_id, base_model=base_model)

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
        # SamplingParams.stop is polymorphic (list[str] | list[int]).
        stop = getattr(sp, "stop", None)
        if stop:
            if all(isinstance(s, int) for s in stop):
                payload["stop_token_ids"] = list(stop)
            elif all(isinstance(s, str) for s in stop):
                payload["stop"] = list(stop)

        url = f"{proxy_url}/v1/completions"
        response = await self._http_client.post(url, json=payload)
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

        return types.SampleOutput(sequences=sequences, prompt_logprobs=None)

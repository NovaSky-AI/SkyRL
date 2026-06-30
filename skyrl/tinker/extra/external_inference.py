import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from cloudpathlib import AnyPath
from sqlmodel.ext.asyncio.session import AsyncSession

from skyrl.backends.renderer import render_model_input
from skyrl.tinker import types
from skyrl.tinker.config import EngineConfig
from skyrl.tinker.db_models import FutureDB, RequestStatus
from skyrl.utils.log import logger
from skyrl.utils.storage import download_and_unpack

if TYPE_CHECKING:
    from skyrl.tinker.api import SampleRequest


_MAX_SAMPLE_RETRIES = 6
_RETRY_BASE_DELAY_SEC = 0.25
_RETRY_MAX_DELAY_SEC = 2.0
_ADAPTER_LOAD_TIMEOUT_SEC = 30.0


def _is_retriable_sample_error(exc: Exception, is_lora: bool) -> bool:
    """Return whether a failed sample request should be retried after a proactive adapter load.

    Transient transport errors are always retried. For LoRA sampling the engine loads the adapter
    lazily, so a freshly extracted adapter can be sampled before it is registered; the engine's
    "adapter not loaded yet" responses (a 404, or a 400 whose body reports the model does not exist)
    are retried too. For base-model sampling those same responses are a permanent configuration error
    (e.g. an unknown base-model name), so they are not retried. Malformed requests and server errors
    are never retried.
    """
    if isinstance(exc, httpx.TransportError):
        return True
    if is_lora and isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        if status_code == 404:
            return True
        if status_code == 400:
            body = exc.response.text.lower()
            return "does not exist" in body or "not found" in body
    return False


def _extract_checkpoint_sync(checkpoint_path: AnyPath, target_dir: Path) -> None:
    """Extract a LoRA checkpoint to disk for vLLM to load.

    This is a blocking operation (filesystem/network I/O) and should be called
    via asyncio.to_thread() to avoid blocking the event loop.
    """
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    # Extract the checkpoint if it doesn't already exist
    if not target_dir.exists():
        try:
            with download_and_unpack(checkpoint_path) as extracted_path:
                extracted_path.rename(target_dir)
        except FileExistsError:
            # This could happen if two processes try to download the file.
            # In that case the other process won the race and created target_dir.
            pass


class ExternalInferenceClient:
    """Client for calling external inference engines (e.g., vLLM)."""

    def __init__(self, engine_config: EngineConfig, db_engine):
        self.base_url = f"{engine_config.external_inference_url}/v1"
        self.api_key = engine_config.external_inference_api_key
        self.checkpoints_base = engine_config.checkpoints_base
        self.lora_base_dir = engine_config.external_inference_lora_base
        self.db_engine = db_engine

    async def call_and_store_result(
        self,
        request_id: int,
        sample_req,
        model_id: str,
        checkpoint_id: str,
        *,
        base_model: str | None = None,
    ):
        """Background task to call external engine and store result in database."""
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=httpx.Timeout(300.0, connect=10.0),  # 5 minutes for inference, 10s for connect
            ) as http_client:
                result = await self._forward_to_engine(
                    sample_req, model_id, checkpoint_id, http_client, base_model=base_model
                )
            result_data = result.model_dump()
            status = RequestStatus.COMPLETED
        except Exception as e:
            logger.exception("External engine error")
            result_data = {"error": str(e), "status": "failed"}
            status = RequestStatus.FAILED

        async with AsyncSession(self.db_engine) as session:
            future = await session.get(FutureDB, request_id)
            future.result_data = result_data
            future.status = status
            future.completed_at = datetime.now(timezone.utc)
            await session.commit()

    async def _request_adapter_load(self, http_client: httpx.AsyncClient, model_name: str) -> None:
        """Ask the engine to load the adapter for ``model_name`` from its extracted directory."""
        try:
            await http_client.post(
                "/load_lora_adapter",
                json={"lora_name": model_name, "lora_path": str(self.lora_base_dir / model_name)},
                timeout=_ADAPTER_LOAD_TIMEOUT_SEC,
            )
        except Exception:
            logger.debug("Proactive load_lora_adapter call failed; retrying sample request")

    async def _post_completion(
        self,
        http_client: httpx.AsyncClient,
        payload: dict,
        headers: dict,
        model_name: str,
        *,
        base_model: str | None,
    ) -> dict:
        """Send the completion request, retrying through the engine's lazy adapter-load race."""
        is_lora = base_model is None
        for attempt in range(_MAX_SAMPLE_RETRIES):
            try:
                response = await http_client.post("/completions", json=payload, headers=headers)
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                if attempt == _MAX_SAMPLE_RETRIES - 1 or not _is_retriable_sample_error(exc, is_lora):
                    raise
                if is_lora:
                    await self._request_adapter_load(http_client, model_name)
                await asyncio.sleep(min(_RETRY_MAX_DELAY_SEC, _RETRY_BASE_DELAY_SEC * (2**attempt)))

    async def _forward_to_engine(
        self,
        request: "SampleRequest",
        model_id: str,
        checkpoint_id: str,
        http_client: httpx.AsyncClient,
        *,
        base_model: str | None = None,
    ) -> types.SampleOutput:
        """Forward request to vLLM with dynamic LoRA loading.

        Extracts the checkpoint to the configured external_inference_lora_base and references it by a model name
        that vLLM can dynamically load via the lora_filesystem_resolver plugin.

        For base model sampling (no LoRA), the request is sent directly using the base model name.
        """
        model_input = request.prompt.to_types()
        prompt_tokens = render_model_input([model_input])[0].prompt_ids

        if base_model:
            # Base model sampling: use the model name directly, no LoRA checkpoint needed
            model_name = base_model
        else:
            # LoRA sampling: extract checkpoint and reference it by name for dynamic loading
            model_name = f"{model_id}_{checkpoint_id}"
            checkpoint_path = self.checkpoints_base / model_id / "sampler_weights" / f"{checkpoint_id}.tar.gz"
            target_dir = self.lora_base_dir / model_name

            await asyncio.to_thread(_extract_checkpoint_sync, checkpoint_path, target_dir)

        payload = {
            "model": model_name,
            "prompt": prompt_tokens,
            "n": request.num_samples,
            "seed": request.sampling_params.seed,
            "max_tokens": request.sampling_params.max_tokens,
            "temperature": request.sampling_params.temperature,
            "top_p": request.sampling_params.top_p,
            "top_k": request.sampling_params.top_k,
            "logprobs": True,
            "stream": False,
            "return_token_ids": True,
        }

        # Pass X-Session-ID for deterministic routing
        headers = {}
        session_id = types.make_routing_session_id(request.sampling_session_id, request.seq_id)
        if session_id is not None:
            headers["X-Session-ID"] = session_id

        result = await self._post_completion(http_client, payload, headers, model_name, base_model=base_model)

        sequences = []
        for choice in result["choices"]:
            lp = choice["logprobs"]
            sequences.append(
                types.GeneratedSequence(
                    tokens=choice["token_ids"],
                    logprobs=lp["token_logprobs"],
                    stop_reason=choice["finish_reason"],
                )
            )

        return types.SampleOutput(sequences=sequences, prompt_logprobs=[])

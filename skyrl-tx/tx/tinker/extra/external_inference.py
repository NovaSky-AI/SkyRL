import httpx
from datetime import datetime, timezone
from sqlmodel.ext.asyncio.session import AsyncSession

from tx.tinker import types
from tx.tinker.db_models import FutureDB, RequestStatus
from tx.utils.log import logger


class ExternalInferenceClient:
    """Client for calling external inference engines (e.g., vLLM)."""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = f"{base_url}/v1"
        self.api_key = api_key

    async def call_and_store_result(
        self,
        db_engine,
        request_id: int,
        sample_req,
        checkpoint_path: str,
    ):
        """Background task to call external engine and store result in database."""
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=httpx.Timeout(300.0, connect=10.0),  # 5 minutes for inference, 10s for connect
            ) as http_client:
                result = await self._forward_to_engine(sample_req, checkpoint_path, http_client)

            async with AsyncSession(db_engine) as session:
                future = await session.get(FutureDB, request_id)
                if future:
                    future.result_data = result.model_dump()
                    future.status = RequestStatus.COMPLETED
                    future.completed_at = datetime.now(timezone.utc)
                    await session.commit()
        except Exception as e:
            logger.exception("External engine error")
            async with AsyncSession(db_engine) as session:
                future = await session.get(FutureDB, request_id)
                if future:
                    future.result_data = {"error": str(e), "status": "failed"}
                    future.status = RequestStatus.FAILED
                    future.completed_at = datetime.now(timezone.utc)
                    await session.commit()

    async def _forward_to_engine(
        self, request, checkpoint_path: str, http_client: httpx.AsyncClient
    ) -> types.SampleOutput:
        """Forward request to vLLM."""
        prompt_tokens = [token for chunk in request.prompt.chunks for token in chunk.tokens]

        payload = {
            # "model": checkpoint_path,
            "model": "Qwen/Qwen3-4B",  # TODO: Currently this is hard coded
            "prompt": prompt_tokens,
            "max_tokens": request.sampling_params.max_tokens,
            "temperature": request.sampling_params.temperature,
            "logprobs": True,
            "stream": False,
            "return_token_ids": True,
            "seed": request.sampling_params.seed,
        }

        if request.sampling_params.stop is not None:
            payload["stop_token_ids"] = request.sampling_params.stop

        response = await http_client.post("/completions", json=payload)
        response.raise_for_status()
        result = response.json()

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

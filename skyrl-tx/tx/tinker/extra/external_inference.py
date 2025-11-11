import httpx
from datetime import datetime, timezone
from sqlmodel.ext.asyncio.session import AsyncSession

from tx.tinker import types
from tx.tinker.db_models import FutureDB, RequestStatus
from tx.utils.log import logger


class ExternalInferenceClient:
    """Client for calling external inference engines (e.g., SGLang)."""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
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
                result = await self._forward_to_engine(sample_req, checkpoint_path, http_client, request_id)

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
        self, request, checkpoint_path: str, http_client: httpx.AsyncClient, request_id: int
    ) -> types.SampleOutput:
        """Forward request to SGLang."""
        prompt_tokens = [token for chunk in request.prompt.chunks for token in chunk.tokens]

        # Build sampling_params for SGLang
        sampling_params = {
            "max_new_tokens": request.sampling_params.max_tokens,
            "temperature": request.sampling_params.temperature,
        }

        if request.sampling_params.seed is not None:
            sampling_params["seed"] = request.sampling_params.seed

        if request.sampling_params.stop is not None:
            sampling_params["stop_token_ids"] = request.sampling_params.stop

        payload = {
            "input_ids": prompt_tokens,
            "sampling_params": sampling_params,
            "return_logprob": True,
            "top_logprobs_num": 1,
            "logprob_start_len": 0,
        }

        logger.info(f"=== EXTERNAL ENGINE REQUEST (request_id={request_id}) ===")
        logger.info(f"Prompt length: {len(prompt_tokens)}, Max tokens: {sampling_params['max_new_tokens']}, Temp: {sampling_params['temperature']}, Seed: {sampling_params.get('seed')}")
        logger.info(f"Stop tokens: {request.sampling_params.stop}")
        logger.info(f"Full payload (excluding input_ids): {dict((k, v) for k, v in payload.items() if k != 'input_ids')}")

        response = await http_client.post("/generate", json=payload)
        response.raise_for_status()
        result = response.json()

        generated_tokens = result['output_ids']
        finish_reason_data = result['meta_info']['finish_reason']
        finish_reason = finish_reason_data.get('type', 'unknown') if isinstance(finish_reason_data, dict) else str(finish_reason_data)

        logger.info(f"=== EXTERNAL ENGINE RESPONSE (request_id={request_id}) ===")
        logger.info(f"Finish reason: {finish_reason}, Matched token: {finish_reason_data.get('matched') if isinstance(finish_reason_data, dict) else 'N/A'}")
        logger.info(f"Generated {len(generated_tokens)} tokens: {generated_tokens[:20]}...")
        if request.sampling_params.stop:
            logger.info(f"Stop token {request.sampling_params.stop[0]} in generated? {request.sampling_params.stop[0] in generated_tokens}")

        # Extract logprobs - SGLang returns logprobs as a list of token logprobs
        logprobs = result.get('meta_info', {}).get('output_token_logprobs', [])

        sequences = [
            types.GeneratedSequence(
                tokens=generated_tokens,
                logprobs=logprobs,
                stop_reason=finish_reason,
            )
        ]

        return types.SampleOutput(sequences=sequences, prompt_logprobs=[])

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
            "no_stop_trim": True,  # Prevent SGLang from trimming stop tokens from output
        }

        if request.sampling_params.seed is not None:
            sampling_params["seed"] = request.sampling_params.seed

        if request.sampling_params.stop is not None:
            sampling_params["stop_token_ids"] = request.sampling_params.stop

            # DEBUG: Check if stop token appears in the top logprobs to see if it's even being considered
            logger.info(f"Stop tokens configured: {request.sampling_params.stop}")

        payload = {
            "input_ids": prompt_tokens,
            "sampling_params": sampling_params,
            "return_logprob": True,
            "top_logprobs_num": 5,  # Get top 5 to see if stop token ever appears
            "logprob_start_len": 0,
        }

        logger.info(f"=== EXTERNAL ENGINE REQUEST (request_id={request_id}) ===")
        logger.info(f"Prompt length: {len(prompt_tokens)}, Max tokens: {sampling_params['max_new_tokens']}, Temp: {sampling_params['temperature']}, Seed: {sampling_params.get('seed')}")
        logger.info(f"Stop token IDs: {request.sampling_params.stop}")
        logger.info(f"Last 10 prompt tokens: {prompt_tokens[-10:]}")
        logger.info(f"Full sampling_params: {sampling_params}")
        logger.info(f"Full payload (excluding input_ids): {dict((k, v) for k, v in payload.items() if k != 'input_ids')}")

        response = await http_client.post("/generate", json=payload)
        response.raise_for_status()
        result = response.json()

        generated_tokens = result['output_ids']
        finish_reason_data = result['meta_info']['finish_reason']
        finish_reason = finish_reason_data.get('type', 'unknown') if isinstance(finish_reason_data, dict) else str(finish_reason_data)

        logger.info(f"=== EXTERNAL ENGINE RESPONSE (request_id={request_id}) ===")
        logger.info(f"Finish reason: {finish_reason}, Matched token: {finish_reason_data.get('matched') if isinstance(finish_reason_data, dict) else 'N/A'}")
        logger.info(f"Generated {len(generated_tokens)} tokens")
        logger.info(f"First 20 tokens: {generated_tokens[:20]}")
        logger.info(f"Last 20 tokens: {generated_tokens[-20:]}")
        if request.sampling_params.stop:
            for stop_token in request.sampling_params.stop:
                logger.info(f"Stop token {stop_token} in generated? {stop_token in generated_tokens}")

        # Extract logprobs - SGLang returns logprobs as a list of [logprob, token_id, ...] tuples
        # We need to extract just the logprob values (first element of each tuple)
        raw_logprobs = result.get('meta_info', {}).get('output_token_logprobs', [])
        if raw_logprobs and isinstance(raw_logprobs[0], list):
            # SGLang format: [[logprob, token_id, ...], ...]
            logprobs = [lp[0] for lp in raw_logprobs]

            # Log first few tokens with their logprobs for debugging
            logger.info(f"First 10 generated (token, logprob): {[(generated_tokens[i], logprobs[i]) for i in range(min(10, len(generated_tokens)))]}")
            logger.info(f"Last 10 generated (token, logprob): {[(generated_tokens[i], logprobs[i]) for i in range(max(0, len(generated_tokens)-10), len(generated_tokens))]}")

            # Check if stop token ever appeared in top 5 alternatives
            if request.sampling_params.stop:
                stop_token = request.sampling_params.stop[0]
                top_logprobs_full = result.get('meta_info', {}).get('output_top_logprobs', [])
                if top_logprobs_full:
                    # Each entry is a list of [logprob, token_id] pairs for top tokens at that position
                    positions_with_stop = []
                    for i, top_tokens in enumerate(top_logprobs_full):
                        if any(token_id == stop_token for _, token_id, *_ in top_tokens):
                            positions_with_stop.append(i)

                    if positions_with_stop:
                        logger.info(f"Stop token {stop_token} appeared in top-5 at positions: {positions_with_stop[:10]}... (showing first 10)")
                    else:
                        logger.warning(f"Stop token {stop_token} NEVER appeared in top-5 alternatives across {len(top_logprobs_full)} positions")
        else:
            # Fallback to raw format if it's already a list of floats
            logprobs = raw_logprobs

        sequences = [
            types.GeneratedSequence(
                tokens=generated_tokens,
                logprobs=logprobs,
                stop_reason=finish_reason,
            )
        ]

        return types.SampleOutput(sequences=sequences, prompt_logprobs=[])

"""SkyRL-Train inference client for Tinker API integration.

This module provides a client that calls skyrl-train's InferenceEngineClient
and handles Tinker type conversion and database storage for the API server.

Architecture:
    skyrl-tx API (/api/v1/asample) -> SkyRLInferenceClient -> InferenceEngineClient.sample()

Usage:
    # From skyrl-train, after initializing inference engines:
    from tx.tinker.extra.skyrl_inference import attach_skyrl_inference

    # Attach to running API server
    attach_skyrl_inference(app, inference_client)
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlmodel.ext.asyncio.session import AsyncSession
from tx.tinker import types
from tx.tinker.db_models import FutureDB, RequestStatus
from tx.utils.log import logger

if TYPE_CHECKING:
    from fastapi import FastAPI
    from skyrl_train.inference_engines.inference_engine_client import (
        InferenceEngineClient,
    )


class SkyRLInferenceClient:
    """Client for calling skyrl-train's inference engines via Tinker API.

    This client:
    1. Converts Tinker pydantic types to/from skyrl-train format
    2. Calls InferenceEngineClient.sample() directly
    3. Stores results in the database for async API requests

    Usage:
        # During app startup
        inference_client = InferenceEngineClient(engines, tokenizer, config)
        skyrl_client = SkyRLInferenceClient(inference_client, db_engine)
        app.state.skyrl_inference_client = skyrl_client

        # In /api/v1/asample endpoint
        asyncio.create_task(skyrl_client.call_and_store_result(request_id, sample_req))
    """

    def __init__(self, inference_client: "InferenceEngineClient", db_engine):
        """Initialize the SkyRL inference client.

        Args:
            inference_client: skyrl-train's InferenceEngineClient with engines initialized.
            db_engine: SQLModel async engine for storing results in FutureDB.
        """
        self.inference_client = inference_client
        self.db_engine = db_engine

    async def call_and_store_result(
        self,
        request_id: int,
        sample_req,
        model_id: str = "",
        checkpoint_id: str = "",
    ):
        """Background task to call skyrl-train inference and store result in database.

        Args:
            request_id: FutureDB request ID to update with results.
            sample_req: SampleRequest from the API endpoint.
            model_id: Model identifier (unused for now, skyrl-train uses pre-loaded model).
            checkpoint_id: Checkpoint identifier (unused for now).
        """
        try:
            result = await self._sample(sample_req)
            result_data = result.model_dump()
            status = RequestStatus.COMPLETED
        except Exception as e:
            logger.exception("SkyRL inference error")
            result_data = {"error": str(e), "status": "failed"}
            status = RequestStatus.FAILED

        async with AsyncSession(self.db_engine) as session:
            future = await session.get(FutureDB, request_id)
            future.result_data = result_data
            future.status = status
            future.completed_at = datetime.now(timezone.utc)
            await session.commit()

    async def _sample(self, request) -> types.SampleOutput:
        """Call skyrl-train's sample() and convert response to Tinker types.

        Args:
            request: SampleRequest from the API endpoint.

        Returns:
            types.SampleOutput with generated sequences.
        """
        # Convert Tinker ModelInput to flat token list
        prompt_tokens = self._extract_prompt_tokens(request.prompt)

        # Convert Tinker SamplingParams to dict
        sampling_params = self._convert_sampling_params(request.sampling_params)

        # Call skyrl-train's InferenceEngineClient directly
        # Note: We don't pass session_id (defaults to None for random load-balancing).
        # Tinker's sampling_session_id/seq_id identify model checkpoints, not conversations,
        # and each sample() call is independent with no KV cache benefit from sticky routing.
        # This matches official Tinker backend behavior.
        result = await self.inference_client.sample(
            prompt_token_ids=prompt_tokens,
            num_samples=request.num_samples,
            sampling_params=sampling_params,
        )

        # Convert result to Tinker types
        return self._convert_to_sample_output(result)

    def _extract_prompt_tokens(self, model_input: types.ModelInput) -> list[int]:
        """Extract flat token list from Tinker ModelInput.

        Args:
            model_input: Tinker ModelInput with chunks of tokens.

        Returns:
            Flat list of token IDs.
        """
        return [token for chunk in model_input.chunks for token in chunk.tokens]

    def _convert_sampling_params(self, params: types.SamplingParams) -> dict:
        """Convert Tinker SamplingParams to dict for skyrl-train.

        Args:
            params: Tinker SamplingParams pydantic model.

        Returns:
            Dict compatible with skyrl-train's sampling.
        """
        result = {
            "temperature": params.temperature,
            "max_tokens": params.max_tokens,
            "top_k": params.top_k,
            "top_p": params.top_p,
        }

        if params.seed is not None:
            result["seed"] = params.seed

        # Handle stop tokens/strings
        if params.stop_tokens:
            result["stop_token_ids"] = params.stop_tokens
        if params.stop_strings:
            result["stop"] = params.stop_strings

        return result

    def _convert_to_sample_output(self, result: dict) -> types.SampleOutput:
        """Convert InferenceEngineOutput to Tinker SampleOutput.

        Args:
            result: InferenceEngineOutput dict from skyrl-train's sample().

        Returns:
            types.SampleOutput with GeneratedSequence list.
        """
        sequences = []
        num_samples = len(result["response_ids"])

        for i in range(num_samples):
            # Map skyrl-train stop reasons to Tinker format
            stop_reason = result["stop_reasons"][i]
            if stop_reason in ("stop", "eos"):
                tinker_stop_reason = "stop"
            else:
                tinker_stop_reason = "length"

            # Extract logprobs if available
            logprobs = []
            if result.get("response_logprobs") and result["response_logprobs"][i]:
                logprobs = result["response_logprobs"][i]

            sequences.append(
                types.GeneratedSequence(
                    tokens=result["response_ids"][i],
                    logprobs=logprobs,
                    stop_reason=tinker_stop_reason,
                )
            )

        # Note: prompt_logprobs not supported yet in skyrl-train's sample()
        return types.SampleOutput(
            sequences=sequences,
            prompt_logprobs=None,
        )


def attach_skyrl_inference(app: "FastAPI", inference_client: "InferenceEngineClient") -> None:
    """Attach SkyRL inference client to an existing FastAPI app.

    This enables the /api/v1/asample endpoint to use skyrl-train's inference
    engines directly instead of the internal JAX backend or external vLLM.

    Args:
        app: The FastAPI app instance (must have db_engine in state).
        inference_client: Initialized InferenceEngineClient from skyrl-train.

    Example:
        # In skyrl-train after engines are initialized:
        from tx.tinker.extra.skyrl_inference import attach_skyrl_inference

        app = get_running_api_app()  # Get the FastAPI app
        attach_skyrl_inference(app, llm_client)
    """
    if not hasattr(app.state, "db_engine"):
        raise RuntimeError("App must have db_engine initialized before attaching SkyRL inference")

    skyrl_client = SkyRLInferenceClient(inference_client, app.state.db_engine)
    app.state.skyrl_inference_client = skyrl_client

    # Also set as external_inference_client so existing endpoint code routes to it
    app.state.external_inference_client = skyrl_client

    logger.info("SkyRL-train inference client attached to API server")

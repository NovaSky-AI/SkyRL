"""SkyRL-Train training client for Tinker API integration.

This module provides a thin wrapper around skyrl-train's TinkerTrainingAdapter
that handles Tinker type conversion and database storage for the API server.

The core training logic lives in skyrl-train's TinkerTrainingAdapter,
keeping skyrl-tx as a lightweight integration layer.

Architecture:
    skyrl-tx API (/api/v1/forward_backward) -> SkyRLTrainingClient -> TinkerTrainingAdapter -> WorkerDispatch

Usage:
    # From skyrl-train, after initializing workers:
    from tx.tinker.extra.skyrl_training import attach_skyrl_training

    # Attach to running API server
    attach_skyrl_training(app, worker_dispatch)
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from sqlmodel.ext.asyncio.session import AsyncSession

from tx.tinker import types
from tx.tinker.db_models import FutureDB, RequestStatus
from tx.utils.log import logger

if TYPE_CHECKING:
    from fastapi import FastAPI
    from skyrl_train.workers.worker_dispatch import WorkerDispatch


class SkyRLTrainingClient:
    """Client for calling skyrl-train's training workers via Tinker API.

    This is a thin wrapper around skyrl-train's TinkerTrainingAdapter that:
    1. Converts Tinker pydantic types to/from plain Python types
    2. Stores results in the database for async API requests

    The core training logic lives in skyrl-train's TinkerTrainingAdapter.

    Usage:
        # During app startup
        worker_dispatch = WorkerDispatch(cfg, policy_actor_group, ...)
        skyrl_client = SkyRLTrainingClient(worker_dispatch, db_engine)
        app.state.skyrl_training_client = skyrl_client

        # In /api/v1/forward_backward endpoint
        asyncio.create_task(skyrl_client.call_forward_backward_and_store(request_id, fwd_bwd_input))
    """

    def __init__(self, worker_dispatch: "WorkerDispatch", db_engine):
        """Initialize the SkyRL training client.

        Args:
            worker_dispatch: skyrl-train's WorkerDispatch with workers initialized.
            db_engine: SQLModel async engine for storing results in FutureDB.
        """
        # Import here to avoid circular imports and allow skyrl-tx to work without skyrl-train
        from skyrl_train.training.tinker_adapter import TinkerTrainingAdapter

        self.adapter = TinkerTrainingAdapter(worker_dispatch)
        self.db_engine = db_engine

    async def call_forward_backward_and_store(
        self,
        request_id: int,
        fwd_bwd_input: types.ForwardBackwardInput,
        model_id: str = "",
    ):
        """Background task to call forward_backward and store result in database.

        Args:
            request_id: FutureDB request ID to update with results.
            fwd_bwd_input: ForwardBackwardInput from the API endpoint.
            model_id: Model identifier (unused for now, uses default policy model).
        """
        try:
            result = await self._forward_backward(fwd_bwd_input)
            result_data = result.model_dump()
            status = RequestStatus.COMPLETED
        except Exception as e:
            logger.exception("SkyRL training forward_backward error")
            result_data = {"error": str(e), "status": "failed"}
            status = RequestStatus.FAILED

        async with AsyncSession(self.db_engine) as session:
            future = await session.get(FutureDB, request_id)
            future.result_data = result_data
            future.status = status
            future.completed_at = datetime.now(timezone.utc)
            await session.commit()

    async def call_optim_step_and_store(
        self,
        request_id: int,
        optim_input: types.OptimStepInput,
        model_id: str = "",
    ):
        """Background task to call optim_step and store result in database.

        Args:
            request_id: FutureDB request ID to update with results.
            optim_input: OptimStepInput from the API endpoint.
            model_id: Model identifier (unused for now, uses default policy model).
        """
        try:
            result = await self._optim_step(optim_input)
            result_data = result.model_dump()
            status = RequestStatus.COMPLETED
        except Exception as e:
            logger.exception("SkyRL training optim_step error")
            result_data = {"error": str(e), "status": "failed"}
            status = RequestStatus.FAILED

        async with AsyncSession(self.db_engine) as session:
            future = await session.get(FutureDB, request_id)
            future.result_data = result_data
            future.status = status
            future.completed_at = datetime.now(timezone.utc)
            await session.commit()

    async def _forward_backward(
        self, fwd_bwd_input: types.ForwardBackwardInput
    ) -> types.ForwardBackwardOutput:
        """Call skyrl-train's forward_backward and convert response to Tinker types.

        Args:
            fwd_bwd_input: ForwardBackwardInput with data and loss_fn.

        Returns:
            types.ForwardBackwardOutput with loss_fn_outputs and metrics.
        """
        # Convert Tinker Datum list to plain Python dicts
        data = self._convert_data(fwd_bwd_input.data)

        # Call skyrl-train's adapter
        result = await self.adapter.forward_backward(
            data=data,
            loss_fn=fwd_bwd_input.loss_fn,
        )

        # Convert result to Tinker types
        return types.ForwardBackwardOutput(
            loss_fn_output_type="per_datum",
            loss_fn_outputs=result.loss_fn_outputs,
            metrics=result.metrics,
        )

    async def _optim_step(
        self, optim_input: types.OptimStepInput
    ) -> types.OptimStepOutput:
        """Call skyrl-train's optim_step and convert response to Tinker types.

        Args:
            optim_input: OptimStepInput with adam_params.

        Returns:
            types.OptimStepOutput (currently empty).
        """
        # Call skyrl-train's adapter
        # Note: SkyRL uses scheduler-based LR, so learning_rate is informational
        await self.adapter.optim_step(
            learning_rate=optim_input.adam_params.learning_rate,
        )

        return types.OptimStepOutput()

    def _convert_data(self, data: List[types.Datum]) -> List[Dict[str, Any]]:
        """Convert Tinker Datum list to plain Python dicts.

        Args:
            data: List of Tinker Datum pydantic models.

        Returns:
            List of dicts compatible with TinkerTrainingAdapter.
        """
        result = []
        for datum in data:
            # Extract tokens from ModelInput
            tokens = []
            for chunk in datum.model_input.chunks:
                tokens.extend(chunk.tokens)

            # Extract loss_fn_inputs
            loss_fn_inputs = {}
            if datum.loss_fn_inputs.target_tokens:
                loss_fn_inputs["target_tokens"] = datum.loss_fn_inputs.target_tokens.data
            if datum.loss_fn_inputs.weights:
                loss_fn_inputs["weights"] = datum.loss_fn_inputs.weights.data
            if datum.loss_fn_inputs.advantages:
                loss_fn_inputs["advantages"] = datum.loss_fn_inputs.advantages.data
            if datum.loss_fn_inputs.logprobs:
                loss_fn_inputs["logprobs"] = datum.loss_fn_inputs.logprobs.data

            result.append({
                "model_input": {"tokens": tokens},
                "loss_fn_inputs": loss_fn_inputs,
            })

        return result


def attach_skyrl_training(app: "FastAPI", worker_dispatch: "WorkerDispatch") -> None:
    """Attach SkyRL training client to an existing FastAPI app.

    This enables the /api/v1/forward_backward and /api/v1/optim_step endpoints
    to use skyrl-train's workers directly instead of the internal JAX backend.

    Args:
        app: The FastAPI app instance (must have db_engine in state).
        worker_dispatch: Initialized WorkerDispatch from skyrl-train.

    Example:
        # In skyrl-train after workers are initialized:
        from tx.tinker.extra.skyrl_training import attach_skyrl_training

        app = get_running_api_app()  # Get the FastAPI app
        attach_skyrl_training(app, worker_dispatch)
    """
    if not hasattr(app.state, "db_engine"):
        raise RuntimeError("App must have db_engine initialized before attaching SkyRL training")

    skyrl_client = SkyRLTrainingClient(worker_dispatch, app.state.db_engine)
    app.state.skyrl_training_client = skyrl_client

    # Also set as external_training_client so existing endpoint code routes to it
    app.state.external_training_client = skyrl_client

    logger.info("SkyRL-train training client attached to API server")

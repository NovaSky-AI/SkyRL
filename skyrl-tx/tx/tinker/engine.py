"""Background engine for processing training requests."""

import argparse
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pydantic import BaseModel
from sqlmodel import create_engine, Session, select, update, func

from tx.tinker.db_models import FutureDB, RequestStatus, CheckpointDB, CheckpointStatus
from tx.tinker import types
from tx.tinker.config import EngineConfig, add_model
from tx.tinker.backends.jax import JaxBackend, JaxBackendConfig
from tx.tinker.backends.utils import log_timing
from tx.tinker.loss_fns import LOSS_TYPES
from tx.utils.log import logger


BACKENDS = {
    "jax": (JaxBackend, JaxBackendConfig),
}


class TinkerEngine:
    """Background engine for processing training requests.

    The engine handles:
    - Database operations (futures, checkpoints)
    - Request finding/scheduling
    - File I/O (download/upload checkpoints)
    - Validating requests against loaded models

    Computation and model management are delegated to the backend.
    """

    def _filter_valid_requests(
        self,
        requests: dict[str, tuple[str, BaseModel]],
    ) -> tuple[dict[str, types.ErrorResponse], dict[str, tuple[str, BaseModel]]]:
        """Filter out requests with invalid model_ids and return error results for them.

        Args:
            requests: Dict mapping request_id to (model_id, request_data) tuples

        Returns:
            Tuple of (error_results, valid_requests)
        """
        results = {}
        valid_requests = {}

        for request_id, (model_id, request_data) in requests.items():
            error = None
            if model_id and not self.backend.has_model(model_id):
                error = f"Model {model_id} not loaded"
            elif not model_id and isinstance(request_data, types.SampleInput):
                if request_data.base_model != self.config.base_model:
                    error = f"Engine is configured for '{self.config.base_model}' but request specified '{request_data.base_model}'"
                elif request_data.checkpoint_id:
                    error = "checkpoint_id must be empty for base model sampling"

            if error:
                results[request_id] = types.ErrorResponse(error=error, status="failed")
            else:
                valid_requests[request_id] = (model_id, request_data)

        return results, valid_requests

    def _prepare_model_pass_batch(
        self,
        requests: dict[str, tuple[str, types.ForwardBackwardInput]],
    ) -> types.PreparedModelPassBatch:
        """Prepare batch data for forward/forward_backward operations.

        Extracts tokens, targets, and metadata from requests into lists
        that the backend will convert to arrays.

        Args:
            requests: Dict mapping request_id to (model_id, request_data) tuples (pre-validated)

        Returns:
            PreparedModelPassBatch with all data extracted from requests
        """
        all_input_ids = []
        all_targets = []
        all_token_weights = []
        all_model_ids = []
        all_sampling_logprobs = []
        all_advantages = []
        all_loss_fn_types = []
        request_batch_slices = []

        for request_id, (model_id, request_data) in requests.items():
            loss_fn_type = LOSS_TYPES[request_data.loss_fn]

            request_start = len(all_input_ids)
            for item in request_data.data:
                tokens = [t for chunk in item.model_input.chunks for t in chunk.tokens]
                all_input_ids.append(tokens)
                loss_fn_inputs = item.loss_fn_inputs
                all_targets.append(loss_fn_inputs.target_tokens.data)
                all_token_weights.append(loss_fn_inputs.weights.data)
                all_sampling_logprobs.append(loss_fn_inputs.logprobs.data)
                all_advantages.append(loss_fn_inputs.advantages.data)
                all_model_ids.append(model_id)
                all_loss_fn_types.append(loss_fn_type)

            request_batch_slices.append((request_id, model_id, request_start, len(all_input_ids)))

        return types.PreparedModelPassBatch(
            all_input_ids=all_input_ids,
            all_targets=all_targets,
            all_token_weights=all_token_weights,
            all_sampling_logprobs=all_sampling_logprobs,
            all_advantages=all_advantages,
            all_model_ids=all_model_ids,
            all_loss_fn_types=all_loss_fn_types,
            request_batch_slices=request_batch_slices,
        )

    def _prepare_sample_batch(
        self,
        requests: dict[str, tuple[str, types.SampleInput]],
    ) -> types.PreparedSampleBatch:
        """Prepare batch data for sample operations.

        Extracts prompts and sampling params from requests into lists
        that the backend will convert to arrays.

        Args:
            requests: Dict mapping request_id to (model_id, request_data) tuples (pre-validated)

        Returns:
            PreparedSampleBatch with all data extracted from requests
        """
        all_prompts = []
        all_sampling_params = []
        all_model_ids = []
        all_checkpoint_ids = []
        all_checkpoint_paths = []
        request_batch_slices = []

        needs_prompt_logprobs = any(request_data.prompt_logprobs for (_, request_data) in requests.values())

        for request_id, (model_id, request_data) in requests.items():
            request_start = len(all_prompts)

            # Expand requests for num_samples
            prompt_tokens = [token for chunk in request_data.prompt.chunks for token in chunk.tokens]
            checkpoint_path = ""
            if model_id and request_data.checkpoint_id:
                checkpoint_path = str(
                    self.config.checkpoints_base / model_id / "sampler_weights" / f"{request_data.checkpoint_id}.tar.gz"
                )
            for _ in range(request_data.num_samples):
                all_prompts.append(prompt_tokens)
                all_sampling_params.append(request_data.sampling_params)
                all_model_ids.append(model_id)
                all_checkpoint_ids.append(request_data.checkpoint_id)
                all_checkpoint_paths.append(checkpoint_path)

            request_batch_slices.append(
                (request_id, model_id, request_start, len(all_prompts), request_data.prompt_logprobs)
            )

        return types.PreparedSampleBatch(
            all_prompts=all_prompts,
            all_sampling_params=all_sampling_params,
            all_model_ids=all_model_ids,
            all_checkpoint_ids=all_checkpoint_ids,
            all_checkpoint_paths=all_checkpoint_paths,
            needs_prompt_logprobs=needs_prompt_logprobs,
            request_batch_slices=request_batch_slices,
        )

    def __init__(
        self,
        config: EngineConfig,
    ):
        """Initialize the engine with a database connection and base model."""
        self.config = config
        self.db_engine = create_engine(config.database_url, echo=False)

        # Initialize the backend (handles model state, computation, and adapter management)
        if config.backend not in BACKENDS:
            raise ValueError(f"Unknown backend: {config.backend}. Available backends: {list(BACKENDS.keys())}")

        backend_class, backend_config_class = BACKENDS[config.backend]
        backend_config = backend_config_class(**config.backend_config)
        self.backend = backend_class(config.base_model, backend_config)

        logger.info(f"Initialized TinkerEngine with backend={type(self.backend).__name__}")

    @property
    def metrics(self) -> types.EngineMetrics:
        """Pass-through to backend metrics for backwards compatibility."""
        return self.backend.metrics

    @contextmanager
    def _checkpoint_status_context(self, model_id: str, checkpoint_id: str, checkpoint_type: types.CheckpointType):
        """Context manager to handle checkpoint DB status updates.

        Fetches the checkpoint entry, yields it, and updates its status to COMPLETED
        or FAILED based on whether an exception occurred.
        """
        with Session(self.db_engine) as session:
            checkpoint_db = session.get(CheckpointDB, (model_id, checkpoint_id, checkpoint_type))
            if checkpoint_db is None:
                raise ValueError(
                    f"Checkpoint entry not found for model '{model_id}', checkpoint '{checkpoint_id}', type '{checkpoint_type}'"
                )

            try:
                yield checkpoint_db
                checkpoint_db.status = CheckpointStatus.COMPLETED
            except Exception as e:
                logger.exception(f"Error saving checkpoint for model {model_id}, checkpoint {checkpoint_id}: {e}")
                checkpoint_db.status = CheckpointStatus.FAILED
                checkpoint_db.error_message = str(e)
                raise
            finally:
                checkpoint_db.completed_at = datetime.now(timezone.utc)
                session.add(checkpoint_db)
                session.commit()

    def find_batchable_model_passes(
        self, session: Session, request_type: types.RequestType
    ) -> dict[str, tuple[str, types.ForwardBackwardInput]]:
        """Find all requests of the given type that come before any destructive update for their model.

        Uses look-ahead scheduling: for each model, only returns operations
        that have no optim_step or load_weights blocking them in the queue.

        Args:
            session: Database session
            request_type: The type of request to find (e.g., FORWARD or FORWARD_BACKWARD)

        Returns:
            Dict mapping request_id to (model_id, request_data) tuples
        """
        # Find the earliest pending optim_step or load_weights per model (these act as barriers)
        barriers_query = (
            select(FutureDB.model_id, func.min(FutureDB.request_id).label("barrier_id"))
            .where(
                (FutureDB.request_type == types.RequestType.OPTIM_STEP)
                | (FutureDB.request_type == types.RequestType.LOAD_WEIGHTS)
            )
            .where(FutureDB.status == RequestStatus.PENDING)
            .group_by(FutureDB.model_id)
        )
        barriers = dict(session.exec(barriers_query).all())

        # Get all pending operations of the requested type ordered by request_id
        query = (
            select(FutureDB)
            .where(FutureDB.request_type == request_type)
            .where(FutureDB.status == RequestStatus.PENDING)
            .order_by(FutureDB.request_id)
        )
        ops = session.exec(query).all()

        # Filter: only include ops that come before their model's barrier
        batchable = [op for op in ops if op.model_id not in barriers or op.request_id < barriers[op.model_id]]

        return {
            str(f.request_id): (f.model_id, types.ForwardBackwardInput.model_validate(f.request_data))
            for f in batchable
        }

    def find_batchable_sample(self, session: Session) -> dict[str, tuple[str, types.SampleInput]]:
        """Find all sample ops that can be safely batched together.

        Returns sample operations ensuring that each model_id has only one checkpoint_id
        to avoid loading different checkpoints for the same model in a single batch.

        If sample_max_num_sequences is configured, limits to that many requests so we don't
        produce partial batches in process_sample_batch. If num_samples > 1 for some requests,
        this may not be perfect, but it's good until we implement continuous batching.

        Args:
            session: Database session

        Returns:
            Dict mapping request_id to (model_id, request_data) tuples
        """
        sample_query = (
            select(FutureDB)
            .where(FutureDB.request_type == types.RequestType.SAMPLE)
            .where(FutureDB.status == RequestStatus.PENDING)
            .order_by(FutureDB.request_id)
        )
        sample_ops = session.exec(sample_query).all()

        batchable = []
        model_checkpoints = {}  # Map from model_id to checkpoint_id of first request to that model
        for op in sample_ops:
            checkpoint_id = op.request_data["checkpoint_id"]
            # Base model requests (empty checkpoint_id) are always compatible, otherwise only
            # take only requests with one checkpoint_id for a given model_id
            if checkpoint_id == "" or model_checkpoints.setdefault(op.model_id, checkpoint_id) == checkpoint_id:
                batchable.append(op)

        # TODO: This leaks the abstraction by accessing backend-specific config.
        # We should find a better way to handle this going forward.
        if isinstance(self.backend, JaxBackend) and self.backend.config.sample_max_num_sequences > 0:
            batchable = batchable[: self.backend.config.sample_max_num_sequences]

        return {str(f.request_id): (f.model_id, types.SampleInput.model_validate(f.request_data)) for f in batchable}

    def find_single_requests(self, session: Session) -> dict[str, tuple[str, types.RequestType, dict]]:
        """Find all requests that need to be processed individually (not batchable).

        Args:
            session: Database session

        Returns:
            Dict mapping request_id to (model_id, request_type, request_data) tuples
        """
        statement = (
            select(FutureDB)
            .where(FutureDB.status == RequestStatus.PENDING)
            .where(FutureDB.request_type != types.RequestType.FORWARD_BACKWARD)
            .where(FutureDB.request_type != types.RequestType.FORWARD)
            .where(FutureDB.request_type != types.RequestType.SAMPLE)
            .where(FutureDB.request_type != types.RequestType.EXTERNAL)
            .order_by(FutureDB.request_id)
        )
        other_futures = session.exec(statement).all()

        return {str(f.request_id): (f.model_id, f.request_type, f.request_data) for f in other_futures}

    def process_create_model(self, model_id: str, request_data: types.CreateModelInput) -> types.CreateModelOutput:
        """Create and initialize a model."""
        # Create model in backend (allocates adapter_index, creates optimizer, and configures adapter)
        self.backend.create_model(model_id, request_data.lora_config)

        logger.info(f"Created LoRA model {model_id}")

        return types.CreateModelOutput(
            model_id=model_id,
            base_model=self.config.base_model,
            lora_config=request_data.lora_config,
        )

    def process_optim_step(self, model_id: str, request_data: types.OptimStepInput) -> types.OptimStepOutput:
        """Process an optim_step request and apply accumulated gradients."""
        if not self.backend.has_model(model_id):
            raise ValueError(f"Model {model_id} not loaded")

        return self.backend.optim_step(model_id, request_data)

    def process_forward_backward(self, requests: dict[str, tuple[str, types.ForwardBackwardInput]]) -> dict:
        """Run forward and backward pass on a batch of requests."""
        prepared = self._prepare_model_pass_batch(requests)
        return self.backend.forward_backward(prepared)

    def process_forward(self, requests: dict[str, tuple[str, types.ForwardBackwardInput]]) -> dict:
        """Run forward-only pass on a batch of requests."""
        prepared = self._prepare_model_pass_batch(requests)
        return self.backend.forward(prepared)

    def process_sample(self, requests: dict[str, tuple[str, types.SampleInput]]) -> dict:
        """Generate samples for a batch of requests."""
        prepared = self._prepare_sample_batch(requests)
        return self.backend.sample(prepared)

    def process_load_weights(self, model_id: str, request_data: types.LoadWeightsInput) -> types.LoadWeightsOutput:
        """Loads a clean, trimmed training checkpoint."""
        if not self.backend.has_model(model_id):
            raise ValueError("Model not loaded. Create the model before loading a checkpoint.")

        checkpoint_path = (
            self.config.checkpoints_base / request_data.source_model_id / f"{request_data.checkpoint_id}.tar.gz"
        )

        self.backend.load_checkpoint(checkpoint_path, model_id)

        return types.LoadWeightsOutput(type="load_weights")

    def process_save_weights(self, model_id: str, request_data: types.SaveWeightsInput) -> types.SaveWeightsOutput:
        """
        Saves a clean training checkpoint by converting the trimmed NNX graph
        to a pure dictionary before serialization, following official Flax docs.
        """
        if not self.backend.has_model(model_id):
            raise ValueError(f"Model {model_id} not loaded")

        checkpoint_id = request_data.path
        output_path = self.config.checkpoints_base / model_id / f"{checkpoint_id}.tar.gz"

        with self._checkpoint_status_context(model_id, checkpoint_id, types.CheckpointType.TRAINING):
            self.backend.save_checkpoint(output_path, model_id)
            logger.info(f"Saved trimmed training checkpoint for model {model_id} to {output_path}")

        return types.SaveWeightsOutput(
            path=f"tinker://{model_id}/weights/{checkpoint_id}",
            type="save_weights",
        )

    def process_save_weights_for_sampler(
        self, model_id: str, request_data: types.SaveWeightsForSamplerInput
    ) -> types.SaveWeightsForSamplerOutput:
        """Process a save_weights_for_sampler request and save model weights."""
        if not self.backend.has_model(model_id):
            raise ValueError(f"Model {model_id} not loaded")

        # Make sure the user cannot store checkpoints in places like ../../<important file>
        checkpoint_id = Path(request_data.path).name
        output_path = self.config.checkpoints_base / model_id / "sampler_weights" / f"{checkpoint_id}.tar.gz"

        with self._checkpoint_status_context(model_id, checkpoint_id, types.CheckpointType.SAMPLER):
            self.backend.save_sampler_checkpoint(output_path, model_id)
            logger.info(f"Saved LoRA adapter weights for model {model_id} to {output_path}")

        return types.SaveWeightsForSamplerOutput(
            path=f"tinker://{model_id}/{checkpoint_id}",
            type="save_weights_for_sampler",
        )

    def _complete_futures(self, results: dict[str, BaseModel]):
        """Helper method to complete multiple futures in the database.

        Args:
            results: Dict mapping request_id to result (Pydantic BaseModel)
        """
        completed_at = datetime.now(timezone.utc)
        params = [
            {
                "request_id": int(request_id),
                "result_data": result.model_dump(),
                "status": RequestStatus.FAILED if isinstance(result, types.ErrorResponse) else RequestStatus.COMPLETED,
                "completed_at": completed_at,
            }
            for request_id, result in results.items()
        ]

        with Session(self.db_engine) as session:
            session.execute(update(FutureDB), params)
            session.commit()

    def process_single_request(self, request_type: types.RequestType, model_id: str, request_data: dict) -> BaseModel:
        match request_type:
            case types.RequestType.CREATE_MODEL:
                return self.process_create_model(model_id, types.CreateModelInput.model_validate(request_data))
            case types.RequestType.OPTIM_STEP:
                return self.process_optim_step(model_id, types.OptimStepInput.model_validate(request_data))
            case types.RequestType.SAVE_WEIGHTS_FOR_SAMPLER:
                return self.process_save_weights_for_sampler(
                    model_id, types.SaveWeightsForSamplerInput.model_validate(request_data)
                )
            case types.RequestType.SAVE_WEIGHTS:
                return self.process_save_weights(model_id, types.SaveWeightsInput.model_validate(request_data))
            case types.RequestType.LOAD_WEIGHTS:
                return self.process_load_weights(model_id, types.LoadWeightsInput.model_validate(request_data))
            case _:
                raise ValueError(f"Unknown request type: {request_type}")

    def process_single_requests(self, requests: dict[str, tuple[str, types.RequestType, dict]]):
        """Process a collection of single (non-batchable) requests.

        Args:
            requests: Dict mapping request_id to (model_id, request_type, request_data) tuples
        """
        if not requests:
            return
        results = {}
        for request_id, (model_id, request_type, request_data) in requests.items():
            with log_timing(f"process_single_request({request_type.value})"):
                try:
                    result = self.process_single_request(request_type, model_id, request_data)
                except Exception as e:
                    logger.exception(f"Error processing request {request_id}: {e}")
                    result = types.ErrorResponse(error=str(e), status="failed")
            results[request_id] = result
        self._complete_futures(results)

    def process_batch_requests(
        self,
        requests: dict[str, tuple[str, BaseModel]],
        processor: Callable[[dict[str, tuple[str, BaseModel]]], dict[str, BaseModel]],
        name: str,
    ):
        """Process a batch of requests with error handling and future completion.

        Args:
            requests: Dict mapping request_id to (model_id, request_data) tuples
            processor: Function that processes requests and returns results dict
            name: Name for logging
        """
        if not requests:
            return
        with log_timing(f"process_batch_requests({name}, n={len(requests)})"):
            try:
                error_results, valid_requests = self._filter_valid_requests(requests)
                if valid_requests:
                    results = processor(valid_requests)
                    results.update(error_results)
                else:
                    results = error_results
            except Exception as e:
                logger.exception(f"Error processing batch: {e}")
                results = {request_id: types.ErrorResponse(error=str(e), status="failed") for request_id in requests}
        self._complete_futures(results)

    def process_pending_requests(self):
        """Main loop to process pending requests."""
        while True:
            # Query for pending requests and extract data within session context
            with Session(self.db_engine) as session:
                # Use look-ahead scheduling to find batchable forward_backward and forward model passes
                forward_backward_requests = self.find_batchable_model_passes(
                    session, types.RequestType.FORWARD_BACKWARD
                )
                forward_requests = self.find_batchable_model_passes(session, types.RequestType.FORWARD)
                # Find pending sample requests that can be batched
                sample_requests = self.find_batchable_sample(session)
                # Get other pending requests (non forward_backward and non sampling)
                other_requests = self.find_single_requests(session)

            # Process batches outside of session context
            self.process_batch_requests(forward_backward_requests, self.process_forward_backward, "forward_backward")
            self.process_batch_requests(forward_requests, self.process_forward, "forward")
            self.process_batch_requests(sample_requests, self.process_sample, "sample")

            # Process other request types individually (in the future we can also batch independent optim_steps)
            self.process_single_requests(other_requests)

            # Poll every 100ms
            time.sleep(0.1)

    def run(self):
        """Entry point to start the engine."""
        logger.info("Starting background engine...")
        self.process_pending_requests()


def main():
    """Entry point for the background engine."""
    # Create argument parser and add Pydantic model fields
    parser = argparse.ArgumentParser(description="SkyRL tx tinker engine for processing requests")
    add_model(parser, EngineConfig)

    # Parse command-line arguments
    args = parser.parse_args()

    # Create EngineConfig from parsed arguments
    config = EngineConfig.model_validate(vars(args))

    # Initialize and run the engine
    TinkerEngine(config).run()


if __name__ == "__main__":
    main()

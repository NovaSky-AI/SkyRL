"""Background engine for processing training requests."""

import argparse
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel
from sqlmodel import create_engine, Session, select, update, func

from flax import nnx
from flax.training import checkpoints

from tx.tinker.db_models import FutureDB, RequestStatus, CheckpointDB, CheckpointStatus
from tx.tinker import types
from tx.tinker.config import EngineConfig, add_model
from tx.tinker.backends import NativeBackend, MaxTextBackend, parse_maxtext_config
from tx.tinker.backends.utils import log_timing
from tx.tinker.loss_fns import LOSS_TYPES
from tx.utils.storage import download_and_unpack
from tx.utils.log import logger


class TinkerEngine:
    """Background engine for processing training requests.

    The engine handles:
    - Database operations (futures, checkpoints)
    - Request finding/scheduling
    - File I/O (download/upload checkpoints)
    - Storing models and optimizers dicts
    - Validating requests against loaded models

    Computation is delegated to the backend (NativeBackend or MaxTextBackend).
    """

    def _filter_valid_requests(
        self,
        requests: dict[str, tuple[str, any]],
    ) -> tuple[dict[str, any], dict[str, tuple[str, any]]]:
        """Filter out requests with invalid model_ids and return error results for them.

        Args:
            requests: Dict mapping request_id to (model_id, request_data) tuples

        Returns:
            Tuple of (error_results, valid_requests)
        """
        results = {}
        valid_requests = {}

        for request_id, (model_id, request_data) in requests.items():
            if model_id and model_id not in self.models:
                results[request_id] = types.ErrorResponse(error=f"Model {model_id} not loaded", status="failed")
            else:
                valid_requests[request_id] = (model_id, request_data)

        return results, valid_requests

    def _prepare_model_pass_batch(
        self,
        requests: dict[str, tuple[str, types.ForwardBackwardInput]],
    ) -> types.PreparedModelPassBatch:
        """Prepare batch data for forward/forward_backward operations.

        Extracts tokens, targets, and metadata from requests into lists
        that the backend will convert to JAX arrays.

        Args:
            requests: Dict mapping request_id to (model_id, request_data) tuples (pre-validated)

        Returns:
            PreparedModelPassBatch with all data extracted from requests
        """
        all_input_ids = []
        all_targets = []
        all_token_weights = []
        all_adapter_indices = []
        all_sampling_logprobs = []
        all_advantages = []
        all_loss_fn_types = []
        request_batch_slices = []

        for request_id, (model_id, request_data) in requests.items():
            adapter_index = self.models[model_id].adapter_index
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
                all_adapter_indices.append(adapter_index)
                all_loss_fn_types.append(loss_fn_type)

            request_batch_slices.append((request_id, model_id, request_start, len(all_input_ids)))

        return types.PreparedModelPassBatch(
            all_input_ids=all_input_ids,
            all_targets=all_targets,
            all_token_weights=all_token_weights,
            all_sampling_logprobs=all_sampling_logprobs,
            all_advantages=all_advantages,
            all_adapter_indices=all_adapter_indices,
            all_loss_fn_types=all_loss_fn_types,
            request_batch_slices=request_batch_slices,
        )

    def _prepare_sample_batch(
        self,
        requests: dict[str, tuple[str, types.SampleInput]],
        adapter_indices: list[int],
    ) -> types.PreparedSampleBatch:
        """Prepare batch data for sample operations.

        Extracts prompts and sampling params from requests into lists
        that the backend will convert to JAX arrays.

        Args:
            requests: Dict mapping request_id to (model_id, request_data) tuples (pre-validated)
            adapter_indices: List of adapter indices corresponding to each request

        Returns:
            PreparedSampleBatch with all data extracted from requests
        """
        all_prompts = []
        all_sampling_params = []
        all_adapter_indices = []
        request_batch_slices = []

        needs_prompt_logprobs = any(
            request_data.prompt_logprobs for (_, request_data) in requests.values()
        )

        for i, (request_id, (model_id, request_data)) in enumerate(requests.items()):
            request_start = len(all_prompts)

            # Expand requests for num_samples
            for _ in range(request_data.num_samples):
                prompt_tokens = [token for chunk in request_data.prompt.chunks for token in chunk.tokens]
                all_prompts.append(prompt_tokens)
                all_sampling_params.append(request_data.sampling_params)
                all_adapter_indices.append(adapter_indices[i])

            request_batch_slices.append((
                request_id, model_id, request_start, len(all_prompts), request_data.prompt_logprobs
            ))

        return types.PreparedSampleBatch(
            all_prompts=all_prompts,
            all_sampling_params=all_sampling_params,
            all_adapter_indices=all_adapter_indices,
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

        # Store LoRA model metadata (model_id -> metadata)
        self.models: dict[str, types.ModelMetadata] = {}
        # Store optimizer instances per LoRA adapter (model_id -> optimizer)
        self.optimizers: dict[str, nnx.Optimizer] = {}

        # Initialize the backend (handles model state and computation)
        if config.maxtext_config_str:
            maxtext_config = parse_maxtext_config(config.maxtext_config_str)
            self.backend = MaxTextBackend(config, maxtext_config)
        else:
            self.backend = NativeBackend(config)

        logger.info(
            f"Initialized TinkerEngine with backend={type(self.backend).__name__}, "
            f"max_lora_adapters={config.max_lora_adapters}, max_lora_rank={config.max_lora_rank}"
        )

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
            f.request_id: (f.model_id, types.ForwardBackwardInput.model_validate(f.request_data)) for f in batchable
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

        if self.config.sample_max_num_sequences > 0:
            batchable = batchable[: self.config.sample_max_num_sequences]

        return {f.request_id: (f.model_id, types.SampleInput.model_validate(f.request_data)) for f in batchable}

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

        return {f.request_id: (f.model_id, f.request_type, f.request_data) for f in other_futures}

    def process_create_model(self, model_id: str, request_data: types.CreateModelInput) -> types.CreateModelOutput:
        """Create and initialize a model."""
        # Assign adapter index for this model_id
        adapter_index = max((m.adapter_index for m in self.models.values()), default=0) + 1

        if adapter_index >= self.config.max_lora_adapters:
            raise ValueError(f"Maximum number of LoRA adapters ({self.config.max_lora_adapters}) reached")

        # Extract LoRA configuration
        lora_config = request_data.lora_config

        # Validate rank doesn't exceed max
        if not (0 < lora_config.rank <= self.config.max_lora_rank):
            raise ValueError(f"LoRA rank {lora_config.rank} must be between 1 and {self.config.max_lora_rank}")

        self.models[model_id] = types.ModelMetadata(
            adapter_index=adapter_index,
            lora_config=lora_config,
        )

        # Create optimizer via backend
        self.optimizers[model_id] = self.backend.create_optimizer(model_id)

        # Configure adapter's rank and scaling in all LoRA layers
        self.backend.configure_adapter(adapter_index, lora_config)

        logger.info(f"Created LoRA model {model_id} with adapter index {adapter_index}, config {lora_config}")

        return types.CreateModelOutput(
            model_id=model_id,
            base_model=self.config.base_model,
            lora_config=request_data.lora_config,
        )

    def process_forward_backward_batch(
        self, requests: dict[str, tuple[str, types.ForwardBackwardInput]]
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        """Process forward_backward requests by delegating to backend."""
        # Filter invalid requests before delegating to backend
        error_results, valid_requests = self._filter_valid_requests(requests)
        if not valid_requests:
            return error_results

        # Prepare batch data
        prepared_batch = self._prepare_model_pass_batch(valid_requests)

        # Delegate computation to backend
        results = self.backend.process_forward_backward_batch(prepared_batch)
        results.update(error_results)
        return results

    def process_forward_batch(
        self, requests: dict[str, tuple[str, types.ForwardBackwardInput]]
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        """Process forward-only requests by delegating to backend."""
        # Filter invalid requests before delegating to backend
        error_results, valid_requests = self._filter_valid_requests(requests)
        if not valid_requests:
            return error_results

        # Prepare batch data
        prepared_batch = self._prepare_model_pass_batch(valid_requests)

        # Delegate computation to backend
        results = self.backend.process_forward_batch(prepared_batch)
        results.update(error_results)
        return results

    def process_sample_batch(
        self, requests: dict[str, tuple[str, types.SampleInput]]
    ) -> dict[str, types.SampleOutput | types.ErrorResponse]:
        """Process multiple sample requests in a single batch.

        Args:
            requests: Dict mapping request_id to (model_id, request_data) tuples

        Returns:
            Dict mapping request_id --> result_data or error info
        """
        if not requests:
            return {}

        # Filter invalid requests before delegating to backend
        error_results, valid_requests = self._filter_valid_requests(requests)
        if not valid_requests:
            return error_results

        # Load sampler weights and get adapter indices
        adapter_indices = self.load_sampler_weights(valid_requests)

        # Prepare batch data
        prepared_batch = self._prepare_sample_batch(valid_requests, adapter_indices)

        # Delegate computation to backend
        results = self.backend.process_sample_batch(prepared_batch)
        results.update(error_results)
        return results

    def process_optim_step(self, model_id: str, request_data: types.OptimStepInput) -> types.OptimStepOutput:
        """Process an optim_step request and apply accumulated gradients."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not loaded")

        adapter_index = self.models[model_id].adapter_index

        return self.backend.process_optim_step(
            model_id, request_data, self.optimizers[model_id], adapter_index
        )

    def process_load_weights(self, model_id: str, request_data: types.LoadWeightsInput) -> types.LoadWeightsOutput:
        """Loads a clean, trimmed training checkpoint."""
        if model_id not in self.models:
            raise ValueError("Model not loaded. Create the model before loading a checkpoint.")

        checkpoint_dir = (
            self.config.checkpoints_base / request_data.source_model_id / f"{request_data.checkpoint_id}.tar.gz"
        )

        with download_and_unpack(checkpoint_dir) as temp_dir:
            checkpoint = checkpoints.restore_checkpoint(
                ckpt_dir=temp_dir,
                target=self.backend.extract_checkpoint_data(model_id, self.models, self.optimizers),
                prefix="checkpoint_",
            )

        if checkpoint is None:
            raise FileNotFoundError(f"Training checkpoint not found in {checkpoint_dir}")

        # Insert checkpoint data into model state via backend
        self.backend.insert_checkpoint_data(model_id, checkpoint, self.models, self.optimizers)

        logger.info(f"Loaded training checkpoint for model {model_id} from {checkpoint_dir}")
        return types.LoadWeightsOutput(type="load_weights")

    def process_save_weights(self, model_id: str, request_data: types.SaveWeightsInput) -> types.SaveWeightsOutput:
        """
        Saves a clean training checkpoint by converting the trimmed NNX graph
        to a pure dictionary before serialization, following official Flax docs.
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not loaded")

        checkpoint_id = request_data.path
        output_path = self.config.checkpoints_base / model_id / f"{checkpoint_id}.tar.gz"

        with self._checkpoint_status_context(model_id, checkpoint_id, types.CheckpointType.TRAINING):
            self.backend.save_checkpoint(output_path, model_id, self.models, self.optimizers)
            logger.info(f"Saved trimmed training checkpoint for model {model_id} to {output_path}")

        return types.SaveWeightsOutput(
            path=f"tinker://{model_id}/weights/{checkpoint_id}",
            type="save_weights",
        )

    def process_save_weights_for_sampler(
        self, model_id: str, request_data: types.SaveWeightsForSamplerInput
    ) -> types.SaveWeightsForSamplerOutput:
        """Process a save_weights_for_sampler request and save model weights."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not loaded")

        lora_model = self.models[model_id]

        # Make sure the user cannot store checkpoints in places like ../../<important file>
        checkpoint_id = Path(request_data.path).name
        output_path = self.config.checkpoints_base / model_id / "sampler_weights" / f"{checkpoint_id}.tar.gz"

        with self._checkpoint_status_context(model_id, checkpoint_id, types.CheckpointType.SAMPLER):
            self.backend.save_sampler_checkpoint(output_path, model_id, self.models)

            logger.info(
                f"Saved LoRA adapter weights for model {model_id} (adapter {lora_model.adapter_index}) to {output_path}"
            )

        return types.SaveWeightsForSamplerOutput(
            path=f"tinker://{model_id}/{checkpoint_id}",
            type="save_weights_for_sampler",
        )

    def load_sampler_weights(self, requests: dict[str, tuple[str, types.SampleInput]]) -> list[int]:
        """Load sampler weights for all requests and return full adapter indices array.

        Args:
            requests: Dict mapping request_id to (model_id, request_data) tuples for the batch

        Returns:
            The adapter_indices array for LoRA sampling [batch_size]
            Uses adapter index 0 for base model sampling (no LoRA)
        """
        adapter_indices = []

        for _, (model_id, request_data) in requests.items():
            base_model = request_data.base_model
            checkpoint_id = request_data.checkpoint_id
            if base_model is None:
                # This code path is for sampling from a LoRA adapter
                assert checkpoint_id != "", "checkpoint_id must be not empty"

                adapter_index = self.models[model_id].adapter_index
                if self.models[model_id].loaded_checkpoint_id == checkpoint_id:
                    # Load model from RAM
                    adapter_indices.append(adapter_index)
                else:
                    # Load model from disk
                    assert adapter_index not in adapter_indices, "Cannot override already used adapter"

                    checkpoint_path = (
                        self.config.checkpoints_base / model_id / "sampler_weights" / f"{checkpoint_id}.tar.gz"
                    )
                    logger.info(f"Loading LoRA sampler checkpoint from {checkpoint_path}")

                    # Use backend to insert sampler weights
                    self.backend.insert_sampler_weights(model_id, checkpoint_id, checkpoint_path, self.models)
                    adapter_indices.append(adapter_index)
            else:
                # This code path is for sampling from the base model
                if base_model != self.config.base_model:
                    raise ValueError(
                        f"Requested base_model '{base_model}' does not match engine's base_model '{self.config.base_model}'"
                    )
                assert model_id == "" and checkpoint_id == ""
                adapter_indices.append(0)

        return adapter_indices

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

    def process_batch_requests(self, requests: dict[str, tuple[str, BaseModel]], batch_processor):
        """Generic function to process a batch of requests.

        Args:
            requests: Dict mapping request_id to (model_id, request_data) tuples
            batch_processor: Function to call to process the batch
        """
        if not requests:
            return
        with log_timing(f"process_batch_requests({batch_processor.__name__}, n={len(requests)})"):
            try:
                results = batch_processor(requests)
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
            self.process_batch_requests(forward_backward_requests, self.process_forward_backward_batch)
            self.process_batch_requests(forward_requests, self.process_forward_batch)
            self.process_batch_requests(sample_requests, self.process_sample_batch)

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

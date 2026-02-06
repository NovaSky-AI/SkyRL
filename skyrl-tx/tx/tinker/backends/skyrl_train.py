"""SkyRL-Train backend for TinkerEngine.

Uses SkyRL-Train infrastructure for supervised training with cross-entropy loss.
Currently supports a single model only.
"""

import asyncio
import os
import tarfile
import tempfile
from typing import Any

import torch
from pydantic import BaseModel
from transformers import AutoTokenizer

from tx.tinker import types
from tx.tinker.backends.backend import AbstractBackend
from tx.utils.log import logger

try:  # Optional dependency: keep other backends importable without ray/skyrl-train.
    import ray
    from ray.util.placement_group import placement_group
    from skyrl_train.training_batch import TrainingInputBatch
    from skyrl_train.trainer import RayPPOTrainer
    from skyrl_train.utils.tracking import Tracking
    from skyrl_train.utils.utils import initialize_ray
    from skyrl_train.utils import get_ray_pg_ready_with_timeout
    from skyrl_train.config.utils import get_default_config
    from skyrl_train.env_vars import SKYRL_RAY_PG_TIMEOUT_IN_S
    from skyrl_train.entrypoints.main_base import create_ray_wrapped_inference_engines_from_config
    from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient

    SKYRL_TRAIN_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in non-ray installs
    ray = None
    placement_group = None
    TrainingInputBatch = Any
    RayPPOTrainer = Any
    Tracking = Any
    initialize_ray = None
    get_ray_pg_ready_with_timeout = None
    get_default_config = None
    SKYRL_RAY_PG_TIMEOUT_IN_S = None
    create_ray_wrapped_inference_engines_from_config = None
    InferenceEngineClient = Any
    SKYRL_TRAIN_AVAILABLE = False


class SkyRLTrainBackendConfig(BaseModel, extra="forbid"):
    """Configuration for the SkyRL-Train backend.

    Note: Currently uses SkyRL's default config for all parameters.
    TODO: Implement proper config management to allow Tinker users to override
    training and inference parameters via backend_config.
    """

    pass


def _build_config(
    base_model: str,
    config: SkyRLTrainBackendConfig,
    lora_config: types.LoraConfig | None = None,
):
    """Build config for SkyRL-Train workers using default config.

    Args:
        base_model: HuggingFace model path
        config: Backend configuration
        lora_config: LoRA configuration if using LoRA
    """
    cfg = get_default_config()
    cfg.trainer.policy.model.path = base_model

    # Disable scheduler - Tinker manages learning rate externally via set_lr()
    cfg.trainer.policy.optimizer_config.scheduler = "constant"
    cfg.trainer.policy.optimizer_config.num_warmup_steps = 0

    return cfg


class SkyRLTrainBackend(AbstractBackend):
    """SkyRL-Train backend for supervised training."""

    def __init__(self, base_model: str, config: SkyRLTrainBackendConfig):
        logger.warning("=" * 80)
        logger.warning("SkyRLTrainBackend is currently EXPERIMENTAL!")
        logger.warning("=" * 80)

        if not SKYRL_TRAIN_AVAILABLE or ray is None:
            raise ImportError(
                "SkyRLTrainBackend requires `ray`. Install the appropriate extras (e.g. `--extra skyrl_train`)."
            )

        self.base_model = base_model
        self.config = config
        self._model_id: str | None = None
        self._model_metadata: types.ModelMetadata | None = None
        self._trainer: RayPPOTrainer | None = None
        self._cfg = None
        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self._inference_engine_client = None  # InferenceEngineClient for sampling

    def has_model(self, model_id: str) -> bool:
        return self._model_id == model_id

    def create_model(self, model_id: str, lora_config: types.LoraConfig) -> None:
        if self._model_id is not None:
            raise ValueError(f"Model '{self._model_id}' already exists. Only one model supported.")

        # Build config
        self._cfg = _build_config(self.base_model, self.config, lora_config)

        if not ray.is_initialized():
            logger.info("Initializing Ray with runtime environment")
            initialize_ray(self._cfg)

        # Create placement group
        colocate_pg = self._create_colocate_pg()

        # Create inference engine client
        logger.info(f"Creating {self._cfg.generator.num_inference_engines} inference engines")
        self._inference_engine_client = InferenceEngineClient(
            create_ray_wrapped_inference_engines_from_config(self._cfg, colocate_pg, self._tokenizer),
            self._tokenizer,
            self._cfg,
        )

        # Create trainer
        tracker = Tracking(
            project_name="tinker",
            experiment_name=model_id,
            backends=[],  # No logging backends
            config=self._cfg,
        )

        self._trainer = RayPPOTrainer(
            cfg=self._cfg,
            tracker=tracker,
            tokenizer=self._tokenizer,
            train_dataset=None,  # Not needed for tinker API
            eval_dataset=None,
            inference_engine_client=self._inference_engine_client,
            generator=None,  # TODO(tyler): Update for sampling + RL
            colocate_pg=colocate_pg,
        )

        # Get worker types based on strategy
        if self._cfg.trainer.strategy in ("fsdp", "fsdp2"):
            from skyrl_train.workers.fsdp.fsdp_worker import PolicyWorker, CriticWorker, RefWorker
        elif self._cfg.trainer.strategy == "megatron":
            from skyrl_train.workers.megatron.megatron_worker import PolicyWorker, CriticWorker, RefWorker
        else:
            raise ValueError(f"Unknown strategy type: {self._cfg.trainer.strategy}")

        logger.info("Building models.")
        self._trainer.build_models(PolicyWorker, CriticWorker, RefWorker)

        logger.info("Initializing weight sync state.")
        self._trainer.init_weight_sync_state()

        self._model_id = model_id
        self._model_metadata = types.ModelMetadata(adapter_index=0, lora_config=lora_config)
        logger.info(f"Created model {model_id} using RayPPOTrainer")

    def _create_colocate_pg(self):
        """Create placement group for colocated training + inference (following main_base.py pattern)."""
        total_gpu_slots = (
            self._cfg.generator.num_inference_engines
            * self._cfg.generator.inference_engine_tensor_parallel_size
            * self._cfg.generator.inference_engine_pipeline_parallel_size
            * self._cfg.generator.inference_engine_data_parallel_size
        )
        logger.info(f"Creating placement group with {total_gpu_slots} GPU slots for colocated training+inference")
        pg = placement_group([{"GPU": 1, "CPU": 1}] * total_gpu_slots, strategy="PACK")

        logger.info("Waiting for placement group to be ready...")
        get_ray_pg_ready_with_timeout(pg, timeout=SKYRL_RAY_PG_TIMEOUT_IN_S)
        logger.info("Placement group ready!")

        return pg

    def delete_model(self, model_id: str) -> None:
        if self._model_id != model_id:
            raise ValueError(f"Model {model_id} not found")
        raise NotImplementedError("Deleting models not yet implemented")

    def _to_training_batch(self, prepared_batch: types.PreparedModelPassBatch) -> TrainingInputBatch:
        """Convert PreparedModelPassBatch to TrainingInputBatch."""
        if not prepared_batch.all_input_ids:
            return TrainingInputBatch({})

        # SkyRL-Train shifts internally, so provide the full sequence length by
        # appending the last target token to each already-shifted input.
        full_sequences = [
            list(input_ids) + ([targets[-1]] if targets else [])
            for input_ids, targets in zip(prepared_batch.all_input_ids, prepared_batch.all_targets)
        ]

        max_seq_len = max(len(seq) for seq in full_sequences)
        max_response_len = max(len(weights) for weights in prepared_batch.all_token_weights)

        sequences, attention_masks, loss_masks, response_masks = [], [], [], []

        for seq, weights in zip(full_sequences, prepared_batch.all_token_weights):
            pad_len = max_seq_len - len(seq)
            sequences.append([self._tokenizer.pad_token_id] * pad_len + list(seq))
            attention_masks.append([0] * pad_len + [1] * len(seq))
            action_pad = max_response_len - len(weights)
            loss_masks.append([0.0] * action_pad + [float(w) for w in weights])
            response_masks.append([0] * action_pad + [1] * len(weights))

        sequences_tensor = torch.tensor(sequences, dtype=torch.long)
        attention_mask_tensor = torch.tensor(attention_masks, dtype=torch.long)
        loss_mask_tensor = torch.tensor(loss_masks, dtype=torch.float32)
        response_mask_tensor = torch.tensor(response_masks, dtype=torch.long)

        batch = TrainingInputBatch(
            {
                "sequences": sequences_tensor,
                "attention_mask": attention_mask_tensor,
                "loss_mask": loss_mask_tensor,
                "response_mask": response_mask_tensor,
            }
        )
        batch.metadata = {"response_length": max_response_len}
        return batch

    def forward_backward(
        self,
        prepared_batch: types.PreparedModelPassBatch,
        loss_fn: str = "cross_entropy",
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        if not prepared_batch.all_input_ids:
            return {}

        batch = self._to_training_batch(prepared_batch)
        data = self._trainer.dispatch.forward_backward("policy", batch, loss_fn=loss_fn)

        results = {}
        for request_id, _, start_idx, end_idx in prepared_batch.request_batch_slices:
            loss_fn_outputs = []
            for i in range(start_idx, end_idx):
                raw_output = data["loss_fn_outputs"][i]
                logprobs = list(raw_output.get("logprobs", []))
                elementwise_loss = list(raw_output.get("elementwise_loss", []))
                loss_fn_outputs.append(
                    {
                        "elementwise_loss": {
                            "data": elementwise_loss,
                            "dtype": "float32",
                            "shape": [len(elementwise_loss)],
                        },
                        "logprobs": {
                            "data": logprobs,
                            "dtype": "float32",
                            "shape": [len(logprobs)],
                        },
                    }
                )
            results[request_id] = types.ForwardBackwardOutput(
                loss_fn_output_type="scalar",
                loss_fn_outputs=loss_fn_outputs,
                metrics={},
            )
        return results

    def forward(
        self,
        prepared_batch: types.PreparedModelPassBatch,
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        raise NotImplementedError("Forward-only pass not supported")

    def optim_step(self, model_id: str, request_data: types.OptimStepInput) -> types.OptimStepOutput:
        if model_id != self._model_id:
            raise ValueError(f"Model {model_id} not found")

        # Apply learning rate from AdamParams before optimizer step
        # Note: beta1, beta2, eps are fixed at optimizer creation and cannot be changed dynamically
        adam_params = request_data.adam_params
        self._trainer.dispatch.set_lr("policy", adam_params.learning_rate)

        grad_norm = self._trainer.dispatch.optim_step("policy")
        logger.info(f"optim_step: lr={adam_params.learning_rate}, grad_norm={grad_norm}")
        return types.OptimStepOutput()

    def sample(
        self,
        prepared_batch: types.PreparedSampleBatch,
    ) -> dict[str, types.SampleOutput | types.ErrorResponse]:
        """Generate samples using InferenceEngineClient.

        NOTE: Weight sync is NOT triggered automatically. The caller must call
        save_weights_for_sampler() explicitly before calling sample() if weights
        have been updated.
        """
        # 1. Validate inference is enabled
        if self._inference_engine_client is None:
            error = types.ErrorResponse(
                error="Sampling not enabled. Inference engines were not initialized (num_inference_engines=0 in SkyRL config).",
                status="error",
            )
            return {req_id: error for req_id, _, _, _, _ in prepared_batch.request_batch_slices}

        # 2. Validate single model
        unique_models = set(prepared_batch.all_model_ids)
        if unique_models != {self._model_id}:
            error = types.ErrorResponse(
                error=f"Model mismatch. Expected {self._model_id}, got {unique_models}", status="error"
            )
            return {req_id: error for req_id, _, _, _, _ in prepared_batch.request_batch_slices}

        # 3. Sample all prompts in parallel
        async def sample_all():
            tasks = []
            for i in range(len(prepared_batch.all_prompts)):
                prompt = prepared_batch.all_prompts[i]
                sampling_params = prepared_batch.all_sampling_params[i]

                # Pass through common fields; only stop needs name translation
                # (Tinker uses stop_strings/stop_tokens, vLLM uses stop/stop_token_ids)
                params_dict = {
                    "temperature": sampling_params.temperature,
                    "max_tokens": sampling_params.max_tokens,
                    "seed": sampling_params.seed,
                    "top_k": sampling_params.top_k,
                    "top_p": sampling_params.top_p,
                }
                if sampling_params.stop_strings:
                    params_dict["stop"] = sampling_params.stop_strings
                if sampling_params.stop_tokens:
                    params_dict["stop_token_ids"] = sampling_params.stop_tokens

                tasks.append(
                    self._inference_engine_client.sample(
                        prompt_token_ids=prompt,
                        num_samples=1,  # Tinker batches multiple samples separately
                        sampling_params=params_dict,
                    )
                )

            return await asyncio.gather(*tasks, return_exceptions=True)

        # Backend runs in engine subprocess with no event loop
        sample_outputs = asyncio.run(sample_all())

        # Note: sample_outputs may contain Exception objects (from return_exceptions=True)
        # We preserve these to include error messages in responses

        # 4. Aggregate results by request
        return self._aggregate_sample_results(prepared_batch, sample_outputs)

    def _aggregate_sample_results(
        self,
        prepared_batch: types.PreparedSampleBatch,
        sample_outputs: list,
    ) -> dict[str, types.SampleOutput | types.ErrorResponse]:
        """Convert InferenceEngineClient outputs to Tinker format."""
        results = {}

        for request_id, model_id, start_idx, end_idx, needs_prompt_logprobs in prepared_batch.request_batch_slices:
            sequences = []
            has_error = False
            error_msg = None

            for i in range(start_idx, end_idx):
                output = sample_outputs[i]

                # Check if sampling failed (Exception or None)
                if isinstance(output, Exception):
                    has_error = True
                    error_msg = f"Sampling failed for sample {i}: {type(output).__name__}: {str(output)}"
                    logger.error(error_msg)
                    break
                elif output is None:
                    has_error = True
                    error_msg = f"Sampling failed for sample {i}: Unknown error (output is None)"
                    logger.error(error_msg)
                    break

                # Extract tokens and logprobs
                response_tokens = output["response_ids"][0]
                response_logprobs = (output.get("response_logprobs") or [[]])[0]
                stop_reason_raw = output["stop_reasons"][0]

                # Map vLLM stop reason to Tinker format
                stop_reason = "stop" if stop_reason_raw in ["stop", "stop_token"] else "length"

                # Ensure logprobs exist (critical for RL)
                if response_logprobs is None or len(response_logprobs) == 0:
                    logger.warning("No logprobs returned - filling with zeros")
                    response_logprobs = [0.0] * len(response_tokens)

                sequences.append(
                    types.GeneratedSequence(
                        tokens=response_tokens,
                        logprobs=response_logprobs,
                        stop_reason=stop_reason,
                    )
                )

            if has_error:
                results[request_id] = types.ErrorResponse(
                    error=error_msg or "Unknown sampling error",
                    status="error",
                )
            else:
                # Note: prompt_logprobs not supported initially
                if needs_prompt_logprobs:
                    logger.warning("Prompt logprobs requested but not yet supported")

                results[request_id] = types.SampleOutput(
                    sequences=sequences,
                    prompt_logprobs=None,
                )

        return results

    def _validate_model_state(self, model_id: str) -> None:
        """Validate that model exists and is initialized."""
        if model_id != self._model_id:
            raise ValueError(f"Model {model_id} not found")
        if self._trainer is None:
            raise RuntimeError("Model not initialized")

    def _create_tar_from_directory(self, source_dir: str, output_path: str) -> None:
        """Create an uncompressed tar archive from a directory."""
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Use uncompressed tar - gzip adds 5-10min CPU time on 6-7GB FSDP checkpoints
        with tarfile.open(output_path, "w") as tar:
            tar.add(source_dir, arcname=".")

    def save_checkpoint(self, output_path, model_id: str) -> None:
        """Save full training checkpoint (model + optimizer + scheduler) as tar."""
        self._validate_model_state(model_id)

        # Create temp directory for checkpoint
        with tempfile.TemporaryDirectory() as temp_dir:
            ckpt_dir = os.path.join(temp_dir, "checkpoint")

            # Save checkpoint directory (includes optimizer state automatically)
            self._trainer.dispatch.save_checkpoint(model="policy", ckpt_dir=ckpt_dir, tokenizer=self._tokenizer)

            # Create tar archive
            self._create_tar_from_directory(ckpt_dir, output_path)

        logger.info(f"Saved checkpoint for {model_id} to {output_path}")

    def load_checkpoint(self, checkpoint_path, model_id: str) -> None:
        """Load full training checkpoint (model + optimizer + scheduler) from tar."""
        self._validate_model_state(model_id)

        # Extract tar to temp directory (filter='data' prevents path traversal attacks)
        with tempfile.TemporaryDirectory() as temp_dir:
            with tarfile.open(checkpoint_path, "r") as tar:
                tar.extractall(temp_dir, filter="data")

            # Load checkpoint (includes optimizer and scheduler states)
            self._trainer.dispatch.load_checkpoint(
                model="policy", ckpt_dir=temp_dir, load_optimizer_states=True, load_lr_scheduler_states=True
            )

        logger.info(f"Loaded checkpoint for {model_id} from {checkpoint_path}")

    def save_sampler_checkpoint(self, output_path, model_id: str, persist: bool = True) -> None:
        """Sync weights to colocated inference engines and optionally save to disk.

        The NCCL broadcast always runs so inference engines have the latest
        policy weights.  When ``persist`` is False (the common hot-path in RL
        loops) the expensive HuggingFace model export is skipped entirely.
        """
        self._validate_model_state(model_id)

        # Always sync weights to inference engines (in-memory NCCL broadcast)
        if self._inference_engine_client is not None:
            asyncio.run(self._trainer.dispatch.save_weights_for_sampler())
            logger.info(f"Synced weights for {model_id} to inference engines via NCCL")

        if persist:
            # Full HuggingFace model export to disk
            with tempfile.TemporaryDirectory() as temp_dir:
                hf_dir = os.path.join(temp_dir, "model")
                self._trainer.dispatch.save_hf_model(model="policy", export_dir=hf_dir, tokenizer=self._tokenizer)
                self._create_tar_from_directory(hf_dir, output_path)
            logger.info(f"Saved sampler checkpoint for {model_id} to {output_path}")
        else:
            # Hot path: write a lightweight marker so the engine's checkpoint
            # bookkeeping stays consistent.  Actual weights live in GPU memory.
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with tarfile.open(output_path, "w"):
                pass  # empty tar — marker only
            logger.info(f"Synced weights for {model_id} (disk save skipped)")

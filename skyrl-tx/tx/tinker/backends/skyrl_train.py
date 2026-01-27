"""SkyRL-Train backend for TinkerEngine.

Uses SkyRL-Train infrastructure for supervised training with cross-entropy loss.
Currently supports a single model only.
"""

from typing import Any

import torch
from pydantic import BaseModel

from tx.tinker import types
from tx.tinker.backends.backend import AbstractBackend
from tx.utils.log import logger

try:  # Optional dependency: keep other backends importable without ray/skyrl-train.
    import ray
    from ray.util.placement_group import placement_group
    from skyrl_train.training_batch import TrainingInputBatch
    from skyrl_train.workers.worker import PPORayActorGroup
    from skyrl_train.workers.worker_dispatch import WorkerDispatch
    from skyrl_train.workers.fsdp.fsdp_worker import PolicyWorker
    from skyrl_train.utils import get_ray_pg_ready_with_timeout
    from skyrl_train.config.utils import get_default_config

    SKYRL_TRAIN_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in non-ray installs
    ray = None
    placement_group = None
    TrainingInputBatch = Any
    PPORayActorGroup = Any
    WorkerDispatch = Any
    PolicyWorker = Any
    get_ray_pg_ready_with_timeout = None
    get_default_config = None
    SKYRL_TRAIN_AVAILABLE = False


class SkyRLTrainBackendConfig(BaseModel, extra="forbid"):
    """Configuration for the SkyRL-Train backend."""

    pass


def _build_config(base_model: str, config: SkyRLTrainBackendConfig, lora_config: types.LoraConfig | None = None):
    """Build config for SkyRL-Train workers using default config."""
    cfg = get_default_config()
    cfg.trainer.policy.model.path = base_model
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
        self._actor_group: PPORayActorGroup | None = None
        self._dispatch: WorkerDispatch | None = None
        self._cfg = None

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

    def has_model(self, model_id: str) -> bool:
        return self._model_id == model_id

    def create_model(self, model_id: str, lora_config: types.LoraConfig) -> None:
        if self._model_id is not None:
            raise ValueError(f"Model '{self._model_id}' already exists. Only one model supported.")

        self._cfg = _build_config(self.base_model, self.config, lora_config)
        num_gpus = self._cfg.trainer.placement.policy_num_gpus_per_node

        pg = placement_group([{"GPU": num_gpus, "CPU": 4}], strategy="PACK")
        get_ray_pg_ready_with_timeout(pg, timeout=30)

        self._actor_group = PPORayActorGroup(
            cfg=self._cfg,
            num_nodes=1,
            num_gpus_per_node=num_gpus,
            ray_actor_type=PolicyWorker,
            pg=pg,
            num_gpus_per_actor=0.75 if num_gpus == 1 else 1.0,
            colocate_all=False,
            sequence_parallel_size=1,
        )
        ray.get(self._actor_group.async_init_model(self.base_model))
        self._dispatch = WorkerDispatch(self._cfg, policy_actor_group=self._actor_group)

        self._model_id = model_id
        self._model_metadata = types.ModelMetadata(adapter_index=0, lora_config=lora_config)
        logger.info(f"Created model {model_id}")

    def delete_model(self, model_id: str) -> None:
        if self._model_id != model_id:
            raise ValueError(f"Model {model_id} not found")
        self._dispatch = None
        self._actor_group = None
        self._model_id = None
        self._model_metadata = None
        self._cfg = None

    def _to_training_batch(self, prepared_batch: types.PreparedModelPassBatch) -> TrainingInputBatch:
        """Convert PreparedModelPassBatch to TrainingInputBatch."""
        if not prepared_batch.all_input_ids:
            return TrainingInputBatch({})

        max_len = max(len(seq) for seq in prepared_batch.all_input_ids)
        num_actions_per_example = [sum(1 for w in weights if w > 0) for weights in prepared_batch.all_token_weights]
        max_num_actions = max(num_actions_per_example, default=0)

        sequences, attention_masks, loss_masks = [], [], []

        for seq, num_actions in zip(prepared_batch.all_input_ids, num_actions_per_example):
            pad_len = max_len - len(seq)
            sequences.append([0] * pad_len + list(seq))
            attention_masks.append([0] * pad_len + [1] * len(seq))
            action_pad = max_num_actions - num_actions
            loss_masks.append([0] * action_pad + [1] * num_actions)

        sequences_tensor = torch.tensor(sequences, dtype=torch.long)
        attention_mask_tensor = torch.tensor(attention_masks, dtype=torch.long)
        loss_mask_tensor = torch.tensor(loss_masks, dtype=torch.long)

        batch = TrainingInputBatch(
            {
                "sequences": sequences_tensor,
                "attention_mask": attention_mask_tensor,
                "loss_mask": loss_mask_tensor,
            }
        )
        batch.metadata = {"response_length": max_num_actions}
        return batch

    def forward_backward(
        self,
        prepared_batch: types.PreparedModelPassBatch,
        loss_fn: str = "cross_entropy",
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        if not prepared_batch.all_input_ids:
            return {}

        batch = self._to_training_batch(prepared_batch)
        data = self._dispatch.forward_backward("policy", batch, loss_fn=loss_fn)

        results = {}
        for request_id, _, start_idx, end_idx in prepared_batch.request_batch_slices:
            loss_fn_outputs = []
            for i in range(start_idx, end_idx):
                raw_output = data["loss_fn_outputs"][i]
                # Convert raw lists to TensorData format expected by the API
                logprobs = raw_output.get("logprobs", [])
                elementwise_loss = raw_output.get("elementwise_loss", [])
                seq_len = len(prepared_batch.all_input_ids[i])
                # SkyRL-Train returns response-only outputs; align to full sequence length.
                elementwise_loss = ([0.0] * max(seq_len - len(elementwise_loss), 0)) + list(elementwise_loss)[-seq_len:]
                logprobs = ([0.0] * max(seq_len - len(logprobs), 0)) + list(logprobs)[-seq_len:]
                loss_fn_outputs.append(
                    {
                        "elementwise_loss": {
                            "data": list(elementwise_loss),
                            "dtype": "float32",
                            "shape": [len(elementwise_loss)],
                        },
                        "logprobs": {
                            "data": list(logprobs),
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
        grad_norm = self._dispatch.optim_step("policy")
        logger.info(f"grad_norm: {grad_norm}")
        return types.OptimStepOutput()

    def sample(
        self,
        prepared_batch: types.PreparedSampleBatch,
    ) -> dict[str, types.SampleOutput | types.ErrorResponse]:
        raise NotImplementedError("Sampling not supported")

    def save_checkpoint(self, output_path, model_id: str) -> None:
        raise NotImplementedError("Saving checkpoints not supported")

    def load_checkpoint(self, checkpoint_path, model_id: str) -> None:
        raise NotImplementedError("Loading checkpoints not supported")

    def save_sampler_checkpoint(self, output_path, model_id: str) -> None:
        raise NotImplementedError("Sampler checkpoints not supported")

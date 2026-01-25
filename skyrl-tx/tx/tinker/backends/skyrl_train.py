"""SkyRL-Train backend for TinkerEngine.

Uses SkyRL-Train infrastructure for supervised training with cross-entropy loss.
Currently supports a single model only.
"""

from pathlib import Path

import ray
import torch
from omegaconf import OmegaConf
from pydantic import BaseModel, Field
from ray.util.placement_group import placement_group

from tx.tinker import types
from tx.tinker.backends.backend import AbstractBackend
from tx.utils.log import logger

from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.workers.worker import PPORayActorGroup
from skyrl_train.workers.fsdp.fsdp_worker import PolicyWorker
from skyrl_train.utils import get_ray_pg_ready_with_timeout


class SkyRLTrainBackendConfig(BaseModel, extra="forbid"):
    """Configuration for the SkyRL-Train backend."""

    num_gpus: int = Field(default=1, description="Number of GPUs to use")
    micro_train_batch_size_per_gpu: int = Field(default=2, description="Micro batch size per GPU")


def _build_config(base_model: str, config: SkyRLTrainBackendConfig, lora_config: types.LoraConfig | None = None):
    """Build minimal config for SkyRL-Train workers."""
    lora_cfg = {
        "rank": lora_config.rank if lora_config else 0,
        "alpha": lora_config.alpha if lora_config else 16,
        "dropout": 0,
        "lora_sync_path": "/tmp/skyrl_lora_sync",
        "target_modules": "all-linear",
        "exclude_modules": None,
        "init_method": "kaiming",
    }

    return OmegaConf.create({
        "trainer": {
            "placement": {"colocate_all": False, "policy_num_nodes": 1, "policy_num_gpus_per_node": config.num_gpus},
            "strategy": "fsdp2",
            "policy": {
                "model": {"path": base_model, "lora": lora_cfg},
                "optimizer_config": {
                    "lr": 1e-5, "adam_betas": [0.9, 0.999], "weight_decay": 0.01,
                    "max_grad_norm": 1.0, "offload_after_step": False, "num_warmup_steps": 0, "scheduler": "constant_with_warmup",
                },
                "fsdp_config": {"cpu_offload": False, "reshard_after_forward": True, "fsdp_size": -1},
                "sequence_parallel_size": 1, "use_torch_compile": False, "record_memory": False, "model_config_kwargs": {},
            },
            "algorithm": {
                "policy_loss_type": "regular", "loss_reduction": "token_mean",
                "eps_clip_low": 0.2, "eps_clip_high": 0.2,
                "use_kl_loss": False, "kl_loss_coef": 0.0, "use_entropy_loss": False, "entropy_loss_coef": 0.0,
            },
            "gradient_checkpointing": True, "gradient_checkpointing_use_reentrant": False,
            "seed": 42, "bf16": True,
            "micro_train_batch_size_per_gpu": config.micro_train_batch_size_per_gpu,
            "micro_forward_batch_size_per_gpu": 4,
            "policy_mini_batch_size": 256,
            "flash_attn": True, "use_sample_packing": False,
            "ckpt_path": "/tmp/skyrl_ckpts", "logger": "console",
        },
        "generator": {
            "n_samples_per_prompt": 1,
            "sampling_params": {"temperature": 1.0},
            "weight_transfer_threshold_cuda_ipc_GB": 1.0,
            "weight_sync_backend": "nccl",
        },
    })


class SkyRLTrainBackend(AbstractBackend):
    """SkyRL-Train backend for supervised training."""

    def __init__(self, base_model: str, config: SkyRLTrainBackendConfig):
        self.base_model = base_model
        self.config = config
        self._model_id: str | None = None
        self._model_metadata: types.ModelMetadata | None = None
        self._actor_group: PPORayActorGroup | None = None

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

    def has_model(self, model_id: str) -> bool:
        return self._model_id == model_id

    def create_model(self, model_id: str, lora_config: types.LoraConfig) -> None:
        if self._model_id is not None:
            raise ValueError(f"Model '{self._model_id}' already exists. Only one model supported.")

        cfg = _build_config(self.base_model, self.config, lora_config)
        num_gpus = self.config.num_gpus

        pg = placement_group([{"GPU": num_gpus, "CPU": num_gpus}], strategy="PACK")
        get_ray_pg_ready_with_timeout(pg, timeout=30)

        self._actor_group = PPORayActorGroup(
            cfg=cfg, num_nodes=1, num_gpus_per_node=num_gpus,
            ray_actor_type=PolicyWorker, pg=pg,
            num_gpus_per_actor=0.75 if num_gpus == 1 else 1.0,
            colocate_all=False, sequence_parallel_size=1,
        )
        ray.get(self._actor_group.async_init_model(self.base_model))

        self._model_id = model_id
        self._model_metadata = types.ModelMetadata(adapter_index=0, lora_config=lora_config)
        logger.info(f"Created model {model_id}")

    def delete_model(self, model_id: str) -> None:
        if self._model_id != model_id:
            raise ValueError(f"Model {model_id} not found")
        self._actor_group = None
        self._model_id = None
        self._model_metadata = None

    def _to_training_batch(self, prepared_batch: types.PreparedModelPassBatch) -> TrainingInputBatch:
        """Convert PreparedModelPassBatch to TrainingInputBatch."""
        if not prepared_batch.all_input_ids:
            return TrainingInputBatch({})

        max_len = max(len(seq) for seq in prepared_batch.all_input_ids)
        sequences, attention_masks, loss_masks = [], [], []

        for seq, weights in zip(prepared_batch.all_input_ids, prepared_batch.all_token_weights):
            pad_len = max_len - len(seq)
            sequences.append([0] * pad_len + list(seq))
            attention_masks.append([0] * pad_len + [1] * len(seq))
            loss_masks.append([0.0] * pad_len + list(weights))

        response_length = max(sum(1 for w in weights if w > 0) for weights in prepared_batch.all_token_weights)

        batch = TrainingInputBatch({
            "sequences": torch.tensor(sequences, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "loss_mask": torch.tensor(loss_masks, dtype=torch.float32),
        })
        batch.metadata = {"response_length": response_length}
        return batch

    def forward_backward(
        self, prepared_batch: types.PreparedModelPassBatch,
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        if not prepared_batch.all_input_ids:
            return {}

        batch = self._to_training_batch(prepared_batch)
        metrics = self._actor_group.run_method("mesh", "forward_backward", batch)

        # Get the loss from metrics and distribute per token
        loss = metrics.get("loss", 0.0)

        results = {}
        for request_id, _, start_idx, end_idx in prepared_batch.request_batch_slices:
            loss_fn_outputs = []
            for i in range(start_idx, end_idx):
                seq_len = len(prepared_batch.all_input_ids[i])
                loss_fn_outputs.append({
                    "elementwise_loss": {"data": [loss] * seq_len, "dtype": "float32", "shape": [seq_len]},
                    "logprobs": {"data": [0.0] * seq_len, "dtype": "float32", "shape": [seq_len]},
                })
            results[request_id] = types.ForwardBackwardOutput(
                loss_fn_output_type="scalar", loss_fn_outputs=loss_fn_outputs, metrics=metrics,
            )
        return results

    def forward(
        self, prepared_batch: types.PreparedModelPassBatch,
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        return self.forward_backward(prepared_batch)

    def optim_step(self, model_id: str, request_data: types.OptimStepInput) -> types.OptimStepOutput:
        if model_id != self._model_id:
            raise ValueError(f"Model {model_id} not found")
        refs = self._actor_group.async_run_ray_method("pass_through", "optim_step")
        ray.get(refs)
        return types.OptimStepOutput()

    def sample(
        self, prepared_batch: types.PreparedSampleBatch,
    ) -> dict[str, types.SampleOutput | types.ErrorResponse]:
        raise NotImplementedError("Sampling not supported")

    def save_checkpoint(self, output_path, model_id: str) -> None:
        if model_id != self._model_id:
            raise ValueError(f"Model {model_id} not found")
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        ray.get([actor.save_checkpoint.remote(output_path) for actor in self._actor_group._actor_handlers])

    def load_checkpoint(self, checkpoint_path, model_id: str) -> None:
        if model_id != self._model_id:
            raise ValueError(f"Model {model_id} not found")
        ray.get([actor.load_checkpoint.remote(Path(checkpoint_path)) for actor in self._actor_group._actor_handlers])

    def save_sampler_checkpoint(self, output_path, model_id: str) -> None:
        raise NotImplementedError("Sampler checkpoints not supported")

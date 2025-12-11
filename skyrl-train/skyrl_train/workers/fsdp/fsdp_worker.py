import asyncio

from skyrl_train.utils.trainer_utils import get_rope_scaling_config, get_rope_theta_config
import ray
import torch
import torch.distributed
from transformers import AutoConfig
from torch.distributed.fsdp.api import ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
import io

try:
    # for torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

from skyrl_train.model_wrapper import HFModelWrapper, get_llm_for_sequence_regression
from skyrl_train.distributed.fsdp_strategy import FSDPStrategy
from skyrl_train.utils import get_physical_gpu_id, str_to_torch_dtype
from skyrl_train.training_batch import TrainingInputBatch, TrainingOutputBatch
from skyrl_train.distributed.fsdp_utils import fsdp_version, get_init_weight_context_manager
from skyrl_train.workers.worker import (
    PolicyWorkerBase,
    CriticWorkerBase,
    RefWorkerBase,
)
from skyrl_train.weight_sync import WeightExtractor, WeightChunk
from skyrl_train.weight_sync.weight_extractor_utils import yield_module_grouped_chunks


class FSDPWeightExtractor(WeightExtractor):
    """Extracts weights from FSDP-sharded models.

    Args:
        model: FSDP model to extract weights from
        group_by_module: If True, group parameters by module (e.g., for FlashRL QKV fusion)
        batch_size_threshold_gb: If > 0, batch complete modules together until threshold is reached
    """

    def __init__(self, model: torch.nn.Module, group_by_module: bool = False, batch_size_threshold_gb: float = 0.0):
        self.model = model
        self.group_by_module = group_by_module
        self.batch_size_threshold_gb = batch_size_threshold_gb

    def extract_weights(self, dtype: torch.dtype):
        """Extract weights from FSDP model.

        Args:
            dtype: Target dtype for inference

        Yields:
            WeightChunk objects (one per parameter, or grouped by module)
        """
        # Configure state_dict type for FSDP v1
        if fsdp_version(self.model) == 1:
            FSDP.set_state_dict_type(
                self.model,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        # Get state dict (handles FSDP sharding)
        params = self.model.state_dict()

        if not self.group_by_module:
            # Simple path: yield one chunk per parameter
            for name, param in params.items():
                tensor = self._gather_tensor(param).to(dtype).detach().contiguous()
                yield WeightChunk(
                    names=[name],
                    dtypes=[str(dtype)],
                    shapes=[list(tensor.shape)],
                    tensors=[tensor],
                )
        else:
            for chunk in yield_module_grouped_chunks(
                params=params,
                dtype=dtype,
                gather_tensor_fn=self._gather_tensor,
                get_shape_fn=lambda name, param, tensor: list(tensor.shape),
                batch_size_threshold_gb=self.batch_size_threshold_gb,
            ):
                yield chunk

    def _gather_tensor(self, param: torch.Tensor) -> torch.Tensor:
        """Gather sharded tensor into full tensor."""
        device = torch.cuda.current_device()
        return param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param


class FSDPPolicyWorkerBase(PolicyWorkerBase):
    def offload_to_cpu(self, pin_memory=True, non_blocking=True, offload_optimizer=True, offload_model=True):
        self._set_numa_affinity(torch.distributed.get_rank() % torch.cuda.device_count())
        self.strategy.offload_to_cpu(
            self.model, self.optimizer, pin_memory, non_blocking, offload_optimizer, offload_model
        )

    def backload_to_gpu(self, non_blocking=True, backload_optimizer=True, backload_model=True):
        self.strategy.backload_to_gpu(self.model, self.optimizer, non_blocking, backload_optimizer, backload_model)

    def init_model(self, model_path, num_training_steps: int = None):
        assert self.cfg.trainer.strategy in ("fsdp", "fsdp2")
        strategy = FSDPStrategy(
            fsdp_config=self.cfg.trainer.policy.fsdp_config,
            optimizer_config=self.cfg.trainer.policy.optimizer_config,
            model_config=self.cfg.trainer.policy.model,
            fsdp_strategy=self.cfg.trainer.strategy,
            seed=self.cfg.trainer.seed,
            micro_train_batch_size_per_gpu=self.cfg.trainer.micro_train_batch_size_per_gpu,
            num_training_steps=num_training_steps,
        )
        strategy.setup_distributed()
        self.strategy = strategy

        self._is_lora = self.cfg.trainer.policy.model.lora.rank > 0

        # Update per-gpu mini batch size based on device mesh
        self._normalize_mini_batch_size()

        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.strategy.device_mesh
        )
        with init_context():

            wrapped_model = HFModelWrapper(
                model_path,
                use_flash_attention_2=self.cfg.trainer.flash_attn,
                # NOTE (sumanthrh): Model initialization should always be in fp32
                # during training
                bf16=False,
                lora_rank=self.cfg.trainer.policy.model.lora.rank,
                lora_alpha=self.cfg.trainer.policy.model.lora.alpha,
                lora_dropout=self.cfg.trainer.policy.model.lora.dropout,
                target_modules=self.cfg.trainer.policy.model.lora.target_modules,
                exclude_modules=self.cfg.trainer.policy.model.lora.exclude_modules,
                sequence_parallel_size=self.cfg.trainer.policy.sequence_parallel_size,
                use_sample_packing=self.cfg.trainer.use_sample_packing,
                use_torch_compile=self.cfg.trainer.policy.use_torch_compile,
                rope_scaling=get_rope_scaling_config(self.cfg.trainer),
                rope_theta=get_rope_theta_config(self.cfg.trainer),
            )
            # in-place patch
            self._seq_parallel_monkey_patch(model=wrapped_model.model)

            if self.cfg.trainer.gradient_checkpointing:
                wrapped_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={
                        "use_reentrant": self.cfg.trainer.gradient_checkpointing_use_reentrant
                    }
                )

        self.model, self.optimizer, self.scheduler = strategy.prepare(
            (wrapped_model, None, None),
        )
        assert (
            self.optimizer is not None and self.scheduler is not None
        ), "FSDP preparation should create optimizer and scheduler"

        # Initialize weight extractor
        self.use_cuda_ipc = self.cfg.generator.weight_sync_backend == "nccl" and self.cfg.trainer.placement.colocate_all
        # TODO(haochen): Now module grouping (in order to support FlashRL) is only enabled for the CUDA IPC
        # transfer strategy, we can enable it for other strategies as well.
        self.weight_extractor = FSDPWeightExtractor(
            self.model.model,
            group_by_module=self.use_cuda_ipc,
            batch_size_threshold_gb=(
                self.cfg.generator.weight_transfer_threshold_cuda_ipc_GB if self.use_cuda_ipc else 0.0
            ),
        )

    async def _save_lora_adapters_and_sync(self, peft_model, lora_sync_path, inference_engine_client):
        """Collect LoRA parameters, save and call inference engine to load."""
        import os
        import json
        from dataclasses import asdict
        from safetensors.torch import save_file
        from skyrl_train.distributed.fsdp_utils import collect_lora_params

        lora_params = collect_lora_params(module=self.model.model)

        if torch.distributed.get_rank() == 0:
            os.makedirs(lora_sync_path, exist_ok=True)

            peft_config = asdict(peft_model.peft_config.get("default", {}))
            peft_config["task_type"] = peft_config["task_type"].value
            peft_config["peft_type"] = peft_config["peft_type"].value
            peft_config["target_modules"] = list(peft_config["target_modules"])

            # Save LoRA parameters and config
            save_file(lora_params, os.path.join(lora_sync_path, "adapter_model.safetensors"))
            with io.open(os.path.join(lora_sync_path, "adapter_config.json"), "w", encoding="utf-8") as f:
                json.dump(peft_config, f, ensure_ascii=False, indent=4)

            # Send LoRA disk loading request to inference engine. `lora_disk_load` is a specific identifier
            # to tell the inference engine to extract the `lora_disk_path`.
            lora_request = {
                "names": ["lora_disk_load"],
                "extras": [{"lora_disk_path": lora_sync_path}],
            }
            await inference_engine_client.update_named_weights(lora_request)

        torch.distributed.barrier()

    async def broadcast_to_inference_engines(self, inference_engine_client):
        use_prefix_cache = self.cfg.generator.enable_prefix_caching
        generator_dtype = str_to_torch_dtype(self.cfg.generator.model_dtype)
        cache_reset_task = None
        if use_prefix_cache and torch.distributed.get_rank() == 0:
            # clear prefix cache
            cache_reset_task = inference_engine_client.reset_prefix_cache()

        torch.cuda.empty_cache()

        # Check if this is a LoRA model
        peft_model = getattr(self.model.model, "_fsdp_wrapped_module", self.model.model)

        if self._is_lora:
            assert hasattr(peft_model, "peft_config"), "LoRA model should have peft_config"

            # assume base model is already synced, sync LoRA adapters
            lora_sync_path = self.cfg.trainer.policy.model.lora.lora_sync_path
            await self._save_lora_adapters_and_sync(peft_model, lora_sync_path, inference_engine_client)
            return

        # Extract weights using the initialized extractor
        if not self.use_cuda_ipc:
            # Broadcast path: one chunk per parameter
            for chunk in self.weight_extractor.extract_weights(generator_dtype):
                # Each chunk contains one parameter
                assert len(chunk) == 1
                name = chunk.names[0]
                tensor = chunk.tensors[0]

                if torch.distributed.get_rank() == 0:
                    # Create legacy update request
                    update_weight_task = asyncio.create_task(
                        inference_engine_client.update_named_weights(
                            {
                                "names": [name],
                                "dtypes": [self.cfg.generator.model_dtype],
                                "shapes": [list(tensor.shape)],
                            }
                        )
                    )

                # Broadcast tensor
                def broadcast_tensor(tensor):
                    if torch.distributed.get_rank() == 0:
                        torch.distributed.broadcast(tensor.data, 0, group=self._model_update_group)

                await asyncio.to_thread(broadcast_tensor, tensor)
                if torch.distributed.get_rank() == 0:
                    await update_weight_task
                torch.distributed.barrier()
        else:
            # CUDA IPC path: batched chunks (batching handled by extractor)
            from torch.multiprocessing.reductions import reduce_tensor

            # Iterate over batched chunks
            for chunk in self.weight_extractor.extract_weights(generator_dtype):
                weights_update_request = {"names": [], "dtypes": [], "shapes": [], "extras": [], "packed": False}

                # Process all parameters in this batch
                # TODO(haochen): Pack tensors into contiguous buffer before creating IPC handle
                # (like Megatron does) to reduce number of IPC handles and file descriptors
                for name, tensor, shape in zip(chunk.names, chunk.tensors, chunk.shapes):
                    # Create IPC handle for tensor
                    ipc_handle = reduce_tensor(tensor)
                    ipc_handle = {get_physical_gpu_id(): ipc_handle}
                    ipc_handle_list = [None] * torch.distributed.get_world_size()
                    torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)

                    if torch.distributed.get_rank() == 0:
                        ipc_handles = {}
                        for d in ipc_handle_list:
                            ipc_handles.update(d)

                        weights_update_request["names"].append(name)
                        weights_update_request["dtypes"].append(self.cfg.generator.model_dtype)
                        weights_update_request["shapes"].append(shape)
                        weights_update_request["extras"].append({"ipc_handles": ipc_handles})

                    torch.distributed.barrier()
                    torch.cuda.synchronize()

                # Send batch
                if torch.distributed.get_rank() == 0:
                    await inference_engine_client.update_named_weights(weights_update_request)
                    torch.cuda.ipc_collect()
                torch.distributed.barrier()
                torch.cuda.synchronize()

        if cache_reset_task is not None:
            await cache_reset_task
        torch.cuda.empty_cache()
        torch.distributed.barrier()

    def get_weight_statistics(self):
        """Compute lightweight statistics for model weights"""
        raise NotImplementedError()

    def _set_pad_token_id(self, pad_token_id):
        # NOTE (sumanthrh): self.model -> HFModelWrapper; self.model -> DeepSpeedEngine, self.model.module -> AutoModelForCausalLM
        self.model.model.config.pad_token_id = pad_token_id

    def forward(
        self,
        data: TrainingInputBatch,
    ) -> TrainingOutputBatch:
        """Run forward pass on data in inference mode.

        Reshard the model after forward pass to redistribute memory and allow for offloading to cpu.
        """
        output = super().forward(data)
        # unshard the root FSDP module (https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes)
        if self._world_size > 1 and fsdp_version(self.model.model) == 1:
            self.model.model._handle.reshard(True)
        return output


class FSDPCriticWorkerBase(CriticWorkerBase):
    def offload_to_cpu(self, pin_memory=True, non_blocking=True, offload_optimizer=True, offload_model=True):
        self._set_numa_affinity(torch.distributed.get_rank() % torch.cuda.device_count())
        self.strategy.offload_to_cpu(
            self.model, self.optimizer, pin_memory, non_blocking, offload_optimizer, offload_model
        )

    def backload_to_gpu(self, non_blocking=True, backload_optimizer=True, backload_model=True):
        self.strategy.backload_to_gpu(self.model, self.optimizer, non_blocking, backload_optimizer, backload_model)

    def init_model(self, model_path, num_training_steps: int = None):
        assert self.cfg.trainer.strategy in ("fsdp", "fsdp2")
        strategy = FSDPStrategy(
            fsdp_config=self.cfg.trainer.critic.fsdp_config,
            optimizer_config=self.cfg.trainer.critic.optimizer_config,
            fsdp_strategy=self.cfg.trainer.strategy,
            seed=self.cfg.trainer.seed,
            micro_train_batch_size_per_gpu=self.cfg.trainer.micro_train_batch_size_per_gpu,
            num_training_steps=num_training_steps,
        )
        strategy.setup_distributed()
        self.strategy = strategy

        # Update per-gpu mini batch size based on device mesh
        self._normalize_mini_batch_size()

        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.strategy.device_mesh
        )
        with init_context():
            critic = get_llm_for_sequence_regression(
                model_path,
                "critic",
                use_flash_attention_2=self.cfg.trainer.flash_attn,
                # NOTE (sumanthrh): Model initialization should always be in fp32
                # during training
                bf16=False,
                lora_rank=self.cfg.trainer.critic.model.lora.rank,
                lora_alpha=self.cfg.trainer.critic.model.lora.alpha,
                lora_dropout=self.cfg.trainer.critic.model.lora.dropout,
                target_modules=self.cfg.trainer.critic.model.lora.target_modules,
                exclude_modules=self.cfg.trainer.critic.model.lora.exclude_modules,
                value_head_prefix=self.cfg.trainer.algorithm.value_head_prefix,
                init_value_head=self.cfg.trainer.policy.model.path == self.cfg.trainer.critic.model.path,
                sequence_parallel_size=self.cfg.trainer.critic.sequence_parallel_size,
                use_sample_packing=self.cfg.trainer.use_sample_packing,
            )
            self._seq_parallel_monkey_patch(model=critic, use_parent_class=True)

            if self.cfg.trainer.gradient_checkpointing:
                critic.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={
                        "use_reentrant": self.cfg.trainer.gradient_checkpointing_use_reentrant
                    }
                )

        # prepare models/optimizers...
        self.model, self.optimizer, self.scheduler = strategy.prepare(
            (critic, None, None),
        )
        assert self.optimizer is not None

    def forward(
        self,
        data: TrainingInputBatch,
    ) -> TrainingOutputBatch:
        """Run forward pass on data in inference mode.

        Reshard the model after forward pass to redistribute memory and allow for offloading to cpu.
        """
        output = super().forward(data)
        # unshard the root FSDP module (https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes)
        if self._world_size > 1 and fsdp_version(self.model.model) == 1:
            self.model.model._handle.reshard(True)
        return output


class FSDPRefWorkerBase(RefWorkerBase):
    def offload_to_cpu(self, pin_memory=True, non_blocking=True, **kwargs):
        self._set_numa_affinity(torch.distributed.get_rank() % torch.cuda.device_count())
        self.strategy.offload_to_cpu(self.model, None, pin_memory, non_blocking)

    def backload_to_gpu(self, non_blocking=True, **kwargs):
        self.strategy.backload_to_gpu(self.model, None, non_blocking)

    def init_model(self, model_path):
        assert self.cfg.trainer.strategy in ("fsdp", "fsdp2")
        strategy = FSDPStrategy(
            fsdp_config=self.cfg.trainer.ref.fsdp_config,
            fsdp_strategy=self.cfg.trainer.strategy,
            seed=self.cfg.trainer.seed,
            micro_train_batch_size_per_gpu=self.cfg.trainer.micro_train_batch_size_per_gpu,
        )
        strategy.setup_distributed()
        self.strategy = strategy

        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.strategy.device_mesh
        )

        with init_context():
            wrapped_model = HFModelWrapper(
                model_path,
                use_flash_attention_2=self.cfg.trainer.flash_attn,
                bf16=self.cfg.trainer.bf16,
                sequence_parallel_size=self.cfg.trainer.ref.sequence_parallel_size,
                use_sample_packing=self.cfg.trainer.use_sample_packing,
                rope_scaling=get_rope_scaling_config(self.cfg.trainer),
                rope_theta=get_rope_theta_config(self.cfg.trainer),
            )
            self._seq_parallel_monkey_patch(model=wrapped_model.model)

        self.model = strategy.prepare(wrapped_model)
        self.model.eval()

    def forward(
        self,
        data: TrainingInputBatch,
    ) -> TrainingOutputBatch:
        """Run forward pass on data in inference mode.

        Reshard the model after forward pass to redistribute memory and allow for offloading to cpu.
        """
        output = super().forward(data)
        # unshard the root FSDP module (https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes)
        if self._world_size > 1 and fsdp_version(self.model.model) == 1:
            self.model.model._handle.reshard(True)
        return output


# Ray remote actors
PolicyWorker = ray.remote(num_gpus=1)(FSDPPolicyWorkerBase)
CriticWorker = ray.remote(num_gpus=1)(FSDPCriticWorkerBase)
RefWorker = ray.remote(num_gpus=1)(FSDPRefWorkerBase)

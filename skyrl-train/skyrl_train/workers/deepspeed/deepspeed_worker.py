import asyncio

import deepspeed
import ray
import torch
import torch.distributed
from loguru import logger
from transformers.trainer import get_scheduler


from skyrl_train.model_wrapper import get_llm_for_sequence_regression, HFModelWrapper
from skyrl_train.distributed.deepspeed_strategy import DeepspeedStrategy
from skyrl_train.utils import get_physical_gpu_id
from skyrl_train.utils.trainer_utils import get_rope_scaling_config, get_rope_theta_config
from skyrl_train.utils.utils import str_to_torch_dtype
from skyrl_train.workers.worker import (
    PolicyWorkerBase,
    CriticWorkerBase,
    RefWorkerBase,
)
from skyrl_train.weight_sync import WeightExtractor, WeightChunk
from skyrl_train.weight_sync.weight_extractor_utils import yield_module_grouped_chunks


class DeepSpeedWeightExtractor(WeightExtractor):
    """Extracts weights from DeepSpeed ZeRO-sharded models.

    Args:
        model: DeepSpeed model to extract weights from
        zero_stage: ZeRO optimization stage (0, 1, 2, or 3)
        group_by_module: If True, group parameters by module (e.g., for FlashRL QKV fusion)
        batch_size_threshold_gb: If > 0, batch complete modules together until threshold is reached
    """

    def __init__(
        self,
        model: torch.nn.Module,
        zero_stage: int,
        group_by_module: bool = False,
        batch_size_threshold_gb: float = 0.0,
    ):
        self.model = model
        self.zero_stage = zero_stage
        self.group_by_module = group_by_module
        self.batch_size_threshold_gb = batch_size_threshold_gb

    def extract_weights(self, dtype: torch.dtype):
        """Extract weights from DeepSpeed model.

        Args:
            dtype: Target dtype for inference

        Yields:
            WeightChunk objects (one per parameter, or grouped by module)
        """
        params = dict(self.model.named_parameters())

        if not self.group_by_module:
            # Simple path: yield one chunk per parameter
            for name, param in params.items():
                tensor = self._gather_tensor(param).to(dtype).detach().contiguous()
                # Get correct shape based on ZeRO stage
                shape = list(param.shape if self.zero_stage != 3 else param.ds_shape)
                yield WeightChunk(
                    names=[name],
                    dtypes=[str(dtype)],
                    shapes=[shape],
                    tensors=[tensor],
                )
        else:
            for chunk in yield_module_grouped_chunks(
                params=params,
                dtype=dtype,
                gather_tensor_fn=self._gather_tensor,
                get_shape_fn=lambda name, param, tensor: list(param.shape if self.zero_stage != 3 else param.ds_shape),
                batch_size_threshold_gb=self.batch_size_threshold_gb,
            ):
                yield chunk

    def _gather_tensor(self, param: torch.nn.Parameter) -> torch.Tensor:
        """Gather sharded parameter (if ZeRO-3)."""
        with deepspeed.zero.GatheredParameters([param], enabled=self.zero_stage == 3):
            return param.data.clone()


class DeepSpeedPolicyWorkerBase(PolicyWorkerBase):
    def offload_to_cpu(self, pin_memory=True, non_blocking=True, **kwargs):
        # NOTE (erictang000): the Deepspeed backend only offloads optimizer states + fp32 params to GPU, so
        # bf16 weights remain on GPU at all times. We thus absorb `offload_optimizer` and `offload_model` into `kwargs`
        # and do not pass them down to the strategy.
        # TODO (erictang000): this is where this was getting called previously - do we need to do this every time?
        self._set_numa_affinity(torch.distributed.get_rank() % torch.cuda.device_count())
        self.strategy.offload_to_cpu(self.model, pin_memory, non_blocking)

    def backload_to_gpu(self, non_blocking=True, **kwargs):
        self.strategy.backload_to_gpu(self.model, non_blocking)

    def init_model(self, model_id_or_path, num_training_steps: int = None):
        assert self.cfg.trainer.strategy in ("deepspeed")
        self.zero_stage = self.cfg.trainer.policy.deepspeed_config.zero_optimization.stage
        if self.cfg.trainer.policy.optimizer_config.max_grad_norm > 0:
            self.cfg.trainer.policy.deepspeed_config.gradient_clipping = (
                self.cfg.trainer.policy.optimizer_config.max_grad_norm
            )
        strategy = DeepspeedStrategy(
            self.cfg.trainer.policy.deepspeed_config,
            seed=self.cfg.trainer.seed,
            micro_train_batch_size_per_gpu=self.cfg.trainer.micro_train_batch_size_per_gpu,
            zero_stage=self.zero_stage,
            bf16=self.cfg.trainer.bf16,
        )
        strategy.setup_distributed()
        self.strategy = strategy

        # Update per-gpu mini batch size based on device mesh
        self._normalize_mini_batch_size()

        ds_config = strategy.get_ds_train_config()
        wrapped_model = HFModelWrapper(
            model_id_or_path,
            use_flash_attention_2=self.cfg.trainer.flash_attn,
            bf16=self.cfg.trainer.bf16,
            target_modules=self.cfg.trainer.policy.model.lora.target_modules,
            exclude_modules=self.cfg.trainer.policy.model.lora.exclude_modules,
            ds_config=ds_config,
            sequence_parallel_size=self.sequence_parallel_size,
            use_sample_packing=self.cfg.trainer.use_sample_packing,
            use_torch_compile=self.cfg.trainer.policy.use_torch_compile,
            rope_scaling=get_rope_scaling_config(self.cfg.trainer),
            rope_theta=get_rope_theta_config(self.cfg.trainer),
        )

        # configure optimizer
        optimizer = strategy.create_optimizer(
            wrapped_model,
            lr=self.cfg.trainer.policy.optimizer_config.lr,
            betas=self.cfg.trainer.policy.optimizer_config.adam_betas,
            weight_decay=self.cfg.trainer.policy.optimizer_config.weight_decay,
            offload_after_step=self.cfg.trainer.policy.optimizer_config.offload_after_step,
        )

        lr_scheduler = get_scheduler(
            self.cfg.trainer.policy.optimizer_config.scheduler,
            optimizer,
            num_warmup_steps=self.cfg.trainer.policy.optimizer_config.num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        if self.cfg.trainer.gradient_checkpointing:
            wrapped_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": self.cfg.trainer.gradient_checkpointing_use_reentrant}
            )

        self._seq_parallel_monkey_patch(model=wrapped_model.model)

        # prepare models/optimizers...
        self.model, self.optimizer, self.scheduler = strategy.prepare(
            (wrapped_model, optimizer, lr_scheduler),
        )

        # Initialize weight extractor
        self.use_cuda_ipc = self.cfg.generator.weight_sync_backend == "nccl" and self.cfg.trainer.placement.colocate_all
        # TODO(haochen): Now module grouping (in order to support FlashRL) is only enabled for the CUDA IPC
        # transfer strategy, we can enable it for other strategies as well.
        self.weight_extractor = DeepSpeedWeightExtractor(
            model=self.model.model.module,
            zero_stage=self.zero_stage,
            group_by_module=self.use_cuda_ipc,
            batch_size_threshold_gb=(
                self.cfg.generator.weight_transfer_threshold_cuda_ipc_GB if self.use_cuda_ipc else 0.0
            ),
        )

        self._model_update_group_name = None

    def process_sequences(self, sequences, input_len, eos_token_id, pad_token_id):
        return self.model.process_sequences(sequences, input_len, eos_token_id, pad_token_id)

    def _set_pad_token_id(self, pad_token_id):
        # NOTE (sumanthrh): self.model -> HFModelWrapper; self.model -> DeepSpeedEngine, self.model.module -> AutoModelForCausalLM
        self.model.model.module.config.pad_token_id = pad_token_id

    def _handle_termination(self):
        logger.info("Received termination signal. Destroying weights update group.")
        if torch.distributed.get_rank() == 0:
            try:
                loop = asyncio.get_running_loop()
                loop.run_until_complete(self.inference_engine_client.teardown())
            except Exception as e:
                logger.error(f"Error destroying weights update group: {e}")

    async def broadcast_to_inference_engines(self, inference_engine_client):
        use_prefix_cache = self.cfg.generator.enable_prefix_caching
        generator_dtype = str_to_torch_dtype(self.cfg.generator.model_dtype)
        cache_reset_task = None
        if use_prefix_cache and torch.distributed.get_rank() == 0:
            # clear prefix cache
            cache_reset_task = inference_engine_client.reset_prefix_cache()

        torch.cuda.empty_cache()

        if not self.use_cuda_ipc:
            # Broadcast path: one chunk per parameter
            for chunk in self.weight_extractor.extract_weights(generator_dtype):
                # Each chunk contains one parameter
                assert len(chunk) == 1
                name = chunk.names[0]
                shape = chunk.shapes[0]
                tensor = chunk.tensors[0]

                if torch.distributed.get_rank() == 0:
                    update_weight_task = asyncio.create_task(
                        inference_engine_client.update_named_weights(
                            {
                                "names": [name],
                                "dtypes": [self.cfg.generator.model_dtype],
                                "shapes": [shape],
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
        stats = {}
        model = self.model.model.module
        for name, param in model.named_parameters():
            with deepspeed.zero.GatheredParameters([param], enabled=self.zero_stage == 3):
                tensor_stats = {
                    "mean": param.data.mean().item(),
                    "std": param.data.std().item(),
                    "norm": param.data.norm().item(),
                    "shape": tuple(param.shape),
                    "max": param.data.max().item(),
                    "min": param.data.min().item(),
                }
                stats[name] = tensor_stats

        return stats


class DeepSpeedCriticWorkerBase(CriticWorkerBase):
    def offload_to_cpu(self, pin_memory=True, non_blocking=True, **kwargs):
        self._set_numa_affinity(torch.distributed.get_rank() % torch.cuda.device_count())
        self.strategy.offload_to_cpu(self.model, pin_memory, non_blocking)

    def backload_to_gpu(self, non_blocking=True, **kwargs):
        self.strategy.backload_to_gpu(self.model, non_blocking)

    def init_model(self, model_id_or_path, num_training_steps: int = None):
        assert self.cfg.trainer.strategy in ("deepspeed")
        self.zero_stage = self.cfg.trainer.critic.deepspeed_config.zero_optimization.stage
        strategy = DeepspeedStrategy(
            self.cfg.trainer.critic.deepspeed_config,
            seed=self.cfg.trainer.seed,
            micro_train_batch_size_per_gpu=self.cfg.trainer.micro_train_batch_size_per_gpu,
            zero_stage=self.zero_stage,
            bf16=self.cfg.trainer.bf16,
        )
        strategy.setup_distributed()
        self.strategy = strategy

        # Update per-gpu mini batch size based on device mesh
        self._normalize_mini_batch_size()

        ds_config = strategy.get_ds_train_config()
        # with torch.device("meta"):
        #     AutoModel.from_pretrained(pretrain, trust_remote_code=True)
        critic = get_llm_for_sequence_regression(
            model_id_or_path,
            "critic",
            use_flash_attention_2=self.cfg.trainer.flash_attn,
            bf16=self.cfg.trainer.bf16,
            target_modules=self.cfg.trainer.critic.model.lora.target_modules,
            exclude_modules=self.cfg.trainer.critic.model.lora.exclude_modules,
            ds_config=ds_config,
            value_head_prefix=self.cfg.trainer.algorithm.value_head_prefix,
            init_value_head=self.cfg.trainer.policy.model.path == self.cfg.trainer.critic.model.path,
            sequence_parallel_size=self.sequence_parallel_size,
            use_sample_packing=self.cfg.trainer.use_sample_packing,
        )
        # configure optimizer
        critic_optim = strategy.create_optimizer(
            critic,
            lr=self.cfg.trainer.critic.optimizer_config.lr,
            betas=self.cfg.trainer.critic.optimizer_config.adam_betas,
            weight_decay=self.cfg.trainer.critic.optimizer_config.weight_decay,
            offload_after_step=self.cfg.trainer.critic.optimizer_config.offload_after_step,
        )

        # configure scheduler
        critic_scheduler = get_scheduler(
            self.cfg.trainer.critic.optimizer_config.scheduler,
            critic_optim,
            num_warmup_steps=self.cfg.trainer.critic.optimizer_config.num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        if self.cfg.trainer.gradient_checkpointing:
            critic.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": self.cfg.trainer.gradient_checkpointing_use_reentrant}
            )
        # We set `use_parent_class` because critic model is of type `CriticModel` which is a subclass of the AutoModel class of interest
        self._seq_parallel_monkey_patch(model=critic, use_parent_class=True)

        # prepare models/optimizers...
        self.model, self.optimizer, self.scheduler = strategy.prepare(
            (critic, critic_optim, critic_scheduler),
        )


class DeepSpeedRefWorkerBase(RefWorkerBase):
    def offload_to_cpu(self, pin_memory=True, non_blocking=True, **kwargs):
        # deepspeed automatically offloads all model parameters to cpu
        # after forward if param_offload is true, and the ref model has no optimizer state
        # so we don't need to call offload_to_cpu here
        pass

    def backload_to_gpu(self, non_blocking=True, **kwargs):
        pass

    def init_model(self, model_path):
        assert self.cfg.trainer.strategy in ("deepspeed")
        self.zero_stage = self.cfg.trainer.ref.deepspeed_config.zero_optimization.stage
        strategy = DeepspeedStrategy(
            self.cfg.trainer.ref.deepspeed_config,
            seed=self.cfg.trainer.seed,
            micro_train_batch_size_per_gpu=self.cfg.trainer.micro_train_batch_size_per_gpu,
            zero_stage=self.zero_stage,
            bf16=self.cfg.trainer.bf16,
        )
        strategy.setup_distributed()
        self.strategy = strategy

        wrapped_model = HFModelWrapper(
            model_path,
            use_flash_attention_2=self.cfg.trainer.flash_attn,
            bf16=self.cfg.trainer.bf16,
            ds_config=strategy.get_ds_eval_config(),
            sequence_parallel_size=self.sequence_parallel_size,
            use_sample_packing=self.cfg.trainer.use_sample_packing,
            rope_scaling=get_rope_scaling_config(self.cfg.trainer),
            rope_theta=get_rope_theta_config(self.cfg.trainer),
        )
        self._seq_parallel_monkey_patch(model=wrapped_model.model)

        self.model = self.strategy.prepare(wrapped_model)
        self.model.eval()


PolicyWorker = ray.remote(num_gpus=1)(DeepSpeedPolicyWorkerBase)
CriticWorker = ray.remote(num_gpus=1)(DeepSpeedCriticWorkerBase)
RefWorker = ray.remote(num_gpus=1)(DeepSpeedRefWorkerBase)

import asyncio
from typing import List, Dict

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


class DeepSpeedWeightExtractor(WeightExtractor):
    """Extracts weights from DeepSpeed ZeRO-sharded models.

    Args:
        model: DeepSpeed model to extract weights from
        zero_stage: ZeRO optimization stage (0, 1, 2, or 3)
        group_by_module: If True, group parameters by module (e.g., for FlashRL QKV fusion)
    """

    def __init__(self, model: torch.nn.Module, zero_stage: int, group_by_module: bool = False):
        self.model = model
        self.zero_stage = zero_stage
        self.group_by_module = group_by_module

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
                tensor = self._prepare_tensor(param, dtype)
                # Get correct shape based on ZeRO stage
                shape = list(param.shape if self.zero_stage != 3 else param.ds_shape)
                yield WeightChunk(
                    names=[name],
                    dtypes=[str(dtype)],
                    shapes=[shape],
                    tensors=[tensor],
                )
        else:
            # Group parameters by module for FlashRL
            # NOTE (sumanthrh): We sync weights module by module. Ex: weights for self attn together, weights for mlp together
            # For FlashRL integration, we allocate new storage for each param. Since q, k and v layer weights are fused internally by vllm,
            # we need to pass the weights for all of these together.
            # Overall, this doesn't hurt perf even in the general case
            module_to_params: Dict[str, List[str]] = {}
            for param_name in params.keys():
                # Extract module name (e.g., "model.layers.0.self_attn" from "model.layers.0.self_attn.q_proj.weight")
                # TODO (sumanthrh): When would this fail? Works for many AutoModelForCausalLM models for now
                module_name = ".".join(param_name.split(".")[:-2])
                if module_name not in module_to_params:
                    module_to_params[module_name] = []
                module_to_params[module_name].append(param_name)

            # Yield chunks grouped by module
            for module_name, param_names in module_to_params.items():
                tensors = []
                names = []
                shapes = []
                dtypes_list = []

                for param_name in param_names:
                    param = params[param_name]
                    tensor = self._prepare_tensor(param, dtype)
                    # Get correct shape based on ZeRO stage
                    shape = list(param.shape if self.zero_stage != 3 else param.ds_shape)
                    tensors.append(tensor)
                    names.append(param_name)
                    shapes.append(shape)
                    dtypes_list.append(str(dtype))

                yield WeightChunk(
                    names=names,
                    dtypes=dtypes_list,
                    shapes=shapes,
                    tensors=tensors,
                    module_name=module_name,
                )

    def _prepare_tensor(self, param: torch.nn.Parameter, dtype: torch.dtype) -> torch.Tensor:
        """Gather (if ZeRO-3), convert dtype, and make tensor contiguous."""
        # For ZeRO-3, gather sharded parameter
        with deepspeed.zero.GatheredParameters([param], enabled=self.zero_stage == 3):
            tensor = param.data.clone()
            tensor = tensor.to(dtype)
            return tensor.detach().contiguous()


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
            target_modules=self.cfg.trainer.target_modules,
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
            # CUDA IPC path: module-grouped chunks for FlashRL
            from torch.multiprocessing.reductions import reduce_tensor

            weights_update_request = {"names": [], "dtypes": [], "shapes": [], "extras": [], "packed": False}
            current_size = 0

            # Iterate over module-grouped chunks
            for chunk in self.weight_extractor.extract_weights(generator_dtype):
                # chunk contains all parameters for one module (e.g., self_attn)
                for i, (name, tensor, shape) in enumerate(zip(chunk.names, chunk.tensors, chunk.shapes)):
                    module_done = i == len(chunk) - 1

                    # Create IPC handle for tensor
                    ipc_handle = reduce_tensor(tensor)
                    ipc_handle = {get_physical_gpu_id(): ipc_handle}
                    ipc_handle_list = [None] * torch.distributed.get_world_size()
                    torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)

                    if torch.distributed.get_rank() == 0:
                        ipc_handles = {}
                        for d in ipc_handle_list:
                            ipc_handles.update(d)

                        current_size += tensor.nbytes
                        weights_update_request["names"].append(name)
                        weights_update_request["dtypes"].append(self.cfg.generator.model_dtype)
                        weights_update_request["shapes"].append(shape)
                        weights_update_request["extras"].append({"ipc_handles": ipc_handles})

                        # We send in batches as an optimization
                        # sync if threshold is reached
                        if (
                            module_done
                            and current_size / (1024**3) > self.cfg.generator.weight_transfer_threshold_cuda_ipc_GB
                        ):
                            await inference_engine_client.update_named_weights(weights_update_request)
                            current_size = 0
                            weights_update_request = {
                                "names": [],
                                "dtypes": [],
                                "shapes": [],
                                "extras": [],
                                "packed": False,
                            }
                            # force collect any sent tensors if possible to be memory efficient
                            torch.cuda.ipc_collect()

                    torch.distributed.barrier()
                    torch.cuda.synchronize()

            # Send any remaining weights
            if len(weights_update_request["names"]) > 0 and torch.distributed.get_rank() == 0:
                await asyncio.create_task(inference_engine_client.update_named_weights(weights_update_request))
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
            target_modules=self.cfg.trainer.target_modules,
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

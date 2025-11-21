import torch
import torch.nn as nn
import torch.distributed
import ray
from transformers import AutoTokenizer, AutoConfig
from huggingface_hub import snapshot_download

import asyncio
import os
from datetime import timedelta
from typing import List, Dict, Any, Optional
from collections import defaultdict
from tqdm import tqdm
from omegaconf import OmegaConf

from megatron.bridge import AutoBridge
import megatron.core.parallel_state as mpu
from megatron.core.optimizer import DistributedOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

from skyrl_train.distributed.megatron.optimizer import (
    init_megatron_optim_config,
    get_megatron_optimizer,
    get_megatron_optimizer_param_scheduler,
)
from skyrl_train.distributed.dispatch import MeshRank
from skyrl_train.distributed.megatron.megatron_strategy import MegatronStrategy
from skyrl_train.distributed.megatron.megatron_utils import print_model_size, broadcast_object_across_pp_ranks
from skyrl_train.utils.utils import update_model_config, str_to_torch_dtype, get_physical_gpu_id
from skyrl_train.utils.constants import SKYRL_WORKER_NCCL_TIMEOUT_IN_S
from skyrl_train.training_batch import TrainingOutputBatch
from skyrl_train.workers.worker_utils import BatchIterator, reduce_metrics
from skyrl_train.workers.worker import (
    PolicyWorkerBase,
    RefWorkerBase,
    CriticWorkerBase,
)
from skyrl_train.workers.megatron.megatron_model_wrapper import MegatronModelWrapper
from skyrl_train.utils.profiler import Profiler


class MegatronWorker:
    def init_configs(
        self, model_path, megatron_config, model_config_kwargs, transformer_config_kwargs, bf16=True, flash_attn=False
    ):
        """
        Initialize the Megatron-Bridge bridge and provider objects + hf_config and tokenizer
        """
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        override_config_kwargs = {
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
        override_config_kwargs.update(model_config_kwargs.get("model_config", {}))
        update_model_config(hf_config, override_config_kwargs=override_config_kwargs)

        # if flash_attn is enabled, we use flash attention backend, otherwise fall back to fused attention backend
        transformer_config_kwargs = OmegaConf.to_container(transformer_config_kwargs, resolve=True)
        transformer_config_kwargs["attention_backend"] = "flash" if flash_attn else "fused"

        bridge = AutoBridge.from_hf_pretrained(model_path, trust_remote_code=True)
        provider = bridge.to_megatron_provider()
        provider.tensor_model_parallel_size = megatron_config.tensor_model_parallel_size
        provider.pipeline_model_parallel_size = megatron_config.pipeline_model_parallel_size
        provider.pipeline_dtype = torch.bfloat16 if bf16 else torch.float32
        provider.context_parallel_size = megatron_config.context_parallel_size
        provider.expert_model_parallel_size = megatron_config.expert_model_parallel_size
        provider.expert_tensor_parallel_size = megatron_config.expert_tensor_parallel_size
        provider.sequence_parallel = megatron_config.tensor_model_parallel_size > 1
        provider.attention_backend = "flash" if flash_attn else "fused"
        provider.variable_seq_lengths = True
        provider.masked_softmax_fusion = True
        provider.moe_token_dispatcher_type = "alltoall"

        for k, v in transformer_config_kwargs.items():
            setattr(provider, k, v)
        provider.finalize()

        self.provider = provider
        self.bridge = bridge

        self.strategy.hf_config = hf_config
        self.tokenizer = tokenizer

    def make_megatron_module(
        self,
        wrap_with_ddp: bool = True,
        ddp_config: Optional[Dict[str, Any]] = None,
        bf16: bool = True,
    ) -> List[nn.Module]:
        """
        Creates a megatron GPTModel (optionally DDP wrapped) using the bridge.
        """
        from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig

        default_ddp_config = DistributedDataParallelConfig()
        if wrap_with_ddp:
            default_ddp_config.use_distributed_optimizer = True
        if ddp_config is not None:
            for k, v in ddp_config.items():
                setattr(default_ddp_config, k, v)
        model = self.provider.provide_distributed_model(
            ddp_config=default_ddp_config, wrap_with_ddp=wrap_with_ddp, bf16=bf16
        )
        return model

    def forward(self, data):
        """
        Override `Worker.forward` to support passing the full mini batch to the MegatronModelWrapper.forward method.
        """
        # Run in micro batches grouped into a single mini-batch
        micro_bsz = self.cfg.trainer.micro_forward_batch_size_per_gpu
        micro_batches = data.chunk(micro_bsz)

        # Build micro-batch dicts expected by policy.forward_mini_batch
        micro_dicts = []
        device = torch.cuda.current_device()
        for micro in micro_batches:
            micro.to(device)
            sequences = micro["sequences"]
            attention_mask = micro["attention_mask"]
            num_actions = micro.metadata["response_length"]
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            micro_dicts.append(
                {
                    "sequences": sequences,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "num_actions": num_actions,
                }
            )

        self.model.eval()
        seq_len = micro_dicts[0]["sequences"].shape[1]
        mbs = micro_dicts[0]["sequences"].shape[0]
        with torch.no_grad():
            log_probs = self.model.forward(
                micro_batches=micro_dicts,
                seq_len=seq_len,
                micro_batch_size=mbs,
                temperature=self.cfg.generator.sampling_params.temperature,
            )

        log_probs = log_probs.to("cpu")
        output = TrainingOutputBatch({"output": log_probs})
        output.metadata = data.metadata
        return output

    def save_hf_model(self, export_dir: str, tokenizer):
        # Save model in HuggingFace safetensors format
        self.strategy.save_hf_model(
            self.bridge,
            self.model,
            export_dir,
            tokenizer=tokenizer,
        )


class MegatronPolicyWorkerBase(MegatronWorker, PolicyWorkerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model: MegatronModelWrapper = None
        self.actor_module: List[nn.Module] = None
        self.scheduler: OptimizerParamScheduler = None
        self.optimizer: DistributedOptimizer = None
        self.profiler: Profiler = None

    def offload_to_cpu(self, pin_memory=True, non_blocking=True, offload_optimizer=True, offload_model=True):
        self._set_numa_affinity(torch.distributed.get_rank() % torch.cuda.device_count())
        self.strategy.offload_to_cpu(
            self.actor_module, self.optimizer, pin_memory, non_blocking, offload_optimizer, offload_model
        )

    def backload_to_gpu(self, non_blocking=True, backload_optimizer=True, backload_model=True):
        self.strategy.backload_to_gpu(
            self.actor_module, self.optimizer, non_blocking, backload_optimizer, backload_model
        )

    def init_worker_process_group(self):
        """
        Override DistributedTorchRayActor.init_worker_process_group to use megatron distributed setup to create the mesh.
        """
        if not torch.distributed.is_initialized():
            # Default torch dist pg init timeout is 10 minutes (600 seconds)
            torch.distributed.init_process_group(
                backend="nccl", timeout=timedelta(seconds=SKYRL_WORKER_NCCL_TIMEOUT_IN_S)
            )

        # Explicitly wrap torch.distributed.broadcast in torch.no_grad() to avoid a warning in Megatron training where the
        # autograd engine tries to track gradients through the default Torch kernel. This fixes a deprecated behaviour in
        # PyTorch, preventing potential silent errors in future versions.

        if not getattr(torch.distributed, "_skyrl_broadcast_no_grad_patched", False):
            _orig_broadcast = torch.distributed.broadcast

            def _broadcast_no_grad(*args, **kwargs):
                with torch.no_grad():
                    return _orig_broadcast(*args, **kwargs)

            torch.distributed.broadcast = _broadcast_no_grad
            torch.distributed._skyrl_broadcast_no_grad_patched = True

        self.strategy = MegatronStrategy(
            megatron_config=self.cfg.trainer.policy.megatron_config,
            optimizer_config=self.cfg.trainer.policy.optimizer_config,
            seed=self.cfg.trainer.seed,
        )
        self.strategy.setup_distributed()

        self.mesh_rank = MeshRank(
            dp=mpu.get_data_parallel_rank(),
            sp=mpu.get_context_parallel_rank(),
            tp=mpu.get_tensor_model_parallel_rank(),
            pp=mpu.get_pipeline_model_parallel_rank(),
            world_size=self._world_size,
            dp_size=mpu.get_data_parallel_world_size(),
            pp_size=mpu.get_pipeline_model_parallel_world_size(),
        )

    def init_model(self, model_path, num_training_steps: int = 1e9):
        """
        Initialize the model, optimizer, and scheduler for the policy worker.
        """
        # initialize the bridge and provider objects
        self.init_configs(
            model_path,
            self.cfg.trainer.policy.megatron_config,
            self.cfg.trainer.policy.megatron_config.model_config_kwargs,
            self.cfg.trainer.policy.megatron_config.transformer_config_kwargs,
            bf16=self.cfg.trainer.bf16,
            flash_attn=self.cfg.trainer.flash_attn,
        )

        # wrap with DDP for training
        self.actor_module = self.make_megatron_module(
            wrap_with_ddp=True,
            ddp_config=self.cfg.trainer.policy.megatron_config.ddp_config,
            bf16=self.cfg.trainer.bf16,
        )

        if self._local_rank == 0 and not os.path.exists(
            model_path
        ):  # if not local path, try downloading model weights from huggingface
            snapshot_download(model_path)  # will be no-op if already downloaded
        torch.distributed.barrier()

        if self._rank == 0:
            print_model_size(self.actor_module[0])

        # create profiler
        if self.cfg.trainer.policy.megatron_config.torch_profiler_config.enable:
            self.profiler = Profiler(self.cfg.trainer.policy.megatron_config.torch_profiler_config)

        # create optimizer
        optim_config = init_megatron_optim_config(
            self.cfg.trainer.policy.optimizer_config, self.cfg.trainer.policy.megatron_config.optimizer_config_kwargs
        )
        self.optimizer = get_megatron_optimizer(self.actor_module, optim_config)

        self._normalize_mini_batch_size()

        # create scheduler
        self.scheduler = get_megatron_optimizer_param_scheduler(
            optimizer=self.optimizer,
            config=self.cfg.trainer.policy.optimizer_config,
            num_training_steps=num_training_steps,
        )

        # create worker model
        self.model = MegatronModelWrapper(
            config=self.cfg,
            actor_module=self.actor_module,
            actor_optimizer=self.optimizer,
            policy_loss_fn=self.policy_loss_fn,
        )

        self.use_cuda_ipc = False
        if self.cfg.generator.weight_sync_backend == "nccl" and self.cfg.trainer.placement.colocate_all:
            self.use_cuda_ipc = True
            # init weight syncing state
            self.weight_conversion_tasks = self.bridge.get_conversion_tasks(self.actor_module)

            # map the params that go in each bucket
            param_info = []

            def calculate_size_in_bytes(param, tp_size, ep_size):
                if param is None:
                    # need to broadcast for other pp ranks
                    size_in_bytes = None
                else:
                    # Calculate size for this parameter
                    prec_to_bytes = {
                        torch.bfloat16: 2,
                        torch.float32: 4,
                    }
                    scale = (
                        prec_to_bytes[torch.bfloat16 if self.cfg.trainer.bf16 else torch.float32]
                        / prec_to_bytes[param.dtype]
                    )
                    size_in_bytes = param.element_size() * param.numel() * tp_size * ep_size * scale

                # Broadcast size_in_bytes across pipeline parallel ranks
                return broadcast_object_across_pp_ranks(size_in_bytes)

            for task in self.weight_conversion_tasks:
                param_info.append(
                    (
                        task,
                        calculate_size_in_bytes(
                            task.param_weight,
                            task.mapping.tp_size,
                            task.mapping.ep_size if task.mapping.is_expert else 1,
                        ),
                    )
                )

            self.param_buckets = [[]]
            curr_size = 0
            for p in param_info:
                task, size = p
                if curr_size + size > self.cfg.generator.weight_transfer_threshold_cuda_ipc_GB * 1024**3:
                    self.param_buckets.append([])
                    curr_size = 0
                self.param_buckets[-1].append(task)
                curr_size += size

    def ppo_train(self, train_data) -> "TrainingOutputBatch":
        """
        Overrides `PolicyWorkerBase.ppo_train` for megatron.

        Since we want megatron to handle gradient accumulation over micro batches, we directly pass mini batches into the
        worker MegatronModelWrapper.forward_backward_mini_batch method.
        """
        dataloader = BatchIterator(
            train_data, sample_batch_size=self.cfg.trainer.micro_train_batch_size_per_gpu, drop_last=False
        )

        micro_batches_per_mini_batch = (
            self.policy_mini_batch_size_per_gpu // self.cfg.trainer.micro_train_batch_size_per_gpu
        )

        status_list = []
        all_metrics = defaultdict(list)
        policy_update_steps = 0

        if self.profiler is not None:
            self.profiler.start()

        for epoch in range(self.cfg.trainer.update_epochs_per_batch):
            self.optimizer.zero_grad()
            pbar = tqdm(
                dataloader,
                desc=f"Policy Train epoch [{epoch + 1}/{self.cfg.trainer.update_epochs_per_batch}]",
                disable=not self.strategy.is_rank_0(),
            )

            micro_buffer = []
            for local_step, experience in enumerate(pbar):
                experience.to_device(torch.cuda.current_device())
                sequences = experience.sequences
                attention_mask = experience.attention_mask
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 0)

                micro_buffer.append(
                    {
                        "sequences": sequences,
                        "attention_mask": attention_mask,
                        "position_ids": position_ids,
                        "num_actions": experience.num_actions,
                        "old_action_log_probs": experience.action_log_probs,
                        "base_action_log_probs": experience.base_action_log_probs,
                        "advantages": experience.advantages,
                        "loss_mask": experience.loss_mask,
                        "rollout_action_logprobs": experience.rollout_logprobs,
                    }
                )

                if len(micro_buffer) == micro_batches_per_mini_batch:
                    # run mini-batch forward-backward and then one optimizer step
                    self.model.train()
                    for chunk in self.actor_module:
                        # if use distributed optimizer, zero grad buffer will be handled by optimizer
                        chunk.zero_grad_buffer()
                    seq_len = micro_buffer[0]["sequences"].shape[1]
                    micro_bsz = micro_buffer[0]["sequences"].shape[0]

                    metrics_list = self.model.forward_backward_mini_batch(
                        micro_batches=micro_buffer,
                        seq_len=seq_len,
                        micro_batch_size=micro_bsz,
                        temperature=self.cfg.generator.sampling_params.temperature,
                    )

                    grad_norm = self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler, name="actor")

                    # within a DP group, metrics are already the same across all workers - we then just all reduce across
                    # the whole world size to get the metrics for the global micro batch
                    for i, metrics in enumerate(metrics_list):
                        status = {
                            "final_loss": metrics["final_loss"],
                            "policy_loss": metrics["policy_loss"],
                            "policy_lr": self.optimizer.param_groups[0]["lr"],
                            "ppo_clip_ratio": metrics["ppo_clip_ratio"],
                            "policy_entropy": metrics["policy_entropy"],
                        }
                        if self.cfg.trainer.algorithm.use_kl_loss:
                            status["policy_kl"] = metrics["policy_kl"]

                        # Attach grad norm only for the last micro in the mini-batch
                        if i == len(metrics_list) - 1 and grad_norm is not None:
                            status["raw_grad_norm"] = grad_norm

                        # attach response_length
                        status["response_length"] = micro_buffer[i]["num_actions"]

                        status = self.strategy.all_reduce(status)
                        status_list.append(status)
                        for k, v in status.items():
                            all_metrics[k].append(v)

                    short_status = {
                        "pg": status_list[-1]["policy_loss"],
                        "glen": status_list[-1]["response_length"],
                        "policy_lr": status_list[-1]["policy_lr"],
                        "ent": status_list[-1]["policy_entropy"],
                    }
                    if "raw_grad_norm" in status_list[-1]:
                        short_status["grad_norm"] = status_list[-1]["raw_grad_norm"]
                    pbar.set_postfix(short_status)

                    policy_update_steps += 1
                    micro_buffer = []

            # drop any trailing micros that don't fill a mini-batch (keep behavior consistent)
            micro_buffer = []

        torch.distributed.barrier()
        if self.profiler is not None:
            self.profiler.stop_and_save()
            self.profiler.stop_trace()

        # not needed beyond status logging
        all_metrics.pop("response_length", None)

        status_mean = reduce_metrics(all_metrics)
        status_mean["policy_update_steps"] = policy_update_steps

        output = TrainingOutputBatch()
        output.metadata = {"train_status": status_mean}
        return output

    async def broadcast_to_inference_engines(self, inference_engine_client):
        from torch.multiprocessing.reductions import reduce_tensor

        use_prefix_cache = self.cfg.generator.enable_prefix_caching
        generator_dtype = str_to_torch_dtype(self.cfg.generator.model_dtype)
        cache_reset_task = None
        if use_prefix_cache and torch.distributed.get_rank() == 0:
            # clear prefix cache
            cache_reset_task = inference_engine_client.reset_prefix_cache()

        torch.cuda.empty_cache()
        weights_update_request = {"names": [], "dtypes": [], "shapes": [], "sizes": [], "extras": []}
        device = torch.cuda.current_device()

        # Non CUDA IPC wt sync
        if not self.use_cuda_ipc:
            # NOTE: need to optimize this to use buckets for non-colocated weight sync as well
            per_tensor_param = self.bridge.export_hf_weights(self.actor_module, show_progress=False)
            for name, param in per_tensor_param:
                if torch.distributed.get_rank() == 0:
                    update_weight_task = asyncio.create_task(
                        inference_engine_client.update_named_weights(
                            {
                                "names": [name],
                                "dtypes": [self.cfg.generator.model_dtype],
                                "shapes": [param.shape],
                            }
                        )
                    )

                def broadcast_param(param):
                    device = torch.cuda.current_device()
                    param = param.to(device, non_blocking=True)
                    param = param.to(generator_dtype)

                    # Broadcast weights from training rank 0 to inference engine ranks via the update group
                    if torch.distributed.get_rank() == 0:
                        torch.distributed.broadcast(param.data, 0, group=self._model_update_group)

                await asyncio.to_thread(broadcast_param, param)
                if torch.distributed.get_rank() == 0:
                    await update_weight_task
                torch.distributed.barrier()
        # CUDA IPC wt sync
        else:
            for bucket in self.param_buckets:
                hf_params_generator = self.bridge.export_hf_weights(
                    self.actor_module,
                    show_progress=False,
                    conversion_tasks=bucket,
                )
                gathered_hf_params = {name: tensor for name, tensor in hf_params_generator}
                gathered_hf_params = {
                    name: tensor.to(device=device, dtype=generator_dtype) for name, tensor in gathered_hf_params.items()
                }

                total_size = sum(tensor.numel() for tensor in gathered_hf_params.values())
                packed_tensor = torch.empty(
                    total_size,
                    device=device,
                    dtype=generator_dtype,
                    requires_grad=False,
                )

                offset = 0
                # Copy tensors into consolidated buffers
                for key, tensor in gathered_hf_params.items():
                    size = tensor.numel()
                    packed_tensor[offset : offset + size].copy_(tensor.detach().view(-1))
                    offset += size
                    weights_update_request["names"].append(key)
                    weights_update_request["dtypes"].append(self.cfg.generator.model_dtype)
                    weights_update_request["shapes"].append(tensor.shape)
                    weights_update_request["sizes"].append(size)

                ipc_handle = reduce_tensor(packed_tensor)
                ipc_handle = {get_physical_gpu_id(): ipc_handle}
                ipc_handle_list = [None] * torch.distributed.get_world_size()
                torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)

                ipc_handles = {}
                for d in ipc_handle_list:
                    ipc_handles.update(d)

                weights_update_request["extras"].append({"ipc_handles": ipc_handles})
                weights_update_request["packed"] = True

                if torch.distributed.get_rank() == 0:
                    await inference_engine_client.update_named_weights(weights_update_request)
                    weights_update_request = {"names": [], "dtypes": [], "shapes": [], "sizes": [], "extras": []}

                # force collect any sent tensors if possible to be memory efficient
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
        # this already gets set in the init_model method
        pass


class MegatronRefWorkerBase(MegatronWorker, RefWorkerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model: MegatronModelWrapper = None
        self.actor_module: List[nn.Module] = None

    def offload_to_cpu(self, pin_memory=True, non_blocking=True, **kwargs):
        self._set_numa_affinity(torch.distributed.get_rank() % torch.cuda.device_count())
        self.strategy.offload_to_cpu(self.actor_module, None, pin_memory, non_blocking)

    def backload_to_gpu(self, non_blocking=True, **kwargs):
        self.strategy.backload_to_gpu(self.actor_module, None, non_blocking)

    def init_worker_process_group(self):
        """
        Override DistributedTorchRayActor.init_worker_process_group to use megatron distributed setup to create the mesh.
        """
        if not torch.distributed.is_initialized():
            # Default torch dist pg init timeout is 10 minutes (600 seconds)
            torch.distributed.init_process_group(
                backend="nccl", timeout=timedelta(seconds=SKYRL_WORKER_NCCL_TIMEOUT_IN_S)
            )

        self.strategy = MegatronStrategy(
            megatron_config=self.cfg.trainer.ref.megatron_config,
            optimizer_config=None,
            seed=self.cfg.trainer.seed,
        )
        self.strategy.setup_distributed()

        self.mesh_rank = MeshRank(
            dp=mpu.get_data_parallel_rank(),
            sp=mpu.get_context_parallel_rank(),
            tp=mpu.get_tensor_model_parallel_rank(),
            pp=mpu.get_pipeline_model_parallel_rank(),
            world_size=self._world_size,
            dp_size=mpu.get_data_parallel_world_size(),
            pp_size=mpu.get_pipeline_model_parallel_world_size(),
        )

    def init_model(self, model_path, num_training_steps: int = 1e9):
        """
        Initialize the model for the ref worker.
        """
        # initialize the bridge and provider objects
        self.init_configs(
            model_path,
            self.cfg.trainer.ref.megatron_config,
            self.cfg.trainer.ref.megatron_config.model_config_kwargs,
            self.cfg.trainer.ref.megatron_config.transformer_config_kwargs,
            bf16=self.cfg.trainer.bf16,
            flash_attn=self.cfg.trainer.flash_attn,
        )

        self.actor_module = self.make_megatron_module(
            wrap_with_ddp=False,
            ddp_config=None,
            bf16=self.cfg.trainer.bf16,
        )

        # download model weights from huggingface (need to be done for ref worker as well, else errors when colocate_all=False)
        if self._local_rank == 0 and not os.path.exists(
            model_path
        ):  # if not local path, try downloading model weights from huggingface
            snapshot_download(model_path)  # will be no-op if already downloaded
        torch.distributed.barrier()

        # load weights
        if self._rank == 0:
            print_model_size(self.actor_module[0])

        # create worker model
        self.model = MegatronModelWrapper(config=self.cfg, actor_module=self.actor_module)

    def get_weight_statistics(self):
        """Compute lightweight statistics for model weights"""
        raise NotImplementedError()

    def _set_pad_token_id(self, pad_token_id):
        # this already gets set in the init_model method
        pass


class MegatronCriticWorkerBase(MegatronWorker, CriticWorkerBase):
    def __init__(self, **kwargs):
        raise NotImplementedError()


PolicyWorker = ray.remote(num_gpus=1)(MegatronPolicyWorkerBase)
RefWorker = ray.remote(num_gpus=1)(MegatronRefWorkerBase)
CriticWorker = ray.remote(num_gpus=1)(MegatronCriticWorkerBase)

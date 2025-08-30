import torch
import torch.nn as nn
import torch.distributed
import ray
from transformers import AutoTokenizer, AutoConfig
import asyncio
from typing import List, Dict

from mbridge import AutoBridge
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
from skyrl_train.distributed.megatron.megatron_utils import freeze_moe_router, print_model_size
from skyrl_train.utils.utils import update_model_config, str_to_torch_dtype, get_physical_gpu_id
from skyrl_train.workers.worker import (
    PolicyWorkerBase,
    RefWorkerBase,
    RewardWorkerBase,
    CriticWorkerBase,
)
from skyrl_train.workers.megatron.megatron_policy import MegatronPPOPolicy
from skyrl_train.dataset.replay_buffer import Experience


class MegatronWorker:
    def init_configs(self, model_path, model_config_kwargs, transformer_config_kwargs):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        hf_config = AutoConfig.from_pretrained(model_path)

        override_config_kwargs = {
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
        override_config_kwargs.update(model_config_kwargs.get("model_config", {}))
        update_model_config(hf_config, override_config_kwargs=override_config_kwargs)

        bridge = AutoBridge.from_config(hf_config)
        bridge.set_extra_args(**transformer_config_kwargs)
        tf_config = bridge.config
        self.bridge = bridge

        self.hf_config = hf_config
        self.tf_config = tf_config
        self.tokenizer = tokenizer

    def make_megatron_module(self, model_config_kwargs, wrap_with_ddp=True):
        model = self.bridge.get_model(
            post_model_creation_callbacks=[],  # don't rely on these since we might switch to Megatron-Bridge
            wrap_with_ddp=wrap_with_ddp,
            ddp_config=self.cfg.trainer.policy.megatron_config.ddp_config,
        )
        if model_config_kwargs.get("moe_config", {}).get("freeze_moe_router", False):
            freeze_moe_router(model)
        return model


class MegatronPolicyWorkerBase(MegatronWorker, PolicyWorkerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model: MegatronPPOPolicy = None
        self.actor_module: List[nn.Module] = None
        self.scheduler: OptimizerParamScheduler = None
        self.optimizer: DistributedOptimizer = None

    def offload_to_cpu(self, pin_memory=True, non_blocking=True):
        self._set_numa_affinity(torch.distributed.get_rank() % torch.cuda.device_count())
        self.strategy.offload_to_cpu(self.actor_module, self.optimizer, pin_memory, non_blocking)

    def backload_to_gpu(self, non_blocking=True):
        self.strategy.backload_to_gpu(self.actor_module, self.optimizer, non_blocking)

    def init_worker_process_group(self):
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        # override the init_process_group method to use megatron distributed setup to create the mesh
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
        # get hf_config and tf_config
        self.init_configs(
            model_path,
            self.cfg.trainer.policy.megatron_config.model_config_kwargs,
            self.cfg.trainer.policy.megatron_config.transformer_config_kwargs,
        )

        # wrap with DDP for training
        self.actor_module = self.make_megatron_module(
            self.cfg.trainer.policy.megatron_config.model_config_kwargs, wrap_with_ddp=True
        )

        # load weights
        # NOTE (erictang000): there is currently a bug in mbridge that causes the model to not load correctly if tie_word_embeddings is set
        # see: https://github.com/NVIDIA/Megatron-LM/issues/533#issuecomment-1760193239
        # this is the case for the Qwen2.5-1.5B and 3B models, but not the 7B model
        self.bridge.load_weights(self.actor_module, model_path)

        if self._rank == 0:
            print_model_size(self.actor_module[0])

        # create optimizer
        optim_config = init_megatron_optim_config(self.cfg.trainer.policy.optimizer_config)
        self.optimizer = get_megatron_optimizer(self.actor_module, optim_config)

        self._normalize_mini_batch_size()

        # create scheduler
        self.scheduler = get_megatron_optimizer_param_scheduler(
            optimizer=self.optimizer,
            config=self.cfg.trainer.policy.optimizer_config,
            num_training_steps=num_training_steps,
        )

        # create worker model
        self.model = MegatronPPOPolicy(
            config=self.cfg,
            hf_config=self.hf_config,
            tf_config=self.tf_config,
            actor_module=self.actor_module,
            actor_optimizer=self.optimizer,
            policy_loss_fn=self.policy_loss_fn,
        )

    def training_step(self, experience: Experience, global_step, local_step, accumulation_steps) -> Dict[str, float]:
        """
        Perform one micro-batch of training, accumulate gradients, and step the optimizer only after `accumulation_steps` micro-batches.
        """
        self.model.train()
        experience.to_device(torch.cuda.current_device())

        sequences = experience.sequences
        num_actions = experience.num_actions
        attention_mask = experience.attention_mask

        metrics = self.model.forward_backward_micro_batch(
            sequences,
            num_actions,
            accumulation_steps,
            old_action_log_probs=experience.action_log_probs,
            base_action_log_probs=experience.base_action_log_probs,
            advantages=experience.advantages,
            loss_mask=experience.loss_mask,
            rollout_action_logprobs=experience.rollout_logprobs,
            attention_mask=attention_mask,
            temperature=self.cfg.generator.sampling_params.temperature,
        )

        grad_norm = None
        if (local_step + 1) % accumulation_steps == 0:
            self.model._restore_and_finalize_once()
            grad_norm = self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler, name="actor")
            self.model._set_finalize_noop()

        if self.record_memory:
            self.save_memory_snapshot(global_step, local_step)

        status = {
            "policy_loss": metrics["policy_loss"],
            "policy_lr": self.optimizer.param_groups[0]["lr"],
            "ppo_clip_ratio": metrics["ppo_clip_ratio"],
            "policy_entropy": metrics["policy_entropy"],
        }
        if self.cfg.trainer.algorithm.use_kl_loss:
            status["policy_kl"] = metrics["policy_kl"]

        if grad_norm is not None:
            status["raw_grad_norm"] = grad_norm

        for k, v in experience.info.items():
            if k == "kl":
                # just use the same value as loss if available
                status[k] = (
                    metrics["policy_kl"].item()
                    if isinstance(metrics["policy_kl"], torch.Tensor)
                    else status["policy_kl"]
                )
            else:
                status[k] = v.mean().item() if isinstance(v, torch.Tensor) else v

        status["response_length"] = num_actions
        return status

    async def broadcast_to_inference_engines(self, inference_engine_client):
        use_prefix_cache = self.cfg.generator.enable_prefix_caching
        generator_dtype = str_to_torch_dtype(self.cfg.generator.model_dtype)
        cache_reset_task = None
        if use_prefix_cache and torch.distributed.get_rank() == 0:
            # clear prefix cache
            cache_reset_task = inference_engine_client.reset_prefix_cache()

        torch.cuda.empty_cache()
        per_tensor_param = self.bridge.export_weights(self.actor_module)

        for name, param in per_tensor_param:
            # NOTE (erictang000) we do not use bucketed weight updates for megatron here, which means this is not compatible with the FlashRL integration
            # in the future we should improve this to use bucketed weight updates and support FlashRL + megatron for large models
            from torch.multiprocessing.reductions import reduce_tensor

            device = torch.cuda.current_device()
            param = param.to(device, non_blocking=True)
            param = param.to(generator_dtype)
            weight = param.data.clone()
            ipc_handle = reduce_tensor(weight)

            ipc_handle = {get_physical_gpu_id(): ipc_handle}

            ipc_handle_list = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)

            if torch.distributed.get_rank() == 0:
                ipc_handles = {}
                for d in ipc_handle_list:
                    ipc_handles.update(d)

                shape = param.shape

                await asyncio.create_task(
                    inference_engine_client.update_named_weights(
                        {
                            "names": [name],
                            "dtypes": [self.cfg.generator.model_dtype],
                            "shapes": [shape],
                            "extras": [
                                {
                                    "ipc_handles": ipc_handles,
                                }
                            ],
                        }
                    )
                )

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
        self.model: MegatronPPOPolicy = None
        self.actor_module: List[nn.Module] = None

    def offload_to_cpu(self, pin_memory=True, non_blocking=True):
        self._set_numa_affinity(torch.distributed.get_rank() % torch.cuda.device_count())
        self.strategy.offload_to_cpu(self.actor_module, None, pin_memory, non_blocking)

    def backload_to_gpu(self, non_blocking=True):
        self.strategy.backload_to_gpu(self.actor_module, None, non_blocking)

    def init_worker_process_group(self):
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        # override the init_process_group method to use megatron distributed setup to create the mesh
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
        # get hf_config and tf_config
        self.init_configs(
            model_path,
            self.cfg.trainer.ref.megatron_config.model_config_kwargs,
            self.cfg.trainer.ref.megatron_config.transformer_config_kwargs,
        )

        # wrap with DDP for training
        self.actor_module = self.make_megatron_module(
            self.cfg.trainer.ref.megatron_config.model_config_kwargs, wrap_with_ddp=False
        )

        # load weights
        self.bridge.load_weights(self.actor_module, model_path)
        if self._rank == 0:
            print_model_size(self.actor_module[0])

        # create worker model
        self.model = MegatronPPOPolicy(
            config=self.cfg, hf_config=self.hf_config, tf_config=self.tf_config, actor_module=self.actor_module
        )

    def get_weight_statistics(self):
        """Compute lightweight statistics for model weights"""
        raise NotImplementedError()

    def _set_pad_token_id(self, pad_token_id):
        # this already gets set in the init_model method
        pass


class MegatronRewardWorkerBase(MegatronWorker, RewardWorkerBase):
    def __init__(self, **kwargs):
        raise NotImplementedError()


class MegatronCriticWorkerBase(MegatronWorker, CriticWorkerBase):
    def __init__(self, **kwargs):
        raise NotImplementedError()


PolicyWorker = ray.remote(num_gpus=1)(MegatronPolicyWorkerBase)
RefWorker = ray.remote(num_gpus=1)(MegatronRefWorkerBase)
CriticWorker = ray.remote(num_gpus=1)(MegatronCriticWorkerBase)
RewardWorker = ray.remote(num_gpus=1)(MegatronRewardWorkerBase)

# This code is adapted from OpenRLHF
# https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/utils/deepspeed/deepspeed.py

import os
import random
from collections import defaultdict
from datetime import timedelta
from typing import List, Union, Optional
from omegaconf import OmegaConf
from jaxtyping import Float

import deepspeed
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from torch import distributed as dist
from torch.optim import Optimizer
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum

from skyrl_train.distributed.strategy import DistributedStrategy
from skyrl_train.model_wrapper import HFModelWrapper
from skyrl_train.distributed.utils import get_optimizer_grouped_parameters, ModelOrModelOptimPair
from skyrl_train.utils.io import io

from safetensors.torch import save_file


def _z3_params_to_fetch(param_list):
    return [p for p in param_list if hasattr(p, "ds_id") and p.ds_status == ZeroParamStatus.NOT_AVAILABLE]


class DeepspeedStrategy(DistributedStrategy):
    """
    The strategy for training with Accelerator.
    """

    def __init__(
        self,
        deepspeed_config,
        seed: int = 42,
        micro_train_batch_size_per_gpu=1,
        train_batch_size=1,
        zero_stage=3,
        bf16=True,
    ) -> None:
        super().__init__()

        self.deepspeed_config = deepspeed_config
        self.stage = zero_stage
        self.train_batch_size = train_batch_size
        self.micro_train_batch_size_per_gpu = micro_train_batch_size_per_gpu
        self.bf16 = bf16
        self.seed = seed

        self.time_steps = defaultdict(int)

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup_distributed(self, timeout=timedelta(minutes=30)) -> None:
        self.set_seed(self.seed)

        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank != -1:
            torch.cuda.set_device(local_rank)

        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        deepspeed.init_distributed(timeout=timeout)
        self.world_size = dist.get_world_size()
        self.accumulated_gradient = self.train_batch_size // self.micro_train_batch_size_per_gpu // self.world_size

    def create_optimizer(self, model, offload_after_step=True, **kwargs) -> Optimizer:
        if isinstance(model, HFModelWrapper):
            model = model.model
        # TODO (sumanthrh): Support this
        if not offload_after_step:
            raise NotImplementedError("Disabling offload after step is not supported for deepspeed")
        # Optimizer
        cpu_optimizer = self.deepspeed_config.zero_optimization.offload_optimizer.device == "cpu"
        AdamOptimizer = DeepSpeedCPUAdam if cpu_optimizer else FusedAdam
        optim_params = get_optimizer_grouped_parameters(model, kwargs["weight_decay"])
        optim = AdamOptimizer(optim_params, **kwargs)
        return optim

    def offload_to_cpu(self, model, pin_memory=True, non_blocking=True):
        """This function guaratees the memory are all released (only torch context cache <100M will remain)."""
        if isinstance(model, HFModelWrapper):
            model = model.model

        if model.config["zero_optimization"]["offload_optimizer"]["device"] == "cpu":
            # if doing optimizer offload, no need to offload states
            return
        elif model.zero_optimization_stage() == 3:
            from deepspeed.runtime.zero.offload_config import OffloadStateTypeEnum

            model.optimizer.offload_states(
                include=[
                    OffloadStateTypeEnum.optim_states,
                    OffloadStateTypeEnum.contiguous_grad_buffer,
                    OffloadStateTypeEnum.hp_params,
                    # this will break for deepspeed < 0.16.5, https://github.com/deepspeedai/DeepSpeed/pull/7050
                    OffloadStateTypeEnum.lp_grads,
                    # OffloadStateTypeEnum.lp_params, # dangerous
                ],
                device=OffloadDeviceEnum.cpu,
                pin_memory=pin_memory,
                non_blocking=non_blocking,
            )
            torch.cuda.synchronize()
            return

        raise NotImplementedError("Zero stage < 3 is not supported")

    def backload_to_gpu(self, model, non_blocking=True):
        # NOTE: this function reloads the weights, ensuring the calculation
        if isinstance(model, HFModelWrapper):
            model = model.model
        else:
            model = model

        if model.config["zero_optimization"]["offload_optimizer"]["device"] == "cpu":
            # if doing optimizer offload, no need to reload states
            return
        if model.zero_optimization_stage() == 3:
            model.reload_states(non_blocking=non_blocking)
            torch.cuda.synchronize()
            return

        raise NotImplementedError("Zero stage < 3 is not supported")

    def backward(self, loss: torch.Tensor, model: nn.Module, optimizer: optim.Optimizer, **kwargs) -> None:
        if isinstance(model, HFModelWrapper):
            model = model.model
        model.backward(loss)

    # TODO(sumanthrh): Support logging grad norm here and verify grad clipping.
    def optimizer_step(
        self,
        optimizer: optim.Optimizer,
        model: nn.Module,
        scheduler,
        name="model",
        **kwargs,
    ) -> Optional[Float[torch.Tensor, "1"]]:
        if isinstance(model, HFModelWrapper):
            model = model.model
        model.step()

    def prepare(
        self, *models_or_model_optim_pairs: ModelOrModelOptimPair
    ) -> Union[List[ModelOrModelOptimPair], ModelOrModelOptimPair]:
        ret = []
        for arg in models_or_model_optim_pairs:
            if isinstance(arg, tuple):
                assert len(arg) == 3, f'Expect (model, optimizer, scheduler) pair, got a tuple with size "{len(arg)}"'
                ret.append(self._ds_init_train_model(*arg))
            else:
                ret.append(self._ds_init_eval_model(arg))

        return ret[0] if len(ret) == 1 else ret

    def _ds_init_train_model(self, model, optim, scheduler):
        is_wrapped = isinstance(model, HFModelWrapper)
        ds_config = self.get_ds_train_config()

        engine, optim, _, scheduler = deepspeed.initialize(
            model=model.model if is_wrapped else model,
            optimizer=optim,
            lr_scheduler=scheduler,
            config=ds_config,
            dist_init_required=True,
        )
        if is_wrapped:
            model.model = engine
        else:
            model = engine

        return model, optim, scheduler

    def _ds_init_eval_model(self, model):
        if not model:
            return model
        is_wrapped = isinstance(model, HFModelWrapper)
        ds_config = self.get_ds_eval_config()

        engine, *_ = deepspeed.initialize(
            model=model.model if is_wrapped else model,
            config=ds_config,
            dist_init_required=True,
        )
        if is_wrapped:
            model.model = engine
        else:
            model = engine
        return model

    def _unwrap_model(self, model) -> nn.Module:
        if isinstance(model, HFModelWrapper):
            return self._unwrap_model(model.model)
        elif hasattr(model, "module"):
            return model.module
        else:
            return model

    def save_checkpoint(
        self,
        model,
        ckpt_dir,
        node_local_rank,
        optimizer=None,
        scheduler=None,
        client_state={},
        tag=None,
        tokenizer=None,
    ):
        if isinstance(model, HFModelWrapper):
            model = model.model

        assert isinstance(model, deepspeed.DeepSpeedEngine)

        if node_local_rank == 0:
            io.makedirs(ckpt_dir, exist_ok=True)

        dist.barrier()

        extra_state_dict = {
            "client_state": client_state,
            "deepspeed_config": OmegaConf.to_container(self.deepspeed_config),
            "rng": self.get_rng_state(),  # Add RNG state for reproducibility
        }

        # Use context manager to handle local vs cloud paths
        with io.local_work_dir(ckpt_dir) as work_dir:
            model.save_checkpoint(work_dir, tag=tag, client_state=extra_state_dict)

            # Save HuggingFace config and tokenizer
            if self.is_rank_0():
                config_save_model = self._unwrap_model(model)
                hf_dir = os.path.join(work_dir, "huggingface")
                self.save_hf_configs(config_save_model.config, hf_dir, tokenizer)

    def load_checkpoint(
        self,
        model,
        ckpt_dir,
        optimizer=None,
        scheduler=None,
        tag=None,
        load_module_strict=True,
        load_optimizer_states=True,
        load_lr_scheduler_states=True,
    ):
        if isinstance(model, HFModelWrapper):
            model = model.model

        assert isinstance(model, deepspeed.DeepSpeedEngine)

        # Use context manager to handle local vs cloud paths
        with io.local_read_dir(ckpt_dir) as read_dir:
            load_path, states = model.load_checkpoint(
                read_dir,
                tag,
                load_module_strict=load_module_strict,
                load_optimizer_states=load_optimizer_states,
                load_lr_scheduler_states=load_lr_scheduler_states,  # DeepSpeed handles this automatically
            )

        if load_path is None:
            raise Exception(f"[deepspeed] failed to resume from checkpoint {ckpt_dir}")

        # Load RNG state for reproducibility (if present)
        if "rng" in states:
            self.load_rng_state(states["rng"])
            if self.is_rank_0():
                self.print(f"[rank-{self.get_rank()}]: Loaded RNG state from checkpoint")

        return load_path, states

    def save_hf_model(self, model: nn.Module, output_dir: str, tokenizer=None, **kwargs) -> None:
        """
        Multi-node safe: gather full FP32 state dict on rank 0 via ZeRO collectives,
        then write a single model.safetensors alongside config/tokenizer.
        """
        # Unwrap model and assert DS engine
        if isinstance(model, HFModelWrapper):
            engine = model.model
        else:
            engine = model
        assert isinstance(engine, deepspeed.DeepSpeedEngine), "Expected a DeepSpeedEngine"

        # Underlying HF model for config/tokenizer
        unwrapped_model = self._unwrap_model(engine)

        # Dist info
        is_dist = dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0
        stage3 = getattr(engine, "zero_optimization_stage", lambda: 0)() == 3

        # Barrier before collecting
        if is_dist:
            dist.barrier()

        # Collect full FP32 state dict on rank 0
        full_state_dict = {}
        for name, param in unwrapped_model.named_parameters():
            # Materialize full param on rank 0 only
            with deepspeed.zero.GatheredParameters([param], modifier_rank=0, enabled=stage3):
                if rank == 0:
                    full_state_dict[name] = param.detach().to(torch.float32).cpu()

        # Buffers (usually small; not ZeRO-sharded)
        if rank == 0:
            for name, buf in unwrapped_model.named_buffers():
                full_state_dict[name] = buf.detach().to(torch.float32).cpu()

            # Handle tied embeddings (keep only input embeddings)
            if getattr(unwrapped_model.config, "tie_word_embeddings", False) and "lm_head.weight" in full_state_dict:
                full_state_dict.pop("lm_head.weight", None)

            # Only rank 0 writes; use io.local_work_dir for local→remote sync
            with io.local_work_dir(output_dir) as work_dir:
                save_file(full_state_dict, os.path.join(work_dir, "model.safetensors"))
                unwrapped_model.config.save_pretrained(work_dir)
                if tokenizer is not None:
                    tokenizer.save_pretrained(work_dir)

        # Final barrier so others wait for upload to complete
        if is_dist:
            dist.barrier()

    def _set_bf16_config(self, ds_config):
        # torch_autocast.enabled should be set in the config file
        # Set DeepSpeed's bf16 config only if torch_autocast is disabled.
        if not ds_config["torch_autocast"]["enabled"]:
            ds_config["bf16"] = {"enabled": self.bf16}

    def get_ds_train_config(self):
        ds_config = OmegaConf.to_container(self.deepspeed_config)
        disable_trace_cache = ds_config.pop("disable_trace_cache", False)
        if disable_trace_cache:
            ds_config["zero_optimization"]["stage3_prefetch_bucket_size"] = 0
            ds_config["zero_optimization"]["stage3_max_live_parameters"] = 0
            ds_config["zero_optimization"]["stage3_max_reuse_distance"] = 0
        ds_config["steps_per_print"] = 100
        self._set_bf16_config(ds_config)

        # these need to be specified for deepspeed setup, but we manually handle
        # gradient accumulation in the training loop
        ds_config["train_micro_batch_size_per_gpu"] = self.micro_train_batch_size_per_gpu
        ds_config["gradient_accumulation_steps"] = 1

        return ds_config

    def get_ds_eval_config(self):
        ds_config = OmegaConf.to_container(self.deepspeed_config)
        ds_config["steps_per_print"] = 100
        self._set_bf16_config(ds_config)
        ds_config["train_micro_batch_size_per_gpu"] = self.micro_train_batch_size_per_gpu
        ds_config["gradient_accumulation_steps"] = 1

        return ds_config

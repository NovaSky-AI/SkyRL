# Utils ported from Verl
# https://github.com/volcengine/verl/blob/e1603dc97f3c20c58feed1f5be34acd5c72a830c/verl/utils/megatron/optimizer.py#L4
# The original copyright is reproduced below:

# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Union

from omegaconf import DictConfig
import torch
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer import get_megatron_optimizer as get_megatron_optimizer_native
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

from skyrl_train.config import OptimizerConfig as SkyRLOptimizerConfig


def init_megatron_optim_config(
    optim_config: Union[SkyRLOptimizerConfig, DictConfig], optimizer_config_kwargs: dict
) -> OptimizerConfig:
    optim_args = {
        "optimizer": getattr(optim_config, "optimizer", "adam"),
        "lr": getattr(optim_config, "lr", 1e-6),
        "min_lr": getattr(optim_config, "min_lr", 0.0),
        "clip_grad": getattr(optim_config, "max_grad_norm", 1.0),
        "weight_decay": getattr(optim_config, "weight_decay", 1e-2),
        "bf16": True,
        "params_dtype": torch.bfloat16,
        "use_distributed_optimizer": True,
    }
    optim_args.update(optimizer_config_kwargs)

    config = OptimizerConfig(**optim_args)
    return config


def get_megatron_optimizer(
    model,
    config: OptimizerConfig,
):
    # Base optimizer.
    return get_megatron_optimizer_native(
        config=config,
        model_chunks=model,
    )


def get_megatron_optimizer_param_scheduler(
    optimizer,
    config: Union[SkyRLOptimizerConfig, DictConfig],
    num_training_steps: int = 1e9,  # default to a large number for constant lr/wd
):
    """
    Get the optimizer parameter scheduler for Megatron.
    """
    # TODO: support other schedulers for Megatron
    if getattr(config, "scheduler", "constant_with_warmup") != "constant_with_warmup":
        raise ValueError("Only constant_with_warmup scheduler is supported for Megatron")

    lr_warmup_steps = config.num_warmup_steps
    if getattr(config, "lr_decay_steps", None) is None:
        lr_decay_steps = num_training_steps
    if getattr(config, "lr_warmup_steps_ratio", None) is not None and (
        getattr(config, "lr_warmup_steps", None) is None or getattr(config, "lr_warmup_steps", None) <= 0
    ):
        lr_warmup_steps = int(config.lr_warmup_steps_ratio * lr_decay_steps)

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=getattr(config, "lr_warmup_init", 0.0),
        max_lr=getattr(config, "lr", 1e-6),
        min_lr=getattr(config, "min_lr", 0.0),
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style="constant",
        start_wd=config.weight_decay,
        end_wd=config.weight_decay,
        wd_incr_steps=num_training_steps,
        wd_incr_style="constant",
        use_checkpoint_opt_param_scheduler=False,
        override_opt_param_scheduler=True,
        wsd_decay_steps=None,
        lr_wsd_decay_style="exponential",
    )

    return opt_param_scheduler


def get_megatron_last_lr(optimizer):
    """
    Get the last learning rate from the optimizer parameter scheduler.
    """
    return optimizer.param_groups[0]["lr"]

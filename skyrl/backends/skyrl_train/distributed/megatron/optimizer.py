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
from typing import Optional, Union

import torch
from megatron.core.optimizer import OptimizerConfig, get_standard_config_overrides
from megatron.core.optimizer import (
    get_megatron_optimizer as get_megatron_optimizer_native,
)
from megatron.core.optimizer.optimizer_config import ParamKey, ParamWithNamePredicate
from megatron.core.optimizer_param_scheduler import (
    OptimizerParamScheduler,
    ParamGroupOverride,
)
from omegaconf import DictConfig

from skyrl.backends.skyrl_train.distributed.megatron.optimizer_dtype import (
    coerce_optimizer_dtype_kwargs,
)
from skyrl.backends.skyrl_train.distributed.megatron.optimizer_offload_to_disk import (
    build_optimizer_offloaded_to_disk,
)
from skyrl.backends.skyrl_train.mtp.grad_clip import is_mtp_param_name
from skyrl.train.config import OptimizerConfig as SkyRLOptimizerConfig
from skyrl.train.config.config import MegatronOptimizerOffloadToDiskConfig


def init_megatron_optim_config(
    optim_config: Union[SkyRLOptimizerConfig, DictConfig], optimizer_config_kwargs: dict
) -> OptimizerConfig:
    # The Tinker server rewrites lr/betas/eps/weight_decay and the per-step clip before every
    # step (set_optimizer_hparams in the workers), so whatever this bakes in is only what runs
    # until the first AdamParams-bearing optim_step arrives.
    adam_betas = getattr(optim_config, "adam_betas", (0.9, 0.999))
    optim_args = {
        "optimizer": getattr(optim_config, "optimizer", "adam"),
        "lr": getattr(optim_config, "lr", 1e-6),
        "min_lr": getattr(optim_config, "min_lr", 0.0),
        "clip_grad": getattr(optim_config, "max_grad_norm", 1.0),
        "weight_decay": getattr(optim_config, "weight_decay", 1e-2),
        "adam_beta1": float(adam_betas[0]),
        "adam_beta2": float(adam_betas[1]),
        "bf16": True,
        "params_dtype": torch.bfloat16,
        "use_distributed_optimizer": True,
    }
    # YAML dtype overrides arrive as strings; Megatron expects torch.dtype.
    optim_args.update(coerce_optimizer_dtype_kwargs(optimizer_config_kwargs))

    config = OptimizerConfig(**optim_args)
    return config


def mtp_lr_config_overrides(config: OptimizerConfig, mtp_lr: float) -> dict:
    """Megatron's standard per-parameter overrides, plus an own learning rate for the MTP heads.

    The heads' group runs the policy's schedule between ``mtp_lr`` and a ``min_lr`` at the policy's
    ``min_lr / lr`` ratio. Matched by name (``is_mtp_param_name``), like the heads' separate grad clip.
    """
    overrides = get_standard_config_overrides(config)
    heads = ParamKey(
        with_name_predicate=ParamWithNamePredicate(name="mtp_heads", fn=lambda param, name: is_mtp_param_name(name))
    )
    min_lr = mtp_lr * config.min_lr / config.lr if config.lr else 0.0
    overrides[heads] = ParamGroupOverride(max_lr=mtp_lr, min_lr=min_lr)
    return overrides


def mtp_heads_lr(optimizer, mtp_lr: Optional[float]) -> float:
    """The MTP heads' current learning rate: their own groups' (found by the ``max_lr`` their override sets) when
    ``mtp_lr`` is set, else the policy's ``param_groups[0]``, which then holds them. Megatron aligns the param groups
    across ranks, so a rank without heads (a PP stage without the MTP layer) still has their groups, empty and
    scheduled. The heads can span several groups (weight decay, expert parallel), all on one schedule."""
    groups = optimizer.param_groups
    if mtp_lr is None:
        return groups[0]["lr"]
    heads = [g["lr"] for g in groups if g.get("max_lr") == mtp_lr]
    assert heads, f"no optimizer param group has the MTP heads' max_lr {mtp_lr}"
    return heads[0]


def get_megatron_optimizer(
    model,
    config: OptimizerConfig,
    offload_to_disk: Optional[MegatronOptimizerOffloadToDiskConfig] = None,
    mtp_lr: Optional[float] = None,
):
    """Build Megatron's optimizer; with ``offload_to_disk`` its state lives on node-local disk, and ``mtp_lr`` gives
    the MTP heads their own learning rate."""
    if offload_to_disk is not None:
        # The disk-offloaded optimizer builds Megatron's standard param groups, without the heads' own group.
        if mtp_lr is not None:
            raise ValueError("optimizer_offload_to_disk does not support trainer.mtp.lr")
        return build_optimizer_offloaded_to_disk(model, config, offload_to_disk)
    # Only an own MTP-head lr adds overrides; otherwise Megatron applies its standard ones itself.
    native_kwargs = {} if mtp_lr is None else {"config_overrides": mtp_lr_config_overrides(config, mtp_lr)}
    return get_megatron_optimizer_native(config=config, model_chunks=model, **native_kwargs)


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

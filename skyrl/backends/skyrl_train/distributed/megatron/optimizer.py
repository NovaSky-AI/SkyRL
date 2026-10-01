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
import logging
from typing import Union

import torch
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer import (
    get_megatron_optimizer as get_megatron_optimizer_native,
)
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from omegaconf import DictConfig

from skyrl.backends.skyrl_train.distributed.megatron.optimizer_dtype import (
    coerce_optimizer_dtype_kwargs,
)
from skyrl.train.config import OptimizerConfig as SkyRLOptimizerConfig

logger = logging.getLogger(__name__)


def init_megatron_optim_config(
    optim_config: Union[SkyRLOptimizerConfig, DictConfig], optimizer_config_kwargs: dict
) -> OptimizerConfig:
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
    # Map the SkyRL scheduler name to a Megatron ``lr_decay_style``. Decaying schedules
    # (cosine/linear) decay ``lr`` -> ``min_lr`` over ``lr_decay_steps`` after the warmup.
    _SCHEDULER_TO_DECAY_STYLE = {
        "constant": "constant",
        "constant_with_warmup": "constant",
        "cosine": "cosine",
        "cosine_with_warmup": "cosine",
        "linear": "linear",
        "linear_with_warmup": "linear",
    }
    scheduler_name = getattr(config, "scheduler", "constant_with_warmup")
    if scheduler_name not in _SCHEDULER_TO_DECAY_STYLE:
        raise ValueError(
            f"Unsupported scheduler {scheduler_name!r} for Megatron; expected one of "
            f"{sorted(_SCHEDULER_TO_DECAY_STYLE)}."
        )
    lr_decay_style = _SCHEDULER_TO_DECAY_STYLE[scheduler_name]

    # Decay over the whole run unless an explicit ``lr_decay_steps`` is given. (Assign
    # unconditionally -- the previous version left this undefined when lr_decay_steps was set.)
    lr_decay_steps = getattr(config, "lr_decay_steps", None)
    if lr_decay_steps is None:
        lr_decay_steps = num_training_steps
    # A decaying schedule needs a finite, real horizon. When num_steps is derived from epochs
    # the trainer passes the true total; if it ever falls back to the ~1e9 default, cosine would
    # be ~flat -- warn rather than silently no-op.
    if lr_decay_style != "constant" and lr_decay_steps >= 1e9:
        logger.warning(
            f"scheduler={scheduler_name!r} but lr_decay_steps={lr_decay_steps} looks like the "
            "unbounded default; the LR will barely decay. Ensure num_steps (or lr_decay_steps) "
            "is set to the real training horizon."
        )

    lr_warmup_steps = config.num_warmup_steps
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
        lr_decay_style=lr_decay_style,
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

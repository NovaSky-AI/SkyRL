"""
uv run --isolated --extra dev pytest tests/gpu/gpu_ci/distributed/test_fsdp_strategy.py
"""

from skyrl_train.model_wrapper import HFModelWrapper
import os
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.distributed_c10d import init_process_group
from skyrl_train.distributed.fsdp_strategy import FSDPStrategy
from skyrl_train.config.utils import get_default_config
from skyrl_train.utils.trainer_utils import get_rope_scaling_config, get_rope_theta_config
from skyrl_train.utils.utils import get_free_port

MODEL_NAME = "llamafactory/tiny-random-Llama-3"


def test_fsdp1_wrap_policy():
    cfg = get_default_config()
    cfg.trainer.policy.model.path = MODEL_NAME
    cfg.trainer.strategy = "fsdp"
    os.environ["RANK"] = "0"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(get_free_port())
    init_process_group(backend="nccl", rank=0, world_size=1)
    strategy = FSDPStrategy(
        fsdp_config=cfg.trainer.policy.fsdp_config,
        optimizer_config=None,
        model_config=cfg.trainer.policy.model,
        fsdp_strategy=cfg.trainer.strategy,
        seed=cfg.trainer.seed,
        micro_train_batch_size_per_gpu=cfg.trainer.micro_train_batch_size_per_gpu,
        num_training_steps=None,
    )
    strategy.setup_distributed()

    wrapped_model = HFModelWrapper(
        MODEL_NAME,
        use_flash_attention_2=cfg.trainer.flash_attn,
        # NOTE (sumanthrh): Model initialization should always be in fp32
        # during training
        bf16=False,
        lora_rank=cfg.trainer.policy.model.lora.rank,
        lora_alpha=cfg.trainer.policy.model.lora.alpha,
        lora_dropout=cfg.trainer.policy.model.lora.dropout,
        target_modules=cfg.trainer.policy.model.lora.target_modules,
        exclude_modules=cfg.trainer.policy.model.lora.exclude_modules,
        sequence_parallel_size=cfg.trainer.policy.sequence_parallel_size,
        use_sample_packing=cfg.trainer.use_sample_packing,
        use_torch_compile=cfg.trainer.policy.use_torch_compile,
        rope_scaling=get_rope_scaling_config(cfg.trainer),
        rope_theta=get_rope_theta_config(cfg.trainer),
    )

    model, _, _ = strategy.prepare(
        (wrapped_model, None, None),
    )
    num_wrapped_modules = 0
    for name, module in model.model.named_modules():
        if isinstance(module, FSDP):
            num_wrapped_modules += 1
    # need at least 1 inner layer to be wrapped
    assert num_wrapped_modules > 1

    dist.destroy_process_group()

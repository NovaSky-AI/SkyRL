"""Megatron worker for offline PiSSA initialization."""

import contextlib
import math

import megatron.core.parallel_state as mpu
import ray
import torch
from loguru import logger
from megatron.bridge.peft.lora_layers import LoRALinear
from megatron.bridge.peft.utils import GroupedExpertLinearAdapter, ParallelLinearAdapter
from megatron.core.distributed.distributed_data_parallel_config import (
    DistributedDataParallelConfig,
)

from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
    MegatronPolicyWorkerBase,
)
from skyrl.train.config.config import get_config_as_dict


def _pissa_factors(weight: torch.Tensor, rank: int, scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    max_rank = min(weight.shape)
    if rank > max_rank:
        raise ValueError(f"PiSSA rank {rank} exceeds maximum rank {max_rank} for weight shape {tuple(weight.shape)}")

    u, s, vh = torch.linalg.svd(weight.float(), full_matrices=False)
    sqrt_s = s[:rank].sqrt()
    linear_out = u[:, :rank] * sqrt_s.unsqueeze(0)
    linear_in = sqrt_s.unsqueeze(1) * vh[:rank, :]
    factor = math.sqrt(scale)
    linear_out /= factor
    linear_in /= factor
    return linear_in, linear_out


def pissa_decompose(weight: torch.Tensor, rank: int, scale: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a weight into exact rank-``rank`` principal factors and a residual."""
    linear_in, linear_out = _pissa_factors(weight, rank, scale)
    residual = weight.float() - scale * (linear_out @ linear_in)
    return linear_in, linear_out, residual


def validate_pissa_config(rank: int, lora_type: str) -> None:
    """Validate the LoRA settings supported by the PiSSA producer."""
    if rank <= 0:
        raise ValueError("PiSSA requires a positive LoRA rank")
    if lora_type != "lora":
        raise ValueError("PiSSA supports only lora_type='lora'")


@contextlib.contextmanager
def zeroed_adapters(model_chunks):
    """Temporarily zero LoRA B matrices for residual-base export."""
    chunks = model_chunks if isinstance(model_chunks, (list, tuple)) else [model_chunks]
    saved = []
    with torch.no_grad():
        for chunk in chunks:
            for module in chunk.modules():
                if isinstance(module, LoRALinear) and isinstance(
                    module.adapter, (ParallelLinearAdapter, GroupedExpertLinearAdapter)
                ):
                    weight = module.adapter.linear_out.weight
                    saved.append((weight, weight.detach().clone()))
                    weight.zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            for weight, original in saved:
                weight.copy_(original)


def pissa_pre_wrap_hook(model):
    """Apply PiSSA after the LoRA transform."""
    apply_pissa_init(model)
    return model


def _all_gather(local: torch.Tensor, dim: int, tp_size: int, tp_group) -> torch.Tensor:
    """Gather an evenly-sharded tensor across the TP group and concat along ``dim``."""
    local = local.contiguous()
    parts = [torch.empty_like(local) for _ in range(tp_size)]
    torch.distributed.all_gather(parts, local, group=tp_group)
    return torch.cat(parts, dim=dim)


def _shard(full: torch.Tensor, dim: int, tp_rank: int, tp_size: int) -> torch.Tensor:
    """Return this TP rank's contiguous shard of ``full`` along ``dim``."""
    if tp_size == 1:
        return full
    size = full.shape[dim]
    if size % tp_size != 0:
        raise ValueError(f"PiSSA: dim {dim} size {size} not divisible by tp_size {tp_size}")
    chunk = size // tp_size
    return full.narrow(dim, tp_rank * chunk, chunk).contiguous()


def _synchronized_pissa_decompose(
    weight: torch.Tensor, rank: int, scale: float, group
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute PiSSA factors once and broadcast them within a parallel group."""
    max_rank = min(weight.shape)
    if rank > max_rank:
        raise ValueError(f"PiSSA rank {rank} exceeds maximum rank {max_rank} for weight shape {tuple(weight.shape)}")

    if torch.distributed.get_rank(group) == 0:
        linear_in, linear_out = _pissa_factors(weight, rank, scale)
    else:
        linear_in = torch.empty((rank, weight.shape[1]), dtype=torch.float32, device=weight.device)
        linear_out = torch.empty((weight.shape[0], rank), dtype=torch.float32, device=weight.device)

    linear_in = linear_in.contiguous()
    linear_out = linear_out.contiguous()
    torch.distributed.broadcast(linear_in, group=group, group_src=0)
    torch.distributed.broadcast(linear_out, group=group, group_src=0)
    residual = weight.float() - scale * (linear_out @ linear_in)
    return linear_in, linear_out, residual


@torch.no_grad()
def _init_one_adapter(base_linear, adapter, tp_size: int, tp_rank: int, tp_group) -> None:
    base_weight = base_linear.weight
    if base_weight.is_meta:
        raise RuntimeError(
            "PiSSA: base weight is on the meta device at adapter-init time; pretrained weights "
            "must be loaded first. Disable meta-device init (init_model_with_meta_device) for PiSSA."
        )

    input_is_parallel = adapter.input_is_parallel
    rank = adapter.dim
    scale = adapter.alpha / adapter.dim

    base_shard_dim = 1 if input_is_parallel else 0
    if tp_size == 1:
        full_w = base_weight.detach().float()
        linear_in_full, linear_out_full, residual_full = pissa_decompose(full_w, rank, scale)
    else:
        full_w = _all_gather(base_weight.detach().float(), base_shard_dim, tp_size, tp_group)
        linear_in_full, linear_out_full, residual_full = _synchronized_pissa_decompose(full_w, rank, scale, tp_group)

    lin_in_shard_dim = 1 if input_is_parallel else 0
    base_dtype = base_weight.dtype
    lora_dtype = adapter.linear_in.weight.dtype

    base_weight.copy_(_shard(residual_full, base_shard_dim, tp_rank, tp_size).to(base_dtype))
    adapter.linear_out.weight.copy_(_shard(linear_out_full, 0, tp_rank, tp_size).to(lora_dtype))
    adapter.linear_in.weight.copy_(_shard(linear_in_full, lin_in_shard_dim, tp_rank, tp_size).to(lora_dtype))


def _grouped_base_weights(base_linear, num_experts: int) -> list[torch.Tensor]:
    """Return writable views of each local expert weight."""
    if getattr(base_linear, "single_grouped_weight", False):
        weight = base_linear.weight
        if weight.shape[0] == num_experts:
            return list(weight.unbind(0))
        if weight.shape[0] % num_experts == 0:
            return list(weight.chunk(num_experts, dim=0))
        raise ValueError(f"PiSSA cannot split grouped weight shape {tuple(weight.shape)} into {num_experts} experts")
    return [getattr(base_linear, f"weight{expert_idx}") for expert_idx in range(num_experts)]


@torch.no_grad()
def _init_grouped_adapter(base_linear, adapter) -> None:
    etp_group = adapter.expert_tp_group
    etp_size = torch.distributed.get_world_size(etp_group)
    etp_rank = torch.distributed.get_rank(etp_group)
    base_weights = _grouped_base_weights(base_linear, adapter.num_local_experts)
    base_shard_dim = 1 if adapter.input_is_parallel else 0
    linear_in_shard_dim = 1 if adapter.input_is_parallel else 0
    scale = adapter.alpha / adapter.dim

    for expert_idx, base_weight in enumerate(base_weights):
        if base_weight.is_meta:
            raise RuntimeError(
                "PiSSA: base weight is on the meta device at adapter-init time; pretrained weights "
                "must be loaded first. Disable meta-device init (init_model_with_meta_device) for PiSSA."
            )
        if etp_size == 1:
            full_weight = base_weight.detach().float()
            linear_in, linear_out, residual = pissa_decompose(full_weight, adapter.dim, scale)
        else:
            full_weight = _all_gather(base_weight.detach().float(), base_shard_dim, etp_size, etp_group)
            linear_in, linear_out, residual = _synchronized_pissa_decompose(full_weight, adapter.dim, scale, etp_group)

        base_weight.copy_(_shard(residual, base_shard_dim, etp_rank, etp_size).to(base_weight.dtype))
        adapter.linear_in.weight[expert_idx].copy_(
            _shard(linear_in, linear_in_shard_dim, etp_rank, etp_size).to(adapter.linear_in.weight.dtype)
        )
        adapter.linear_out.weight[expert_idx].copy_(
            _shard(linear_out, 0, etp_rank, etp_size).to(adapter.linear_out.weight.dtype)
        )


@torch.no_grad()
def apply_pissa_init(model_chunks) -> None:
    """Overwrite supported Megatron LoRA adapters with PiSSA factors."""
    tp_size = mpu.get_tensor_model_parallel_world_size()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    tp_group = mpu.get_tensor_model_parallel_group()

    chunks = model_chunks if isinstance(model_chunks, (list, tuple)) else [model_chunks]
    adapters = []
    grouped_adapters = []
    unsupported = []
    for chunk in chunks:
        for module in chunk.modules():
            if not isinstance(module, LoRALinear):
                continue
            if isinstance(module.adapter, ParallelLinearAdapter):
                adapters.append((module.to_wrap, module.adapter))
            elif isinstance(module.adapter, GroupedExpertLinearAdapter):
                grouped_adapters.append((module.to_wrap, module.adapter))
            else:
                unsupported.append(type(module.adapter).__name__)

    if unsupported:
        adapter_types = ", ".join(sorted(set(unsupported)))
        raise ValueError(f"PiSSA does not support adapter types: {adapter_types}")
    if not adapters and not grouped_adapters:
        raise ValueError("PiSSA found no supported LoRA adapters")

    for base_linear, adapter in adapters:
        _init_one_adapter(base_linear, adapter, tp_size, tp_rank, tp_group)
    for base_linear, adapter in grouped_adapters:
        _init_grouped_adapter(base_linear, adapter)

    logger.info(f"PiSSA: initialized {len(adapters) + len(grouped_adapters)} LoRA adapter(s)")


class PiSSAInitWorkerBase(MegatronPolicyWorkerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        validate_pissa_config(
            self.cfg.policy.model.lora.rank,
            self.cfg.policy.megatron_config.lora_config.lora_type,
        )

    def make_megatron_module(
        self,
        wrap_with_ddp=True,
        ddp_config=None,
        lora_config=None,
        lora_type="lora",
        bf16=True,
    ):
        self.configure_lora(lora_config, lora_type)

        def lora_pre_wrap_hook(model):
            lora_model = self.lora_cls(model, training=True)
            self.lora_cls.set_params_to_save(lora_model)
            return lora_model

        self.provider.register_pre_wrap_hook(lora_pre_wrap_hook)
        self.provider.register_pre_wrap_hook(pissa_pre_wrap_hook)

        resolved_ddp_config = DistributedDataParallelConfig()
        if wrap_with_ddp:
            resolved_ddp_config.use_distributed_optimizer = True
        if ddp_config is not None:
            for key, value in get_config_as_dict(ddp_config).items():
                setattr(resolved_ddp_config, key, value)
        return self.provider.provide_distributed_model(
            ddp_config=resolved_ddp_config,
            wrap_with_ddp=wrap_with_ddp,
            bf16=bf16,
        )

    def save_pissa_residual(self, export_dir: str, tokenizer) -> None:
        with zeroed_adapters(self.model.actor_module):
            self.strategy.save_hf_model(
                self.bridge,
                self.model,
                export_dir,
                tokenizer=tokenizer,
            )


PiSSAInitWorker = ray.remote(num_gpus=1)(PiSSAInitWorkerBase)

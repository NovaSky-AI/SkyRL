"""Keep optimizer master parameters consistent with loaded model weights."""

import torch


def _dequantize_model_param(optimizer, param):
    """Use the same quantized-parameter handling as Megatron's native reload."""
    if optimizer._is_distopt_quantized_param(param):
        if optimizer._is_grouped_quantized_tensor(param):
            return param.float()
        from megatron.core.fp8_utils import dequantize_fp8_tensor

        return dequantize_fp8_tensor(param)
    return param


@torch.no_grad()
def reload_optimizer_model_params(optimizer) -> None:
    """Refresh masters without changing Adam history or parameter-group settings.

    Megatron's native reload handles ordinary mixed-precision masters. With CPU
    offload, also refresh the distributed master shards and their host copies.
    For precision-aware FusedAdam, leave lazy state alone and reset BF16 remainder
    bits to zero. Pass a copy when setting full masters because FusedAdam can
    rescale its input in place.
    """
    optimizer.reload_model_params()
    pending = [optimizer]
    while pending:
        sub_optimizer = pending.pop()
        if hasattr(sub_optimizer, "chained_optimizers"):
            pending.extend(sub_optimizer.chained_optimizers)
            continue
        config = sub_optimizer.config
        inner_optimizer = sub_optimizer.optimizer
        if inner_optimizer is None:
            continue
        precision_aware = getattr(config, "use_precision_aware_optimizer_no_fp8_or_ds_fp8", False)
        if getattr(config, "optimizer_cpu_offload", False):
            if not precision_aware and config.use_distributed_optimizer:
                for model_group, master_group in zip(
                    sub_optimizer.model_float16_groups, sub_optimizer.shard_fp32_from_float16_groups
                ):
                    for model_param, master_param in zip(model_group, master_group):
                        param_range = sub_optimizer._get_model_param_range_map(model_param)["param"]
                        model_param = _dequantize_model_param(sub_optimizer, model_param)
                        shard = model_param.view(-1)[param_range.start : param_range.end]
                        master_param.copy_(shard)
            for param, inner_param in inner_optimizer.param_to_inner_param.items():
                inner_param.copy_(param)
            continue
        if not precision_aware:
            continue
        for group in inner_optimizer.param_groups:
            for param in group["params"]:
                if "master_param" not in inner_optimizer.state.get(param, {}):
                    continue
                if inner_optimizer.store_param_remainders and param.dtype == torch.bfloat16:
                    master = torch.zeros_like(param, dtype=torch.int16)
                else:
                    master = _dequantize_model_param(sub_optimizer, param).detach().to(dtype=torch.float32, copy=True)
                inner_optimizer.set_scaled_state(param, "master_param", master)

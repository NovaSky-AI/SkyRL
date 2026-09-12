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

    Megatron's public reload handles ordinary mixed-precision optimizers and
    ChainedOptimizer. The pinned implementation needs additional copies for
    HybridDeviceOptimizer and Transformer Engine precision-aware masters.
    """
    optimizer.reload_model_params()
    for sub_optimizer in getattr(optimizer, "chained_optimizers", [optimizer]):
        config = sub_optimizer.config
        inner_optimizer = sub_optimizer.optimizer
        if inner_optimizer is None:
            # Some pipeline/expert ranks have no trainable optimizer shards.
            continue
        precision_aware = getattr(config, "use_precision_aware_optimizer_no_fp8_or_ds_fp8", False)
        if getattr(config, "optimizer_cpu_offload", False):
            if not precision_aware and config.use_distributed_optimizer:
                # Native reload returns through HybridDeviceOptimizer before
                # updating DistributedOptimizer's separate FP32 shard groups.
                for model_group, master_group in zip(
                    sub_optimizer.model_float16_groups, sub_optimizer.shard_fp32_from_float16_groups
                ):
                    for model_param, master_param in zip(model_group, master_group):
                        param_range = sub_optimizer._get_model_param_range_map(model_param)["param"]
                        model_param = _dequantize_model_param(sub_optimizer, model_param)
                        shard = model_param.view(-1)[param_range.start : param_range.end]
                        master_param.copy_(shard)
            # Unlike param_to_fp32_param, this mapping also includes CPU
            # mirrors of native FP32 parameters. Adam history remains intact.
            for param, inner_param in inner_optimizer.param_to_inner_param.items():
                inner_param.copy_(param)
            continue
        if not precision_aware:
            continue
        for group in inner_optimizer.param_groups:
            for param in group["params"]:
                if "master_param" not in inner_optimizer.state.get(param, {}):
                    # Uninitialized FusedAdam state is seeded from the current
                    # model parameters on its first step.
                    continue
                if inner_optimizer.store_param_remainders and param.dtype == torch.bfloat16:
                    # A loaded BF16 value has no additional FP32 remainder bits.
                    master = torch.zeros_like(param, dtype=torch.int16)
                else:
                    # FusedAdam may rescale this input in-place when its master
                    # storage is FP16. Do not alias a native FP32 model shard.
                    master = _dequantize_model_param(sub_optimizer, param).detach().to(dtype=torch.float32, copy=True)
                inner_optimizer.set_scaled_state(param, "master_param", master)

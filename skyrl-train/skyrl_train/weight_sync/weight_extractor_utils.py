"""Utility functions for weight extraction."""

from collections import defaultdict
from typing import Dict, List, Callable, Iterator, Any
import torch

from skyrl_train.weight_sync import WeightChunk


def yield_module_grouped_chunks(
    params: Dict[str, Any],
    dtype: torch.dtype,
    gather_tensor_fn: Callable[[Any], torch.Tensor],
    get_shape_fn: Callable[[str, Any, torch.Tensor], List[int]],
    batch_size_threshold_gb: float = 0.0,
) -> Iterator[WeightChunk]:
    """Yield WeightChunk objects grouped by module.

    This helper function eliminates duplication between different weight extractors
    that need to group parameters by module (e.g., for FlashRL QKV fusion).

    Groups parameters by their parent module by removing the last two components
    from the parameter name. For example:
    "model.layers.0.self_attn.q_proj.weight" -> "model.layers.0.self_attn"

    Args:
        params: Dictionary mapping parameter names to parameter objects
        dtype: Target dtype for inference
        gather_tensor_fn: Backend-specific function to gather sharded tensors into full tensors
        get_shape_fn: Function to extract shape from param_name, param, and prepared tensor
        batch_size_threshold_gb: If > 0, batch complete modules together until threshold is reached

    Yields:
        WeightChunk objects containing all parameters for each module (or batched modules if threshold set)
    """
    # Group parameters by module for FlashRL
    # NOTE (sumanthrh): We sync weights module by module. Ex: weights for self attn together, weights for mlp together
    # For FlashRL integration, we allocate new storage for each param. Since q, k and v layer weights are fused internally by vllm,
    # we need to pass the weights for all of these together.
    # Overall, this doesn't hurt perf even in the general case
    module_to_params: Dict[str, List[str]] = defaultdict(list)
    for param_name in params.keys():
        # Extract module name (e.g., "model.layers.0.self_attn" from "model.layers.0.self_attn.q_proj.weight")
        # TODO (sumanthrh): When would this fail? Works for many AutoModelForCausalLM models for now
        module_name = ".".join(param_name.split(".")[:-2])
        module_to_params[module_name].append(param_name)

    # Accumulate complete modules until threshold reached
    batch_tensors = []
    batch_names = []
    batch_shapes = []
    batch_dtypes = []
    current_size = 0
    threshold_bytes = batch_size_threshold_gb * 1024**3

    for module_name, param_names in module_to_params.items():
        module_tensors = []
        module_names = []
        module_shapes = []
        module_dtypes = []
        module_size = 0

        # Prepare all tensors for this module
        # TODO: Allow gather_tensor_fn to accept a list of params for batched gathering.
        # This would be more efficient for DeepSpeed ZeRO-3 where GatheredParameters
        # can gather multiple params in a single all-gather collective.
        for param_name in param_names:
            param = params[param_name]
            tensor = gather_tensor_fn(param)
            tensor = tensor.to(dtype).detach().contiguous()
            shape = get_shape_fn(param_name, param, tensor)
            module_tensors.append(tensor)
            module_names.append(param_name)
            module_shapes.append(shape)
            module_dtypes.append(str(dtype))
            module_size += tensor.nbytes

        # Check if adding this module would exceed threshold
        if current_size > 0 and current_size + module_size > threshold_bytes:
            # Yield current batch before adding this module
            yield WeightChunk(
                names=batch_names,
                dtypes=batch_dtypes,
                shapes=batch_shapes,
                tensors=batch_tensors,
            )
            # Start new batch
            batch_tensors = []
            batch_names = []
            batch_shapes = []
            batch_dtypes = []
            current_size = 0

        # Add module to current batch
        batch_tensors.extend(module_tensors)
        batch_names.extend(module_names)
        batch_shapes.extend(module_shapes)
        batch_dtypes.extend(module_dtypes)
        current_size += module_size

    # Yield final batch if non-empty
    if batch_tensors:
        yield WeightChunk(
            names=batch_names,
            dtypes=batch_dtypes,
            shapes=batch_shapes,
            tensors=batch_tensors,
        )

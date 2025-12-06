"""Utility functions for weight extraction."""

from typing import Dict, List, Callable, Iterator, Any
import torch

from skyrl_train.weight_sync import WeightChunk


def yield_module_grouped_chunks(
    params: Dict[str, Any],
    dtype: torch.dtype,
    prepare_tensor_fn: Callable[[Any, torch.dtype], torch.Tensor],
    get_shape_fn: Callable[[str, Any, torch.Tensor], List[int]],
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
        prepare_tensor_fn: Function to prepare tensor (gather, convert dtype, make contiguous)
        get_shape_fn: Function to extract shape from param_name, param, and prepared tensor

    Yields:
        WeightChunk objects containing all parameters for each module
    """
    # Group parameters by module for FlashRL
    # NOTE (sumanthrh): We sync weights module by module. Ex: weights for self attn together, weights for mlp together
    # For FlashRL integration, we allocate new storage for each param. Since q, k and v layer weights are fused internally by vllm,
    # we need to pass the weights for all of these together.
    # Overall, this doesn't hurt perf even in the general case
    module_to_params: Dict[str, List[str]] = {}
    for param_name in params.keys():
        # Extract module name (e.g., "model.layers.0.self_attn" from "model.layers.0.self_attn.q_proj.weight")
        # TODO (sumanthrh): When would this fail? Works for many AutoModelForCausalLM models for now
        module_name = ".".join(param_name.split(".")[:-2])
        if module_name not in module_to_params:
            module_to_params[module_name] = []
        module_to_params[module_name].append(param_name)

    # Yield chunks grouped by module
    for module_name, param_names in module_to_params.items():
        tensors = []
        names = []
        shapes = []
        dtypes_list = []

        for param_name in param_names:
            param = params[param_name]
            tensor = prepare_tensor_fn(param, dtype)
            shape = get_shape_fn(param_name, param, tensor)
            tensors.append(tensor)
            names.append(param_name)
            shapes.append(shape)
            dtypes_list.append(str(dtype))

        yield WeightChunk(
            names=names,
            dtypes=dtypes_list,
            shapes=shapes,
            tensors=tensors,
            module_name=module_name,
        )

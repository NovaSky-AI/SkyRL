from typing import Dict, Any, Optional, Union
import random
import hashlib
from omegaconf import DictConfig, ListConfig
from skyrl_train.inference_engines.inference_engine_client_http_endpoint import ErrorResponse, ErrorInfo
from typing import List
from http import HTTPStatus
from typing import Tuple
import ray
from ray.util.placement_group import PlacementGroupSchedulingStrategy


def get_vllm_sampling_params(sampling_params: DictConfig) -> Dict[str, Any]:
    stop_val = sampling_params.get("stop", None)
    vllm_sampling_params = {
        "min_tokens": 1,
        "skip_special_tokens": True,
        "include_stop_str_in_output": True,
        "max_tokens": sampling_params.max_generate_length,
        "temperature": sampling_params.temperature,
        "top_p": sampling_params.top_p,
        "top_k": sampling_params.top_k,
        "min_p": sampling_params.min_p,
        "logprobs": sampling_params.logprobs,
        "stop": list(stop_val) if stop_val is not None else None,
    }
    exclude_keys = ["max_generate_length"]
    for key, value in sampling_params.items():
        if key not in vllm_sampling_params and key not in exclude_keys:
            # Convert OmegaConf ListConfig to regular list if needed
            if isinstance(value, ListConfig):
                value = list(value)
            vllm_sampling_params[key] = value
    return vllm_sampling_params


def get_sglang_sampling_params(sampling_params: DictConfig) -> Dict[str, Any]:
    # `min_tokens` in vllm is equivalent to `min_new_tokens` in sglang. However `min_new_tokens` and
    # `stop` are not supported when `skip_tokenizer_init` is True, which we need for token-in-token-out.
    # See this issue for more: https://github.com/sgl-project/sglang/issues/9039#issuecomment-3218331087
    sglang_sampling_params = {
        "skip_special_tokens": True,
        "no_stop_trim": True,  # equivalent to include_stop_str_in_output=True
        "max_new_tokens": sampling_params.max_generate_length,
        "temperature": sampling_params.temperature,
        "top_p": sampling_params.top_p,
        "top_k": sampling_params.top_k,
        "min_p": sampling_params.min_p,
    }
    # logprobs not supported with sglang for now
    exclude_keys = ["max_generate_length", "logprobs"]
    for key, value in sampling_params.items():
        if key not in sglang_sampling_params and key not in exclude_keys:
            # Convert OmegaConf ListConfig to regular list if needed
            if isinstance(value, ListConfig):
                value = list(value)
            sglang_sampling_params[key] = value
    return sglang_sampling_params


def get_sampling_params_for_backend(backend: str, sampling_params: DictConfig) -> Dict[str, Any]:
    if backend == "vllm":
        return get_vllm_sampling_params(sampling_params)
    elif backend == "sglang":
        return get_sglang_sampling_params(sampling_params)
    else:
        raise ValueError(f"Unsupported generation backend: {backend}")


def hash_with_sha256(x: Union[int, str]) -> int:
    return int.from_bytes(hashlib.sha256(str(x).encode()).digest(), "big")


def route_prompts_to_engines(
    num_prompts: int, num_inference_engines: int, session_ids: Optional[Union[list[int], list[str]]]
) -> dict[int, list[int]]:
    """
    Given the number of prompts, number of inference engines, and the session_id, return a mapping
    from engine index to the list of prompt IDs the engine will process.

    Args:
    - num_prompts: int - The number of prompts.
    - num_inference_engines: int - The number of inference engines.
    - session_ids: Optional[Union[list[int], list[str]]] - The session IDs.

    Required:
    - num_prompts > 0
    - num_inference_engines > 0
    - session_ids is a list of integers or strings if provided
    - len(session_ids) == num_prompts if provided

    Returns:
    - dict[int, list[int]] - A mapping from engine index to the list of prompt IDs the engine will process.
    """
    # 0. Validation
    assert num_prompts > 0, "Number of prompts must be greater than 0"
    assert num_inference_engines > 0, "Number of inference engines must be greater than 0"
    if session_ids is not None:
        assert isinstance(session_ids, list) and all(
            isinstance(sid, (int, str)) for sid in session_ids
        ), "Session ID must be a list of integers or strings"
        assert len(session_ids) == num_prompts, "Session ID must have the same length as the number of prompts"

    # 1. session_id not provided, with a single prompt: route to a random engine for a naive load balancing.
    if session_ids is None and num_prompts == 1:
        engine_idx = random.randint(0, num_inference_engines - 1)
        return {engine_idx: [0]}

    # 2. session_id not provided, with a batched prompt: split evenly across engines.
    engine_idx_to_prompt_ids: dict[int, list[int]] = {}
    if session_ids is None:
        dp_item_size = (num_prompts + num_inference_engines - 1) // num_inference_engines
        for dp_rank in range(num_inference_engines):
            start_idx = dp_rank * dp_item_size
            end_idx = min((dp_rank + 1) * dp_item_size, num_prompts)
            prompt_ids = list(range(start_idx, end_idx))
            if len(prompt_ids) > 0:
                engine_idx_to_prompt_ids[dp_rank] = prompt_ids
        return engine_idx_to_prompt_ids

    # 3. session_id provided, we route by session_id
    for i, cur_sid in enumerate(session_ids):
        engine_idx = hash_with_sha256(str(cur_sid)) % num_inference_engines
        engine_idx_to_prompt_ids.setdefault(engine_idx, []).append(i)
    return engine_idx_to_prompt_ids


def postprocess_completion_request(
    prompt: Union[List[int], List[List[int]], List[str], str],
    session_id_value: Optional[Union[List[int], List[str], int, str]],
) -> tuple[Optional[Union[List[int], List[str], ErrorResponse]], Union[List[List[int]], List[str]]]:
    """
    Postprocess the session_id value and raise error if invalid.

    Returns a list of session_ids, or None if session_id_value is None, or ErrorResponse if invalid.
    Also returns the processed prompt, where if the prompt is a single request, we make it
    a singleton list of a single request. That is, List[int] becomes List[List[int]] of length 1,
    and str becomes List[str] of length 1.

    Postconditions:
    - If session_id_value is None, we return None.
    - If session_id_value and prompt do not match, we return ErrorResponse.
    - The returned session_id_list has the same length as the prompt.
    - The returned prompt is either List[List[int]], or List[str], whether batched or not.
    """

    def _is_list_of_ints(x):
        return isinstance(x, list) and all(isinstance(y, int) for y in x)

    # Determine if this is a single or batched request (a List[str] of length 1 is considered batched)
    is_single = isinstance(prompt, str) or _is_list_of_ints(prompt)
    if is_single:
        prompt = [prompt]

    if session_id_value is None:
        return None, prompt

    if isinstance(session_id_value, (int, str)):
        session_id_value = [session_id_value]

    if len(session_id_value) != len(prompt):
        return (
            ErrorResponse(
                error=ErrorInfo(
                    message=(
                        "For /completions request with a single prompt, request.session_id must "
                        f"be a single integer/string or a singleton list.\nFor batched requests, "
                        f"request.session_id must have the same length as request.prompt."
                        f"However, received (len(session_id_value): {len(session_id_value)}, len(prompt): {len(prompt)})."
                    ),
                    type=HTTPStatus.BAD_REQUEST.phrase,
                    code=HTTPStatus.BAD_REQUEST.value,
                ),
            ),
            prompt,
        )

    return session_id_value, prompt


def aggregate_completion_usage_info(
    results: List[Dict[str, Any]],
    backend: str,
) -> Dict[str, Any]:
    """
    Aggregate the completion usage info and return the final usage info, since our
    inference engine client breaks down a batched request into sub-requests and routes to engines,
    where each engine only returns its sub-request's usage info. When we return the final response,
    we need to aggregate the usage info.

    NOTE(Charlie): we don't explicitly import vllm here for ease of CPU test. Whether these fields
    are still compatible with newer vllm versions can be checked in our GPU tests, where we explicitly
    check the dictionary with `CompletionResponse.model_validate()`.
    """
    if backend == "vllm":
        # required fields
        usage_info = {
            "prompt_tokens": sum(result["usage"]["prompt_tokens"] for result in results),
            "total_tokens": sum(result["usage"]["total_tokens"] for result in results),
        }
        # optional fields
        if results[0]["usage"].get("completion_tokens") is not None:
            usage_info["completion_tokens"] = sum(result["usage"]["completion_tokens"] for result in results)
        if results[0]["usage"].get("prompt_tokens_details") is not None:
            if results[0]["usage"]["prompt_tokens_details"].get("cached_tokens") is not None:
                usage_info["prompt_tokens_details"] = {
                    "cached_tokens": sum(
                        result["usage"]["prompt_tokens_details"]["cached_tokens"] for result in results
                    )
                }
        return usage_info
    elif backend == "sglang":
        raise NotImplementedError("SGLang is not supported yet")
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def get_rendezvous_addr_port(placement_group, pg_index: int) -> Tuple[str, int]:
    """
    Minimal helper to get a rendezvous addr:port in `placement_group`'s bundle at index `pg_index`.
    """

    @ray.remote(num_cpus=0, num_gpus=0)
    def get_addr_port():
        from ray.experimental.collective.util import get_address_and_port

        return get_address_and_port()

    master_sched = PlacementGroupSchedulingStrategy(
        placement_group=placement_group,
        placement_group_capture_child_tasks=True,
        placement_group_bundle_index=pg_index,
    )
    # Get DP group rendezvous (addr, port) on the same node as index `pg_index`'s bundle.
    data_parallel_address, data_parallel_rpc_port = ray.get(
        get_addr_port.options(scheduling_strategy=master_sched).remote()
    )
    return data_parallel_address, data_parallel_rpc_port

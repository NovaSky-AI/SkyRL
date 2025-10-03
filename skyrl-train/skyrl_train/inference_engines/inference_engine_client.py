from skyrl_train.inference_engines.base import (
    InferenceEngineInterface,
    InferenceEngineInput,
    InferenceEngineOutput,
    NamedWeightsUpdateRequest,
)
from skyrl_train.inference_engines.inference_engine_client_http_endpoint import ErrorResponse, ErrorInfo
from transformers import PreTrainedTokenizerBase
import asyncio
from typing import List, Any, Optional, Dict, Union
from skyrl_train.inference_engines.utils import (
    route_prompts_to_engines,
    hash_with_sha256,
    postprocess_completion_request,
    aggregate_completion_usage_info,
)
from omegaconf import DictConfig
import threading
from loguru import logger
import random


class InferenceEngineClient(InferenceEngineInterface):
    """
    Client to talk to a set of InferenceEngines.

    Note that InferenceEngineClient sub-classes InferenceEngineInterface so it can be used as if talking to a single engine.
    """

    def __init__(
        self, engines: List[InferenceEngineInterface], tokenizer: PreTrainedTokenizerBase, full_config: DictConfig
    ):
        """
        Args:
            engines: List[InferenceEngineInterface] - The inference engines, remote or local.
            tokenizer: PreTrainedTokenizerBase - The tokenizer to use.
            full_config: DictConfig - See ppo_base_config.yaml
        """
        self.engines = engines
        self.tokenizer = tokenizer
        self.model_name = full_config.trainer.policy.model.path
        self.backend = full_config.generator.backend
        self.enable_http_endpoint = full_config.generator.enable_http_endpoint
        self.http_endpoint_host = full_config.generator.http_endpoint_host
        self.http_endpoint_port = full_config.generator.http_endpoint_port
        if self.enable_http_endpoint:
            self._spin_up_http_endpoint()

        logger.info(f"InferenceEngineClient initialized with {len(engines)} engines.")

    async def _run_on_all_engines(self, method_name: str, *args, **kwargs):
        """
        Call a method on all engines concurrently and gather the results.
        """
        assert len(self.engines) > 0, "No engines to call method on"

        awaitables = [getattr(engine, method_name)(*args, **kwargs) for engine in self.engines]
        return await asyncio.gather(*awaitables)

    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        # 0. Extract input
        prompts = input_batch.get("prompts")
        prompt_token_ids = input_batch.get("prompt_token_ids")
        session_ids = input_batch.get("session_ids")
        sampling_params = input_batch.get("sampling_params")

        if (prompts is None and prompt_token_ids is None) or (prompts is not None and prompt_token_ids is not None):
            raise ValueError("Either `prompts` or `prompt_token_ids` must be provided, but not both.")
        if prompt_token_ids is None:
            prompt_token_ids = self.tokenizer.apply_chat_template(
                prompts,
                add_generation_prompt=True,
                add_special_tokens=False,
                return_dict=True,
                tokenize=True,
            )["input_ids"]

        num_prompts = len(prompt_token_ids)
        num_inference_engines = len(self.engines)

        # 1. Route prompts to engines
        engine_idx_to_prompt_ids: dict[int, list[int]] = route_prompts_to_engines(
            num_prompts=num_prompts,
            num_inference_engines=num_inference_engines,
            session_ids=session_ids,
        )

        # 2. Generate responses concurrently
        tasks: list[asyncio.Task] = []
        indices_list: list[list[int]] = []  # the original prompt indices that each task works on
        for engine_idx, prompt_ids in engine_idx_to_prompt_ids.items():
            # index prompt_token_ids with prompt_ids
            cur_prompt_token_ids = [prompt_token_ids[i] for i in prompt_ids]
            engine_input = InferenceEngineInput(
                prompt_token_ids=cur_prompt_token_ids,
                sampling_params=sampling_params,
            )
            tasks.append(asyncio.create_task(self.engines[engine_idx].generate(engine_input)))
            indices_list.append(prompt_ids)

        results = await asyncio.gather(*tasks)

        # 3. Reconstruct output in original order
        n = len(prompt_token_ids)
        responses: list[str] = [""] * n
        stop_reasons: list[str] = [""] * n
        response_logprobs: List[Optional[List[float]]] = [None for _ in range(n)]
        response_ids: List[List[int]] = [[] for _ in range(n)]
        # a bit hacky for now
        add_resp_logprobs = False

        for indices, result in zip(indices_list, results):
            for local_idx, original_idx in enumerate(indices):
                responses[original_idx] = result["responses"][local_idx]
                stop_reasons[original_idx] = result["stop_reasons"][local_idx]
                response_ids[original_idx] = result["response_ids"][local_idx]
                if result.get("response_logprobs", None):
                    add_resp_logprobs = True
                    response_logprobs[original_idx] = result["response_logprobs"][local_idx]

        return InferenceEngineOutput(
            responses=responses,
            stop_reasons=stop_reasons,
            response_ids=response_ids,
            response_logprobs=response_logprobs if add_resp_logprobs else None,
        )

    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        session_id = request_payload["json"].pop("session_id", None)
        if session_id is None:
            # if session_id is not provided, we'll use a random engine
            engine_idx = random.randint(0, len(self.engines) - 1)
        else:
            assert isinstance(session_id, (str, int)), "Session ID must be an integer or string for `/chat/completions`"
            engine_idx = hash_with_sha256(str(session_id)) % len(self.engines)
        return await self.engines[engine_idx].chat_completion(request_payload)

    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handles an OpenAI /completions request.

        Since `request["prompt"]` can be `Union[list[int], list[list[int]], str, list[str]]`,
        (i.e. {batched, single} x {string, token IDs}), we need to route the request to engines
        differently, based on whether it's a single or batched request, and whether `request["session_id"]`
        is provided. This is similar to `generate()` method.

        For single, we do the same routing logic as `chat_completion()`. For batched, we route by
        `request["session_id"]` if present, and if not we split evenly across engines.

        Regardless, the order will be maintained, i.e. `output["choices"][i]` corresponds to `request["prompt"][i]`.
        """
        body = request_payload.get("json", {})

        # NOTE(Charlie): do not reuse headers here as the single request may become various new requests
        headers = {"Content-Type": "application/json"}

        # 1. Postprocess prompt, session_id, and validate request.
        prompt = body.get("prompt")
        session_id_value = body.pop("session_id", None)
        ret = postprocess_completion_request(prompt, session_id_value)
        session_id_list: Optional[Union[List[int], List[str], ErrorResponse]] = ret[0]
        prompt: Union[List[List[int]], List[str]] = ret[1]
        if isinstance(session_id_list, ErrorResponse):
            return session_id_list.model_dump()

        num_prompts = len(prompt)
        num_inference_engines = len(self.engines)
        assert num_prompts > 0, "Number of prompts must be greater than 0"

        # 1. Route prompts to engines
        engine_idx_to_prompt_ids: dict[int, list[int]] = route_prompts_to_engines(
            num_prompts=num_prompts,
            num_inference_engines=num_inference_engines,
            session_ids=session_id_list,
        )

        # 2. Generate responses concurrently
        tasks: list[asyncio.Task] = []
        indices_list: list[list[int]] = []  # the original prompt indices that each task works on
        for engine_idx, prompt_ids in engine_idx_to_prompt_ids.items():
            cur_prompt = [prompt[i] for i in prompt_ids]
            # reuse the exact same request except for the prompt
            cur_json = dict(body)
            cur_json["prompt"] = cur_prompt
            coro = self.engines[engine_idx].completion({"json": cur_json, "headers": headers})
            tasks.append(asyncio.create_task(coro))
            indices_list.append(prompt_ids)

        results = await asyncio.gather(*tasks)

        # 3. Check for errors.
        # results can be ErrorResponse or CompletionResponse. If one of the sub-requests fails, we
        # return an error response. That is, there is no partial success, following vLLM and SGLang's behavior.
        for result in results:
            if "error" in result or result.get("object", "") == "error":
                # former is vllm format, latter is sglang format
                error_details = result.get("error", result)  # resolves vllm/sglang format difference
                error_code = error_details["code"]
                error_type = error_details["type"]
                return ErrorResponse(
                    error=ErrorInfo(
                        message=f"In one of the engines that SkyRL manages, an error occurred: {error_details['message']}",
                        type=error_type,
                        code=error_code,
                    ),
                ).model_dump()

        # 4. Combine choices and preserve original order.
        # If there is only one result, we return it directly.
        if len(results) == 1:
            return results[0]

        # Use the first result as base response. There are some fields that cannot be shared
        # across sub-requests. For now it is just the usage field.
        final_response = dict(results[0])
        final_response["usage"] = aggregate_completion_usage_info(results, self.backend)

        # Aggregate choices. TODO(Charlie): improve logic when we need to support n > 1
        # vLLM sets index positions per sub-batch, so we reset indices to be 0..n-1 for the combined response.
        combined_choices: list[Dict[str, Any]] = [None] * num_prompts
        for indices, result in zip(indices_list, results):
            # indices are the original prompt indices that the task's response corresponds to
            for local_idx, original_idx in enumerate(indices):
                choice = result["choices"][local_idx]
                choice["index"] = original_idx  # overwrite index with the global position
                combined_choices[original_idx] = choice

        # sanity check that the index is correct
        for new_idx in range(len(combined_choices)):
            assert combined_choices[new_idx]["index"] == new_idx

        final_response["choices"] = combined_choices
        return final_response

    async def wake_up(self, *args: Any, **kwargs: Any):
        return await self._run_on_all_engines("wake_up", *args, **kwargs)

    async def sleep(self, *args: Any, **kwargs: Any):
        return await self._run_on_all_engines("sleep", *args, **kwargs)

    async def init_weight_update_communicator(
        self,
        master_addr,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend,
        override_existing: bool = False,
    ):
        tasks = []
        rank_offset_count = rank_offset

        for engine in self.engines:
            tasks.append(
                engine.init_weight_update_communicator(
                    master_addr=master_addr,
                    master_port=master_port,
                    rank_offset=rank_offset_count,
                    world_size=world_size,
                    group_name=group_name,
                    backend=backend,
                    override_existing=override_existing,
                )
            )
            rank_offset_count += engine.tp_size()
        await asyncio.gather(*tasks)

    async def update_named_weights(self, request: NamedWeightsUpdateRequest):
        return await self._run_on_all_engines("update_named_weights", request=request)

    async def reset_prefix_cache(self):
        return await self._run_on_all_engines("reset_prefix_cache")

    async def teardown(self):
        return await self._run_on_all_engines("teardown")

    def tp_size(self) -> int:
        raise NotImplementedError("InferenceEngineClient does not implement tp_size()")

    def dp_size(self) -> int:
        raise NotImplementedError("InferenceEngineClient does not implement dp_size()")

    # ----------------------------
    # HTTP endpoint related methods
    # ----------------------------

    def __del__(self):
        """
        Destructor to shut down the HTTP endpoint if it was started.
        """
        # TODO(Charlie): __del__ is not guaranteed to be called in general. Add to `teardown` method
        # when the `_handle_termination` flow is implemented. See `skyrl_train/workers/worker.py`
        # comments on `_handle_termination` for more details.
        if (
            self.enable_http_endpoint
            and hasattr(
                self, "_server_thread"
            )  # don't want to shut down the server when it is pickled as a ray method argument.
            and self._server_thread is not None
        ):
            try:
                from skyrl_train.inference_engines.inference_engine_client_http_endpoint import shutdown_server

                shutdown_server(
                    host=self.http_endpoint_host,
                    port=self.http_endpoint_port,
                    max_wait_seconds=10,
                )
                if hasattr(self, "_server_thread") and self._server_thread.is_alive():
                    self._server_thread.join(timeout=10)
            except Exception as e:
                logger.error(f"Error shutting down HTTP endpoint: {e}")

    def __getstate__(self):
        """
        Override to avoid pickling the server thread, which is not picklable.
        Needed when passing InferenceEngineClient as an argument to async_run_ray_method().
        """
        state = self.__dict__.copy()
        state["_server_thread"] = None
        return state

    def _spin_up_http_endpoint(self):
        from skyrl_train.inference_engines.inference_engine_client_http_endpoint import (
            serve,
            wait_for_server_ready,
        )

        self._server_thread = threading.Thread(
            target=serve,
            args=(self,),
            kwargs={
                "host": self.http_endpoint_host,
                "port": self.http_endpoint_port,
                "log_level": "warning",
            },
            daemon=True,
        )
        self._server_thread.start()
        wait_for_server_ready(
            host=self.http_endpoint_host,
            port=self.http_endpoint_port,
            max_wait_seconds=30,
        )
        logger.info(
            f"InferenceEngineClient HTTP endpoint started on {self.http_endpoint_host}:{self.http_endpoint_port}"
        )

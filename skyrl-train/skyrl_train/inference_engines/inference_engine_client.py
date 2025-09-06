from skyrl_train.inference_engines.base import (
    InferenceEngineInterface,
    InferenceEngineInput,
    InferenceEngineOutput,
    NamedWeightsUpdateRequest,
)
from transformers import PreTrainedTokenizerBase
import asyncio
from typing import List, Any, Optional, Dict
from omegaconf import DictConfig
import threading


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

        print(f"InferenceEngineClient initialized with {len(engines)} engines.")

    async def _run_on_all_engines(self, method_name: str, *args, **kwargs):
        """
        Call a method on all engines concurrently and gather the results.
        """
        assert len(self.engines) > 0, "No engines to call method on"

        awaitables = [getattr(engine, method_name)(*args, **kwargs) for engine in self.engines]
        return await asyncio.gather(*awaitables)

    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        prompts = input_batch.get("prompts")
        prompt_token_ids = input_batch.get("prompt_token_ids")
        trajectory_ids = input_batch.get("trajectory_ids")
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

        # TODO(tgriggs): If there are no traj ids, we'd still like to load balance instead of landing on a single engine.
        if trajectory_ids is not None:
            # Route based on trajectory_ids
            return await self._generate_with_trajectory_routing(prompt_token_ids, trajectory_ids, sampling_params)
        else:
            # Split evenly across engines
            return await self._generate_batched(prompt_token_ids, sampling_params)

    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        trajectory_id = request_payload["json"].pop("trajectory_id", 0)
        engine_idx = abs(hash(str(trajectory_id))) % len(self.engines)
        return await self.engines[engine_idx].chat_completion(request_payload)

    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        # TODO(Charlie): completion might be batched. Need different trajectory id
        # routing logic.
        body = request_payload.get("json", {})
        headers = request_payload.get("headers", {})
        prompt = body.get("prompt")

        def is_list_of_ints(x):
            return isinstance(x, list) and (len(x) == 0 or all(isinstance(y, int) for y in x))

        def is_list_of_list_of_ints(x):
            return (
                isinstance(x, list)
                and len(x) > 0
                and all(isinstance(y, list) and all(isinstance(z, int) for z in y) for y in x)
            )

        def is_list_of_str(x):
            return isinstance(x, list) and (len(x) == 0 or all(isinstance(y, str) for y in x))

        # Determine if this is a single or batched request
        is_single = isinstance(prompt, str) or is_list_of_ints(prompt)
        is_batched = is_list_of_str(prompt) or is_list_of_list_of_ints(prompt)

        trajectory_id_value = body.pop("trajectory_id", None)

        if is_single:
            # Single completion request, route by a single trajectory_id (if provided)
            if isinstance(trajectory_id_value, list) and len(trajectory_id_value) != 1:
                raise ValueError(
                    "For single /completions request, trajectory_id must be a single integer or a singleton list."
                )
            trajectory_id = (
                trajectory_id_value[0] if isinstance(trajectory_id_value, list) else (trajectory_id_value or 0)
            )
            engine_idx = abs(hash(str(trajectory_id))) % len(self.engines)
            # Forward as-is (body already has trajectory_id removed)
            return await self.engines[engine_idx].completion({"json": body, "headers": headers})

        if not is_batched:
            raise ValueError(
                "Invalid prompt type for /completions. Expected str, list[int], list[str] or list[list[int]]."
            )

        # Batched request
        num_items = len(prompt)

        if trajectory_id_value is not None:
            # If trajectory ids are specified, they must be provided for each prompt
            if not isinstance(trajectory_id_value, list) or len(trajectory_id_value) != num_items:
                raise ValueError(
                    "For batched /completions, trajectory_id must be a list with the same length as prompt."
                )

            # Group prompts by engine index
            engine_groups: dict[int, dict[str, list]] = {}
            for i, (cur_prompt, cur_tid) in enumerate(zip(prompt, trajectory_id_value)):
                engine_idx = abs(hash(str(cur_tid))) % len(self.engines)
                group = engine_groups.setdefault(engine_idx, {"prompts": [], "indices": []})
                group["prompts"].append(cur_prompt)
                group["indices"].append(i)

            # Dispatch batched requests per engine
            tasks: list[asyncio.Task] = []
            indices_list: list[list[int]] = []
            for engine_idx, group in engine_groups.items():
                sub_json = dict(body)
                sub_json["prompt"] = group["prompts"]
                coro = self.engines[engine_idx].completion({"json": sub_json, "headers": headers})
                tasks.append(asyncio.create_task(coro))
                indices_list.append(group["indices"])

            results = await asyncio.gather(*tasks)

        else:
            # No trajectory ids: split evenly across engines
            num_inference_engines = len(self.engines)
            dp_item_size = (num_items + num_inference_engines - 1) // num_inference_engines

            tasks = []
            indices_list: list[list[int]] = []
            for dp_rank in range(num_inference_engines):
                start_idx = dp_rank * dp_item_size
                end_idx = min((dp_rank + 1) * dp_item_size, num_items)
                sub_prompts = prompt[start_idx:end_idx]
                if not sub_prompts:
                    continue
                sub_json = dict(body)
                sub_json["prompt"] = sub_prompts
                coro = self.engines[dp_rank].completion({"json": sub_json, "headers": headers})
                tasks.append(asyncio.create_task(coro))
                indices_list.append(list(range(start_idx, end_idx)))

            results = await asyncio.gather(*tasks)

        # Combine choices preserving original order. vLLM sets index positions per sub-batch;
        # we will reset indices to be 0..n-1 for the combined response.
        combined_choices: list[Dict[str, Any]] = [None] * num_items  # type: ignore
        base_response = None
        for group_indices, result in zip(indices_list, results):
            if base_response is None:
                base_response = result
            for local_idx, original_idx in enumerate(group_indices):
                choice = dict(result["choices"][local_idx])
                # overwrite index with the global position
                choice["index"] = original_idx
                combined_choices[original_idx] = choice

        # Reindex sequentially 0..n-1 to comply with OpenAI-style responses
        # while preserving order of prompts
        for new_idx in range(len(combined_choices)):
            combined_choices[new_idx]["index"] = new_idx

        # Build final response using the first result as base
        if base_response is None:
            base_response = {
                "id": "",
                "object": "text_completion",
                "created": 0,
                "model": self.model_name,
                "choices": [],
            }
        final_response = dict(base_response)
        final_response["model"] = self.model_name
        final_response["choices"] = combined_choices
        return final_response

    async def _generate_with_trajectory_routing(
        self, prompt_token_ids, trajectory_ids, sampling_params
    ) -> InferenceEngineOutput:
        """
        Route prompts to engines based on trajectory_ids and return results in the original order of the prompts.
        """
        # Group prompts by engine
        engine_groups: dict[int, dict[str, list]] = {}
        assert len(prompt_token_ids) == len(
            trajectory_ids
        ), f"Mismatch between number of prompts ({len(prompt_token_ids)}) and trajectory_ids ({len(trajectory_ids)})"
        for i, (token_ids, traj_id) in enumerate(zip(prompt_token_ids, trajectory_ids)):
            engine_idx = abs(hash(str(traj_id))) % len(self.engines)
            group = engine_groups.setdefault(engine_idx, {"token_ids": [], "indices": []})
            group["token_ids"].append(token_ids)
            group["indices"].append(i)

        # Build two parallel lists: one of tasks, one of the index‐lists
        tasks: list[asyncio.Task] = []
        indices_list: list[list[int]] = []
        for engine_idx, group in engine_groups.items():
            inp = InferenceEngineInput(
                prompt_token_ids=group["token_ids"],
                sampling_params=sampling_params,
            )
            coro = self.engines[engine_idx].generate(inp)
            tasks.append(asyncio.create_task(coro))
            indices_list.append(group["indices"])

        results = await asyncio.gather(*tasks)

        # Reconstruct output in original order
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

    async def _generate_batched(self, prompt_token_ids, sampling_params) -> InferenceEngineOutput:
        """
        Split prompts evenly across engines and return results in the original order of the prompts.
        """
        num_inference_engines = len(self.engines)
        dp_item_size = (len(prompt_token_ids) + num_inference_engines - 1) // num_inference_engines

        tasks = []
        for dp_rank in range(num_inference_engines):
            start_idx = dp_rank * dp_item_size
            end_idx = (dp_rank + 1) * dp_item_size
            dp_items = prompt_token_ids[start_idx:end_idx]

            if len(dp_items) <= 0:
                continue

            engine_input = InferenceEngineInput(
                prompt_token_ids=dp_items,
                sampling_params=sampling_params,
            )
            tasks.append(self.engines[dp_rank].generate(engine_input))

        all_outputs = await asyncio.gather(*tasks)

        # Flatten results
        responses = []
        stop_reasons = []
        response_ids = []
        response_logprobs = []
        for output in all_outputs:
            responses.extend(output["responses"])
            stop_reasons.extend(output["stop_reasons"])
            response_ids.extend(output["response_ids"])
            if output.get("response_logprobs", None):
                response_logprobs.extend(output["response_logprobs"])

        return InferenceEngineOutput(
            responses=responses,
            stop_reasons=stop_reasons,
            response_ids=response_ids,
            response_logprobs=response_logprobs if len(response_logprobs) else None,
        )

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
            assert engine.tp_size is not None, "Engine must have a tp_size"
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
            rank_offset_count += engine.tp_size
        await asyncio.gather(*tasks)

    async def update_named_weights(self, request: NamedWeightsUpdateRequest):
        return await self._run_on_all_engines("update_named_weights", request=request)

    async def reset_prefix_cache(self):
        return await self._run_on_all_engines("reset_prefix_cache")

    async def teardown(self):
        return await self._run_on_all_engines("teardown")

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
                print(f"Error shutting down HTTP endpoint: {e}")

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
        print(f"InferenceEngineClient HTTP endpoint started on {self.http_endpoint_host}:{self.http_endpoint_port}")

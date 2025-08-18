from skyrl_train.inference_engines.base import (
    InferenceEngineInterface,
    InferenceEngineInput,
    InferenceEngineOutput,
    NamedWeightUpdateRequest,
)
import asyncio
import threading
from typing import List, Any, Dict


class InferenceEngineClient(InferenceEngineInterface):
    """
    Client to talk to a set of InferenceEngines.

    Note that InferenceEngineClient sub-classes InferenceEngineInterface so it can be used as if talking to a single engine.
    """

    def __init__(self, engines: List[InferenceEngineInterface], generator_config: Dict[str, Any]):
        self.engines = engines
        self.generator_config = generator_config
        self.model_name = generator_config.get("model_name")
        self.backend = generator_config.get("backend")

        self.use_http_server_inference_engine_client = generator_config.get(
            "use_http_server_inference_engine_client", False
        )
        self.http_server_inference_engine_client_host = generator_config.get(
            "http_server_inference_engine_client_host", "127.0.0.1"
        )
        self.http_server_inference_engine_client_port = generator_config.get(
            "http_server_inference_engine_client_port", 8000
        )

        if self.use_http_server_inference_engine_client:
            self._spin_up_http_server()

        print(f"InferenceEngineClient initialized with {len(engines)} engines.")

    def _spin_up_http_server(self):
        from skyrl_train.inference_engines.launch_inference_engine_http_server import serve, wait_for_server_ready

        self._server_thread = threading.Thread(
            target=serve,
            args=(self,),
            kwargs={
                "host": self.http_server_inference_engine_client_host,
                "port": self.http_server_inference_engine_client_port,
                "log_level": "warning",
                "backend": self.backend,
            },
            daemon=True,
        )
        self._server_thread.start()
        wait_for_server_ready(
            host=self.http_server_inference_engine_client_host,
            port=self.http_server_inference_engine_client_port,
            max_wait_seconds=30,
        )
        print(
            f"InferenceEngineClient HTTP server started on {self.http_server_inference_engine_client_host}:{self.http_server_inference_engine_client_port}"
        )

    def __del__(self):
        """
        Destructor to shut down the HTTP server if it was started.
        """
        # TODO(Charlie): __del__ is not guaranteed to be called in general. Add to `teardown` method
        # when the `_handle_termination` flow is implemented. See `worker.py` comments on
        # `_handle_termination` for more details.
        if (
            self.use_http_server_inference_engine_client
            # don't want to shut down the server when it is pickled as a ray method argument.
            and hasattr(self, "_server_thread")
            and self._server_thread is not None
        ):
            try:
                from skyrl_train.inference_engines.launch_inference_engine_http_server import shutdown_server

                shutdown_server(
                    host=self.http_server_inference_engine_client_host,
                    port=self.http_server_inference_engine_client_port,
                    max_wait_seconds=5,
                )
                if hasattr(self, "_server_thread") and self._server_thread.is_alive():
                    self._server_thread.join(timeout=5)
            except Exception as e:
                print(f"Error shutting down HTTP server: {e}")

    def __getstate__(self):
        """
        Override to avoid pickling the server thread.
        Needed when passing InferenceEngineClient as an argument to async_run_ray_method().
        """
        state = self.__dict__.copy()
        # Thread objects are not picklable – just drop the reference.
        state["_server_thread"] = None
        return state

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

        # TODO(tgriggs): If there are no traj ids, we'd still like to load balance instead of landing on a single engine.
        if trajectory_ids is not None:
            # Route based on trajectory_ids
            return await self._generate_with_trajectory_routing(
                prompts, prompt_token_ids, trajectory_ids, sampling_params
            )
        else:
            # Split evenly across engines
            return await self._generate_batched(prompts, prompt_token_ids, sampling_params)

    async def _generate_with_trajectory_routing(self, prompts, prompt_token_ids, trajectory_ids, sampling_params):
        """
        Route prompts to engines based on trajectory_ids and return results in the original order of the prompts.
        """
        # Group prompts by engine
        engine_groups: dict[int, dict[str, list]] = {}
        prompts_or_tokens = prompts if prompts is not None else prompt_token_ids
        for i, (prompt_or_token, traj_id) in enumerate(zip(prompts_or_tokens, trajectory_ids)):
            engine_idx = abs(hash(str(traj_id))) % len(self.engines)
            group = engine_groups.setdefault(engine_idx, {"prompt_or_token": [], "indices": []})
            group["prompt_or_token"].append(prompt_or_token)
            group["indices"].append(i)

        # Build two parallel lists: one of tasks, one of the index‐lists
        tasks: list[asyncio.Task] = []
        indices_list: list[list[int]] = []
        for engine_idx, group in engine_groups.items():
            inp = InferenceEngineInput(
                prompts=group["prompt_or_token"] if prompts is not None else None,
                prompt_token_ids=group["prompt_or_token"] if prompt_token_ids is not None else None,
                sampling_params=sampling_params,
            )
            coro = self.engines[engine_idx].generate(inp)
            tasks.append(asyncio.create_task(coro))
            indices_list.append(group["indices"])

        results = await asyncio.gather(*tasks)

        # Reconstruct output in original order
        n = len(prompts_or_tokens)
        responses: list[str] = [""] * n
        stop_reasons: list[str] = [""] * n

        for indices, result in zip(indices_list, results):
            for local_idx, original_idx in enumerate(indices):
                responses[original_idx] = result["responses"][local_idx]
                stop_reasons[original_idx] = result["stop_reasons"][local_idx]

        return InferenceEngineOutput(responses=responses, stop_reasons=stop_reasons)

    async def _generate_batched(self, prompts, prompt_token_ids, sampling_params):
        """
        Split prompts evenly across engines and return results in the original order of the prompts.
        """
        num_inference_engines = len(self.engines)
        prompts_or_tokens = prompts if prompts is not None else prompt_token_ids
        dp_item_size = (len(prompts_or_tokens) + num_inference_engines - 1) // num_inference_engines

        tasks = []
        for dp_rank in range(num_inference_engines):
            start_idx = dp_rank * dp_item_size
            end_idx = (dp_rank + 1) * dp_item_size
            dp_items = prompts_or_tokens[start_idx:end_idx]

            if len(dp_items) <= 0:
                continue

            engine_input = InferenceEngineInput(
                prompts=dp_items if prompts is not None else None,
                prompt_token_ids=dp_items if prompt_token_ids is not None else None,
                sampling_params=sampling_params,
            )
            tasks.append(self.engines[dp_rank].generate(engine_input))

        all_outputs = await asyncio.gather(*tasks)

        # Flatten results
        responses = []
        stop_reasons = []
        for output in all_outputs:
            responses.extend(output["responses"])
            stop_reasons.extend(output["stop_reasons"])

        return InferenceEngineOutput(responses=responses, stop_reasons=stop_reasons)

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

    async def update_named_weight(self, request: NamedWeightUpdateRequest):
        return await self._run_on_all_engines("update_named_weight", request=request)

    async def reset_prefix_cache(self):
        return await self._run_on_all_engines("reset_prefix_cache")

    async def teardown(self):
        return await self._run_on_all_engines("teardown")

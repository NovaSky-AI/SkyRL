from skyrl_train.inference_engines.base import (
    InferenceEngineInterface,
    InferenceEngineInput,
    InferenceEngineOutput,
    NamedWeightUpdateRequest,
)
import asyncio
from typing import List, Any
from dataclasses import dataclass

@dataclass
class TaskMetadata:
    engine_idx: int
    traj_ids: list[str]
    indices: list[int]
    prompt_or_tokens: list[Any]

class InferenceEngineClient(InferenceEngineInterface):
    """
    Client to talk to a set of InferenceEngines.

    Note that InferenceEngineClient sub-classes InferenceEngineInterface so it can be used as if talking to a single engine.
    """

    def __init__(self, engines: List[InferenceEngineInterface]):
        self.engines = engines
        self.engine_health = [True] * len(engines)
        
        print(f"InferenceEngineClient initialized with {len(engines)} engines.")

    # TODO: Not used yet because ther isn't an easy way to restart engines when inference is mid flight rn
    async def _long_standing_engine_health_checker(self):
        while True:
            tasks = []
            for i in range(len(self.engines)):
                tasks.append(asyncio.create_task(self.engines[i].check_health(timeout=10.0)))
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"Engine {i} is dead")
                    self.engines[i].teardown()
                    self.engines[i] = None
                    self.engines.pop(i)
            print(f"Engine health check complete. {len(self.engines)} engines remaining.")
            # check every 5 seconds if any engine is dead
            await asyncio.sleep(5.0)


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

    def _consistent_hash_healthy_engines(self, key: str) -> int:
        # consistent hash across all healthy engines
        healthy_engines = [i for i, is_healthy in enumerate(self.engine_health) if is_healthy]
        healthy_engine_idx = abs(hash(str(key))) % len(healthy_engines)
        return healthy_engines[healthy_engine_idx]

    async def _generate_with_trajectory_routing(self, prompts, prompt_token_ids, trajectory_ids, sampling_params):
        """
        Route prompts to engines based on trajectory_ids and return results in the original order of the prompts.
        """
        # Group prompts by engine
        engine_groups: dict[int, dict[str, list]] = {}
        prompts_or_tokens = prompts if prompts is not None else prompt_token_ids
        for i, (prompt_or_token, traj_id) in enumerate(zip(prompts_or_tokens, trajectory_ids)):
            engine_idx = self._consistent_hash_healthy_engines(str(traj_id))
            group = engine_groups.setdefault(engine_idx, {"prompt_or_token": [], "indices": [], "traj_ids": []})
            group["prompt_or_token"].append(prompt_or_token)
            group["indices"].append(i)
            group["traj_ids"].append(traj_id)

        # Build two parallel lists: one of tasks, one of the index‐lists
        tasks: list[asyncio.Task] = []
        task_metadata = {}
        for engine_idx, group in engine_groups.items():
            inp = InferenceEngineInput(
                prompts=group["prompt_or_token"] if prompts is not None else None,
                prompt_token_ids=group["prompt_or_token"] if prompt_token_ids is not None else None,
                sampling_params=sampling_params,
            )
            coro = self.engines[engine_idx].generate(inp)
            task = asyncio.create_task(coro)
            tasks.append(task)
            task_metadata[task] = TaskMetadata(engine_idx, group["traj_ids"], group["indices"], inp)

        results = []
        result_indices = []
        # detect errors as they happen and mark that engine as dead
        while tasks:
            # Wait for at least one task to complete
            done, _ = await asyncio.wait(
                tasks, 
                return_when=asyncio.FIRST_COMPLETED
            )
            # Process all completed tasks
            for completed_task in done:
                metadata = task_metadata[completed_task]
                # remove from task queue also in case this worker is actually alive and we get duplicated results
                tasks.remove(completed_task)
                try:
                    result = await completed_task
                    results.append(result)
                    result_indices.append(metadata.indices)
                except Exception:
                    # engine raised an exception, mark as dead and redistribute tasks
                    self.engine_health[metadata.engine_idx]= False
                    # remove relevant engine group because we will redistribute tasks
                    del engine_groups[metadata.engine_idx]

                    assert len(metadata.traj_ids) > 0 # metadata should have at least one traj id

                    traj_id = metadata.traj_ids[0]
                    # set a new engine idx for the metadata
                    metadata.engine_idx = self._consistent_hash_healthy_engines(str(traj_id))
                    coro = self.engines[metadata.engine_idx].generate(metadata.inp)
                    task = asyncio.create_task(coro)
                    tasks.append(task)
                    task_metadata[task] = metadata
                    # TODO: redistribute in smaller chunks to each engine
                    # for prompt_or_token, idx in zip(metadata.prompt_or_tokens, metadata.indices):
                    #     redistributed_engine_idx = self._consistent_hash_healthy_engines(str(traj_id))

                    #     tasks.append(self.engines[redistributed_engine_idx].generate(prompt_or_token))
                    #     task_metadata[tasks[-1]] = TaskMetadata(redistributed_engine_idx, [idx], prompt_or_token)
                

        # Reconstruct output in original order
        n = len(prompts_or_tokens)
        responses: list[str] = [""] * n
        stop_reasons: list[str] = [""] * n

        for indices, result in zip(result_indices, results):
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

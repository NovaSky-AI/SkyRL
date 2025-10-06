import asyncio
from typing import Callable, Any, Dict
from loguru import logger

DoFnType = Callable[[int, int], Any]  # batch_idx, trajectory_id
DispatcherType = Callable[[DoFnType, DoFnType, DoFnType], Any]

# Dispatcher Registry
DISPATCHER_REGISTRY: Dict[str, DispatcherType] = {}


def register_dispatcher(name):
    def decorator(fn):
        DISPATCHER_REGISTRY[name] = fn
        return fn

    return decorator


# Async Pipeline Dispatcher (Producer-Consumer Pipelining)
@register_dispatcher("async_pipeline")
async def async_pipeline_dispatcher(
    cfg, trajectories: Dict[str, Dict[str, Any]], init_fn: str, run_fn: str, eval_fn: str
):
    async def pipeline():
        """Pipeline dispatcher for async processing of init, run, and eval functions."""
        # Initialize queues
        init_queue = asyncio.Queue()
        run_queue = asyncio.Queue()
        eval_queue = asyncio.Queue()

        # Get the generator instance from the init function
        max_parallel_agents = cfg["max_parallel_agents"]
        max_eval_parallel_agents = cfg.get("max_eval_parallel_agents", max_parallel_agents)

        num_instances = cfg["num_instances"]
        num_trajectories = cfg["num_trajectories"]
        total_instances = num_instances

        max_eval_parallel_agents = min(total_instances * num_trajectories, max_eval_parallel_agents)
        max_parallel_agents = min(total_instances * num_trajectories, max_parallel_agents)

        logger.info(
            f"Using max_parallel_agents of {max_parallel_agents} for {total_instances} instances with {num_trajectories} trajectories each"
        )
        logger.info(
            f"Using max_eval_parallel_agents of {max_eval_parallel_agents} for {total_instances} instances with {num_trajectories} trajectories each"
        )

        # Fill the init queue with tasks
        for trajectory_id in range(num_trajectories):
            for instance_id in trajectories.keys():
                await init_queue.put((instance_id, trajectory_id))

        async def initialize_one():
            while True:
                instance_id, trajectory_id = await init_queue.get()
                await getattr(trajectories[instance_id][trajectory_id], init_fn)()
                await run_queue.put((instance_id, trajectory_id))
                init_queue.task_done()

        async def run_one():
            while True:
                instance_id, trajectory_id = await run_queue.get()
                await getattr(trajectories[instance_id][trajectory_id], run_fn)()
                await eval_queue.put((instance_id, trajectory_id))
                run_queue.task_done()

        async def eval_one():
            while True:
                instance_id, trajectory_id = await eval_queue.get()
                await getattr(trajectories[instance_id][trajectory_id], eval_fn)()
                eval_queue.task_done()

        # Create tasks for initialization, running and evaluation
        init_tasks = [asyncio.create_task(initialize_one()) for _ in range(max_parallel_agents)]
        run_tasks = [asyncio.create_task(run_one()) for _ in range(max_parallel_agents)]
        eval_tasks = [asyncio.create_task(eval_one()) for _ in range(max_eval_parallel_agents)]

        # Wait until all initialization tasks are done
        print("Waiting for initialization tasks to complete...")
        await init_queue.join()
        for task in init_tasks:
            task.cancel()

        print("Initialization tasks completed. Waiting for run tasks to complete...")
        # Wait until all running tasks are done
        await run_queue.join()
        for task in run_tasks:
            task.cancel()

        print("Run tasks completed. Waiting for evaluation tasks to complete...")
        # Wait until all evaluation tasks are done
        await eval_queue.join()
        for task in eval_tasks:
            task.cancel()

    await pipeline()


# Async Batch Dispatcher
@register_dispatcher("async_batch")
async def async_batch_dispatcher(cfg, trajectories: Dict[int, Dict[int, Any]], init_fn: str, run_fn: str, eval_fn: str):
    async def run_all():
        tasks = []
        num_trajectories = cfg["num_trajectories"]

        async def one_traj(instance_id, trajectory_id):
            traj = trajectories[instance_id][trajectory_id]
            if init_fn is not None:
                await getattr(traj, init_fn)()
            await getattr(traj, run_fn)()
            await getattr(traj, eval_fn)()

        for instance_id in trajectories.keys():
            for trajectory_id in range(num_trajectories):
                tasks.append(asyncio.create_task(one_traj(instance_id, trajectory_id)))

        await asyncio.gather(*tasks)

    await run_all()


# Async FixedEnv Pool Dispatcher (Env Pool Reuse)
@register_dispatcher("async_fix_pool")
async def async_fix_pool_dispatcher(cfg, init_fn, run_fn, eval_fn):
    """
    Dispatcher for pre-initialized environments. Each trajectory is assigned
    to a free env. When finished, the env is returned to the pool.
    """

    async def dispatcher():
        envs = cfg["envs"]  # List of pre-initialized environments
        num_envs = len(envs)
        num_instances = cfg["num_instances"]
        num_trajectories = cfg["num_trajectories"]
        total_trajectories = num_instances * num_trajectories

        # Queue to keep track of available env_ids
        env_queue = asyncio.Queue()
        for env_id in range(num_envs):
            await env_queue.put(env_id)

        # Queue of all pending (batch_idx, trajectory_id)
        work_queue = asyncio.Queue()
        for trajectory_id in range(num_trajectories):
            for batch_idx in range(num_instances):
                await work_queue.put((batch_idx, trajectory_id))

        logger.info(f"FixedEnv dispatcher with {num_envs} envs for {total_trajectories} total trajectories.")

        async def worker():
            while True:
                try:
                    batch_idx, trajectory_id = await work_queue.get()
                    env_id = await env_queue.get()

                    # Reset and assign env
                    await init_fn(batch_idx, trajectory_id, env_id)

                    # Run and eval
                    await run_fn(batch_idx, trajectory_id, env_id)
                    await eval_fn(batch_idx, trajectory_id, env_id)

                    # Mark trajectory and env as done
                    work_queue.task_done()
                    await env_queue.put(env_id)
                except Exception as e:
                    logger.exception(f"Worker failed for ({batch_idx}, {trajectory_id}): {e}")
                    work_queue.task_done()
                    await env_queue.put(env_id)

        # Launch one worker per environment
        workers = [asyncio.create_task(worker()) for _ in range(num_envs)]

        print("Waiting for all work to complete...")
        await work_queue.join()

        # Cancel all workers
        for w in workers:
            w.cancel()

    await dispatcher()

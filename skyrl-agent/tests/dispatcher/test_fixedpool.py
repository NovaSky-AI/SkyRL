import asyncio
import random
from skyrl_agent.dispatcher.dispatchers import async_fix_pool_dispatcher


# Dummy env class
class DummyEnv:
    def __init__(self, env_id):
        self.env_id = env_id
        self.reset_called = 0
        self.run_log = []

    async def reset(self):
        self.reset_called += 1


# Dummy init_fn, run_fn, eval_fn
async def init_fn(batch_idx, trajectory_id, env_id):
    await envs[env_id].reset()
    print(f"[init] env{env_id} <- (batch{batch_idx}, traj{trajectory_id})")


async def run_fn(batch_idx, trajectory_id, env_id):
    await asyncio.sleep(random.uniform(0.01, 0.1))  # simulate async work
    envs[env_id].run_log.append(("run", batch_idx, trajectory_id))
    print(f"[run] env{env_id} <- (batch{batch_idx}, traj{trajectory_id})")


async def eval_fn(batch_idx, trajectory_id, env_id):
    await asyncio.sleep(random.uniform(0.01, 0.05))  # simulate async eval
    envs[env_id].run_log.append(("eval", batch_idx, trajectory_id))
    print(f"[eval] env{env_id} <- (batch{batch_idx}, traj{trajectory_id})")


async def test_async_fix_pool_dispatcher():
    global envs
    num_envs = 3
    num_instances = 2
    num_trajectories = 3
    envs = [DummyEnv(env_id=i) for i in range(num_envs)]

    cfg = {
        "envs": envs,
        "num_instances": num_instances,
        "num_trajectories": num_trajectories,
    }

    # Run the dispatcher
    await async_fix_pool_dispatcher(cfg, init_fn, run_fn, eval_fn)

    # Check each env's reset call count and logs
    total_tasks = num_instances * num_trajectories
    total_logs = sum(len(env.run_log) for env in envs)
    assert total_logs == 2 * total_tasks, f"Expected {2 * total_tasks} log entries, got {total_logs}"

    print("\n All tasks processed correctly.")


if __name__ == "__main__":
    asyncio.run(test_async_fix_pool_dispatcher())

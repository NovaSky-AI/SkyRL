# Copyright Sierra

from skyrl_gym.envs.tau_bench.tau_core.base import Env
from skyrl_gym.envs.tau_bench.tau_core.retail.data import load_data
from skyrl_gym.envs.tau_bench.tau_core.retail.rules import RULES
from skyrl_gym.envs.tau_bench.tau_core.retail.tools import ALL_TOOLS
from skyrl_gym.envs.tau_bench.tau_core.retail.wiki import WIKI
from typing import Optional
from skyrl_gym.envs.tau_bench.tau_core.user import BaseUserSimulationEnv


class MockRetailDomainEnv(Env):
    def __init__(
        self,
        user: BaseUserSimulationEnv,
        task_split: str = "test",
        task_index: Optional[int] = None,
    ):
        match task_split:
            case "test":
                from skyrl_gym.envs.tau_bench.tau_core.retail.tasks_test import TASKS_TEST as tasks
            case "train":
                from skyrl_gym.envs.tau_bench.tau_core.retail.tasks_train import TASKS_TRAIN as tasks
            case "dev":
                from skyrl_gym.envs.tau_bench.tau_core.retail.tasks_dev import TASKS_DEV as tasks
            case _:
                raise ValueError(f"Unknown task split: {task_split}")
        super().__init__(
            data_load_func=load_data,
            tools=ALL_TOOLS,
            tasks=tasks,
            wiki=WIKI,
            rules=RULES,
            user=user,
            task_index=task_index,
        )
        self.terminate_tools = ["transfer_to_human_agents"]

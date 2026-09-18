# Adapted from Sierra's tau-bench (sierra-research/tau-bench).
# The upstream LLM user simulators call litellm directly; SkyRL injects its own
# user simulator (see ``skyrl_gym.envs.tau_bench.user_simulator``) so we keep only
# the abstract interface here and drop the litellm dependency.

import abc
import enum
from typing import Optional


class BaseUserSimulationEnv(abc.ABC):
    """Interface the vendored retail ``Env`` expects for the simulated user.

    ``reset`` returns the opening user utterance for a task; ``step`` returns the
    user's reply to the agent's latest message (or ``###STOP###`` to end).
    """

    metadata: dict = {}

    @abc.abstractmethod
    def reset(self, instruction: Optional[str] = None) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, content: str) -> str:
        raise NotImplementedError

    def get_total_cost(self) -> float:
        return 0.0


class UserStrategy(enum.Enum):
    HUMAN = "human"
    LLM = "llm"
    REACT = "react"
    VERIFY = "verify"
    REFLECTION = "reflection"

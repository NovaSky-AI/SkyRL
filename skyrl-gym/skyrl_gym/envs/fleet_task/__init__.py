"""Fleet Task Environment for SkyRL-Gym.

Provides a multi-turn tool-use environment backed by Fleet-hosted environments,
using OpenEnv's FleetTaskEnv as the abstraction layer.
"""

from skyrl_gym.envs.fleet_task.env import FleetTaskEnv
from skyrl_gym.envs.fleet_task.tool_call_parser import parse_tool_call

__all__ = ["FleetTaskEnv", "parse_tool_call"]

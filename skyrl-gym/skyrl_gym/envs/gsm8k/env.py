from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.gsm8k import utils
from typing import Dict, Any
from omegaconf import DictConfig
from skyrl_gym.tools import RLMExecutorToolGroup
import re


class GSM8kEnv(BaseTextEnv):
    """
    Environment for Math execution tasks with RLM tool support.
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"

        # Initialize RLM tool for Python execution
        self.rlm_tool = RLMExecutorToolGroup()
        self.init_tool_groups([self.rlm_tool])

        self.ground_truth = extras["reward_spec"]["ground_truth"]

    def _parse_action(self, action: str) -> tuple:
        """
        Parse action to extract tool calls.
        Expects format: <tool><tool_name>input</tool_name></tool>
        Example: <tool><rlm>print(2 + 2)</rlm></tool>
        """
        # Try to find tool block
        tool_block_match = re.search(r"<tool>(.*?)</tool>", action, re.DOTALL)
        if not tool_block_match:
            # No tool call found, treat entire action as final answer
            return None, None, None

        tool_content = tool_block_match.group(1).strip()
        inner_tag_match = re.search(r"<(\w+)>(.*?)</\1>", tool_content, re.DOTALL)

        if not inner_tag_match:
            return None, None, None

        # Extract the tool name and input
        tool_name = inner_tag_match.group(1)
        tool_input = inner_tag_match.group(2).strip()

        # Get the tool group name from the tool name
        tool_group_name = self.tool_to_toolgroup.get(tool_name)

        return tool_group_name, tool_name, tool_input

    def _get_reward(self, action: str) -> float:
        """Compute reward by comparing action to ground truth."""
        return utils.compute_score(action, self.ground_truth)

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """
        Execute one step in the environment.
        If action contains a tool call, execute it and return observation.
        Otherwise, compute reward and mark as done.
        """
        # Try to parse tool call from action
        tool_group_name, tool_name, tool_input = self._parse_action(action)

        if tool_group_name and tool_name and tool_input:
            # Execute the tool
            try:
                observation = self._execute_tool(tool_group_name, tool_name, [tool_input])
                # Return observation, not done yet
                return BaseTextEnvStepOutput(
                    observations=[{"role": "user", "content": observation}],
                    reward=0.0,
                    done=False,
                    metadata={"tool_used": tool_name},
                )
            except Exception as e:
                # Tool execution failed, return error observation
                error_msg = f"Tool execution error: {str(e)}"
                return BaseTextEnvStepOutput(
                    observations=[{"role": "user", "content": error_msg}],
                    reward=0.0,
                    done=False,
                    metadata={"error": str(e)},
                )
        else:
            # No tool call, treat as final answer
            reward = self._get_reward(action)
            return BaseTextEnvStepOutput(observations=[], reward=reward, done=True, metadata={})

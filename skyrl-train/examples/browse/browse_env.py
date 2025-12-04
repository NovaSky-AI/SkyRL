from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from typing import Any
from skyrl_gym.envs.search.utils import compute_score
from examples.browse.browse_tool import BraveSearch
import re
from typing import Dict, Optional, List
from omegaconf import DictConfig
from examples.browse.tool_parser import Qwen3ToolParser, GLM4ToolParser
import json


class BrowseEnv(BaseTextEnv):
    """
    Environment for Brave Search execution tasks using the native tool call parser.
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        self.ground_truth = extras["reward_spec"]["ground_truth"]
        self.max_turns = extras["max_turns"] if "max_turns" in extras else 2

        # Initialize the tools
        self.searcher = BraveSearch()
        self.function_mapping = {
            "brave_search": self.searcher.search,
        }

        # Chat history
        # role (user, assistant), content (tool observation or LLM response)
        self.chat_history: ConversationType = []

        self.tool_mapping = {
            "qwen3": Qwen3ToolParser(),
            "glm4": GLM4ToolParser(),
        }
        assert (
            env_config.tool_call_parser in self.tool_mapping
        ), f"Invalid tool call parser: {env_config.tool_call_parser}. Valid parsers are: {self.tool_mapping.keys()}"

        self.tool_parser = self.tool_mapping[env_config.tool_call_parser]

    def _parse_action(self, action: str) -> List[Optional[str]]:
        match = None
        if "<search>" in action and "</search>" in action:
            match = re.search(r"<search>(.*?)</search>", action, re.DOTALL)
        return [match.group(1)] if match else [None]

    def _get_reward(self, action: str, done: bool) -> float:
        if done:
            # Concat all chat history into a single string and compute reward
            chat_history_str = "".join([item["content"] for item in self.chat_history])
            return compute_score(chat_history_str, self.ground_truth)
        else:
            # No reward for intermediate steps for Search tasks
            return 0

    def _is_done(self, action: str) -> bool:
        if self.turns >= self.max_turns:
            return True
        return "<answer>" in action and "</answer>" in action  # or "<tool_call>" not in action

    def _validate_action(self, action: str):
        stop_tags = ["</search>", "</answer>"]
        for tag in stop_tags:
            if tag in action:
                assert action.split(tag, 1)[1] == "", (
                    f"{tag} detected in the response but it is not the last string generated. "
                    f"Use {stop_tags} as stop strings in the configuration."
                )

    def _execute_tool(self, tool_group_name: str, tool_name: str, tool_input: Any) -> str:
        tool_output = super()._execute_tool(tool_group_name, tool_name, tool_input)

        return tool_output

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1

        self.chat_history.append({"role": "assistant", "content": action})

        error = None
        done = self._is_done(action)
        reward = self._get_reward(action, done)

        if done:
            return BaseTextEnvStepOutput(observations=[], reward=reward, done=done, metadata={})

        try:
            result = self.tool_parser.extract_tool_calls(action)
        except Exception as e:
            error = str(e)
            return BaseTextEnvStepOutput(
                observations=[
                    {
                        "role": "tool",
                        "content": "Error: Failed to parse/extract tool calls. Make sure follow the correct tool call format.",
                    }
                ],
                reward=reward,
                done=False,
                metadata={},
            )

        if not result["tools_called"]:
            return BaseTextEnvStepOutput(
                observations=[
                    {
                        "role": "user",
                        "content": "Error: No tool calls nor answer found in the response. You must either call a tool wrapped with <tool_call> and </tool_call> tags or provide an answer wrapped with <answer> and </answer> tags.",
                    }
                ],
                reward=reward,
                done=False,
                metadata={},
            )

        observations = []
        infos = []

        for tool_call in result["tool_calls"]:
            try:
                if tool_call["name"] in self.function_mapping:
                    observation = json.dumps(self.function_mapping[tool_call["name"]](**tool_call["arguments"]))
                else:
                    observation = f"Error: Unknown tool call: {tool_call['name']}. The only supported tools are: {self.function_mapping.keys()}"
            except Exception as e:
                error = str(e)
                observation = error
            tool_msg_dict = {"role": "tool", "name": tool_call["name"], "content": observation}
            observations.append(tool_msg_dict)
            self.chat_history.append(tool_msg_dict)

            infos.append(
                {
                    "tool_group": "BraveSearchToolGroup",
                    "tool_name": tool_call["name"],
                    "tool_input": tool_call["arguments"],
                }
            )

        return BaseTextEnvStepOutput(
            observations=observations,
            reward=reward,
            done=done,
            metadata=infos,
        )

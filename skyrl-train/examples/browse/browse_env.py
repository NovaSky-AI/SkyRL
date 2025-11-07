from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from typing import Any
from skyrl_gym.envs.search.utils import compute_score
from examples.browse.browse_tool import BraveSearchToolGroup, DEFAULT_TIMEOUT
from inference_utils.tools import BraveSearch
import re
from typing import Dict, Optional, List
from omegaconf import DictConfig
from examples.browse.tool_parser import Qwen3XMLToolParser
import json


class BrowseEnv(BaseTextEnv):
    """
    Environment for Search execution tasks.

    Based on Verl + Search-R1 integration
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        self.ground_truth = extras["reward_spec"]["ground_truth"]
        self.max_turns = extras["max_turns"] if "max_turns" in extras else 2

        # Initialize the tools
        # name is hardcoded to "BraveSearchToolGroup", with tool name "browse"
        self.tool_group = BraveSearchToolGroup(
            search_url=env_config.get("search_url", "http://127.0.0.1:8000/retrieve"),
            topk=env_config.get("topk", 3),
            timeout=env_config.get("timeout", DEFAULT_TIMEOUT),
            log_requests=env_config.get("log_requests", True),
        )
        self.init_tool_groups([self.tool_group])

        # Chat history
        # role (user, assistant), content (tool observation or LLM response)
        self.chat_history: ConversationType = []

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
        return "<answer>" in action and "</answer>" in action

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

        return "\n<information>" + tool_output + "</information>\n"

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        self._validate_action(action)
        self.chat_history.append({"role": "assistant", "content": action})

        error = None
        done = self._is_done(action)
        reward = self._get_reward(action, done)

        if done:
            return BaseTextEnvStepOutput(observations=[], reward=reward, done=done, metadata={})

        try:
            query = self._parse_action(action)
            observation = self._execute_tool("BraveSearchToolGroup", "brave_search", query)
        except Exception as e:
            error = str(e)
            observation = None

        # Wrap the observation properly as a message
        if observation:
            new_obs = {"role": "user", "content": observation}
        elif error:
            # Give error as observation if any
            new_obs = {"role": "user", "content": error}
        else:
            new_obs = None

        info = {
            "tool_group": "BraveSearchToolGroup",
            "tool_name": "brave_search",
            "tool_input": query,
        }

        # Update chat history
        if new_obs:
            self.chat_history.append(new_obs)

        return BaseTextEnvStepOutput(
            observations=[new_obs] if new_obs else [],
            reward=reward,
            done=done,
            metadata=info,
        )


class NativeBrowseEnv(BaseTextEnv):
    """
    Environment for Search execution tasks.

    Based on Verl + Search-R1 integration
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        self.ground_truth = extras["reward_spec"]["ground_truth"]
        self.max_turns = extras["max_turns"] if "max_turns" in extras else 2

        # Initialize the tools
        # name is hardcoded to "BraveSearchToolGroup", with tool name "browse"
        self.tool_group = BraveSearchToolGroup(
            search_url=env_config.get("search_url", "http://127.0.0.1:8000/retrieve"),
            topk=env_config.get("topk", 3),
            timeout=env_config.get("timeout", DEFAULT_TIMEOUT),
            log_requests=env_config.get("log_requests", True),
        )
        self.init_tool_groups([self.tool_group])

        # Chat history
        # role (user, assistant), content (tool observation or LLM response)
        self.chat_history: ConversationType = []

        self.tool_parser = Qwen3XMLToolParser()
        self.searcher = BraveSearch()
        self.function_mapping = {
            "brave_search": self.searcher.search,
        }

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
        # print(f"action: {action}")
        self.turns += 1
        # print('turns', self.turns)
        # print('action', action)
        # self._validate_action(action)
        self.chat_history.append({"role": "assistant", "content": action})

        error = None
        done = self._is_done(action)
        reward = self._get_reward(action, done)

        # print(f"action: \n{action}")

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
                        "role": "tool",
                        "content": "Error: Failed to parse/extract tool calls. Make sure follow the correct tool call format.",
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
                # observation = self._execute_tool("BraveSearchToolGroup", tool_call['name'], tool_call['arguments'])
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

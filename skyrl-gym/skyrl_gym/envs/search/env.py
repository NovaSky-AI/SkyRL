from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from typing import Tuple, Any
from skyrl_gym.envs.search.utils import compute_score
from skyrl_gym.tools import SearchToolGroup
import re
from typing import Dict
from omegaconf import DictConfig


class SearchEnv(BaseTextEnv):
    """
    Environment for Search execution tasks.

    Reference Search-R1 implementation and VeRL + Search-R1 integration
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        self.ground_truth = extras["reward_spec"]["ground_truth"]
        self.max_turns = extras["max_turns"] if "max_turns" in extras else 5

        # Initialize the tools
        self.tool_group = SearchToolGroup()
        self.init_tool_groups([self.tool_group])

        # Chat history
        # role (user, assistant), content (tool observation or LLM response)
        self.chat_history: ConversationType = []

    def _parse_action(self, action: str) -> Tuple[str, str, Any]:
        """
        Return tool group name, tool name, and input arguments to tool function
        """
        tool_name = None
        query_match = None

        if "<query>" in action and "</query>" in action:
            match = re.search(r"<query>(.*?)</query>", action, re.DOTALL)
            if match:
                query_match = match.group(1)
            else:
                query_match = None

        tool_group_name = self.tool_group.get_name()
        tool_name = self.tool_group.get_tool_names()[0]
        return tool_group_name, tool_name, [query_match]

    def _get_reward(self, action: str, done: bool) -> float:
        if done:
            # Concat all chat history into a single string and compute reward
            chat_history_str = "\n".join([item["content"] for item in self.chat_history])
            return compute_score(chat_history_str, self.ground_truth)
        else:
            # No reward for intermediate steps for Search tasks
            return 0

    def _is_done(self, action: str) -> bool:
        if self.turns >= self.max_turns:
            return True
        return "<answer>" in action and "</answer>" in action

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        self.chat_history.append({"role": "assistant", "content": action})

        error = None
        done = self._is_done(action)
        reward = self._get_reward(action, done)

        if done:
            return BaseTextEnvStepOutput(observations=[], reward=reward, done=done, metadata={})

        try:
            tool_group_name, tool_name, tool_input = self._parse_action(action)
            observation = self._execute_tool(tool_group_name, tool_name, tool_input)
        except Exception as e:
            error = str(e)
            observation = None
            tool_group_name = None
            tool_name = None
            tool_input = ""

        # Wrap the observation properly as a message
        if observation:
            new_obs = {"role": "user", "content": observation}
        elif error:
            # Give error as observation if any
            new_obs = {"role": "user", "content": error}
        else:
            new_obs = None

        info = {"tool_group": tool_group_name, "tool_name": tool_name, "tool_input": tool_input}

        # Update chat history
        if new_obs:
            self.chat_history.append(new_obs)

        return BaseTextEnvStepOutput(observations=[new_obs] if new_obs else [], reward=reward, done=done, metadata=info)

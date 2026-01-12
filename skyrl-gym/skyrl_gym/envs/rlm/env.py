from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.rlm import utils
from typing import Dict, Any
from omegaconf import DictConfig
from skyrl_gym.tools import RLMExecutorToolGroup
import re
import json


class RLMExecutorEnv(BaseTextEnv):
    """
    Environment for Math execution tasks.
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"

        self.rlm_tool = RLMExecutorToolGroup()
        self.init_tool_groups([self.rlm_tool])

        rlm_base_url = None
        rlm_model = None
        rlm_api_key = None
        if env_config is not None:
            rlm_base_url = env_config.get("rlm_base_url", None)
            rlm_model = env_config.get("rlm_model", None)
            rlm_api_key = env_config.get("rlm_api_key", None)

        if (rlm_base_url and rlm_model) or rlm_api_key:
            self.rlm_tool.rlm_setup(
                base_url=rlm_base_url or "https://api.openai.com/v1",
                model=rlm_model or "gpt-4o-mini",
                openai_api_key=rlm_api_key,
                init_prompt="",
            )

        self.ground_truth = extras["reward_spec"]["ground_truth"]
        self.current_turn = 0
        self.max_turns = env_config.get("max_turns", 3) if env_config else 3
        self.original_query = extras.get("prompt", "")
        self._has_called_rlm = False

    def init(self, prompt):
        """Capture the original query from the initial prompt."""
        if prompt and len(prompt) > 0:
            self.original_query = prompt[0].get("content", "")
        print(f"[GSM8K ENV] Captured original query: {self.original_query[:100]}...")
        return super().init(prompt)

    def _get_reward(self, action: str) -> float:
        return utils.compute_score(action, self.ground_truth)

    def _parse_tool_call(self, action: str):
        # Try to match <tool><tool_name>input</tool_name></tool> format
        tool_block_match = re.search(r"<tool>(.*?)</tool>", action, re.DOTALL)
        if not tool_block_match:
            return None, None, None

        tool_content = tool_block_match.group(1).strip()
        inner_tag_match = re.search(r"<(\w+)>(.*?)</\1>", tool_content, re.DOTALL)

        if not inner_tag_match:
            return None, None, None

        tool_name = inner_tag_match.group(1)
        tool_input = inner_tag_match.group(2).strip()
        tool_group_name = self.tool_to_toolgroup.get(tool_name)

        return tool_group_name, tool_name, tool_input

    def _force_rlm_tool_call(self, action: str) -> str:

        print(f"[GSM8K ENV] Turn {self.current_turn}: Forcing rlm call")
        query = "Solve this math problem step by step and provide the final numerical answer."
        return f"<tool><rlm>{query}</rlm></tool>"

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.current_turn += 1
        action_with_tool = action
        if not self._has_called_rlm and not re.search(r"<tool>", action, re.DOTALL):
            action_with_tool = self._force_rlm_tool_call(action)

        tool_group_name, tool_name, tool_input = self._parse_tool_call(action_with_tool)

        if tool_group_name and tool_name and self.current_turn <= self.max_turns:
            try:
                tool_payload: Any
                if tool_input == "":
                    tool_payload = []
                else:
                    s = tool_input.strip()
                    looks_like_json = s and s[0] in "{[" and s[-1] in "}]"
                    if looks_like_json:
                        try:
                            tool_payload = json.loads(s)
                        except Exception:
                            tool_payload = tool_input
                    else:
                        tool_payload = tool_input

                if tool_name == "rlm":
                    if isinstance(tool_payload, dict):
                        tool_payload.setdefault("query", tool_input)
                        tool_payload.setdefault("context", self.original_query or "")
                    else:
                        tool_payload = {"query": tool_input, "context": self.original_query or ""}

                    print(f"tool_payload: {tool_payload}")
                    self._has_called_rlm = True

                observation = self._execute_tool(tool_group_name, tool_name, tool_payload)
                print(f"observation: {observation}")

                return BaseTextEnvStepOutput(
                    observations=[{"role": "user", "content": observation}],
                    reward=0.0,  # No reward until done
                    done=False,
                    metadata={"turn": self.current_turn},
                )

            except Exception as e:
                # If tool execution fails, return error observation and continue
                import traceback

                error_msg = f"Tool execution error: {str(e)}"
                print("[RLM ENV] Tool execution failed!")
                print(f"[RLM ENV] Tool: {tool_name}")
                print(f"[RLM ENV] Error: {error_msg}")
                print(f"[RLM ENV] Traceback:\n{traceback.format_exc()}")
                return BaseTextEnvStepOutput(
                    observations=[{"role": "user", "content": error_msg}],
                    reward=0.0,
                    done=False,
                    metadata={"turn": self.current_turn, "error": str(e)},
                )

        # Either no tool call, max turns reached, or final answer
        done = True
        reward = self._get_reward(action)
        return BaseTextEnvStepOutput(observations=[], reward=reward, done=done, metadata={"turn": self.current_turn})

    def reset(self, **kwargs):
        """Reset turn counter/state when episode resets."""
        self.current_turn = 0
        self._has_called_rlm = False
        # Capture the original query if provided
        if "prompt" in kwargs:
            self.original_query = kwargs["prompt"]
        return super().reset(**kwargs)

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from typing import Any, Tuple
from skyrl_gym.envs.allocated_code import utils
import re
from typing import Dict, Optional, List
from omegaconf import DictConfig
import httpx
import atexit
import signal
import weakref


class AllocatedCodeToolGroup:
    def __init__(self, manager_url: str = "http://localhost:5000"):
        self.manager_url = manager_url
        self.allocated_container: Optional[int] = None
        self.client = httpx.Client(timeout=30.0)
    
    def allocate_container(self):
        if self.allocated_container is not None:
            return
        
        # Retry container allocation up to 3 times
        for attempt in range(3):
            try:
                response = self.client.post(f"{self.manager_url}/allocate")
                response.raise_for_status()
                result = response.json()
                self.allocated_container = result["container_id"]
                return
            except Exception as e:
                # Convert HTTPStatusError to a more Ray-friendly exception
                error_msg = f"Failed to allocate container (attempt {attempt + 1}/3): {str(e)}"
                if hasattr(e, 'response') and e.response is not None:
                    error_msg += f" (Status: {e.response.status_code})"
                
                if attempt == 2:  # Last attempt
                    raise RuntimeError(error_msg)
                else:
                    print(f"WARNING: {error_msg}, retrying...")
                    import time
                    time.sleep(1)  # Wait 1 second before retry
    
    def deallocate_container(self):
        if self.allocated_container is None:
            return
        
        self.client.post(f"{self.manager_url}/deallocate/{self.allocated_container}")
        self.allocated_container = None
    
    def execute_code(self, code: str) -> str:
        if self.allocated_container is None:
            raise RuntimeError("No container allocated")
        
        try:
            response = self.client.post(
                f"{self.manager_url}/session/{self.allocated_container}/execute",
                json={"code": code}
            )
            response.raise_for_status()
            result = response.json()
            raw_output = result.get("outputs", [])
            # Clean and format the Jupyter output
            output = utils.clean_jupyter_output(raw_output)
            return output
        except Exception as e:
            # Convert HTTPStatusError to a more Ray-friendly exception
            error_msg = f"Failed to execute code: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                error_msg += f" (Status: {e.response.status_code})"
            raise RuntimeError(error_msg)
    
    def get_tool_names(self):
        return ["python"]


class AllocatedCodeEnv(BaseTextEnv):
    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        self.ground_truth = extras["reward_spec"]["ground_truth"]
        self.max_turns = extras["max_turns"] if "max_turns" in extras else 2
        self.query = extras["extra_info"]["question"]
        self.data_source = extras["data_source"]
        
        # Configure whether to enable code execution reward, default False (keep original behavior)
        self.use_code_execution_reward = env_config.get("use_code_execution_reward", False)
        self.code_execution_reward_value = env_config.get("code_execution_reward_value", 0.1)
        
        # Configure whether to use discrete reward scoring, default False (keep original behavior)
        self.use_discrete_reward = env_config.get("use_discrete_reward", False)
        
        manager_url = env_config.get("manager_url", "http://localhost:5000")
        self.tool_group = AllocatedCodeToolGroup(manager_url)
        
        self.tool_group.allocate_container()
        
        self.chat_history: ConversationType = []

    def reset(self, **kwargs):
        self.turns = 0
        self.chat_history = []
        self.tool_group.allocate_container()
        return []

    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        self.turns = 0
        self.chat_history = []
        self.tool_group.allocate_container()
        return prompt, {}

    def _parse_action(self, action: str) -> Optional[str]:
        match = None
        if "<python>" in action and "</python>" in action:
            match = re.search(r"<python>(.*?)</python>", action, re.DOTALL)
            if match:
                code = match.group(1)
                if "```python" in code and "```" in code:
                    inner_match = re.search(r"```python(.*?)```", code, re.DOTALL)
                    if inner_match:
                        return inner_match.group(1)
                return code
        return None

    def _extract_hypothesis(self, response: str) -> str:
        """Extract hypothesis from model response."""
        # Look for <answer> tags first
        if '<answer>' in response and '</answer>' in response:
            match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # If no answer tags, return the full response
        return ''
    
    def _get_reward(self, action: str, done: bool, tool_output: str = None) -> float:
        if done:
            # Final reward: LLM-based scoring of hypothesis quality
            chat_history_str = "".join([item["content"] for item in self.chat_history])
            if self.data_source == "qrdata":
                return utils.compute_score(action, self.ground_truth)
            hypothesis = self._extract_hypothesis(chat_history_str)
            if self.use_discrete_reward:
                return utils.compute_llm_score_discrete(hypothesis, chat_history_str, self.ground_truth, self.query)
            else:
                return utils.compute_llm_score(hypothesis, chat_history_str, self.ground_truth, self.query)
        else:
            # Intermediate reward: conditionally give reward for successful code execution
            if (self.use_code_execution_reward and 
                tool_output and 
                utils.is_code_execution_successful(tool_output)):
                return self.code_execution_reward_value
            return 0

    def _is_done(self, action: str) -> bool:
        max_turns_reached = self.turns >= self.max_turns
        has_answer_tags = "<answer>" in action and "</answer>" in action
        
        result = max_turns_reached or has_answer_tags
        
        return result

    def _postprocess_action(self, action: str) -> str:
        if "</python>" in action:
            return action.split("</python>")[0] + "</python>"
        elif "</answer>" in action:
            return action.split("</answer>")[0] + "</answer>"
        else:
            return action

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        action = self._postprocess_action(action)
        self.chat_history.append({"role": "assistant", "content": action})

        error = None
        tool_output = None
        done = self._is_done(action)

        if done:
            reward = self._get_reward(action, done)
            self.tool_group.deallocate_container()
            return BaseTextEnvStepOutput(
                observations=[], reward=reward, done=done, metadata={}, postprocessed_action=action
            )

        try:
            code = self._parse_action(action)
            if code:
                tool_output = self.tool_group.execute_code(code)
                observation = "\n<information>" + tool_output + "</information>\n"
            else:
                observation = None
        except Exception as e:
            error = str(e)
            observation = None
        
        # Calculate reward (including intermediate rewards)
        reward = self._get_reward(action, done, tool_output)

        if observation:
            new_obs = {"role": "user", "content": observation}
        elif error:
            new_obs = {"role": "user", "content": error}
        else:
            new_obs = None

        info = {
            "tool_group": "AllocatedCodeToolGroup", 
            "tool_name": "python",
            "tool_input": code,
            "allocated_container": self.tool_group.allocated_container,
        }

        if new_obs:
            self.chat_history.append(new_obs)

        # print(f"DEBUG: Step output - done: {done}, reward: {reward}, new_obs: {new_obs['content'][:20] if new_obs else 'None'}")
        return BaseTextEnvStepOutput(
            observations=[new_obs] if new_obs else [],
            reward=reward,
            done=done,
            metadata=info,
            postprocessed_action=action,
        )

    def close(self):
        self.tool_group.deallocate_container()
        super().close() 
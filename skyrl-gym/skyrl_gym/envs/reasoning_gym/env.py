from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from typing import Dict, Any
from omegaconf import DictConfig
import json
import pickle
import base64
from reasoning_gym.utils import extract_answer
from reasoning_gym import create_dataset


class ReasoningGymEnv(BaseTextEnv):
    """
    Environment for ReasoningGym tasks.
    Handles reward calculation using ReasoningGym's scoring methods.
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "extra_info" in extras, "extra_info field is required"
        self.extra_info = extras["extra_info"]
        
        
        # Deserialize the data source from the stored serialized version
        data_source_serialized = self.extra_info.get("data_source_serialized")
        self.reasoning_gym_data_source = pickle.loads(base64.b64decode(data_source_serialized))

        try:
            self.original_entry = json.loads(self.extra_info["dataset_entry"])
        except (json.JSONDecodeError, TypeError):
            self.original_entry = self.extra_info["dataset_entry"]


    def _get_reward(self, action: str) -> float:
        """
        Calculate reward using ReasoningGym's built-in scoring logic.
        """
        
        try:
            found_answer = extract_answer(action, tag_name="answer")
        except Exception as e:
            print(f"Warning: Error extracting answer between <answer></answer> tags from model output, scoring the entire model output: {e}")
            found_answer = action

        reward = self.reasoning_gym_data_source.score_answer(found_answer, entry=self.original_entry)
        return float(reward)

        
    def step(self, action: str) -> BaseTextEnvStepOutput:
        """
        Process one step in the reasoning environment.
        For reasoning tasks, we typically complete in one step.
        """
        done = True  # Most reasoning tasks complete in one step
        reward = self._get_reward(action)

        # No additional observations needed for reasoning tasks
        return BaseTextEnvStepOutput(
            observations=[], 
            reward=reward, 
            done=done, 
            metadata={}
        )

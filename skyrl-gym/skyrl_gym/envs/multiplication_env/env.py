from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from typing import Any, Dict
import re

class MultiplyEnv(BaseTextEnv):
    # only need to override the step method
    def _parse_action(self, action: str) -> str:
        """Extract answer from the action string, which is of format \\boxed{result}"""
        match = re.search(r"\\boxed\{([^}]+)\}", action)
        return match.group(1) if match else None
    
    def step(self, action:str) -> BaseTextEnvStepOutput:
        answer = self._parse_action(action)
        is_correct = answer is not None and answer.strip() == str(self.ground_truth).strip()

        return BaseTextEnvStepOutput(
            observations=[],
            reward=1.0 if is_correct else 0.0,
            done=True,
            metadata={"parsed_answer": answer}
        )
        

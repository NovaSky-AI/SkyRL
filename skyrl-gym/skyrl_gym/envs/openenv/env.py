from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from typing import Any, Dict, Type, Any
import json
from omegaconf import DictConfig
# Demonstrate five different environments for now 
from envs.echo_env import EchoEnv, EchoAction
from envs.coding_env import CodingEnv, CodeAction
from envs.openspiel_env import OpenSpielEnv, OpenSpielAction
from envs.atari_env import AtariEnv, AtariAction
import re

class OpenEnv(BaseTextEnv):
    """
    Environment for LiveCodeBench execution environment.
    """

    def __init__(
        self,
        env_config: DictConfig,
        extras: Dict[str, Any] = {},
    ):
        super().__init__()

        self.extras = extras 
        self.max_turns = extras["max_turns"] if "max_turns" in extras else 1
        
        # NOTE: find a way to get the env class 
        self.env_name = extras["env_name"]
        self.env_type = self._get_env_class(self.env_name)
        self.env = self.env_type.from_docker_image(self.env_name+":latest")
         
        # Reset before start 
        self.initial_step_result = self.env.reset()
        
        # Look at the state of the environment
        # self.env.state()
        
        # Chat history
        # role (user, assistant), content (tool observation or LLM response)
        self.chat_history: ConversationType = []

    def _get_env_class(self, env_name: str) -> Type:
        if env_name == "echo_env":
            return EchoEnv
        elif env_name == "coding_env":
            return CodingEnv
        elif env_name == "openspiel-env":
            return OpenSpielEnv
        elif env_name == "atari-env":
            return AtariEnv
        # TODO: handle ChatEnv, maybe use ChatEnv to maintain message history?
        else:
            raise ValueError(f"Unknown environment '{env_name}'")
        
    def _get_openenv_action(self, env_name: str, action: str):
        """
        Parse the action string to detect things to pass into the OpenEnv environment.
        Assume a simple fixed format: <action>...</action>
        
        Returns:
            Action object to pass into the OpenEnv environment.
        """
        match = None
        if "<action>" in action and "</action>" in action:
            match = re.search(r"<action>(.*?)</action>", action, re.DOTALL)
        
        action = match.group(1) if match else None
        
        if env_name == "echo_env":
            return EchoAction(message=action)
        elif env_name == "coding_env":
            return CodeAction(code=action)
        elif env_name == "openspiel-env":
            # NOTE: optionally pass in game names
            if "game_name" in self.extras: 
                return OpenSpielAction(action_id=int(action), game_name=self.extras["game_name"])
            else:
                return OpenSpielAction(action_id=int(action))
        elif env_name == "atari-env":
            if not action.isdigit():
                raise ValueError(f"Atari action must be numeric, got: {action}")
            return AtariAction(action_id=int(action))
        else:
            raise ValueError(f"Unknown environment '{env_name}'")
    
    def _is_done(self) -> bool:
        if self.turns >= self.max_turns:
            return True
        
    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.chat_history.append({"role": "assistant", "content": action})
        
        # Check max turns reached
        done = self._is_done()
        if done:
            return BaseTextEnvStepOutput(observations=[], reward=0, done=done, metadata={})
        
        try:
            action = self._get_openenv_action(self.env_name, action)
            result = self.env.step(action)
            observation = result.observation
            reward = result.reward 
            done = result.done 
        except Exception as e:
            error = str(e)
            observation = None 
            reward = -1 
        
        if observation:
            new_obs = {"role": "user", "content": observation}
        elif error:
            new_obs = {"role": "user", "content": error}
        else:
            new_obs = None
        
        if new_obs:
            self.chat_history.append(new_obs)
        
        info = {
            "env_class": self.env_name,
            "action": action,
            "observation": observation,
        }
        
        return BaseTextEnvStepOutput(
            observations=[new_obs] if new_obs else [],
            reward=reward,
            done=done,
            metadata=info,
        )
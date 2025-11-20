from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from typing import Dict, Type, Any
from omegaconf import DictConfig

# Demonstrate five different environments for now
from envs.echo_env import EchoEnv, EchoAction
from envs.coding_env import CodingEnv, CodeAction
from envs.openspiel_env import OpenSpielEnv, OpenSpielAction
from envs.atari_env import AtariEnv, AtariAction
from envs.sumo_rl_env import SumoRLEnv, SumoAction
from envs.finrl_env import FinRLEnv, FinRLAction
import re
import ast

import dataclasses


def serialize_observation(obs: Any, max_list_len: int = 20) -> str:
    """
    Convert any Observation (dataclass or dict) into a flat string.
    Long lists are truncated for readability.
    """
    if obs is None:
        return "Observation: None"

    # Convert to dict representation
    if dataclasses.is_dataclass(obs):
        obs_dict = dataclasses.asdict(obs)
    elif hasattr(obs, "__dict__"):
        obs_dict = vars(obs)
    elif isinstance(obs, dict):
        obs_dict = obs
    else:
        # Fallback for unexpected types
        return f"Observation: {str(obs)}"

    flat_lines = []
    for k, v in obs_dict.items():
        if isinstance(v, list):
            display = v[:max_list_len]
            suffix = " ..." if len(v) > max_list_len else ""
            flat_lines.append(f"{k}: {display}{suffix} (len={len(v)})")
        else:
            flat_lines.append(f"{k}: {v}")

    return "\n".join(flat_lines) + "\n given this information, try again."


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
        self.env = self.env_type.from_docker_image(self.env_name + ":latest")

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
        elif env_name == "sumo-rl-env":
            return SumoRLEnv
        elif env_name == "finrl-env":
            return FinRLEnv
        else:
            raise ValueError(f"Unknown environment '{env_name}'")

    def _get_openenv_action(self, env_name: str, action: str):
        """
        Parse the action string to detect things to pass into the OpenEnv environment.
        Assume a simple fixed format: <action>...</action>

        Returns:
            Action object to pass into the OpenEnv environment.
        """
        matches = []
        if "<action>" in action and "</action>" in action:
            matches = re.findall(r"<action>(.*?)</action>", action)

        action = matches[-1] if len(matches) > 0 else None

        if not action:
            raise ValueError(f"No action found in action string: {action}")

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
        elif env_name == "sumo-rl-env":
            return SumoAction(phase_id=int(action))
        elif env_name == "finrl-env":
            try:
                actions_list = ast.literal_eval(action)
            except Exception as e:
                raise ValueError(f"Invalid FinRL action format '{action}', needs to be a list of floats: {e}")

            return FinRLAction(actions=list(actions_list))
        else:
            raise ValueError(f"Unknown environment '{env_name}'")

    def _is_done(self) -> bool:
        return self.turns >= self.max_turns

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        self.chat_history.append({"role": "assistant", "content": action})

        max_turns_reached = self._is_done()

        error = None
        try:
            action = self._get_openenv_action(self.env_name, action)
            result = self.env.step(action)
            observation = serialize_observation(result.observation)
            reward = 0.0 if not result.reward else result.reward
            done = result.done
        except Exception as e:
            error = str(e)
            observation = None
            reward = -1

        if max_turns_reached:
            # If reached max turns, just return the reward and done
            return BaseTextEnvStepOutput(observations=[], reward=reward, done=max_turns_reached, metadata={})
        else:
            # Return observation for multi-turn interaction
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
            # print("chat history: ", self.chat_history)

            return BaseTextEnvStepOutput(
                observations=[new_obs] if new_obs else [],
                reward=reward,
                done=done,
                metadata=info,
            )

    def close(self):
        self.env.close()

"""The original SDPO science/tool-use reward functions as local SkyRL envs."""

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput

from . import rewards_mcq, rewards_tooluse


class SDPOEnvironment(BaseTextEnv):
    scorer = staticmethod(rewards_mcq.compute_score)

    def __init__(self, env_config=None, extras=None):
        super().__init__()
        self.ground_truth = extras["reward_model"]["ground_truth"]
        self.result = {}

    def step(self, action):
        self.result = self.scorer(action, self.ground_truth)
        return BaseTextEnvStepOutput(
            observations=[],
            reward=self.result["score"],
            done=True,
            metadata=self.result,
        )

    def get_metrics(self):
        return self.result


class SDPOToolUseEnvironment(SDPOEnvironment):
    scorer = staticmethod(rewards_tooluse.compute_score)


def register_sdpo_envs():
    from skyrl_gym.envs import register
    from skyrl_gym.envs.registration import registry

    for name, cls in (
        ("sdpo_mcq", "SDPOEnvironment"),
        ("sdpo_tooluse", "SDPOToolUseEnvironment"),
    ):
        if name not in registry:
            register(id=name, entry_point=f"examples.train.sdpo.reproduction.envs:{cls}")

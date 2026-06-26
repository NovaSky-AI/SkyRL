r"""BIRD SQL env for skyrl-gym, mirroring the verl PR #6 reward contract.

Single-turn env: the model emits a response containing one SQL query inside a
``<think>...</think>`` block followed by a ```` ```sql ... ``` ```` block (the
arctic_text_to_sql_r1 format). We hand the full response and the gold SQL +
per-sample SQLite path to ``.bird_reward.compute_score`` — the *same*
reward function the validated verl PR #6 run used — and surface its ``score``
field as the GRPO reward.

The reward function is vendored at
``integrations/arctic_rl/envs/bird_reward.py`` so the integration doesn't
depend on a private Arctic-Platform recipe branch.

Why a SkyRL-side env at all?  The Arctic RL server is reward-agnostic; reward
scoring runs client-side via skyrl-gym (see ``ArcticGenerator.generate``).
This keeps the protocol simple (server takes (sequences, rewards, loss_mask))
and lets us reuse verl PR #6's reward function unmodified.

Required ``extras`` (forwarded automatically by ``PromptDataset`` from the
verl-format parquet — no schema rewriting needed):

  - ``reward_model.ground_truth`` : gold SQL string
  - ``extra_info.db_path``        : absolute path to the SQLite DB file
  - ``extra_info``                : passed through to ``compute_score``
                                    as the ``extra_info`` arg (it needs
                                    ``db_path`` and optionally
                                    ``alternative_answers``).
  - ``data_source``               : passed through (used by some reward fns
                                    for routing; benign for BIRD).
"""

from typing import Any, Dict

from .bird_reward import compute_score
from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput


class BirdEnv(BaseTextEnv):
    """Single-turn BIRD SQL env using the verl PR #6 reward fn verbatim."""

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = {}):
        super().__init__()

        # ``reward_model`` is the verl-canonical name; some SkyRL envs use
        # ``reward_spec``. Accept either to stay compatible across schemas.
        reward_block = extras.get("reward_model") or extras.get("reward_spec") or {}
        ground_truth = reward_block.get("ground_truth")
        if ground_truth is None:
            raise ValueError(
                "BirdEnv requires `reward_model.ground_truth` (or "
                "`reward_spec.ground_truth`) in env_extras — gold SQL is "
                "missing from this prompt's parquet row."
            )
        self.ground_truth: str = ground_truth

        extra_info = extras.get("extra_info") or {}
        if "db_path" not in extra_info:
            raise ValueError(
                "BirdEnv requires `extra_info.db_path` in env_extras — the "
                "per-sample SQLite path is missing. Re-run preprocess_bird.py "
                "to regenerate the parquet with `db_path` populated."
            )
        self.extra_info: Dict[str, Any] = dict(extra_info)
        # ``data_source`` lets reward fns route across SQL dialects; default to
        # "bird" since this env is BIRD-specific.
        self.data_source: str = extras.get("data_source", "bird")

    def _get_reward(self, response: str) -> float:
        result = compute_score(
            data_source=self.data_source,
            solution_str=response,
            ground_truth=self.ground_truth,
            extra_info=self.extra_info,
        )
        return float(result["score"])

    def step(self, action: str) -> BaseTextEnvStepOutput:
        # Single-turn: one model response -> one reward -> done.
        # The reward function reads <think>...</think> and ```sql ...``` blocks
        # itself, so we hand it the raw action without parsing.
        reward = self._get_reward(action)
        return BaseTextEnvStepOutput(
            observations=[],
            reward=reward,
            done=True,
            metadata={},
        )

"""Registers the internal gym envs."""

from skyrl_gym.envs.registration import register

register(
    id="aime",
    entry_point="skyrl_gym.envs.aime.env:AIMEEnv",
)

register(
    id="gsm8k",
    entry_point="skyrl_gym.envs.gsm8k.env:GSM8kEnv",
)

register(
    id="gsm8k_multi_turn",
    entry_point="skyrl_gym.envs.gsm8k.multi_turn_env:GSM8kMultiTurnEnv",
)

register(
    id="text2sql",
    entry_point="skyrl_gym.envs.sql.env:SQLEnv",
)

register(
    id="search",
    entry_point="skyrl_gym.envs.search.env:SearchEnv",
)

register(
    id="lcb",
    entry_point="skyrl_gym.envs.lcb.env:LCBEnv",
)

register(
    id="searchcode",
    entry_point="skyrl_gym.envs.searchcode.env:SearchCodeEnv",
)

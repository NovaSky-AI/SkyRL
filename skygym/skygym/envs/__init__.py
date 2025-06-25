"""Registers the internal gym envs."""

from skygym.envs.registration import register

register(
    id="gsm8k",
    entry_point="skygym.envs.gsm8k.env:GSM8kEnv",
)

register(
    id="text2sql",
    entry_point="skygym.envs.sql.env:SQLEnv",
)

register(
    id="search",
    entry_point="skygym.envs.search.env:SearchEnv",
)

register(
    id="lcb",
    entry_point="skygym.envs.lcb.env:LCBEnv",
)

register(
    id="searchcode",
    entry_point="skygym.envs.searchcode.env:SearchCodeEnv",
)

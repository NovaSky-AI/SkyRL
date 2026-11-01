"""Tests for the tau-bench (retail) environment.

These run on CPU with a scripted user simulator (no model server needed). They
verify the action protocol, the multi-turn step loop, and that the vendored
upstream reward (DB-state match + required outputs) is wired correctly: replaying
a task's own gold actions yields reward 1.0, while an empty trajectory yields 0.0.
"""

import json

import pytest

import skyrl_gym
from skyrl_gym.envs.tau_bench.env import TauBenchEnvConfig
from skyrl_gym.envs.tau_bench.user_simulator import ScriptedUserSimulator

DUMMY_PROMPT = [{"role": "user", "content": "(placeholder)"}]


def _make_env(task_index: int, replies):
    env = skyrl_gym.make(
        "tau_bench",
        env_config=TauBenchEnvConfig(),
        extras={"task_index": task_index, "task_split": "test", "max_turns": 50},
    )
    scripted = ScriptedUserSimulator(replies)
    # Replace the HTTP user simulator with the scripted one (used by init/step).
    env.user_sim = scripted
    env.tau_env.user = scripted
    return env


def _tool_call(action) -> str:
    return f'<tool_call>{json.dumps({"name": action.name, "arguments": action.kwargs})}</tool_call>'


def test_init_builds_system_prompt_and_first_user_message():
    env = _make_env(0, ["Hi, I need some help with my order."])
    convo, meta = env.init(DUMMY_PROMPT)
    assert convo[0]["role"] == "system"
    assert convo[1]["role"] == "user"
    assert convo[1]["content"] == "Hi, I need some help with my order."
    # System prompt should carry the retail policy + tool schemas + protocol.
    assert "<tool_call>" in convo[0]["content"]
    assert "exchange_delivered_order_items" in convo[0]["content"]
    assert meta["task_index"] == 0


def test_gold_trajectory_gets_reward_one():
    env = _make_env(0, ["Hi, I'd like to make some changes to my order."])
    env.init(DUMMY_PROMPT)

    gold_actions = env.tau_env.task.actions
    assert len(gold_actions) > 0
    for action in gold_actions:
        out = env.step(_tool_call(action))
        assert out["done"] is False, f"episode ended early on gold action {action.name}"

    # A plain message -> respond -> scripted user is exhausted -> '###STOP###' -> done.
    out = env.step("Everything is taken care of now, thank you!")
    assert out["done"] is True
    assert out["reward"] == 1.0
    metrics = env.get_metrics()
    assert metrics["success"] == 1.0
    assert metrics["num_tool_errors"] == 0.0


def test_empty_trajectory_gets_reward_zero():
    env = _make_env(0, ["Hi, I'd like to make some changes to my order."])
    env.init(DUMMY_PROMPT)
    # Immediately end without performing any of the required actions.
    out = env.step("I changed my mind, goodbye.")
    assert out["done"] is True
    assert out["reward"] == 0.0
    assert env.get_metrics()["success"] == 0.0


def test_malformed_tool_call_returns_error_observation():
    env = _make_env(0, ["Hi."])
    env.init(DUMMY_PROMPT)
    out = env.step("<tool_call>{not valid json}</tool_call>")
    assert out["done"] is False
    assert len(out["observations"]) == 1
    assert "Error: could not parse tool call" in out["observations"][0]["content"]
    assert env.get_metrics()["num_tool_errors"] == 1.0


def test_tool_call_returns_observation_and_continues():
    env = _make_env(0, ["Hi."])
    env.init(DUMMY_PROMPT)
    # A read-only lookup should return an observation and not end the episode.
    action = next(a for a in env.tau_env.task.actions if a.name.startswith("find_user_id"))
    out = env.step(_tool_call(action))
    assert out["done"] is False
    assert len(out["observations"]) == 1
    assert out["observations"][0]["role"] == "user"


def test_max_turns_terminates():
    env = _make_env(0, ["Hi."])
    env.init(DUMMY_PROMPT)
    env.max_turns = 1
    out = env.step('<tool_call>{"name": "list_all_product_types", "arguments": {}}</tool_call>')
    assert out["done"] is True

"""How each Harbor agent is pointed at its trajectory's skycap URL (``agents.py``)."""

import os

from examples.train_integrations.harbor_skycap.agents import (
    PLACEHOLDER_API_KEY,
    binding,
)

URL = "http://skycap:1/t/abc/v1"


def test_terminus_gets_the_url_in_its_kwargs_and_the_salt_in_extra_body() -> None:
    terminus = binding("terminus-2")
    agent = {"kwargs": {}}
    terminus.configure(agent, URL, "salt")

    assert not terminus.in_sandbox and terminus.raw_content
    assert terminus.model_name("Qwen3-8B") == "hosted_vllm/Qwen3-8B"
    assert agent["kwargs"]["api_base"] == URL
    assert agent["kwargs"]["llm_kwargs"] == {"api_key": PLACEHOLDER_API_KEY, "extra_body": {"cache_salt": "salt"}}


def test_mini_swe_agent_gets_the_url_through_its_environment(monkeypatch) -> None:
    monkeypatch.delenv("MSWEA_API_KEY", raising=False)
    mswea = binding("mini-swe-agent")
    mswea.prepare()
    agent = {"env": {"KEEP": "1"}}
    mswea.configure(agent, URL, "salt")

    # It runs inside the sandbox and acts through tool calls, which skycap must parse.
    assert mswea.in_sandbox and not mswea.raw_content
    assert mswea.model_name("Qwen3-8B") == "openai/Qwen3-8B"
    assert agent["env"] == {
        "KEEP": "1",
        "OPENAI_API_BASE": URL,
        "OPENAI_BASE_URL": URL,
        "OPENAI_API_KEY": PLACEHOLDER_API_KEY,
        "MSWEA_COST_TRACKING": "ignore_errors",
    }
    assert os.environ["MSWEA_API_KEY"] == PLACEHOLDER_API_KEY


def test_an_unlisted_agent_is_an_installed_one_speaking_openai() -> None:
    other = binding("claude-code")
    agent = {}
    other.configure(agent, URL, None)

    assert other.in_sandbox and other.model_name("m") == "openai/m"
    assert agent["env"]["OPENAI_BASE_URL"] == URL


def test_no_agent_name_means_terminus_2() -> None:
    assert binding(None) is binding("terminus-2")


def test_the_generator_refuses_an_in_sandbox_agent_without_an_exposure() -> None:
    from types import SimpleNamespace

    import pytest

    from examples.train_integrations.harbor_skycap.harbor_generator import (
        HarborSkycapGenerator,
    )

    generator_cfg = SimpleNamespace(
        step_wise_trajectories=True,
        merge_stepwise_output=False,
        inference_engine=SimpleNamespace(served_model_name="policy"),
        rate_limit=None,
    )
    with pytest.raises(ValueError, match="skycap.exposure.type"):
        HarborSkycapGenerator(generator_cfg, {"agent": {"name": "mini-swe-agent"}}, ["http://x"])

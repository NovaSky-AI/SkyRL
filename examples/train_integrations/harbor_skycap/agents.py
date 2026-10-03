"""How each Harbor agent is pointed at its trajectory's skycap URL.

Agents take their model endpoint in different ways: Terminus-2 through its own
kwargs, the installed agents (mini-swe-agent, ...) through the environment they
run with in the sandbox, Harbor's ``AgentConfig.env``. A binding says which, and
whether the agent calls the model from inside its sandbox, in which case skycap
must be reachable from there.
"""

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

#: skycap authenticates nothing, but LiteLLM won't build a client without a key.
PLACEHOLDER_API_KEY = "skycap"


@dataclass(frozen=True)
class AgentBinding:
    #: Whether the agent calls the model from inside its sandbox rather than from this process.
    in_sandbox: bool
    #: Points one trial's ``agent`` section at ``base_url``, carrying ``cache_salt`` when the agent can.
    configure: Callable[[Dict[str, Any], str, Optional[str]], None]
    #: The model name Harbor hands the agent, from the served model name.
    model_name: Callable[[str], str]
    #: Whether skycap should answer with the completion's raw text (``use_raw_content``) rather
    #: than parsed reasoning and tool calls. Terminus-2 parses its own actions out of the text;
    #: an agent that acts through tool calls needs them parsed.
    raw_content: bool = False
    #: Runs once in this process before trials start.
    prepare: Callable[[], None] = lambda: None


def _terminus(agent: Dict[str, Any], base_url: str, cache_salt: Optional[str]) -> None:
    kwargs = agent.setdefault("kwargs", {})
    kwargs["api_base"] = base_url
    llm_kwargs = kwargs.setdefault("llm_kwargs", {})
    # Terminus-2 takes `api_base` itself but passes a key only through `llm_kwargs`.
    llm_kwargs["api_key"] = PLACEHOLDER_API_KEY
    if cache_salt is not None:
        # LiteLLM merges `extra_body` into the request body, where skycap reads `cache_salt` and forwards it.
        extra_body = llm_kwargs.setdefault("extra_body", {})
        if not isinstance(extra_body, dict):
            raise TypeError("harbor_trial_config.agent.kwargs.llm_kwargs.extra_body must be a mapping")
        extra_body["cache_salt"] = cache_salt


def _openai_env(agent: Dict[str, Any], base_url: str, cache_salt: Optional[str]) -> None:
    """For an installed agent whose LiteLLM or OpenAI client reads the endpoint from its environment.

    ``cache_salt`` has no way in from there, so it isn't sent: prefixes cached under older weights
    can be reused. Synchronous training resets the engine's prefix cache at every weight sync, so
    this only matters with asynchronous training.
    """
    agent["env"] = {
        **(agent.get("env") or {}),
        "OPENAI_API_BASE": base_url,
        "OPENAI_BASE_URL": base_url,
        "OPENAI_API_KEY": PLACEHOLDER_API_KEY,
    }


def _mini_swe_agent(agent: Dict[str, Any], base_url: str, cache_salt: Optional[str]) -> None:
    _openai_env(agent, base_url, cache_salt)
    # mini-swe-agent refuses to run a model it has no price for unless told to carry on.
    agent["env"]["MSWEA_COST_TRACKING"] = "ignore_errors"


def _mini_swe_agent_prepare() -> None:
    # Harbor checks for a key in its own process environment before it starts the agent.
    os.environ.setdefault("MSWEA_API_KEY", PLACEHOLDER_API_KEY)


def _hosted_vllm(served: str) -> str:
    return f"hosted_vllm/{served}"


def _openai(served: str) -> str:
    # The `openai/` provider is what reads OPENAI_API_BASE.
    return f"openai/{served}"


_TERMINUS = AgentBinding(in_sandbox=False, configure=_terminus, model_name=_hosted_vllm, raw_content=True)
BINDINGS: Dict[str, AgentBinding] = {
    "terminus-2": _TERMINUS,
    "terminus-1": _TERMINUS,
    "terminus": _TERMINUS,
    "mini-swe-agent": AgentBinding(
        in_sandbox=True, configure=_mini_swe_agent, model_name=_openai, prepare=_mini_swe_agent_prepare
    ),
}


def binding(agent_name: Optional[str]) -> AgentBinding:
    """The binding for ``agent_name``; an unlisted agent is taken to be an installed one speaking OpenAI's API."""
    return BINDINGS.get(agent_name or "terminus-2") or AgentBinding(
        in_sandbox=True, configure=_openai_env, model_name=_openai
    )

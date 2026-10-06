"""Stand-ins for the tokenizer, SkyRL's router and a Harbor trial.

- ``FakeRenderer``: characters are tokens (``ord(c) + 100``) and three specials
  frame a message, ``START role NL content END NL``.
- ``MockRouter``: speaks ``/skyrl/v1/generate`` with packed routed experts
  (row ``p`` holds ``p % 256``) and sampler support (``[token, token + 1]``,
  padded to top_k with -1), and records ``/finish_session``.
- ``FakeTrial``: what Harbor's ``Trial`` is to the generator. Its task path picks
  a script: a linear chat, a summarization (the history is rewritten), a
  discarded reply asked again for (mini-swe-agent's format-error retry), a
  timeout, a crash, or a sandbox that times out starting on the first attempt
  (``slow_start``; "first" counts every trial of that task path since
  ``FakeTrial.configs`` was reset, so a test runs one such trial). Results carry Harbor's phase timings: the sandbox takes
  ``SETUP`` seconds to start, the agent ``AGENT`` and the verifier ``VERIFY``.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any

import aiohttp
import numpy as np
import orjson
from aiohttp import web

from skycap.tokens.renderer import Rendered, join_pieces
from skyrl.backends.skyrl_train.inference_servers.generate_wire import (
    pack_routed_experts,
    pack_sample_support,
)

START, END, NL = 1, 2, 3
SETUP, AGENT, VERIFY, START_TIMEOUT = 5.0, 10.0, 2.0, 600.0
EPOCH = datetime(2026, 1, 1)
LAYERS, EXPERTS_PER_TOKEN, TOP_K = 2, 2, 3


def encode(text: str) -> list[int]:
    return [ord(c) + 100 for c in text]


def decode(tokens: Sequence[int]) -> str:
    return "".join(chr(t - 100) for t in tokens if t >= 100)


class FakeRenderer:
    name = "fake"

    def _message(self, message: Mapping[str, Any]) -> list[int]:
        return [START, *encode(str(message.get("role"))), NL, *encode(message.get("content") or ""), END, NL]

    def render(self, messages: Sequence[Mapping[str, Any]], tools: Any) -> Rendered:
        tokens: list[int] = []
        indices: list[int] = []
        for index, message in enumerate(messages):
            chunk = self._message(message)
            tokens += chunk
            indices += [index] * len(chunk)
        prompt = [START, *encode("assistant"), NL]
        return Rendered(token_ids=tokens + prompt, tail_indices=indices + [-1] * len(prompt))

    def bridge(self, previous_prompt, previous_completion, new_messages, tools) -> Rendered | None:
        if not previous_completion or previous_completion[-1] != END:
            return None
        if any(m.get("role") == "assistant" for m in new_messages):
            return None
        tail = self.render(new_messages, tools)
        return Rendered(
            token_ids=[*previous_prompt, *previous_completion, NL, *tail.token_ids],
            tail_indices=[-1, *tail.tail_indices],
            reused=len(previous_prompt) + len(previous_completion),
        )

    def parse(self, completion_ids: Sequence[int], tools: Any) -> dict[str, Any]:
        return {"role": "assistant", "content": decode([t for t in completion_ids if t != END])}

    def stop_token_ids(self) -> list[int]:
        return [END]

    def decode_spans(self, token_ids: Sequence[int], context: Sequence[int] = ()) -> tuple[str, list[int]]:
        special = {START: "<s>", END: "</s>", NL: "\n"}
        return join_pieces([special.get(t) or chr(t - 100) for t in token_ids])


class MockRouter:
    def __init__(self) -> None:
        self.url = ""
        #: Set to answer every call with this text (plus the end token) instead of ``re<n>``.
        self.reply: str | None = None
        #: Routed experts on every reply; row ``p`` names position ``p`` of its call, so a test can check alignment.
        self.routes = True
        self.requests: list[dict[str, Any]] = []
        self.sessions: list[str] = []
        self.released: list[str] = []

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/skyrl/v1/generate", self.generate)
        app.router.add_post("/finish_session", self.finish_session)
        return app

    async def generate(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.requests.append(body)
        self.sessions.append(request.headers.get("X-Session-ID", ""))
        prompt = body["token_ids"]
        completion = [*encode(self.reply if self.reply is not None else f"re{len(prompt)}"), END]
        total = len(prompt) + len(completion)
        # As vLLM does: rows from `routed_experts_prompt_start` on, for every position but the last.
        start = body["sampling_params"].get("routed_experts_prompt_start", 0)
        rows = np.arange(start, total - 1) % 256
        routed = np.broadcast_to(rows[:, None, None], (len(rows), LAYERS, EXPERTS_PER_TOKEN)).astype(np.uint8)
        choice: dict[str, Any] = {
            "token_ids": completion,
            "finish_reason": "stop",
            "logprobs": {"content": [{"logprob": -0.5} for _ in completion]},
            "routed_experts": pack_routed_experts(routed) if self.routes else None,
            "rollout_sample_support": None,
        }
        if body.get("return_sample_support"):
            support = np.full((len(completion), TOP_K), -1, dtype=np.int32)
            support[:, 0] = completion
            support[:, 1] = np.asarray(completion) + 1
            choice["rollout_sample_support"] = pack_sample_support(support)
        return web.Response(body=orjson.dumps({"choices": [choice]}), content_type="application/json")

    async def finish_session(self, request: web.Request) -> web.Response:
        self.released.append(request.query["session_id"])
        return web.json_response({})


def timing(start: float, seconds: float) -> SimpleNamespace:
    """A Harbor ``TimingInfo`` from ``start`` to ``start + seconds``, in seconds after ``EPOCH``."""
    return SimpleNamespace(
        started_at=EPOCH + timedelta(seconds=start), finished_at=EPOCH + timedelta(seconds=start + seconds)
    )


def verified(reward: float, exception: str | None = None, timed: bool = True) -> SimpleNamespace:
    """A trial's result. Timed: every phase ran, except the verifier after an exception."""
    return SimpleNamespace(
        exception_info=SimpleNamespace(exception_type=exception) if exception else None,
        verifier_result=None if exception else SimpleNamespace(rewards={"reward": reward}),
        environment_setup=timing(0, SETUP) if timed else None,
        agent_execution=timing(SETUP, AGENT) if timed else None,
        verifier=timing(SETUP + AGENT, VERIFY) if timed and not exception else None,
    )


def start_timed_out() -> SimpleNamespace:
    """What Harbor returns when the sandbox doesn't start in time: the setup's timing, nothing after."""
    return SimpleNamespace(
        exception_info=SimpleNamespace(exception_type="EnvironmentStartTimeoutError"),
        verifier_result=None,
        environment_setup=timing(0, START_TIMEOUT),
        agent_execution=None,
        verifier=None,
    )


class FakeTrial:
    """Replaces ``harbor.trial.trial.Trial``. Every created trial's config is kept in ``configs``."""

    configs: list[dict[str, Any]] = []

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @classmethod
    async def create(cls, config: Any) -> "FakeTrial":
        cls.configs.append(config.model_dump() if hasattr(config, "model_dump") else config)
        return cls(cls.configs[-1])

    async def run(self) -> SimpleNamespace:
        script = str(self.config["task"]["path"])
        kwargs = self.config["agent"]["kwargs"]
        # As the agent finds its endpoint: an installed one in the sandbox's environment, Terminus-2 in its kwargs.
        base_url = (self.config["agent"].get("env") or {}).get("OPENAI_API_BASE") or kwargs["api_base"]
        extra = kwargs.get("llm_kwargs", {}).get("extra_body", {})

        async def chat(session: aiohttp.ClientSession, messages: list[dict[str, Any]]) -> dict[str, Any]:
            body = {"model": "policy", "messages": messages, **extra}
            async with session.post(f"{base_url}/chat/completions", json=body) as response:
                assert response.status == 200, await response.text()
                return (await response.json())["choices"][0]["message"]

        attempt = sum(str(c["task"]["path"]) == script for c in self.configs) - 1

        async with aiohttp.ClientSession() as session:
            if script == "crash":
                raise RuntimeError("sandbox did not start")
            if script == "silent":
                # Verified, but the agent never called the model; and Harbor recorded no timings.
                return verified(1.0, timed=False)
            if script == "slow_start" and attempt == 0:
                return start_timed_out()
            history = [{"role": "user", "content": script}]
            history.append(await chat(session, history))
            if script == "timeout":
                return verified(0.0, "AgentTimeoutError")
            history += [{"role": "user", "content": "ok"}]
            if script == "discard":
                # As mini-swe-agent on a format error: the reply is dropped from the history and the
                # agent asks again with an error message in its place.
                await chat(session, history)
                history += [{"role": "user", "content": "format error"}]
            history.append(await chat(session, history))
            if script == "summarize":
                # The agent compacts: the next call starts from a rewritten history.
                history = [{"role": "user", "content": "summary"}]
                history.append(await chat(session, history))
            return verified(1.0)

"""User simulators for the tau-bench environment.

The vendored retail ``Env`` (``tau_core.base.Env``) drives a simulated user: it calls
``user.reset(instruction)`` to obtain the opening customer message for a task and
``user.step(agent_message)`` to obtain the customer's reply to each agent turn.

Two implementations are provided:

- ``HTTPUserSimulator``: talks to an OpenAI-compatible chat-completions endpoint
  (a self-hosted vLLM server) over HTTP. This is the production simulator used
  during rollouts/eval.
- ``ScriptedUserSimulator``: returns pre-canned replies. Used for unit tests so
  the environment can be exercised without a live model.

The system prompt / role conventions mirror Sierra's upstream ``LLMUserSimulationEnv``
(sierra-research/tau-bench): from the simulator's point of view the *agent* speaks in
the ``user`` role and the simulated customer replies in the ``assistant`` role.
"""

import sys
from typing import Any, Dict, List, Optional

import requests

from skyrl_gym.envs.tau_bench.tau_core.user import BaseUserSimulationEnv


def build_user_system_prompt(instruction: Optional[str]) -> str:
    instruction_display = ("\n\nInstruction: " + instruction + "\n") if instruction is not None else ""
    return f"""You are a user interacting with an agent.{instruction_display}
Rules:
- Just generate one line at a time to simulate the user's message.
- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.
- Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.
- If the instruction goal is satisified, generate '###STOP###' as a standalone message without anything else to end the conversation.
- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.
- Try to make the conversation as natural as possible, and stick to the personalities in the instruction."""


class HTTPUserSimulator(BaseUserSimulationEnv):
    """LLM user simulator backed by an OpenAI-compatible chat-completions endpoint.

    Args:
        endpoint: Base URL of the OpenAI-compatible API, e.g. ``http://127.0.0.1:8001/v1``.
            ``/chat/completions`` is appended automatically.
        model: Model name to request from the server.
        max_tokens: Max tokens for each user turn.
        temperature: Sampling temperature for the user model.
        timeout: Per-request timeout in seconds.
        api_key: Optional bearer token (vLLM ignores it but accepts it).
    """

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:8001/v1",
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout: int = 120,
        api_key: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.url = endpoint.rstrip("/") + "/chat/completions"
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.api_key = api_key
        self.messages: List[Dict[str, Any]] = []

    def _generate(self) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": self.messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        try:
            resp = requests.post(self.url, json=payload, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            # A single user-sim hiccup (e.g. an over-long context returning HTTP 400)
            # must not abort the whole eval. End this conversation gracefully instead;
            # the episode is then scored as-is (typically a failure for incomplete tasks).
            body = ""
            try:
                body = resp.text[:300]  # type: ignore[name-defined]
            except Exception:
                pass
            print(
                f"[tau_bench user-sim] request failed ({e}); body={body!r}; " "ending conversation with ###STOP###",
                file=sys.stderr,
            )
            content = "###STOP###"
        # Record the simulated user's own reply so the next turn has full context.
        self.messages.append({"role": "assistant", "content": content})
        return content

    def reset(self, instruction: Optional[str] = None) -> str:
        self.messages = [
            {"role": "system", "content": build_user_system_prompt(instruction)},
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        return self._generate()

    def step(self, content: str) -> str:
        self.messages.append({"role": "user", "content": content})
        return self._generate()


class ScriptedUserSimulator(BaseUserSimulationEnv):
    """Deterministic user simulator for tests.

    Returns ``replies`` in order on each ``reset``/``step``; once exhausted it returns
    ``###STOP###`` so episodes terminate cleanly.
    """

    def __init__(self, replies: List[str]) -> None:
        super().__init__()
        self._replies = list(replies)
        self._idx = 0

    def _next(self) -> str:
        if self._idx < len(self._replies):
            reply = self._replies[self._idx]
            self._idx += 1
            return reply
        return "###STOP###"

    def reset(self, instruction: Optional[str] = None) -> str:
        self._idx = 0
        return self._next()

    def step(self, content: str) -> str:
        return self._next()

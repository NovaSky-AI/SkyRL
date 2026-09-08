"""Compatibility test against Harbor's pinned LiteLLM integration."""

import pytest

from skyrl.train.generators.tito.proxy import TITOProxy
from skyrl.train.generators.tito.renderer import RenderedPrompt
from skyrl.train.generators.tito.trace import Trace


class _Renderer:
    def __init__(self, assistant_message=None):
        self._assistant_message = assistant_message or {
            "role": "assistant",
            "content": "done",
            "reasoning_content": "thinking",
        }

    def render(self, messages, *, tools=None):
        return RenderedPrompt((10, 99), (0, -1))

    def bridge(self, previous_prompt_ids, previous_completion_ids, new_messages, *, tools=None):
        return None

    def parse_response(self, token_ids, *, tools=None):
        return self._assistant_message

    def get_stop_token_ids(self):
        return [2]

    def decode_token(self, token_id):
        return str(token_id)


class _InferenceEngine:
    def __init__(self):
        self.finished_sessions = []

    async def generate(self, input_batch, model=None):
        return {
            "responses": ["done"],
            "response_ids": [[50, 51]],
            "stop_reasons": ["stop"],
            "response_logprobs": [[-0.2, -0.3]],
            "prompt_logprobs": None,
            "rollout_expert_indices": None,
        }

    async def finish_session(self, session_id):
        self.finished_sessions.append(session_id)


@pytest.mark.asyncio
async def test_pinned_harbor_collects_proxy_rollout_details():
    pytest.importorskip("harbor")
    from harbor.llms.lite_llm import LiteLLM

    engine = _InferenceEngine()
    proxy = TITOProxy(engine, _Renderer())
    trace = Trace()

    async with proxy.serving():
        async with proxy.register(trace, router_session_id="router-session", cache_salt=None, model="model") as handle:
            llm = LiteLLM(
                model_name="hosted_vllm/model",
                api_base=f"{handle.base_url}/v1",
                session_id="harbor-session",
                collect_rollout_details=True,
                model_info={
                    "max_input_tokens": 1024,
                    "max_output_tokens": 128,
                    "input_cost_per_token": 0.0,
                    "output_cost_per_token": 0.0,
                },
                api_key="EMPTY",
                timeout=30,
            )
            response = await llm.call("hello")

    assert response.content == "done"
    assert response.prompt_token_ids == [10, 99]
    assert response.completion_token_ids == [50, 51]
    assert response.logprobs == [-0.2, -0.3]
    assert response.reasoning_content == "thinking"
    assert len(trace.committed_turns()) == 1
    assert engine.finished_sessions == ["router-session"]


@pytest.mark.asyncio
async def test_pinned_litellm_parses_proxy_tool_calls():
    litellm = pytest.importorskip("litellm")

    engine = _InferenceEngine()
    renderer = _Renderer(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"key":"x"}',
                    },
                }
            ],
        }
    )
    proxy = TITOProxy(engine, renderer)
    trace = Trace()

    async with proxy.serving():
        async with proxy.register(trace, router_session_id="router-session", cache_salt=None, model="model") as handle:
            response = await litellm.acompletion(
                model="hosted_vllm/model",
                api_base=f"{handle.base_url}/v1",
                api_key="EMPTY",
                messages=[{"role": "user", "content": "look up x"}],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {
                                "type": "object",
                                "properties": {"key": {"type": "string"}},
                            },
                        },
                    }
                ],
                tool_choice="auto",
            )

    tool_call = response.choices[0].message.tool_calls[0]
    assert tool_call.id == "call-1"
    assert tool_call.function.name == "lookup"
    assert tool_call.function.arguments == '{"key":"x"}'
    assert len(trace.committed_turns()) == 1

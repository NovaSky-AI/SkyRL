"""Harbor generator wiring tests for the TITO proxy."""

from types import SimpleNamespace

import httpx
import pytest

from skyrl.train.generators.base import TrajectoryID
from skyrl.train.generators.tito.proxy import TITOProxy
from skyrl.train.generators.tito.renderer import RenderedPrompt
from skyrl.train.generators.tito.trace import Trace
from skyrl.train.generators.tito.types import ModelTurnResult


class _Renderer:
    def render(self, messages, *, tools=None):
        return RenderedPrompt((10, 99), (0, -1))

    def bridge(self, previous_prompt_ids, previous_completion_ids, new_messages, *, tools=None):
        return None

    def parse_response(self, token_ids, *, tools=None):
        return {"role": "assistant", "content": "done"}

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
            "response_ids": [[50]],
            "stop_reasons": ["stop"],
            "response_logprobs": [[-0.2]],
            "prompt_logprobs": None,
            "rollout_expert_indices": None,
        }

    async def finish_session(self, session_id):
        self.finished_sessions.append(session_id)


class _RateLimiter:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False


@pytest.mark.asyncio
async def test_harbor_attempt_uses_registered_proxy_endpoint(monkeypatch):
    pytest.importorskip("harbor")
    from examples.train_integrations.harbor_tito import (
        harbor_generator as harbor_module,
    )

    captured_config = None

    class FakeTrial:
        def __init__(self, config):
            self.config = config

        @classmethod
        async def create(cls, config):
            nonlocal captured_config
            captured_config = config
            return cls(config)

        async def run(self):
            body = {
                "model": "model",
                "messages": self.config["task"]["path"],
                "logprobs": True,
                "return_token_ids": True,
            }
            api_base = self.config["agent"]["kwargs"]["api_base"]
            async with httpx.AsyncClient() as client:
                response = await client.post(f"{api_base}/chat/completions", json=body)
            response.raise_for_status()
            data = response.json()
            choice = data["choices"][0]
            return SimpleNamespace(
                exception_info=None,
                verifier_result=SimpleNamespace(rewards={"reward": 1.0}),
                agent_result=SimpleNamespace(
                    rollout_details=[
                        {
                            "prompt_token_ids": [data["prompt_token_ids"]],
                            "completion_token_ids": [choice["token_ids"]],
                            "logprobs": [[item["logprob"] for item in choice["logprobs"]["content"]]],
                        }
                    ],
                    metadata={"n_episodes": 1},
                ),
            )

    monkeypatch.setattr(harbor_module, "Trial", FakeTrial)
    monkeypatch.setattr(
        harbor_module.TrialConfig,
        "model_validate",
        staticmethod(lambda config: config),
    )

    engine = _InferenceEngine()
    generator = harbor_module.TITOHarborGenerator.__new__(harbor_module.TITOHarborGenerator)
    generator._harbor_trial_config_template = {
        "agent": {
            "kwargs": {
                "llm_kwargs": {},
            }
        }
    }
    generator._served_model_name = "model"
    generator._rate_limiter = _RateLimiter()
    proxy = TITOProxy(engine, _Renderer())

    async with proxy.serving():
        output = await generator._tito_harbor_agent_loop(
            proxy=proxy,
            prompt=[{"role": "user", "content": "hello"}],
            trajectory_id=TrajectoryID("instance", 0),
            cache_salt="salt",
        )

    assert captured_config["agent"]["kwargs"]["api_base"].startswith("http://127.0.0.1:")
    assert output.reward == 1.0
    assert output.trace is not None
    assert len(output.trace.committed_turns()) == 1
    assert len(engine.finished_sessions) == 1


def test_parity_matches_multiple_harbor_segments_to_transitions():
    from examples.train_integrations.harbor_tito.harbor_generator import (
        TITOHarborGenerator,
    )

    trace = Trace()
    calls = [
        ([{"role": "user", "content": "main"}], [1, 9], [0, -1], [2], "main-answer"),
        ([{"role": "user", "content": "aux"}], [3, 9], [0, -1], [4], "aux-answer"),
        ([{"role": "user", "content": "next"}], [5, 9], [0, -1], [6], "next-answer"),
    ]
    for index, (messages, prompt, attribution, completion, content) in enumerate(calls):
        pending = trace.prepare_turn(messages)
        trace.commit(
            pending,
            ModelTurnResult(
                prompt_token_ids=tuple(prompt),
                prompt_message_indices=tuple(attribution),
                reused_prefix_length=0,
                completion_ids=tuple(completion),
                completion_logprobs=(-0.1,),
                assistant_message={"role": "assistant", "content": content},
                stop_reason="stop",
            ),
        )

    transitions = trace.transitions()
    rollout_details = [
        {
            "prompt_token_ids": [
                list(transitions[0].prompt_token_ids),
                list(transitions[2].prompt_token_ids),
            ],
            "completion_token_ids": [
                list(transitions[0].completion_ids),
                list(transitions[2].completion_ids),
            ],
            "logprobs": [
                list(transitions[0].completion_logprobs),
                list(transitions[2].completion_logprobs),
            ],
        },
        {
            "prompt_token_ids": [list(transitions[1].prompt_token_ids)],
            "completion_token_ids": [list(transitions[1].completion_ids)],
            "logprobs": [list(transitions[1].completion_logprobs)],
        },
    ]

    assert TITOHarborGenerator._validate_trace_parity(trace, rollout_details) == [
        [0, 2],
        [1],
    ]


def test_parity_allows_tito_only_truncated_transition():
    from examples.train_integrations.harbor_tito.harbor_generator import (
        TITOHarborGenerator,
    )

    trace = Trace()
    for index, stop_reason in enumerate(("length", "stop")):
        pending = trace.prepare_turn([{"role": "user", "content": f"call-{index}"}])
        trace.commit(
            pending,
            ModelTurnResult(
                prompt_token_ids=(index + 1, 9),
                prompt_message_indices=(0, -1),
                reused_prefix_length=0,
                completion_ids=(index + 10,),
                completion_logprobs=(-0.1,),
                assistant_message={"role": "assistant", "content": f"answer-{index}"},
                stop_reason=stop_reason,
            ),
        )

    successful = trace.transition(1)
    rollout_details = [
        {
            "prompt_token_ids": [list(successful.prompt_token_ids)],
            "completion_token_ids": [list(successful.completion_ids)],
            "logprobs": [list(successful.completion_logprobs)],
        }
    ]

    assert TITOHarborGenerator._validate_trace_parity(trace, rollout_details) == [[1]]


def test_parity_rejects_tito_only_non_truncated_transition():
    from examples.train_integrations.harbor_tito.harbor_generator import (
        TITOHarborGenerator,
    )

    trace = Trace()
    for index in range(2):
        pending = trace.prepare_turn([{"role": "user", "content": f"call-{index}"}])
        trace.commit(
            pending,
            ModelTurnResult(
                prompt_token_ids=(index + 1, 9),
                prompt_message_indices=(0, -1),
                reused_prefix_length=0,
                completion_ids=(index + 10,),
                completion_logprobs=(-0.1,),
                assistant_message={"role": "assistant", "content": f"answer-{index}"},
                stop_reason="stop",
            ),
        )

    successful = trace.transition(1)
    rollout_details = [
        {
            "prompt_token_ids": [list(successful.prompt_token_ids)],
            "completion_token_ids": [list(successful.completion_ids)],
            "logprobs": [list(successful.completion_logprobs)],
        }
    ]

    with pytest.raises(ValueError, match="Non-truncated TITO transitions"):
        TITOHarborGenerator._validate_trace_parity(trace, rollout_details)

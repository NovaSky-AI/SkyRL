import asyncio
from types import SimpleNamespace

import pytest

from skyrl.backends.fireworks.inference import FireworksInferenceClient
from skyrl.backends.fireworks.runtime import FireworksRuntime
from skyrl.train.config import FireworksConfig


class _Future:
    def __init__(self, value):
        self.value = value

    def result(self, timeout=None):
        return self.value


class _Tokenizer:
    def decode(self, tokens, skip_special_tokens=True):
        if not skip_special_tokens:
            return "".join(f"<{token}>" for token in tokens)
        return ":".join(str(token) for token in tokens)

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        chat_template,
        return_dict,
    ):
        assert messages == [{"role": "user", "content": "hello"}]
        assert tokenize is True
        assert add_generation_prompt is True
        assert return_dict is False
        assert chat_template is None
        return [11, 12]


class _Sampler:
    def __init__(self):
        self.prompts = []
        self.closed = False

    async def sample_async(self, *, prompt, num_samples, sampling_params):
        assert num_samples == 1
        self.prompts.append(prompt)
        sequence = SimpleNamespace(
            tokens=[7, 8], logprobs=[-0.7, -0.8], stop_reason="stop"
        )
        return SimpleNamespace(sequences=[sequence])

    def close(self):
        self.closed = True


class _Service:
    def __init__(self):
        self.sampler = _Sampler()
        self._managed_handle = SimpleNamespace(
            deployment=SimpleNamespace(
                inference_model="accounts/test/deployments/rollout"
            ),
            deployment_manager=SimpleNamespace(
                inference_url="https://api.fireworks.ai"
            ),
        )

    def create_sampling_client(self, **kwargs):
        return self.sampler

    def hotload_sampler_snapshot(self, path):
        return None

    def close(self):
        return None


class _TrainingClient:
    def save_weights_for_sampler(self, name):
        return _Future(SimpleNamespace(path=f"snapshot://{name}"))


@pytest.mark.asyncio
async def test_generate_returns_exact_tokens_and_logprobs() -> None:
    runtime = FireworksRuntime(
        service=_Service(),
        training_client=_TrainingClient(),
        tokenizer=_Tokenizer(),
        config=FireworksConfig(),
    )
    await runtime.publish_sampler_weights()
    client = FireworksInferenceClient(
        runtime=runtime,
        default_sampling_params={
            "max_tokens": 8,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "logprobs": 1,
        },
    )

    output = await client.generate(
        {
            "prompt_token_ids": [[1, 2], [3, 4]],
            "sampling_params": None,
            "session_ids": ["a", "b"],
            "prompts": None,
            "mm_features": None,
            "cache_salt": None,
        }
    )

    assert output["responses"] == ["7:8", "7:8"]
    assert output["response_ids"] == [[7, 8], [7, 8]]
    assert output["response_logprobs"] == [[-0.7, -0.8], [-0.7, -0.8]]
    assert output["stop_reasons"] == ["stop", "stop"]
    assert client.weight_version == 0
    await client.teardown()


def test_client_exposes_native_openai_endpoint() -> None:
    runtime = FireworksRuntime(
        service=_Service(),
        training_client=_TrainingClient(),
        tokenizer=_Tokenizer(),
        config=FireworksConfig(),
    )
    client = FireworksInferenceClient(runtime=runtime, default_sampling_params={})

    assert client.get_endpoint_url() == "https://api.fireworks.ai/inference/v1"
    assert client.model_name == "accounts/test/deployments/rollout"


@pytest.mark.asyncio
async def test_generate_requires_published_sampler() -> None:
    runtime = FireworksRuntime(
        service=_Service(),
        training_client=_TrainingClient(),
        tokenizer=_Tokenizer(),
        config=FireworksConfig(),
    )
    client = FireworksInferenceClient(
        runtime=runtime, default_sampling_params={"max_tokens": 8}
    )

    with pytest.raises(RuntimeError, match="have not been published"):
        await client.generate(
            {
                "prompt_token_ids": [[1]],
                "sampling_params": None,
                "session_ids": None,
                "prompts": None,
                "mm_features": None,
                "cache_salt": None,
            }
        )
    await runtime.close()


@pytest.mark.asyncio
async def test_pause_generation_blocks_new_admissions() -> None:
    service = _Service()
    runtime = FireworksRuntime(
        service=service,
        training_client=_TrainingClient(),
        tokenizer=_Tokenizer(),
        config=FireworksConfig(),
    )
    await runtime.publish_sampler_weights()
    client = FireworksInferenceClient(
        runtime=runtime, default_sampling_params={"max_tokens": 8}
    )
    await client.pause_generation()

    task = asyncio.create_task(
        client.generate(
            {
                "prompt_token_ids": [[1]],
                "sampling_params": None,
                "session_ids": None,
                "prompts": None,
                "mm_features": None,
                "cache_salt": None,
            }
        )
    )
    await asyncio.sleep(0.01)
    assert service.sampler.prompts == []

    await client.resume_generation()
    await task
    assert len(service.sampler.prompts) == 1
    await runtime.close()

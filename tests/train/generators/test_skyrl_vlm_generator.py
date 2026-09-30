"""
CPU tests for SkyRLGymGenerator with ``generator.vision_language_generator=True``.

The inference server's ``/v1/chat/completions/render`` is mocked with the model's HF chat template.
Each image expands to ``IMAGE_TOKENS`` placeholder tokens and carries its URL as its hash and
serialized kwargs, so tests can check which images reach the engine and the trainer.

uv run --extra dev --isolated pytest tests/train/generators/test_skyrl_vlm_generator.py -v
"""

import copy
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from loguru import logger
from transformers import AutoTokenizer

from skyrl.train.config import (
    ChatTemplateConfig,
    GeneratorConfig,
    SamplingParams,
    SkyRLGymConfig,
)
from skyrl.train.generators.base import GeneratorInput, GeneratorOutput
from skyrl.train.generators.chat_renderer import (
    VLLMChatRenderer,
    append_mm_features,
    shift_mm_features,
    truncate_mm_features,
)
from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator
from skyrl_gym.envs import deregister, register
from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.registration import registry

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
IMAGE_TOKENS = 4
DECODE_MM_KWARGS = "skyrl.backends.renderer.decode_mm_kwargs"


def _image_message(url: str, text: str) -> Dict[str, Any]:
    return {
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": text}],
    }


# ---------------------------------------------------------------------------
# Test environments
# ---------------------------------------------------------------------------


class CPUVLMTestEnv(BaseTextEnv):
    """3-turn env with text observations."""

    def __init__(self, env_config: Any, extras: Dict[str, Any] = {}):
        super().__init__()
        self.max_turns = 3

    def init(self, prompt):
        return prompt, {}

    def step(self, action: str):
        self.turns += 1
        done = self.turns >= self.max_turns
        return BaseTextEnvStepOutput(
            observations=[{"role": "user", "content": f"{self.turns}"}] if not done else [],
            reward=1.0 if done else 0.0,
            done=done,
            metadata={},
        )


class CPUVLMImageObsEnv(BaseTextEnv):
    """3-turn env that returns a new image observation after every step, including the last."""

    def __init__(self, env_config: Any, extras: Dict[str, Any] = {}):
        super().__init__()
        self.max_turns = 3

    def init(self, prompt):
        return [*prompt, _image_message("img://prompt", "look")], {}

    def step(self, action: str):
        self.turns += 1
        done = self.turns >= self.max_turns
        return BaseTextEnvStepOutput(
            observations=[_image_message(f"img://obs{self.turns}", f"step {self.turns}")],
            reward=1.0 if done else 0.0,
            done=done,
            metadata={},
        )


for _env_id, _cls in (("cpu_vlm_test_env", "CPUVLMTestEnv"), ("cpu_vlm_image_obs_env", "CPUVLMImageObsEnv")):
    if _env_id not in registry:
        register(id=_env_id, entry_point=f"tests.train.generators.test_skyrl_vlm_generator:{_cls}")


@pytest.fixture(autouse=True, scope="module")
def deregister_test_env():
    yield
    deregister("cpu_vlm_test_env")
    deregister("cpu_vlm_image_obs_env")


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generator_cfg(**overrides) -> GeneratorConfig:
    kwargs = dict(
        sampling_params=SamplingParams(max_generate_length=200, logprobs=None),
        max_input_length=4096,
        batched=False,
        max_turns=3,
        zero_reward_on_non_stop=False,
        apply_overlong_filtering=False,
        use_conversation_multi_turn=True,
        chat_template=ChatTemplateConfig(source="name", name_or_path=None),
        step_wise_trajectories=False,
        vision_language_generator=True,
    )
    kwargs.update(overrides)
    return GeneratorConfig(**kwargs)


class MockRenderServer:
    """Stands in for vLLM's /render: HF chat template plus a fixed-size placeholder per image."""

    def __init__(self, tokenizer, image_numbering: bool = False):
        self.tokenizer = tokenizer
        self.image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.image_numbering = image_numbering
        self.requests: List[Dict[str, Any]] = []

    async def __call__(self, request_payload):
        body = request_payload["json"]
        self.requests.append(copy.deepcopy(body))
        messages = body["messages"]
        chat_template_kwargs = dict(body.get("chat_template_kwargs") or {})
        if self.image_numbering:
            chat_template_kwargs["add_vision_id"] = True
        token_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=body.get("add_generation_prompt", True),
            tokenize=True,
            return_dict=False,
            **chat_template_kwargs,
        )
        urls = [
            part["image_url"]["url"]
            for m in messages
            if isinstance(m["content"], list)
            for part in m["content"]
            if part.get("type") == "image_url"
        ]
        expanded: List[int] = []
        placeholders = []
        for t in token_ids:
            if t == self.image_pad_id:
                placeholders.append({"offset": len(expanded), "length": IMAGE_TOKENS})
                expanded.extend([self.image_pad_id] * IMAGE_TOKENS)
            else:
                expanded.append(t)
        assert len(placeholders) == len(urls)
        features = (
            {
                "mm_hashes": {"image": list(urls)},
                "mm_placeholders": {"image": placeholders},
                "kwargs_data": {"image": [f"kwargs:{u}" for u in urls]},
            }
            if urls
            else None
        )
        return {"token_ids": expanded, "features": features}


class MockLLM:
    """Records every generate request and returns a fixed response per turn."""

    def __init__(self, tokenizer, response_text: str = "b", response_suffix: Optional[str] = None):
        self.response_text = response_text
        # Text returned by the engine; defaults to the response without the eos string.
        self.returned_text = response_text if response_suffix is None else response_text + response_suffix
        self.response_ids = tokenizer.encode(response_text + tokenizer.eos_token, add_special_tokens=False)
        self.requests: List[Dict[str, Any]] = []

    async def __call__(self, input_batch, model=None):
        self.requests.append(copy.deepcopy(input_batch))
        num_prompts = len(input_batch["prompt_token_ids"])
        return {
            "responses": [self.returned_text] * num_prompts,
            "stop_reasons": ["stop"] * num_prompts,
            "response_logprobs": None,
            "response_ids": [list(self.response_ids) for _ in range(num_prompts)],
        }


def _build_generator(tokenizer, render: MockRenderServer, llm: MockLLM, **cfg_overrides) -> SkyRLGymGenerator:
    mock_client = MagicMock()
    mock_client.model_name = MODEL_NAME
    mock_client.finish_session = AsyncMock()
    mock_client.render_chat_completion = AsyncMock(side_effect=render.__call__)
    mock_client.generate = AsyncMock(side_effect=llm.__call__)
    return SkyRLGymGenerator(
        generator_cfg=_generator_cfg(**cfg_overrides),
        skyrl_gym_cfg=SkyRLGymConfig(max_env_workers=0),
        inference_engine_client=mock_client,
        tokenizer=tokenizer,
    )


def _input_batch(env_class: str = "cpu_vlm_test_env", sampling_params=None) -> GeneratorInput:
    batch: GeneratorInput = {
        "prompts": [[{"role": "user", "content": "a"}]],
        "env_extras": [{"answer": "4"}],
        "env_classes": [env_class],
    }
    if sampling_params is not None:
        batch["sampling_params"] = sampling_params
    return batch


def _fake_decode(kwargs_data):
    """Records which serialized images reached the trainer."""
    return {"pixel_values": list((kwargs_data or {}).get("image", [])) or None, "image_grid_thw": None}


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"step_wise_trajectories": True}, "step_wise_trajectories"),
        ({"use_conversation_multi_turn": False}, "use_conversation_multi_turn"),
        ({"chat_template": ChatTemplateConfig(source="name", name_or_path="qwen3_without_thinking")}, "custom chat"),
        ({"vision_language_generator": False, "vision_language_rerender_check": True}, "rerender_check"),
    ],
)
def test_vlm_validate_cfg_refusals(tokenizer, overrides, match):
    with pytest.raises(ValueError, match=match):
        _build_generator(tokenizer, MockRenderServer(tokenizer), MockLLM(tokenizer), **overrides)


# ---------------------------------------------------------------------------
# Text observations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch(DECODE_MM_KWARGS, side_effect=_fake_decode)
async def test_vlm_text_obs_are_token_in_token_out(_mock_decode, tokenizer):
    """Each turn's engine prompt extends the previous one by the generated ids plus the observation."""
    render, llm = MockRenderServer(tokenizer), MockLLM(tokenizer)
    generator = _build_generator(tokenizer, render, llm)
    output: GeneratorOutput = await generator.generate(_input_batch())

    prompts = [r["prompt_token_ids"][0] for r in llm.requests]
    assert len(prompts) == 3
    for prev, cur in zip(prompts, prompts[1:]):
        assert cur[: len(prev)] == prev
        assert cur[len(prev) : len(prev) + len(llm.response_ids)] == llm.response_ids

    response_ids = output["response_ids"][0]
    loss_mask = output["loss_masks"][0]
    assert output["prompt_token_ids"][0] + response_ids == prompts[-1] + llm.response_ids
    assert [t for t, m in zip(response_ids, loss_mask) if m] == llm.response_ids * 3
    # Earlier turns are never re-rendered: no render request contains an assistant message.
    assert not any(m["role"] == "assistant" for req in render.requests for m in req["messages"])

    rewards = output["rewards"][0]
    assert len(rewards) == len(response_ids)
    assert [(i, r) for i, r in enumerate(rewards) if r] == [(len(response_ids) - 1, 1.0)]
    assert "pixel_values" not in output


@pytest.mark.asyncio
@patch(DECODE_MM_KWARGS, side_effect=_fake_decode)
async def test_vlm_text_obs_tokens_match_the_chat_template(_mock_decode, tokenizer):
    """The observation tokens equal what the chat template renders for the full conversation."""
    llm = MockLLM(tokenizer)
    generator = _build_generator(tokenizer, MockRenderServer(tokenizer), llm)
    output: GeneratorOutput = await generator.generate(_input_batch())

    conversation = [{"role": "user", "content": "a"}]
    for turn in range(1, 3):
        conversation += [{"role": "assistant", "content": llm.response_text}, {"role": "user", "content": f"{turn}"}]
    expected = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True, return_dict=False)
    assert llm.requests[-1]["prompt_token_ids"][0] == expected
    assert output["prompt_token_ids"][0] == tokenizer.apply_chat_template(
        conversation[:1], add_generation_prompt=True, tokenize=True, return_dict=False
    )


@pytest.mark.asyncio
@patch(DECODE_MM_KWARGS, side_effect=_fake_decode)
async def test_vlm_response_text_with_eos_does_not_double_eos(_mock_decode, tokenizer):
    llm = MockLLM(tokenizer, response_suffix=tokenizer.eos_token)
    generator = _build_generator(tokenizer, MockRenderServer(tokenizer), llm)
    output: GeneratorOutput = await generator.generate(_input_batch())

    eos_id = tokenizer.eos_token_id
    ids = output["response_ids"][0]
    assert all(not (a == eos_id and b == eos_id) for a, b in zip(ids, ids[1:])), "doubled eos in response_ids"


@pytest.mark.asyncio
@patch(DECODE_MM_KWARGS, side_effect=_fake_decode)
async def test_vlm_sampling_params_are_passed_through(_mock_decode, tokenizer):
    """Like the text path, the engine gets the request's sampling params (None means engine defaults)."""
    llm = MockLLM(tokenizer)
    generator = _build_generator(tokenizer, MockRenderServer(tokenizer), llm)
    await generator.generate(_input_batch())
    assert all(r["sampling_params"] is None for r in llm.requests)

    llm.requests.clear()
    await generator.generate(_input_batch(sampling_params={"max_tokens": 7, "temperature": 0.5}))
    assert all(r["sampling_params"] == {"max_tokens": 7, "temperature": 0.5} for r in llm.requests)


@pytest.mark.asyncio
@patch(DECODE_MM_KWARGS, side_effect=_fake_decode)
async def test_vlm_chat_template_kwargs_reach_the_render_request(_mock_decode, tokenizer):
    render = MockRenderServer(tokenizer)
    generator = _build_generator(tokenizer, render, MockLLM(tokenizer), chat_template_kwargs={"foo": "bar"})
    await generator.generate(_input_batch())
    assert render.requests and all(r["chat_template_kwargs"] == {"foo": "bar"} for r in render.requests)
    assert all(r["model"] == MODEL_NAME for r in render.requests)


# ---------------------------------------------------------------------------
# Image observations
# ---------------------------------------------------------------------------


def _placeholder_runs(token_ids: List[int], image_pad_id: int) -> List[Dict[str, int]]:
    runs, i = [], 0
    while i < len(token_ids):
        if token_ids[i] == image_pad_id:
            start = i
            while i < len(token_ids) and token_ids[i] == image_pad_id:
                i += 1
            runs.append({"offset": start, "length": i - start})
        else:
            i += 1
    return runs


@pytest.mark.asyncio
@patch(DECODE_MM_KWARGS, side_effect=_fake_decode)
async def test_vlm_image_obs_features_accumulate(_mock_decode, tokenizer):
    """Each generate call carries every image so far, with offsets at its placeholder tokens."""
    render, llm = MockRenderServer(tokenizer), MockLLM(tokenizer)
    generator = _build_generator(tokenizer, render, llm)
    output: GeneratorOutput = await generator.generate(_input_batch("cpu_vlm_image_obs_env"))

    urls = ["img://prompt", "img://obs1", "img://obs2"]
    assert len(llm.requests) == 3
    for turn, request in enumerate(llm.requests):
        prompt_ids = request["prompt_token_ids"][0]
        features = request["mm_features"][0]
        assert features["mm_placeholders"]["image"] == _placeholder_runs(prompt_ids, render.image_pad_id)
        assert features["mm_hashes"]["image"] == urls[: turn + 1]
        assert features["kwargs_data"]["image"] == [f"kwargs:{u}" for u in urls[: turn + 1]]

    # The final observation (with img://obs3) is dropped from the response, and so is its image.
    sequence = output["prompt_token_ids"][0] + output["response_ids"][0]
    assert len(_placeholder_runs(sequence, render.image_pad_id)) == 3
    assert output["pixel_values"] == [[f"kwargs:{u}" for u in urls]]


@pytest.mark.asyncio
@patch(DECODE_MM_KWARGS, side_effect=_fake_decode)
async def test_vlm_length_stop_drops_the_last_observation_image(_mock_decode, tokenizer):
    full_llm = MockLLM(tokenizer)
    await _build_generator(tokenizer, MockRenderServer(tokenizer), full_llm).generate(
        _input_batch("cpu_vlm_image_obs_env")
    )
    second_turn_prompt_length = len(full_llm.requests[1]["prompt_token_ids"][0])

    # The third turn's prompt exceeds the limit, so the loop stops after two turns.
    render, llm = MockRenderServer(tokenizer), MockLLM(tokenizer)
    generator = _build_generator(tokenizer, render, llm, max_input_length=second_turn_prompt_length)
    output: GeneratorOutput = await generator.generate(_input_batch("cpu_vlm_image_obs_env"))

    assert output["stop_reasons"][0] == "length"
    assert len(llm.requests) == 2
    sequence = output["prompt_token_ids"][0] + output["response_ids"][0]
    assert len(_placeholder_runs(sequence, render.image_pad_id)) == 2
    assert output["pixel_values"] == [["kwargs:img://prompt", "kwargs:img://obs1"]]


@pytest.mark.asyncio
@patch(DECODE_MM_KWARGS, side_effect=_fake_decode)
async def test_vlm_refuses_image_numbering_templates(_mock_decode, tokenizer):
    """Qwen-VL's add_vision_id numbers images across the conversation, which appending cannot reproduce."""
    generator = _build_generator(tokenizer, MockRenderServer(tokenizer, image_numbering=True), MockLLM(tokenizer))
    with pytest.raises(ValueError, match="add_vision_id"):
        await generator.generate(_input_batch("cpu_vlm_image_obs_env"))


@pytest.mark.asyncio
@patch(DECODE_MM_KWARGS, side_effect=_fake_decode)
async def test_vlm_rerender_check_logs_a_mismatch(_mock_decode, tokenizer):
    # The engine returns ids that re-tokenizing the text does not reproduce.
    llm = MockLLM(tokenizer, response_text="hello")
    llm.response_ids = tokenizer.encode("hel", add_special_tokens=False) + tokenizer.encode(
        "lo" + tokenizer.eos_token, add_special_tokens=False
    )
    assert llm.response_ids != tokenizer.encode("hello" + tokenizer.eos_token, add_special_tokens=False)
    generator = _build_generator(tokenizer, MockRenderServer(tokenizer), llm, vision_language_rerender_check=True)
    messages = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        await generator.generate(_input_batch())
    finally:
        logger.remove(handler_id)
    assert any("differs from the token-in-token-out sequence" in m for m in messages)


@pytest.mark.asyncio
@patch(DECODE_MM_KWARGS, side_effect=_fake_decode)
async def test_vlm_generate_batched(_mock_decode, tokenizer):
    """Batched generation renders each prompt through /render and sends and returns its images."""
    render, llm = MockRenderServer(tokenizer), MockLLM(tokenizer)
    generator = _build_generator(tokenizer, render, llm, batched=True, max_turns=1)
    prompts = [
        [_image_message("img://a", "first")],
        [{"role": "user", "content": "text only"}],
        [_image_message("img://b", "second"), _image_message("img://c", "third")],
    ]
    output: GeneratorOutput = await generator.generate(
        {"prompts": prompts, "env_extras": [{}] * 3, "env_classes": ["cpu_vlm_test_env"] * 3}
    )

    assert len(llm.requests) == 1
    request = llm.requests[0]
    assert request["prompt_token_ids"] == output["prompt_token_ids"]
    expected_urls = [["img://a"], [], ["img://b", "img://c"]]
    for prompt_ids, features, urls in zip(request["prompt_token_ids"], request["mm_features"], expected_urls):
        if not urls:
            assert features is None
            continue
        assert features["mm_placeholders"]["image"] == _placeholder_runs(prompt_ids, render.image_pad_id)
        assert features["mm_hashes"]["image"] == urls
    assert output["pixel_values"] == [["kwargs:img://a"], None, ["kwargs:img://b", "kwargs:img://c"]]
    assert output["response_ids"] == [llm.response_ids] * 3


# ---------------------------------------------------------------------------
# VLLMChatRenderer and feature helpers
# ---------------------------------------------------------------------------


def _renderer(render_fn) -> VLLMChatRenderer:
    client = MagicMock()
    client.render_chat_completion = AsyncMock(side_effect=render_fn)
    return VLLMChatRenderer(
        client=client,
        base_conversation=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        eos_token_id=9,
    )


@pytest.mark.asyncio
async def test_renderer_raises_when_the_render_does_not_start_with_the_base():
    async def render(payload):
        messages = payload["json"]["messages"]
        # The base renders as [1, 2, 9]; with an observation after it the base renders differently.
        return {"token_ids": [1, 2, 9] if len(messages) == 2 else [1, 3, 9, 5, 9], "features": None}

    with pytest.raises(ValueError, match="does not start with the rendered base"):
        await _renderer(render).render_observation([{"role": "user", "content": "o"}], is_done=False)


@pytest.mark.asyncio
async def test_renderer_raises_on_a_placeholder_inside_the_base():
    async def render(payload):
        messages = payload["json"]["messages"]
        if len(messages) == 2:
            return {"token_ids": [1, 2, 9], "features": None}
        features = {"mm_hashes": {"image": ["h"]}, "mm_placeholders": {"image": [{"offset": 1, "length": 1}]}}
        return {"token_ids": [1, 2, 9, 5, 9], "features": {**features, "kwargs_data": None}}

    with pytest.raises(ValueError, match="inside the base conversation"):
        await _renderer(render).render_observation([{"role": "user", "content": "o"}], is_done=False)


@pytest.mark.asyncio
async def test_renderer_drops_tokens_after_the_base_eos_and_skips_empty_done_obs():
    async def render(payload):
        messages = payload["json"]["messages"]
        # The base ends with eos (9) followed by a newline token (8), which belongs to the observation.
        return {"token_ids": [1, 2, 9, 8] if len(messages) == 2 else [1, 2, 9, 8, 5, 9, 8, 7], "features": None}

    renderer = _renderer(render)
    obs = await renderer.render_observation([{"role": "user", "content": "o"}], is_done=False)
    assert obs.token_ids == [8, 5, 9, 8, 7]
    assert (await renderer.render_observation([], is_done=True)).token_ids == []


def _features(offsets, names):
    return {
        "mm_hashes": {"image": list(names)},
        "mm_placeholders": {"image": [{"offset": o, "length": 2} for o in offsets]},
        "kwargs_data": {"image": [f"k{n}" for n in names]},
    }


def test_shift_and_append_mm_features():
    first = _features([3], ["a"])
    second = shift_mm_features(_features([1], ["b"]), 10)
    merged = append_mm_features(first, second)
    assert merged == {
        "mm_hashes": {"image": ["a", "b"]},
        "mm_placeholders": {"image": [{"offset": 3, "length": 2}, {"offset": 11, "length": 2}]},
        "kwargs_data": {"image": ["ka", "kb"]},
    }
    # Inputs are not modified.
    assert first == _features([3], ["a"])
    assert append_mm_features(None, None) is None
    assert append_mm_features(first, None) is first


def test_truncate_mm_features():
    features = _features([2, 10], ["a", "b"])
    assert truncate_mm_features(features, 20) == features
    assert truncate_mm_features(features, 10) == _features([2], ["a"])
    assert truncate_mm_features(features, 2) is None
    with pytest.raises(ValueError, match="straddles"):
        truncate_mm_features(features, 11)

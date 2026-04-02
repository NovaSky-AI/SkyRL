"""
uv run --extra dev --extra skyrl-train --isolated pytest tests/train/generators/test_output_builders.py
"""

import pytest

from skyrl.train.generators.output_builders import (
    RetokenizedTrajectory,
    TokenizedTrajectory,
    build_generator_output_from_messages,
    build_generator_output_from_tokenized,
)
from skyrl.train.generators.utils import (
    encode_messages_subset,
    get_generation_prompt_ids,
    get_response_ids_and_loss_mask_from_messages,
)

dummy_chat_template = (
    "{%- for message in messages %}"
    "{%- if message['role'] == 'user' %}"
    "<USER>{{ message['content'] }}</s>\n"
    "{%- elif message['role'] == 'assistant' %}"
    "<ASSISTANT>{{ message['content'] }}</s>\n"
    "{%- elif message['role'] == 'system' %}"
    "<SYSTEM>{{ message['content'] }}</s>\n"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "<ASSISTANT>"
    "{%- endif %}"
)


class FakeTokenizer:
    eos_token_id = 0

    def apply_chat_template(
        self,
        messages,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=False,
        chat_template=None,
    ):
        role_prefixes = {
            "system": [101],
            "user": [102],
            "assistant": [103],
        }
        token_ids = []
        for message in messages:
            token_ids.extend(role_prefixes[message["role"]])
            token_ids.extend(ord(char) + 200 for char in message["content"])
            token_ids.append(self.eos_token_id)
        if add_generation_prompt:
            token_ids.extend(role_prefixes["assistant"])
        return token_ids


@pytest.fixture
def tokenizer_w_dummy_template():
    return FakeTokenizer()


def _assistant_generated_token_count(message, tokenizer, chat_template):
    generation_prompt_ids = get_generation_prompt_ids(tokenizer, chat_template=chat_template)
    cur_token_ids = encode_messages_subset([message], tokenizer, chat_template=chat_template)
    if tokenizer.eos_token_id in cur_token_ids:
        last_eos_idx = len(cur_token_ids) - 1 - cur_token_ids[::-1].index(tokenizer.eos_token_id)
        return last_eos_idx + 1 - len(generation_prompt_ids)
    return len(cur_token_ids) - len(generation_prompt_ids)


def test_build_generator_output_from_messages_matches_low_level_helper(tokenizer_w_dummy_template):
    prompt_messages = [{"role": "user", "content": "What is 2 + 2?"}]
    response_messages = [
        {"role": "assistant", "content": "The answer is 4."},
        {"role": "user", "content": "Explain why."},
        {"role": "assistant", "content": "Because 2 + 2 sums to 4."},
    ]

    output = build_generator_output_from_messages(
        [
            RetokenizedTrajectory(
                prompt_messages=prompt_messages,
                response_messages=response_messages,
                reward=1.0,
            )
        ],
        tokenizer_w_dummy_template,
        chat_template=dummy_chat_template,
    )

    expected_prompt_token_ids = tokenizer_w_dummy_template.apply_chat_template(
        prompt_messages,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=False,
        chat_template=dummy_chat_template,
    )
    expected_response_ids, expected_loss_mask, expected_rollout_logprobs = get_response_ids_and_loss_mask_from_messages(
        response_messages,
        tokenizer_w_dummy_template,
        chat_template=dummy_chat_template,
    )

    assert output["prompt_token_ids"] == [expected_prompt_token_ids]
    assert output["response_ids"] == [expected_response_ids]
    assert output["loss_masks"] == [expected_loss_mask]
    assert output["rewards"] == [1.0]
    assert output["stop_reasons"] is None
    assert output["rollout_logprobs"] == expected_rollout_logprobs
    assert output["rollout_metrics"] is not None


def test_build_generator_output_from_messages_supports_assistant_message_logprobs(tokenizer_w_dummy_template):
    prompt_messages = [{"role": "user", "content": "Say hi."}]
    response_messages = [
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "Say bye."},
        {"role": "assistant", "content": "Bye now!"},
    ]
    assistant_messages = [message for message in response_messages if message["role"] == "assistant"]
    assistant_logprobs = [
        [-(i + 1) / 10.0 for i in range(_assistant_generated_token_count(message, tokenizer_w_dummy_template, dummy_chat_template))]
        for message in assistant_messages
    ]

    output = build_generator_output_from_messages(
        [
            RetokenizedTrajectory(
                prompt_messages=prompt_messages,
                response_messages=response_messages,
                reward=1.0,
                assistant_message_logprobs=assistant_logprobs,
            )
        ],
        tokenizer_w_dummy_template,
        chat_template=dummy_chat_template,
    )

    _, _, expected_rollout_logprobs = get_response_ids_and_loss_mask_from_messages(
        response_messages,
        tokenizer_w_dummy_template,
        assistant_logprobs=assistant_logprobs,
        chat_template=dummy_chat_template,
    )

    assert output["rollout_logprobs"] == [expected_rollout_logprobs]


def test_build_generator_output_from_tokenized_computes_metrics_and_preserves_optional_fields():
    trajectories = [
        TokenizedTrajectory(
            prompt_token_ids=[1, 2],
            response_ids=[3, 4],
            loss_mask=[1, 1],
            reward=1.0,
            stop_reason="complete",
            rollout_logprobs=[-0.1, -0.2],
            rollout_expert_indices=[[[0]], [[1]], [[2]], [[3]]],
        ),
        TokenizedTrajectory(
            prompt_token_ids=[5],
            response_ids=[6, 7, 8],
            loss_mask=[1, 0, 1],
            reward=0.5,
            stop_reason="length",
            rollout_logprobs=[-0.3, -0.4, -0.5],
            rollout_expert_indices=[[[4]], [[5]], [[6]], [[7]]],
        ),
    ]

    output = build_generator_output_from_tokenized(trajectories)

    assert output["prompt_token_ids"] == [[1, 2], [5]]
    assert output["response_ids"] == [[3, 4], [6, 7, 8]]
    assert output["loss_masks"] == [[1, 1], [1, 0, 1]]
    assert output["stop_reasons"] == ["complete", "length"]
    assert output["rollout_logprobs"] == [[-0.1, -0.2], [-0.3, -0.4, -0.5]]
    assert output["rollout_expert_indices"] == [
        [[[0]], [[1]], [[2]], [[3]]],
        [[[4]], [[5]], [[6]], [[7]]],
    ]
    assert output["rollout_metrics"] is not None
    assert "generate/avg_num_tokens" in output["rollout_metrics"]


def test_build_generator_output_from_tokenized_preserves_token_level_rewards():
    output = build_generator_output_from_tokenized(
        [
            TokenizedTrajectory(
                prompt_token_ids=[1],
                response_ids=[2, 3],
                loss_mask=[1, 1],
                reward=[0.0, 1.0],
            )
        ]
    )

    assert output["rewards"] == [[0.0, 1.0]]


def test_build_generator_output_from_tokenized_respects_rollout_metrics_override():
    output = build_generator_output_from_tokenized(
        [
            TokenizedTrajectory(
                prompt_token_ids=[1],
                response_ids=[2],
                loss_mask=[1],
                reward=1.0,
            )
        ],
        rollout_metrics={"custom/metric": 123.0},
    )

    assert output["rollout_metrics"] == {"custom/metric": 123.0}


def test_build_generator_output_from_tokenized_requires_all_or_none_stop_reasons():
    with pytest.raises(ValueError, match="stop_reasons"):
        build_generator_output_from_tokenized(
            [
                TokenizedTrajectory(prompt_token_ids=[1], response_ids=[2], loss_mask=[1], reward=1.0),
                TokenizedTrajectory(
                    prompt_token_ids=[3],
                    response_ids=[4],
                    loss_mask=[1],
                    reward=0.0,
                    stop_reason="complete",
                ),
            ]
        )


def test_build_generator_output_from_tokenized_requires_all_or_none_rollout_logprobs():
    with pytest.raises(ValueError, match="rollout_logprobs"):
        build_generator_output_from_tokenized(
            [
                TokenizedTrajectory(
                    prompt_token_ids=[1],
                    response_ids=[2],
                    loss_mask=[1],
                    reward=1.0,
                    rollout_logprobs=[-0.1],
                ),
                TokenizedTrajectory(prompt_token_ids=[3], response_ids=[4], loss_mask=[1], reward=0.0),
            ]
        )


def test_build_generator_output_from_tokenized_validates_rollout_expert_index_length():
    with pytest.raises(ValueError, match="rollout_expert_indices"):
        build_generator_output_from_tokenized(
            [
                TokenizedTrajectory(
                    prompt_token_ids=[1, 2],
                    response_ids=[3, 4],
                    loss_mask=[1, 1],
                    reward=1.0,
                    rollout_expert_indices=[[[0]], [[1]], [[2]]],
                )
            ]
        )

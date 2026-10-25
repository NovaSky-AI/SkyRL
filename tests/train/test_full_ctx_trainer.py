"""
uv run --isolated --extra skyrl-train --extra dev pytest tests/train/test_full_ctx_trainer.py
"""

import random
from unittest.mock import MagicMock

import torch

from examples.train_scripts.full_context.trainer_full_ctx import (
    make_dummy_generator_output,
)
from skyrl.train.trainer import RayPPOTrainer
from tests.train.util import example_dummy_config

VOCAB_SIZE = 1000
NUM_PROMPTS = 3
N_SAMPLES_PER_PROMPT = 4
PROMPT_LENGTH = 64
RESPONSE_LENGTH = 128


def _make_batch(seed: int):
    return make_dummy_generator_output(
        rng=random.Random(seed),
        vocab_size=VOCAB_SIZE,
        num_prompts=NUM_PROMPTS,
        n_samples_per_prompt=N_SAMPLES_PER_PROMPT,
        prompt_length=PROMPT_LENGTH,
        response_length=RESPONSE_LENGTH,
    )


def test_dummy_batch_layout():
    """Checks the batch shapes, uids, rewards and loss masks."""
    output, uids = _make_batch(seed=0)
    num_samples = NUM_PROMPTS * N_SAMPLES_PER_PROMPT

    assert uids == [str(i) for i in range(NUM_PROMPTS) for _ in range(N_SAMPLES_PER_PROMPT)]
    assert len(output["prompt_token_ids"]) == num_samples
    assert len(output["response_ids"]) == num_samples
    assert all(len(prompt) == PROMPT_LENGTH for prompt in output["prompt_token_ids"])
    assert all(len(response) == RESPONSE_LENGTH for response in output["response_ids"])
    assert output["loss_masks"] == [[1] * RESPONSE_LENGTH] * num_samples
    assert len(output["rewards"]) == num_samples
    for reward in output["rewards"]:
        assert reward[:-1] == [0] * (RESPONSE_LENGTH - 1)
        assert reward[-1] in (0, 1)


def test_dummy_batch_random_tokens():
    """Checks that token ids are in range and vary across positions, and that each group shares its prompt."""
    output, _ = _make_batch(seed=0)
    sequences = output["prompt_token_ids"] + output["response_ids"]

    assert all(0 <= token_id < VOCAB_SIZE for sequence in sequences for token_id in sequence)
    # Not a single repeated token per sequence.
    assert all(len(set(sequence)) > 1 for sequence in sequences)

    prompts = output["prompt_token_ids"]
    groups = [prompts[i : i + N_SAMPLES_PER_PROMPT] for i in range(0, len(prompts), N_SAMPLES_PER_PROMPT)]
    for group in groups:
        assert all(prompt == group[0] for prompt in group)
    assert len({tuple(group[0]) for group in groups}) == NUM_PROMPTS
    assert len({tuple(response) for response in output["response_ids"]}) == len(output["response_ids"])


def test_dummy_batch_seeded():
    """Checks that the batch is reproducible for a given seed."""
    assert _make_batch(seed=0) == _make_batch(seed=0)
    assert _make_batch(seed=0) != _make_batch(seed=1)


def test_dummy_batch_converts_to_full_length_training_input():
    """Checks that every sequence in the training batch is full length with no padding."""
    cfg = example_dummy_config()
    cfg.trainer.train_batch_size = NUM_PROMPTS
    cfg.trainer.policy_mini_batch_size = NUM_PROMPTS
    cfg.generator.n_samples_per_prompt = N_SAMPLES_PER_PROMPT
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    trainer = RayPPOTrainer(
        cfg=cfg,
        tracker=None,
        tokenizer=tokenizer,
        train_dataset=None,
        eval_dataset=None,
        inference_engine_client=None,
        generator=MagicMock(),
    )
    trainer.dispatch = MagicMock()
    trainer.dispatch.get_lcm_dp_size.return_value = 1

    output, uids = _make_batch(seed=0)
    training_input = trainer.convert_to_training_input(output, uids)

    num_samples = NUM_PROMPTS * N_SAMPLES_PER_PROMPT
    expected_sequences = torch.tensor(
        [prompt + response for prompt, response in zip(output["prompt_token_ids"], output["response_ids"])]
    )
    assert torch.equal(training_input["sequences"], expected_sequences)
    assert torch.equal(training_input["attention_mask"], torch.ones(num_samples, PROMPT_LENGTH + RESPONSE_LENGTH))
    assert torch.equal(training_input["response_mask"], torch.ones(num_samples, RESPONSE_LENGTH))
    assert torch.equal(training_input["loss_mask"], torch.ones(num_samples, RESPONSE_LENGTH))

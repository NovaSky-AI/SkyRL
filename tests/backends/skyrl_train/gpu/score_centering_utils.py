"""Training batches for the score-centering GPU tests, shared by the FSDP and Megatron suites."""

import functools
import math

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.utils.sample_support import (
    SAMPLE_SUPPORT_FIELD,
    CompactSampleSupport,
)
from skyrl.train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl.utils.tok import get_tokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"
# Selected token, its strongest competitor, and the strongest token in the other TP=2 vocabulary half.
SUPPORT_WIDTH = 3
# The sampler column is the model's own support distribution mixed with uniform, so most ratios sit
# inside the rollout_is band while the sampler and trainer still disagree.
SAMPLER_MODEL_WEIGHT = 0.9


@functools.cache
def response_supports() -> tuple[list[list[int]], list[list[int]], list[np.ndarray], list[np.ndarray]]:
    """Prompts, responses, and per-token support ids and sampler logprobs drawn from the model itself."""
    tokenizer = get_tokenizer(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32).eval()
    vocab_half = AutoConfig.from_pretrained(MODEL_NAME).vocab_size // 2
    prompts = [
        [{"role": "user", "content": "What is 2 + 2?"}],
        [{"role": "user", "content": "Name the capital of France."}],
    ]
    prompt_ids = [
        tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True, return_dict=False)
        for prompt in prompts
    ]
    responses = [
        tokenizer.encode(" The answer is 4.", add_special_tokens=False) + [tokenizer.eos_token_id],
        tokenizer.encode(" Let us calculate it carefully.", add_special_tokens=False) + [tokenizer.eos_token_id],
    ]
    support_ids, sampler_logprobs = [], []
    for prompt, response in zip(prompt_ids, responses):
        with torch.no_grad():
            logits = model(torch.tensor([prompt + response])).logits[0, len(prompt) - 1 : -1]
        selected = torch.tensor(response)
        rows = torch.arange(len(response))
        competitor = logits.scatter(1, selected[:, None], -math.inf).argmax(-1)
        other_half = (torch.arange(logits.shape[-1]) >= vocab_half)[None, :] != (selected >= vocab_half)[:, None]
        straddler = logits.masked_fill(~other_half, -math.inf)
        straddler[rows, competitor] = -math.inf
        ids = torch.stack([selected, competitor, straddler.argmax(-1)], dim=-1)
        model_probs = logits.gather(1, ids).softmax(-1)
        sampler = SAMPLER_MODEL_WEIGHT * model_probs + (1 - SAMPLER_MODEL_WEIGHT) / SUPPORT_WIDTH
        support_ids.append(ids.numpy().astype(np.int32))
        sampler_logprobs.append(sampler.log().numpy().astype(np.float32))
    return prompt_ids, responses, support_ids, sampler_logprobs


def training_batch() -> TrainingInputBatch:
    """Two responses whose support rows straddle a TP=2 vocabulary split."""
    tokenizer = get_tokenizer(MODEL_NAME)
    prompt_ids, responses, support_ids, sampler_logprobs = response_supports()
    supports = [
        CompactSampleSupport.from_masked_rows(ids, np.ones_like(ids, dtype=bool), scores)
        for ids, scores in zip(support_ids, sampler_logprobs)
    ]

    sequences, attention_mask, response_mask, _, loss_mask, rollout_logprobs, _, sample_support = (
        convert_prompts_responses_to_batch_tensors(
            pad_token_id=tokenizer.pad_token_id,
            prompts=prompt_ids,
            responses=responses,
            rewards=[[0.0] * len(response) for response in responses],
            loss_masks=[[1] * len(response) for response in responses],
            logprobs=[scores[:, 0].tolist() for scores in sampler_logprobs],
            rollout_sample_support=supports,
        )
    )
    batch = TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "loss_mask": loss_mask,
            "rollout_logprobs": rollout_logprobs,
            # One response with a positive and one with a negative advantage.
            "advantages": loss_mask.float() * (loss_mask.new_tensor([[1.0], [-1.0]]) / loss_mask.sum()),
            "action_log_probs": None,
            "base_action_log_probs": None,
            SAMPLE_SUPPORT_FIELD: sample_support,
        }
    )
    batch.metadata = {"response_length": response_mask.shape[1], "global_step": 0}
    return batch

"""Training batches for the support-logprob-term GPU tests, shared by the FSDP and Megatron suites.

The support rows are the score-centering rows (the selected token, its strongest competitor, and the strongest
token in the other TP=2 vocabulary half) with a member-value column, as TT-OPD's trainer writes it: the selected
token's slot gets 0 and the two candidates their weights.
"""

import functools

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.utils.sample_support import (
    SAMPLE_SUPPORT_FIELD,
    CompactSampleSupport,
)
from skyrl.train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl.utils.tok import get_tokenizer
from tests.backends.skyrl_train.gpu.score_centering_utils import (
    MODEL_NAME,
    response_supports,
)


def _member_weights(num_rows: int, num_tokens: int) -> np.ndarray:
    """``[rows, 3]`` weights ``[0, c_1, c_2]``, pre-scaled by the batch's token count (token_mean)."""
    candidates = np.stack([np.linspace(0.4, -0.3, num_rows), np.linspace(-0.2, 0.5, num_rows)], axis=1)
    return (np.concatenate([np.zeros((num_rows, 1)), candidates], axis=1) / num_tokens).astype(np.float32)


def _weighted_rows():
    prompt_ids, responses, support_ids, sampler_logprobs = response_supports()
    num_tokens = sum(len(response) for response in responses)
    weights = [_member_weights(len(response), num_tokens) for response in responses]
    return prompt_ids, responses, support_ids, sampler_logprobs, weights


@functools.cache
def expected_support_loss() -> float:
    """``-sum_t sum_v w_v log pi(v)`` over every response token, from a float32 forward of the model."""
    prompt_ids, responses, support_ids, _, weights = _weighted_rows()
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32).eval()
    total = 0.0
    for prompt, response, ids, values in zip(prompt_ids, responses, support_ids, weights):
        with torch.no_grad():
            log_probs = (
                model(torch.tensor([prompt + response])).logits[0, len(prompt) - 1 : -1].double().log_softmax(-1)
            )
        total -= float((torch.from_numpy(values).double() * log_probs.gather(1, torch.from_numpy(ids).long())).sum())
    return total


def weighted_training_batch() -> TrainingInputBatch:
    """Two responses whose weighted support rows straddle a TP=2 vocabulary split."""
    tokenizer = get_tokenizer(MODEL_NAME)
    prompt_ids, responses, support_ids, sampler_logprobs, weights = _weighted_rows()
    supports = [
        CompactSampleSupport.from_masked_rows(ids, np.ones_like(ids, dtype=bool), scores, member_values=values)
        for ids, scores, values in zip(support_ids, sampler_logprobs, weights)
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
            "advantages": loss_mask.float() * (loss_mask.new_tensor([[1.0], [-1.0]]) / loss_mask.sum()),
            # importance_sampling's old policy: the sampler's logprob of each selected token.
            "action_log_probs": rollout_logprobs.clone(),
            "base_action_log_probs": None,
            SAMPLE_SUPPORT_FIELD: sample_support,
        }
    )
    batch.metadata = {"response_length": response_mask.shape[1], "global_step": 0}
    return batch

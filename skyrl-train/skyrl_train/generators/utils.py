import torch
from typing import List, Tuple
from collections import defaultdict
import numpy as np
from skyrl_train.generators.base import GeneratorOutput

CUSTOM_CHAT_TEMPLATES = {
    # chat template for qwen3 thinking mode to remove think tokens similar to generation phase
    "qwen3_thinking": (
        "{% for message in messages %}"
        "{% if (message['role'] != 'assistant') %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% elif (message['role'] == 'assistant')%}"
        "{{'<|im_start|>' + message['role'] + '\n'}}"
        "{% generation %}"
        "{% set full_content = message['content'] %}"
        "{% set mycontent = message['content'] %}"
        "{% set is_last_message = loop.last and messages[-1]['role'] == 'assistant' %}"
        "{% if '</think>' in full_content and not is_last_message %}"
        "{% set mycontent = full_content.split('</think>')[-1].lstrip('\n') %}"
        "{% endif %}"
        "{{mycontent + '<|im_end|>'}}"
        "{% endgeneration %}"
        "{{'\n'}}"
        "{% endif %}"
        "{% endfor %}"
    ),
}


def get_custom_chat_template(model_name: str) -> str:
    if "Qwen3" in model_name:
        return CUSTOM_CHAT_TEMPLATES["qwen3_thinking"]
    else:
        return None


def get_generation_prompt_ids(tokenizer) -> List[int]:
    """
    Helper function to get the generation prompt ids for a given tokenizer.
    """
    empty_user = tokenizer.apply_chat_template([{"role": "user", "content": ""}], tokenize=True)
    empty_user_with_generation_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}], add_generation_prompt=True, tokenize=True
    )

    generation_prompt_ids = empty_user_with_generation_prompt[len(empty_user) :]
    return generation_prompt_ids


@torch.no_grad()
def get_metrics_from_generator_output(generator_output: GeneratorOutput, uids: List[str]) -> Tuple[float, float]:
    """
    Get `mean_raw_reward` (or avg_score), `pass_at_n` from generator output.
    """
    rewards: List[float] = generator_output["rewards"]

    # Compute pass@N metrics
    pass_at_n_dict = defaultdict(list)
    for i, reward in enumerate(rewards):
        pass_at_n_dict[uids[i]].append(reward)

    mean_raw_reward = np.mean(rewards)

    # pass@N metric
    pass_at_n = sum(1 for v in pass_at_n_dict.values() if np.sum(v) > 0) / len(pass_at_n_dict)

    return mean_raw_reward, pass_at_n


def concatenate_generator_outputs(generator_outputs: List[GeneratorOutput]) -> GeneratorOutput:
    """
    Used in eval to concatenate the generator outputs of multiple batches.

    `rollout_metrics` are not concatenated because they are already aggregated.
    """
    assert len(generator_outputs) > 0
    result: GeneratorOutput = {
        "prompt_token_ids": sum([output["prompt_token_ids"] for output in generator_outputs], []),
        "response_ids": sum([output["response_ids"] for output in generator_outputs], []),
        "rewards": sum([output["rewards"] for output in generator_outputs], []),
        "loss_masks": sum([output["loss_masks"] for output in generator_outputs], []),
    }
    if "stop_reasons" in generator_outputs[0]:
        result["stop_reasons"] = sum([output["stop_reasons"] for output in generator_outputs], [])

    return result

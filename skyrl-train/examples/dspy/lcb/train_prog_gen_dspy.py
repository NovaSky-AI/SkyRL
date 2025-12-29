import json
import random

import dspy
from datasets import load_dataset

import arbor
from arbor import ArborGRPO, ArborProvider

# Start Arbor server (starts in background)
import pickle
import json
from collections import Counter

from lcb_utils import post_process_tests_inputs, check_test
from live_code_bench_execute import check_correctness
from utils import post_process_code
from program import NaiveCodeGenerator_dspy
from data import lcb_data


import os

# Export your Weights & Biases API key
os.environ["WANDB_API_KEY"] = "6a25ce5b41815c557d6fe8aecb8bac2dd6b1bea0"
os.environ["TOKENIZERS_PARALLELISM"] = "true"


# try:
#     vllm_version = pkg_resources.get_distribution("vllm").version
#     print(f"vllm package version: {vllm_version}")
# except pkg_resources.DistributionNotFound:
#     print("vllm package is not installed.")


arbor_server_info = arbor.init()
trainset, testset = lcb_data()


random.Random(43).shuffle(trainset)

# print(trainset[0])

# Get Arbor server info from init()
provider = ArborProvider()

student_lm_name = "Qwen/Qwen2.5-Coder-3B-Instruct"
# student_lm_name = "Qwen/Qwen2.5-7B-Instruct"
# student_lm_name = "Qwen/Qwen2.5-14B-Instruct"
# student_lm_name = "Qwen/Qwen2.5-32B-Instruct"
prog_lm = dspy.LM(
    model=f"openai/arbor:{student_lm_name}",
    provider=provider,
    api_base=arbor_server_info["base_url"],
    api_key="arbor",
    max_tokens=2048,
    temperature=1.0,
    top_p=1.0,
    top_k=-1,
    repetition_penalty=1.0,
    cache=False
)


prog = NaiveCodeGenerator_dspy(cot=False)

prog.set_lm(prog_lm)


with open("gold_preds_train.pkl", "rb") as f: #TODO: use test during test time
    gold_preds = pickle.load(f)


def reward_fn(example, pred, trace=None):
    timeout = 50
    result_dict = check_correctness(
        example,
        post_process_code(pred.code),
        timeout,
        is_extracted=True,
        fast_check=True,
    )
    return result_dict["passed"]


train_kwargs = {
    "per_device_train_batch_size": 2,
    "temperature": 1.0,
    "top_k": -1,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
    "beta": 0.001,
    "learning_rate": 1e-6,
    "gradient_checkpointing": True,
    "gradient_accumulation_steps": 2,
    "generation_batch_size": 16,
    # "steps_per_generation" = generation_batch_size // (per_device_train_batch_size * num_train_gpu)
    "fp16": False,
    "lr_scheduler_type": "constant_with_warmup",
    "warmup_steps": 1,
    "shuffle_dataset": True, 
    "log_completions": True,
    "max_prompt_length": 2048,
    "max_seq_len": 2048,  # how long the entire sequence may be
    "max_tokens": 2048,   # Maximum number of tokens to generate per turn
    "soft_completion_penalty_length": None,
    "max_grad_norm": 1.0,
    "report_to": "wandb",
    "log_completions": True,
    "max_steps": 1000,
    "logging_steps": 1,
    "loss_type": "dapo", #"dr_grpo",
    "mask_truncated_completions": True,
    "scale_rewards": False,
    "num_training_gpus": 6,
    "num_inference_gpus": 1,
    # "lora_config": {
    #     "lora_alpha": 16,
    #     "lora_dropout": 0.05,
    #     "r": 8,
    #     "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    # },
}

compiler = ArborGRPO(
    metric=reward_fn,
    num_dspy_examples_per_grpo_step=16,
    num_rollouts_per_grpo_step=12,
    exclude_demos=True,
    num_train_steps=100,
    num_threads=3,
    train_kwargs=train_kwargs,
    checkpoint="single-best",
)

unique_chars_rl = compiler.compile(
    student=prog,
    trainset=trainset,
)

# print(unique_chars_rl(english="hello"))
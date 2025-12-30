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

from .lcb_utils import post_process_tests_inputs, check_test

from .program import CodeGeneratorWithRankerAssertTestGen_single_dspy
from .data import lcb_data


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
generator_lm = dspy.LM(
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



prog = CodeGeneratorWithRankerAssertTestGen_single_dspy(
        generator_lm=generator_lm,
        num_tests=5,
        num_programs=5,
        use_chain_of_thought=False
    )

# student_unique_chars = unique_chars.deepcopy()
# student_unique_chars.set_lm(student_lm)

# def _unique_letter_reward(input, pred, trace=None) -> float:
#     letters = [ch.lower() for ch in pred.french if ch.isalpha()]
#     return float(len(set(letters)))


with open("gold_preds_train.pkl", "rb") as f: #TODO: use test during test time
    gold_preds = pickle.load(f)

def reward_fn(input, pred, trace=None):
    print("pred", pred)


    # TODO: add a check to see the output is empty
    
    prompt = input["prompt"]
    task_id = input["task_id"]
    is_stdin = input["is_stdin"]
    gold_pred = gold_preds[task_id]

    try:
        tests = post_process_tests_inputs(pred.tests, is_stdin)
    except Exception:
        print("test parsing failed")
        return 0

    if len(tests) == 0:
        print("no test found!")
        return 0
    


    tests_as_strings = [json.dumps(test, sort_keys=True) for test in tests]
    tests_counter = Counter(tests_as_strings)
    tests = [
        {"test": json.loads(test_str), "count": count}
        for test_str, count in tests_counter.items()
    ]
    

    num_passed = 0
    for test in tests:
        results = check_test([test["test"]], gold_pred, 0, prompt, "dummy", runtime_debug=True)
        passed = results[0]
        if passed:
            num_passed += 1
    print("reward:", num_passed / len(tests))
    return num_passed / len(tests)

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
    "generation_batch_size": 8,
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
    "num_training_gpus": 4,
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
    num_rollouts_per_grpo_step=8,
    exclude_demos=True,
    num_train_steps=140,
    num_threads=3,
    train_kwargs=train_kwargs,
    checkpoint="single-best",
)

unique_chars_rl = compiler.compile(
    student=prog,
    trainset=trainset,
)

# print(unique_chars_rl(english="hello"))
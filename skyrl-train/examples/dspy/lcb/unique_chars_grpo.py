import random

import dspy
from datasets import load_dataset

import arbor
from arbor import ArborGRPO, ArborProvider

# Start Arbor server (starts in background)
import pdb; pdb.set_trace()

arbor_server_info = arbor.init()
arbor_port = arbor_server_info["port"]
print("arbor_port!!!!!!!!!!!!!!!!!: ", arbor_port)
print("arbor_base_url!!!!!!!!!!!!!!!!!: ", arbor_server_info["base_url"])

raw_dataset = load_dataset("Helsinki-NLP/opus_books", "en-fr")
raw_data = [
    dspy.Example(english=ex["translation"]["en"], french=ex["translation"]["fr"]).with_inputs("english")
    for ex in raw_dataset["train"]
][:2000]

random.Random(43).shuffle(raw_data)
trainset = raw_data[:1000]
testset = raw_data[1000:1100]


import requests
from arbor.server.api.schemas import GRPOStatus

def initialize_grpo(
    model, url=f"http://127.0.0.1:{arbor_port}/v1/fine_tuning/grpo/initialize"
) -> GRPOStatus:
    headers = {"Content-Type": "application/json"}
    trainer_config = {
        "num_generations": 8,
        "temperature": 1.0,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-6,
        "beta": 0.01,
        "max_grad_norm": 1.0,
        "max_prompt_length": 512,
        "max_seq_len": 1024,
        "mask_truncated_completions": True,
        "gradient_checkpointing": True,
        "lr_scheduler_type": "constant_with_warmup",
        "warmup_steps": 10,
        "max_steps": 1000,
        "bf16": True,
        "report_to": "wandb",
        "logging_steps": 10,
        # "scale_rewards": False,
    }

    data = {
        "run_name": "checkpointing-demo",
        "model": model,
        "trainer_config": trainer_config,
        "inference_config": {
            "model": model,
            "max_context_length": 2048,
        },
        "gpu_config": {
            "type": "multi",
            "multi": {
                "num_inference_gpus": 1,
                "num_training_gpus": 1,
            },
        },
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return GRPOStatus.model_validate(response.json())


unique_chars = dspy.Predict(f"english -> french")

# Get Arbor server info from init()
provider = ArborProvider()

student_lm_name = "Qwen/Qwen2.5-1.5B-Instruct"
# student_lm_name = "Qwen/Qwen2.5-7B-Instruct"
# student_lm_name = "Qwen/Qwen2.5-14B-Instruct"
# student_lm_name = "Qwen/Qwen2.5-32B-Instruct"
student_lm = dspy.LM(
    model=f"openai/arbor:{student_lm_name}",
    provider=provider,
    api_base=arbor_server_info["base_url"],
    api_key="arbor",
    max_tokens=512,
    temperature=1.0,
    top_p=1.0,
    top_k=-1,
    repetition_penalty=1.0,
)

initialize_grpo(model=student_lm_name)

student_unique_chars = unique_chars.deepcopy()
student_unique_chars.set_lm(student_lm)

import pdb; pdb.set_trace()

student_unique_chars(english="hello")
import pdb; pdb.set_trace()

def _unique_letter_reward(input, pred, trace=None) -> float:
    letters = [ch.lower() for ch in pred.french if ch.isalpha()]
    return float(len(set(letters)))


train_kwargs = {
    "per_device_train_batch_size": 8,
    "temperature": 1.0,
    "top_k": -1,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
    "beta": 0.00,
    "learning_rate": 1e-6,
    "gradient_checkpointing": True,
    "gradient_accumulation_steps": 1,
    "fp16": True,
    "lr_scheduler_type": "constant_with_warmup",
    # "warmup_steps": 50,
    # "max_prompt_length": None,
    "max_completion_length": 512,
    "soft_completion_penalty_length": None,
    "max_grad_norm": 1.0,
    "report_to": "wandb",
    "log_completions": True,
    "max_context_length": None,
    "max_steps": 1000,
    "logging_steps": 10,
    "loss_type": "dapo", #"dr_grpo",
    "mask_truncated_completions": False,
    "scale_rewards": False,
    "num_training_gpus": 1,
    "num_inference_gpus": 1,
    # "lora_config": {
    #     "lora_alpha": 16,
    #     "lora_dropout": 0.05,
    #     "r": 8,
    #     "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    # },
}

compiler = ArborGRPO(
    metric=_unique_letter_reward,
    num_dspy_examples_per_grpo_step=1,
    num_rollouts_per_grpo_step=8,
    exclude_demos=True,
    num_train_steps=1000,
    num_threads=1,
    use_train_as_val=False,
    num_steps_for_val=500,
    train_kwargs=train_kwargs,
    checkpoint="single-best",
)

unique_chars_rl = compiler.compile(
    student=student_unique_chars,
    trainset=trainset,
    valset=testset,
)

print(unique_chars_rl(english="hello"))
from skyrl_agent import AutoAgentRunner
from transformers import AutoTokenizer
from datasets import load_dataset
import asyncio
import os

DOCKER_IMAGE_PREFIX = os.environ.get("EVAL_DOCKER_IMAGE_PREFIX", "xingyaoww/")


def get_instance_docker_image(instance_id) -> str:
    image_name = "sweb.eval.x86_64." + instance_id
    image_name = image_name.replace("__", "_s_")  # to comply with docker image naming convention
    name = (DOCKER_IMAGE_PREFIX.rstrip("/") + "/" + image_name).lower()
    return name


os.environ["OPENAI_API_KEY"] = "sc"
model = "Qwen/Qwen3-32B"
tokenizer = AutoTokenizer.from_pretrained(model)
dataset_file = "/data/sycao/r2e-all/train.parquet"
dataset = load_dataset("parquet", data_files=dataset_file)["train"]
print(f"Loaded dataset with {len(dataset)} instances")

agent_generator = AutoAgentRunner.from_task("./examples/test_vllm_oh.yaml", infer_engine=None, tokenizer=tokenizer)

output = asyncio.run(agent_generator.run(dataset, val_mode=True))

print(output["rewards"])
print(output["rollout_metrics"])
print("Output keys: ", list(output.keys()))


print("Average reward: ", sum(output["rewards"]) / len(output["rewards"]))

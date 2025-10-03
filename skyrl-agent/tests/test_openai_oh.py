from skyrl_agent import AutoAgentRunner
from transformers import AutoTokenizer
from datasets import load_dataset
import asyncio
import os

os.environ["OPENAI_API_KEY"] = "dummy"
model = "qwen/qwen2-0.5b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model)
dataset_file = "/path/to/parquet/file"

# read a few samples from the dataset
dataset = load_dataset("parquet", data_files=dataset_file)["train"].select(range(1, 2))

agent_generator = AutoAgentRunner.from_task("./tests/test_openai_oh.yaml", infer_engine=None, tokenizer=tokenizer)

output = asyncio.run(agent_generator.run(dataset))

print(output["rewards"])
print(output["rollout_metrics"])

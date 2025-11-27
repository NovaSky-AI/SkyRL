import os
from skyrl_agent import AutoAgentRunner
from pathlib import Path
from transformers import AutoTokenizer
import datasets
import asyncio

os.environ["OPENAI_API_KEY"] = "sc"  # dummy key, assumes an unath'ed vLLM service running locally
model = "Qwen/Qwen3-32B"

tokenizer = AutoTokenizer.from_pretrained(model)
dataset = "datasets/browsecomp-plus/browsecomp-plus-skyagent.parquet"
# read a few samples from the dataset
dataset = datasets.load_dataset("parquet", data_files=dataset)["train"].select(range(10))
print(dataset[0])

yaml_path = str(Path(__file__).parent / "openai_react.yaml")

agent_generator = AutoAgentRunner.from_task(
    yaml_path,
    # no explicit inference engine with OpenAI
    infer_engine=None,
    tokenizer=tokenizer,
)

output = asyncio.run(agent_generator.run(dataset, val_mode=True))
print(output["rewards"])

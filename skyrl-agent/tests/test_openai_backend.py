"""
uv run --frozen tests/test_openai_backend.py
"""

from skyrl_agent.integrations.openai import OpenAIBackend
from transformers import AutoTokenizer
import datasets
import asyncio
import logging

logging.basicConfig(level=logging.DEBUG)

# model = "NovaSky-AI/SWE-Gym-OpenHands-7B-Agent"
model = "qwen/qwen2-0.5b-instruct"
dataset = "/path/to/parquet/file"
api_url = "http://localhost:6002"

tokenizer = AutoTokenizer.from_pretrained(model)


# read a few samples from the dataset
dataset = datasets.load_dataset("parquet", data_files=dataset)["train"].select(range(2))
print(dataset[0])

backend = OpenAIBackend(infer_engine=None, cfg={"model_name": model, "api_url": api_url})

prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Hello, how are you?"}], tokenize=True, add_generation_prompt=True
)
output = asyncio.run(
    backend.async_generate_ids(prompt, sampling_params={"temperature": 0.0, "top_p": 1.0, "max_tokens": 1024})
)
print(output)

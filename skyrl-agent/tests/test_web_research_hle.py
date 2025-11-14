#!/usr/bin/env python3
"""
Test Web Research with HLE data
Usage: python tests/test_web_research_hle.py
"""
import os
import json
import asyncio
from skyrl_agent import AutoAgentRunner
from transformers import AutoTokenizer
import numpy as np

# Save results in WebThinker format
import datetime
from datasets import load_dataset

# Set dummy OPENAI_API_KEY for local models
os.environ["OPENAI_API_KEY"] = "dummy"

# Check environment
assert os.getenv("GOOGLE_SEARCH_KEY"), "Please set GOOGLE_SEARCH_KEY"

# Set web summary service to use the auxiliary model on port 8003
os.environ["WEB_SUMMARY_API_BASE"] = "http://localhost:8003/v1"
os.environ["WEB_SUMMARY_MODEL"] = "Qwen/Qwen2.5-32B-Instruct"

# Load unified HLE data from parquet (with proper prompt format)
dataset = load_dataset("parquet", data_files="skyagent/dataset/hle_test_unified.parquet")["train"]
print(f"✅ Loaded HLE dataset with {len(dataset)} total samples")

# Select samples to test
test_dataset = dataset.select(range(500))  # Run all 500 samples
test_data = test_dataset.to_pandas().to_dict("records")
print(f"📊 Testing with {len(test_data)} HLE samples:")
for i, sample in enumerate(test_data):
    print(f"  Sample {i+1}:")
    print(f"    - ID: {sample.get('id', 'N/A')}")
    print(f"    - Category: {sample.get('category', 'N/A')}")
    print(f"    - Question preview: {sample.get('Question', sample.get('prompt', 'N/A'))[:100]}...")
    if "answer" in sample:
        print(f"    - Ground truth answer: {sample['answer'][:50]}...")

# Initialize tokenizer - match the model running on port 8000
model = "Qwen/Qwen3-8B"  # Main model uses Qwen3-32B
try:
    tokenizer = AutoTokenizer.from_pretrained(model)
except Exception as e:
    print(f"Warning: Failed to load tokenizer for {model}: {e}")
    # Try alternate model names
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B")

# Initialize runner with yaml config
runner = AutoAgentRunner.from_task("./tests/web_research_test.yaml", infer_engine=None, tokenizer=tokenizer)

# Run - pass the dataset directly
print(f"Testing {len(test_dataset)} HLE samples...")
output = asyncio.run(runner.run(test_dataset))

# Prepare results in WebThinker format
results = []
for i, sample in enumerate(test_data):
    # Get the model's output
    if "response_ids" in output and i < len(output["response_ids"]):
        try:
            model_output = tokenizer.decode(output["response_ids"][i], skip_special_tokens=False)
        except Exception:
            model_output = "Error decoding response"
    else:
        model_output = "No response generated"

    # Create result entry in WebThinker format
    result_entry = {
        **sample,  # Include all original HLE fields
        "input": f"Please answer the following question. You should provide your final answer in the format \\boxed{{YOUR_ANSWER}}.\n\nQuestion:\n{sample.get('Question', sample.get('prompt'))}\n\n",
        "Output": model_output,
    }
    results.append(result_entry)


# Convert numpy types to Python types for JSON serialization
def convert_numpy_types(obj):
    """Recursively convert numpy types to Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


# Convert results to JSON-serializable format
results = convert_numpy_types(results)

timestamp = datetime.datetime.now().strftime("%m.%d,%H:%M")
output_file = f"test.{timestamp}.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"\n✅ Results saved to: {output_file} (WebThinker format)")
print("📊 Summary:")
rewards = output.get("rewards", [])
if rewards:
    print(f"  - Average Score: {sum(rewards)/len(rewards):.2f}")
    print(f"  - Scores: {rewards}")

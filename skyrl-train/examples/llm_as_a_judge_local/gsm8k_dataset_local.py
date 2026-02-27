"""
Preprocess the GSM8k dataset for the local LLM-as-a-Judge example.

Identical to ``examples/llm_as_a_judge/gsm8k_dataset_judge.py`` except
``env_class`` is set to ``llm_as_a_judge_local``.

Usage:
    uv run examples/llm_as_a_judge_local/gsm8k_dataset_local.py \\
        --output_dir ~/data/gsm8k_llm_judge_local
"""

import argparse
import re
import os

import datasets


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="~/data/gsm8k_llm_judge_local")

    args = parser.parse_args()
    args.output_dir = os.path.expanduser(args.output_dir)

    data_source = "openai/gsm8k"
    dataset = datasets.load_dataset(data_source, "main")

    train_dataset = dataset["train"]
    val_dataset = dataset["test"]

    instruction_following = 'Let\'s think step by step and output the final answer after "####".'

    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")
            question = question_raw + " " + instruction_following

            answer_raw = example.pop("answer")
            solution = extract_solution(answer_raw)
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "env_class": "llm_as_a_judge_local",
                "reward_spec": {
                    "method": "rule",
                    "ground_truth": solution,
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    val_dataset = val_dataset.map(function=make_map_fn("test"), with_indices=True)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(output_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(output_dir, "validation.parquet"))
    print(f"Dataset saved to {output_dir}")

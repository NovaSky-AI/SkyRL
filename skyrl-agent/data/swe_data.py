import os
from datasets import load_dataset, Dataset
import argparse
from tqdm import tqdm
from collections import defaultdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="R2E-Gym/R2E-Gym-Subset")
    parser.add_argument("--output", type=str, default="dataset/r2e-all")
    args = parser.parse_args()

    def extract_func_part(instance_id: str) -> str:
        parts = instance_id.split(".")
        if len(parts) >= 3:
            func_part = parts[2].split("__")[0]
            return func_part
        return None

    method_set = set()

    input_dataset = load_dataset(args.input, trust_remote_code=True)
    image_name_count = defaultdict(int)
    mapping = {"docker_image": "instance_id", "repo_name": "repo", "commit_hash": "base_commit"}
    for split in ["train"]:
        output_dataset = []
        cur_dataset = input_dataset[split]
        idx = -1
        for data_entry in tqdm(cur_dataset):
            for old_key, new_key in mapping.items():
                if old_key in data_entry:
                    data_entry[new_key] = data_entry.pop(old_key)

            if not data_entry["problem_statement"]:
                continue

            image_name = data_entry["repo"]

            cur_data = {
                "prompt": data_entry["problem_statement"],
                "data_source": "r2e-gym",
                "ability": "coding",
                "instance": data_entry,
            }

            image_name_count[image_name] += 1
            output_dataset.append(cur_data)

        print(f"Selected {len(output_dataset)} instances out of {len(cur_dataset)}")

        output_dataset = Dataset.from_list(output_dataset)

        os.makedirs(args.output, exist_ok=True)
        output_dataset.to_parquet(os.path.join(args.output, f"{split}.parquet"))

    # For validation set, keep the original logic
    input_dataset_name = "princeton-nlp/SWE-bench_Verified"
    input_dataset = load_dataset(input_dataset_name)
    for split in ["validation"]:
        output_dataset = []
        cur_dataset = input_dataset["test"]
        for data_entry in tqdm(cur_dataset):
            cur_data = {
                "prompt": data_entry["problem_statement"],
                "data_source": "swe-bench",
                "ability": "coding",
                "instance": data_entry,
            }
            output_dataset.append(cur_data)

        print(f"Selected {len(output_dataset)} validation instances")
        output_dataset = Dataset.from_list(output_dataset)

        output_dataset.to_parquet(os.path.join(args.output, f"{split}.parquet"))

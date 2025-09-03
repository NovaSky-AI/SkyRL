"""
Preprocess the SWEBench dataset to SkyRL format
"""

import argparse
import os

import datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="~/data/swe_gym")

    args = parser.parse_args()

    args.output_dir = os.path.expanduser(args.output_dir)

    data_source = "SumanthRH/SWE-Gym"
    eval_data_source = "SumanthRH/SWE-bench_Verified"

    dataset = datasets.load_dataset(data_source, "default")
    train_dataset = dataset["train"]
    # Use only the instance ids in SkyRL-v0-293-data
    from huggingface_hub import snapshot_download
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        snapshot_download("NovaSky-AI/SkyRL-v0-293-data", local_dir=tmpdir, repo_type="dataset")
        skyrl_v0_dataset = datasets.load_dataset("parquet", data_files=[os.path.join(tmpdir, "train.parquet")])["train"]
        valid_instance_ids = set([skyrl_v0_dataset["instance"][i]["instance_id"] for i in range(len(skyrl_v0_dataset))])
    train_dataset = train_dataset.filter(lambda x: x["instance_id"] in valid_instance_ids)

    val_dataset = datasets.load_dataset(eval_data_source, "default")["test"]

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            data = {
                "data_source": data_source if split == "train" else eval_data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": example["problem_statement"],
                    }
                ],
                "env_class": "null",
                "instance": example,
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    val_dataset = val_dataset.map(function=make_map_fn("test"), with_indices=True)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(output_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(output_dir, "validation.parquet"))

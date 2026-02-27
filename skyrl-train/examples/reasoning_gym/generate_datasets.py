import os
import argparse
import random
from skyrl_gym.envs.reasoning_gym.dataset import ReasoningGymDataset
from reasoning_gym import create_dataset

# you can run this script to generate the training and validation datasets for ReasoningGym
# example:
# uv run examples/reasoning_gym/generate_datasets.py --dataset_name leg_counting --size 10000 --developer_prompt "You are a helpful assistant that can solve problems. Place your final answer between <answer></answer> tags." --file_path $HOME/data/reasoning_gym

# ARGS:
# --dataset_name: the name of the dataset to generate
# --size: the size of the dataset to generate
# --developer_prompt: the prompt to use for the developer
# --file_path: the path to the directory to save the datasets
# --training_seed: OPTIONAL, the seed to use for the training dataset
# --validation_seed: OPTIONAL, the seed to use for the validation dataset

def main(args):
    dataset_name = args.dataset_name
    size = args.size
    developer_prompt = args.developer_prompt
    output_dir = os.path.expanduser(args.file_path)
    
    os.makedirs(output_dir, exist_ok=True)

    training_seed = args.training_seed if args.training_seed is not None else random.randint(0, 1000000)
    validation_seed = args.validation_seed if args.validation_seed is not None else random.randint(0, 1000000)

    training_dataset = generate_dataset(dataset_name, size=size, developer_prompt=developer_prompt, seed=training_seed)
    training_dataset.write_to_parquet(os.path.join(output_dir, "train.parquet"))
    print(f"Dataset with {len(training_dataset)} examples saved to {output_dir}/train.parquet")

    validation_dataset = generate_dataset(dataset_name, size=size, developer_prompt=developer_prompt, seed=validation_seed)
    validation_dataset.write_to_parquet(os.path.join(output_dir, "validation.parquet"))
    print(f"Dataset with {len(validation_dataset)} examples saved to {output_dir}/validation.parquet")

def generate_dataset(dataset_name, size, developer_prompt, seed):
    """
    Generate dataset and include the reasoning_gym dataset in extras for efficient environment usage.
    """
    ds = create_dataset(dataset_name, size=size, seed=seed)
    
    rg = ReasoningGymDataset(
        procedural_dataset=ds,
        developer_prompt=developer_prompt,
        dataset_name=dataset_name,
        size=size,
        seed=seed
    )
    
    return rg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="leg_counting")
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--developer_prompt", type=str, default="Think step by step. Return ONLY the final numeric answer wrapped in <answer> tags, like <answer>42</answer>.")
    parser.add_argument("--file_path", type=str, default="~/data/reasoning_gym")
    parser.add_argument("--training_seed", type=int, help="Seed for training dataset generation")
    parser.add_argument("--validation_seed", type=int, help="Seed for validation dataset generation")
    
    args = parser.parse_args()
    main(args)
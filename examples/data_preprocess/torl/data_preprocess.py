# Usage: python SkyRL/examples/data_preprocess/torl/data_preprocess.py --output dataset/SkyRL-ToRL-data
# Directly use the dataset from the original repo
# https://github.com/GAIR-NLP/ToRL/blob/main/data/torl_data

import os
import argparse
from datasets import Dataset, load_dataset
from tqdm import tqdm
import pandas as pd

def download_torl_data(output_dir):
    """Download ToRL dataset from the GAIR-NLP/ToRL repository"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # URLs for the ToRL dataset files
    base_url = "https://github.com/GAIR-NLP/ToRL/raw/main/data/torl_data/"
    files = ["train.parquet",  "test_0524.parquet"]
    
    for file_name in files:
        print(f"Downloading {file_name}...")
        try:
            # Load the parquet file directly from the URL
            df = pd.read_parquet(f"{base_url}{file_name}")
            
            # Convert to the SkyRL format
            output_dataset = []
            print(f"Columns: {df.columns.tolist()}")
            print(f"Shape: {df.shape}")
            print(f"First row:")
            for col in df.columns:
                print(f"  {col}: {df.iloc[0][col]}")
            # print(df.head())
            # assert False
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {file_name}"):
                new_prompt = row["prompt"]
                assert len(new_prompt) == 2, "Dacheng: new_prompt should be a list of two elements"
                new_prompt[0] = {
                    "role": "system",
                    "content" : "A conversation between User and Assistant. The user asks a question, and the Assistant solves it."
                }
                new_prompt[1] = {
                    "role": "user",
                    "content": "Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}.\n" + new_prompt[1]["content"] + "\n\n"
                }
                cur_data = {
                    "data_source": "torl",
                    "prompt": new_prompt,  
                    "ability": "torl", 
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": row["reward_model"]["ground_truth"],
                    },
                    "extra_info": {
                        "split": row["extra_info"]["split"],
                        "id": row["extra_info"]["id"]
                    }
                }
                output_dataset.append(cur_data)
            
            # Save as parquet in SkyRL format
            if file_name == "train.parquet":
                split_name = "train"
            elif file_name == "test_0524.parquet":
                split_name = "test"
            else:
                raise ValueError(f"Unknown file name: {file_name}")
            output_dataset = Dataset.from_list(output_dataset)
            output_path = os.path.join(output_dir, f"{split_name}.parquet")
            output_dataset.to_parquet(output_path)
            
            # print the first 10 examples after processing
            print(f"First 10 examples after processing:")
            for i in range(10):
                print(output_dataset[i])
            print(f"Saved {len(output_dataset)} examples to {output_path}")
            
        except Exception as e:
            print(f"Error downloading {file_name}: {e}")
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='dataset/SkyRL-ToRL-data', 
                       help='Output directory for processed ToRL dataset')
    args = parser.parse_args()
    
    print(f"Downloading ToRL dataset to {args.output}")
    download_torl_data(args.output)
    print("Download and preprocessing complete!")


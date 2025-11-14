import pandas as pd
import polars as pl
from pathlib import Path

# Define input and output files
DATA_DIR = Path.home() / "data/dapo"
FILES = {
    "dapo-math-17k": "dapo-math-17k.parquet",
    "aime-2024": "aime-2024.parquet",
}

for name, filename in FILES.items():
    in_path = f"{DATA_DIR}/{filename}"
    out_path = f"{DATA_DIR}/{name}-cleaned.parquet"

    # Read using pandas
    df = pd.read_parquet(in_path)

    # Convert to Polars for fast deduplication and group operations
    pl_df = pl.from_pandas(df).unique(subset=["data_source", "prompt", "ability", "reward_model"])

    # Count number of reward_models per prompt
    pl_df = pl_df.with_columns(pl.col("reward_model").n_unique().over("prompt").alias("n_rm"))

    # Keep only prompts with one reward_model
    cleaned = pl_df.filter(pl.col("n_rm") == 1).drop("n_rm")

    # Convert back to pandas and save
    cleaned.to_pandas().to_parquet(out_path)
    print(f"Cleaned file saved to: {out_path}")

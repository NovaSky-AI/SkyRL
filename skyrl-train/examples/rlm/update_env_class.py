#!/usr/bin/env python3
"""Update the env_class column in the GSM8K dataset to use rlm_ex instead of gsm8k."""

import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path


def update_env_class(input_file: str, output_file: str, new_env_class: str = "rlm_ex"):
    """Update the env_class column in a parquet file.

    Args:
        input_file: Path to input parquet file
        output_file: Path to output parquet file
        new_env_class: New value for env_class column
    """
    table = pq.read_table(input_file)
    data_dict = table.to_pydict()

    print(f"Original env_class values: {set(data_dict['env_class'])}")
    data_dict["env_class"] = [new_env_class] * len(data_dict["env_class"])
    print(f"Updated env_class values: {set(data_dict['env_class'])}")
    new_table = pa.Table.from_pydict(data_dict)

    pq.write_table(new_table, output_file)
    print(f"Updated dataset written to: {output_file}")
    print(f"Total rows: {len(data_dict['env_class'])}")


if __name__ == "__main__":
    import sys

    data_dir = Path.home() / "data" / "gsm8k"

    input_val = data_dir / "validation.parquet"
    output_val = data_dir / "validation_rlm.parquet"

    if input_val.exists():
        update_env_class(str(input_val), str(output_val), "rlm_ex")
    else:
        print(f"Error: {input_val} not found")
        sys.exit(1)

    input_train = data_dir / "train.parquet"
    output_train = data_dir / "train_rlm.parquet"

    if input_train.exists():
        update_env_class(str(input_train), str(output_train), "rlm_ex")
    else:
        print(f"Note: {input_train} not found (skipping)")

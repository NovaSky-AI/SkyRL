"""Convert AIME 2025 (I + II, 30 problems) to the DAPO AIME eval format.

Reads `aime-2025.jsonl` (rows of ``{"prompt": [{"role": "user", "content": ...}], "label": ...}``) and writes
`aime-2025-cleaned.parquet` with the same columns, prompt template and reward style as `aime-2024-cleaned.parquet`,
so it is graded by the `aime` environment. Rows use ``data_source="aime-2025"``.
"""

import json
import re

import pandas as pd
import polars as pl
from pathlib import Path
import argparse

# Same prompt template as the BytedTsinghua-SIA/AIME-2024 parquet, so the `aime` env grader
# (last `Answer: X` line) applies unchanged.
AIME_PROMPT_PREFIX = (
    "Solve the following math problem step by step. The last line of your response should be of the form "
    "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
)
AIME_PROMPT_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'


def normalize_aime_label(label: str) -> str:
    """Normalize an AIME answer label to a plain integer string in [0, 999].

    The upstream AIME-2025 file stores one answer as ``336^\\circ``; AIME answers are integers.
    """
    label = re.sub(r"\^\s*\{?\\circ\}?$", "", label.strip()).strip()
    if not (re.fullmatch(r"[0-9]+", label) and int(label) <= 999):
        raise ValueError(f"AIME label is not an integer in [0, 999]: {label!r}")
    return str(int(label))


def aime_2025_jsonl_to_df(jsonl_path: str) -> pd.DataFrame:
    """Convert `aime-2025.jsonl` into the DAPO AIME format (same columns as `aime-2024.parquet`).

    Rows keep the source order, and ``extra_info.index`` is the source row index: 0-14 are
    AIME 2025 I problems 1-15 and 15-29 are AIME 2025 II problems 1-15.
    """
    rows = []
    with open(jsonl_path) as f:
        for i, line in enumerate(line for line in f if line.strip()):
            record = json.loads(line)
            (message,) = record["prompt"]
            problem = message["content"]
            rows.append(
                {
                    "data_source": "aime-2025",
                    "prompt": [{"content": AIME_PROMPT_PREFIX + problem + AIME_PROMPT_SUFFIX, "role": "user"}],
                    "ability": "MATH",
                    "reward_model": {
                        "ground_truth": normalize_aime_label(record["label"]),
                        "style": "rule-lighteval/MATH_v2",
                    },
                    "extra_info": {"index": i, "raw_problem": problem, "split": None},
                }
            )
    if len(rows) != 30:
        raise ValueError(f"Expected 30 AIME-2025 problems (I + II), got {len(rows)}")
    return pd.DataFrame(rows)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows and prompts that have more than one reward model, keeping the input row order.

    Same deduplication as `data_preprocess_dapo_aime.py`.
    """
    # Convert to Polars for fast deduplication and group operations
    pl_df = pl.from_pandas(df).unique(subset=["data_source", "prompt", "ability", "reward_model"], maintain_order=True)

    # Count number of reward_models per prompt
    pl_df = pl_df.with_columns(pl.col("reward_model").n_unique().over("prompt").alias("n_rm"))

    # Keep only prompts with one reward_model
    return pl_df.filter(pl.col("n_rm") == 1).drop("n_rm").to_pandas()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(Path.home() / "data/dapo"))
    parser.add_argument(
        "--input-file", type=str, default=None, help="Path to `aime-2025.jsonl` (default: <data-dir>/aime-2025.jsonl)"
    )
    args = parser.parse_args()

    in_path = args.input_file or f"{args.data_dir}/aime-2025.jsonl"
    out_path = f"{args.data_dir}/aime-2025-cleaned.parquet"

    clean(aime_2025_jsonl_to_df(in_path)).to_parquet(out_path)
    print(f"Cleaned file saved to: {out_path}")

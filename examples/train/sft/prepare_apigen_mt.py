"""Convert Salesforce/APIGen-MT-5k to OpenAI messages format and save locally.

APIGen-MT-5k ships in ShareGPT shape (``conversations`` column with
``from``/``value`` keys, plus per-row ``tools`` JSON-string and ``system``
text columns). ``SFTTrainer`` expects an OpenAI-style ``messages`` column
with ``role``/``content`` keys. This script does the rename + role mapping
and writes the result as Parquet so ``load_dataset(<dir>)`` picks it up.

Role mapping:

* ``human``         -> ``user``
* ``gpt``           -> ``assistant`` (text content)
* ``function_call`` -> ``assistant`` with ``tool_calls`` (content="")
* ``observation``   -> ``tool``

Usage::

    uv run examples/train/sft/prepare_apigen_mt.py \\
        --output-dir ~/data/apigen-mt-5k-openai
"""

import argparse
import os

from datasets import load_dataset


def _convert_turn(turn: dict) -> dict:
    """Map one ShareGPT turn to an OpenAI-style message dict.

    Raises ``ValueError`` on unknown roles so a dataset-schema change is
    caught loudly instead of silently dropping turns.
    """
    role, value = turn["from"], turn["value"]
    if role == "human":
        return {"role": "user", "content": value}
    if role == "gpt":
        return {"role": "assistant", "content": value}
    if role == "function_call":
        # APIGen ships the call as a JSON-encoded single dict
        # ({"name": ..., "arguments": ...}).
        return {"role": "assistant", "content": "", "tool_calls": value}
    if role == "observation":
        return {"role": "tool", "content": value}
    raise ValueError(
        f"Unknown ShareGPT role {role!r} in APIGen-MT-5k turn; expected one of human/gpt/function_call/observation"
    )


def convert_example(example: dict) -> dict:
    messages = [_convert_turn(turn) for turn in example["conversations"]]
    return {
        "messages": messages,
        # ``tools`` is a JSON-encoded string in the source; ``_coerce_tools``
        # in sft_trainer.py handles that shape, so leave it untouched.
        "tools": example["tools"],
        "system": example["system"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=os.path.expanduser("~/data/apigen-mt-5k-openai"),
        help="Directory to write the converted parquet shard into.",
    )
    parser.add_argument(
        "--num-rows",
        type=int,
        default=None,
        help="Optional cap on number of rows (for smoke tests). Default: all rows",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    parquet_path = os.path.join(output_dir, "train.parquet")

    split = "train" if args.num_rows is None else f"train[:{args.num_rows}]"
    print(f"[prepare_apigen_mt] Loading Salesforce/APIGen-MT-5k split={split} ...")
    ds = load_dataset("Salesforce/APIGen-MT-5k", split=split)

    print(f"[prepare_apigen_mt] Converting {len(ds)} rows ShareGPT → OpenAI messages ...")
    ds = ds.map(
        convert_example,
        remove_columns=ds.column_names,
        desc="convert",
    )

    os.makedirs(output_dir, exist_ok=True)
    print(f"[prepare_apigen_mt] Writing {parquet_path} ...")
    ds.to_parquet(parquet_path)
    print(f"[prepare_apigen_mt] Done. {len(ds)} rows -> {parquet_path}")


if __name__ == "__main__":
    main()

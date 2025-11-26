"""
Unified converter for MemAgent-style datasets.
Generates SkyRL-Agent formatted training and evaluation parquet files.

Default behavior (hardcoded sources like original scripts):
  - Train URL:
      https://huggingface.co/datasets/BytedTsinghua-SIA/hotpotqa/resolve/main/hotpotqa_train_32k.parquet
  - Eval URLs:
      https://huggingface.co/datasets/BytedTsinghua-SIA/hotpotqa/resolve/main/eval_{num_docs}.json
      for num_docs in [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]

Run to produce BOTH train and eval outputs into a directory:
  python data/memagent.py --output-dir ./outputs
"""

import argparse
import json
import os
import re
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import requests
from transformers import AutoTokenizer
from pathlib import Path

# Global tokenizer for multiprocessing
tokenizer = None


def init_tokenizer():
    """Initialize tokenizer in each worker process."""
    global tokenizer
    if tokenizer is None:
        print("Loading Qwen tokenizer in worker process...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)


def _np_to_list_inplace(record: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in list(record.items()):
        if isinstance(value, np.ndarray):
            record[key] = value.tolist()
    return record


# ========== Optional chunking utility (currently unused but kept) ==========
def chunk_documents_by_tokens(context: str, max_tokens: int = 4000) -> list[str]:
    """Split context by 'Document X:' and group up to max_tokens using tokenizer lengths."""
    global tokenizer
    if tokenizer is None:
        init_tokenizer()

    doc_pattern = r"(Document \d+:)"
    parts = re.split(doc_pattern, context or "")

    documents = []
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            doc = parts[i] + "\n" + parts[i + 1]
            documents.append(doc.strip())
        elif i < len(parts):
            documents.append(parts[i].strip())

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_tokens = 0

    for doc in documents:
        doc_tokens = len(tokenizer.encode(doc))
        if current_tokens + doc_tokens > max_tokens and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [doc]
            current_tokens = doc_tokens
        else:
            current_chunk.append(doc)
            current_tokens += doc_tokens

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    return chunks


# ========== Training conversion ==========
def _process_train_row(args: Tuple[int, pd.Series]) -> Dict[str, Any]:
    idx, row = args
    try:
        extra_info = row.get("extra_info", {}) if isinstance(row, dict) else row["extra_info"]
    except Exception:
        extra_info = {}
    original_index = extra_info.get("index", idx) if isinstance(extra_info, dict) else idx

    try:
        reward_model = row.get("reward_model") if isinstance(row, dict) else row["reward_model"]
    except Exception:
        reward_model = {}
    ground_truth = (reward_model or {}).get("ground_truth", [])

    question = ""
    try:
        if isinstance(extra_info, dict):
            question = extra_info.get("question", "")
    except Exception:
        pass

    try:
        data_source = (
            ("ruler_" + str(row.get("data_source"))) if isinstance(row, dict) else ("ruler_" + str(row["data_source"]))
        )
    except Exception:
        data_source = "ruler_hotpotqa"

    try:
        prompt = row.get("prompt") if isinstance(row, dict) else row["prompt"]
    except Exception:
        prompt = [{"role": "user", "content": question}]

    try:
        ability = row.get("ability") if isinstance(row, dict) else row["ability"]
    except Exception:
        ability = "memory"

    try:
        context = row.get("context") if isinstance(row, dict) else row["context"]
    except Exception:
        context = ""

    record = {
        "id": f"ruler_hotpotqa_{original_index}",
        "question": question,
        "golden_answers": ground_truth,
        "data_source": data_source,
        "prompt": prompt,
        "ability": ability,
        "reward_model": reward_model,
        "context": context,
        "extra_info": extra_info,
        "metadata": None,
    }
    return _np_to_list_inplace(record)


def convert_train_to_skyagent(input_parquet: str, output_parquet: str, processes: int | None = None) -> pd.DataFrame:
    """Convert training parquet to SkyRL-Agent format parquet."""
    print(f"Loading training dataset from: {input_parquet}")
    df = pd.read_parquet(input_parquet)
    print(f"Loaded: shape={df.shape}, columns={list(df.columns)}")

    rows_data = [(idx, row) for idx, row in df.iterrows()]
    nproc = processes or cpu_count()
    print(f"Processing with {nproc} worker processes...")

    with Pool(processes=nproc, initializer=init_tokenizer) as pool:
        sky_records: list[Dict[str, Any]] = []
        for i, rec in enumerate(pool.imap(_process_train_row, rows_data, chunksize=10)):
            sky_records.append(rec)
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1:,} / {len(df):,} records ({(i+1)/len(df)*100:.1f}%)")

    sky_df = pd.DataFrame(sky_records)
    print(f"Converted training: shape={sky_df.shape}")
    _save_parquet(sky_df, output_parquet)
    return sky_df


# ========== Eval conversion ==========
def _process_eval_record(args: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
    idx, record = args
    original_index = record.get("index", idx)
    question = record.get("input", "")
    answers = record.get("answers", [])
    context = record.get("context", "")
    num_docs = record.get("num_docs", 0)

    prompt = [{"content": question, "role": "user"}]
    reward_model = {"ground_truth": answers, "type": "exact_match"}
    extra_info = {"question": question, "index": original_index, "num_docs": num_docs}

    sky = {
        "id": f"ruler_hotpotqa_{original_index}",
        "question": question,
        "golden_answers": answers,
        "data_source": "ruler_hotpotqa",
        "prompt": prompt,
        "ability": "memory",
        "reward_model": reward_model,
        "context": context,
        "extra_info": extra_info,
        "metadata": None,
    }
    return _np_to_list_inplace(sky)


def _load_eval_json(input_path: str) -> list[Dict[str, Any]]:
    """Load eval JSON from local path or HTTP URL; tolerate /blob/ -> /resolve/."""
    if input_path.startswith("http://") or input_path.startswith("https://"):
        url = input_path.replace("/blob/", "/resolve/")
        print(f"Downloading eval JSON: {url}")
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
    else:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    if not isinstance(data, list):
        data = [data]
    return data


def convert_eval_to_skyagent(input_json: str, output_parquet: str, processes: int | None = None) -> pd.DataFrame:
    """Convert evaluation JSON to SkyRL-Agent format parquet."""
    data = _load_eval_json(input_json)
    print(f"Loaded eval records: {len(data):,}")
    if data:
        print(f"Sample eval keys: {list(data[0].keys())}")

    records_data = [(idx, rec) for idx, rec in enumerate(data)]
    nproc = processes or cpu_count()
    print(f"Processing with {nproc} worker processes...")

    with Pool(processes=nproc, initializer=init_tokenizer) as pool:
        sky_records: list[Dict[str, Any]] = []
        for i, rec in enumerate(pool.imap(_process_eval_record, records_data, chunksize=10)):
            sky_records.append(rec)
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1:,} / {len(data):,} records ({(i+1)/len(data)*100:.1f}%)")

    sky_df = pd.DataFrame(sky_records)
    print(f"Converted eval: shape={sky_df.shape}")
    _save_parquet(sky_df, output_parquet)
    return sky_df


# ========== IO helpers ==========
def _save_parquet(df: pd.DataFrame, output_parquet: str) -> None:
    os.makedirs(os.path.dirname(output_parquet) or ".", exist_ok=True)
    df.to_parquet(output_parquet, index=False, engine="pyarrow", row_group_size=len(df))
    print(f"Saved to {output_parquet}")
    # quick verify
    df2 = pd.read_parquet(output_parquet)
    print(f"Verified load: shape={df2.shape}")


# ========== CLI ==========
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MemAgent data converter to SkyRL-Agent format (train + eval)")
    p.add_argument("--output-dir", required=True, help="Directory to write train and eval parquet files")
    p.add_argument("--processes", type=int, default=None, help="Number of worker processes (default: cpu_count)")
    p.add_argument("--skip-train", action="store_true", help="Skip train conversion")
    p.add_argument("--skip-eval", action="store_true", help="Skip eval conversion")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Hardcoded sources as in the original scripts
    train_url = "https://huggingface.co/datasets/BytedTsinghua-SIA/hotpotqa/resolve/main/hotpotqa_train_32k.parquet"
    eval_sizes = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]

    if not args.skip_train:
        train_out = str(out_dir / "ruler_hotpotqa_train_32k_skyagent.parquet")
        convert_train_to_skyagent(train_url, train_out, processes=args.processes)

    if not args.skip_eval:
        for num_docs in eval_sizes:
            input_json = f"https://huggingface.co/datasets/BytedTsinghua-SIA/hotpotqa/resolve/main/eval_{num_docs}.json"
            output_parquet = str(out_dir / f"ruler_hotpotqa_eval_{num_docs}_skyagent.parquet")
            convert_eval_to_skyagent(input_json, output_parquet, processes=args.processes)


if __name__ == "__main__":
    main()

# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Preprocess SQL RL training data to verl-compatible parquet format.

Supports three data sources:
  - BIRD     (default): reads from raw BIRD JSON + SQLite databases
  - Spider:  reads from raw Spider JSON + SQLite databases
  - GretelAI: reads from original parquet, builds per-sample SQLite databases

Produces prompts in the arctic_text_to_sql_r1 format and applies a Qwen3
token-length filter (default 16,384 tokens) to drop the long-tail BIRD DBs
(e.g. works_cycles, movie_3) whose full SQLite DDL exceeds 100K tokens and
won't fit in context.

Usage:
    # BIRD only with default 16K-token cap (recommended for long-prompt RL):
    python preprocess_bird.py

    # Different token cap or tokenizer:
    python preprocess_bird.py --max_tokens 8192 --tokenizer Qwen/Qwen3-1.7B

    # All three sources:
    python preprocess_bird.py --sources bird spider gretelai


This script is vendored from
``arctic_platform.rl.projects.txt2sql.preprocess_bird`` so the BIRD pipeline
in this integration is self-contained and doesn't require the
Arctic-Platform private RL recipe checkout. Keep the upstream copy as the
source of truth — re-sync if it changes.
"""

import argparse
import csv
import os
import re
import sqlite3

import pandas as pd

# ---------------------------------------------------------------------------
# R1 prompt template (arctic_text_to_sql_r1)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a data science expert. Below, you are provided with a database schema "
    "and a natural language question. Your task is to understand the schema and generate "
    "a valid SQL query to answer the question."
)

INSTRUCT_INFO = """
Please provide a detailed chain-of-thought reasoning process and include your thought process within `<think>` tags. Your final answer should be enclosed within `<answer>` tags.

Ensure that your SQL query follows the correct syntax and is formatted as follows:

```sql
-- Your SQL query here
```

Example format:
<think> Step-by-step reasoning, including self-reflection and corrections if necessary. [Limited by 4K tokens] </think>
<answer> Summary of the thought process leading to the final SQL query. [Limited by 1K tokens]

```sql
Correct SQL query here
```
</answer>""".strip()


def build_r1_messages(schema_ddl: str, question: str, engine: str = "SQLite") -> list[dict]:
    """Build prompt messages matching the arctic_text_to_sql_r1 format."""
    user_content = f"""
Database Engine:
{engine}

Database Schema:
{schema_ddl}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{question}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.

Output Format:
{INSTRUCT_INFO}
    """.strip()

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Schema DDL extraction from SQLite databases
# ---------------------------------------------------------------------------


def _flatten_inline(s: str) -> str:
    """Collapse whitespace so the value fits on a single comment line. No length cap."""
    return " ".join(s.split())


def _short_value(v, max_len: int = 60) -> str:
    """Render a single cell for the sample-rows table; trim only very long values."""
    s = str(v) if isinstance(v, (int, float)) else repr(v)
    return s[: max_len - 3] + "..." if len(s) > max_len else s


_FK_RE = re.compile(
    r"FOREIGN\s+KEY\s*\(\s*([^)]+?)\s*\)\s*REFERENCES\s+" r"[\"'`]?([A-Za-z_][\w]*)[\"'`]?\s*(?:\(\s*([^)]+?)\s*\))?",
    re.IGNORECASE,
)


def _extract_fks(create_sql: str) -> list[str]:
    """Extract FK relationships from a CREATE TABLE block, formatted compactly."""
    fks = []
    for m in _FK_RE.finditer(create_sql):
        local_cols = m.group(1).replace('"', "").replace("`", "").replace("'", "").strip()
        ref_table = m.group(2)
        ref_cols = m.group(3) or ""
        ref_cols = ref_cols.replace('"', "").replace("`", "").replace("'", "").strip()
        if ref_cols:
            fks.append(f"{local_cols} -> {ref_table}.{ref_cols}")
        else:
            fks.append(f"{local_cols} -> {ref_table}")
    # Also handle inline column-level "REFERENCES" syntax (no FOREIGN KEY clause).
    # Skip lines that already declare a FOREIGN KEY (handled above), PRIMARY KEY, etc.
    for line in create_sql.split("\n"):
        s = line.strip()
        if not s:
            continue
        first = s.split()[0].upper()
        if first in {"FOREIGN", "PRIMARY", "UNIQUE", "CONSTRAINT", "CHECK", "CREATE", ")", "("}:
            continue
        m = re.search(
            r"^\s*[\"\'`]?([A-Za-z_][\w]*)[\"\'`]?\s+[^,]*?\bREFERENCES\s+"
            r"[\"\'`]?([A-Za-z_][\w]*)[\"\'`]?(?:\s*\(\s*([^)]+?)\s*\))?",
            line,
            re.IGNORECASE,
        )
        if m:
            local_col = m.group(1)
            ref_table = m.group(2)
            ref_cols = (m.group(3) or "").replace('"', "").replace("`", "").replace("'", "").strip()
            if ref_cols:
                fks.append(f"{local_col} -> {ref_table}.{ref_cols}")
            else:
                fks.append(f"{local_col} -> {ref_table}")
    # Dedup while preserving order.
    seen = set()
    uniq = []
    for fk in fks:
        if fk not in seen:
            seen.add(fk)
            uniq.append(fk)
    return uniq


def load_db_descriptions(db_dir: str) -> dict:
    """Load BIRD's database_description/*.csv as {(table_lower, col_lower): {...}}.

    BIRD ships per-table CSVs with column-level semantic descriptions and value
    semantics. Returns an empty dict if the folder is missing.
    """
    desc_dir = os.path.join(db_dir, "database_description")
    out: dict = {}
    if not os.path.isdir(desc_dir):
        return out

    for fname in os.listdir(desc_dir):
        if not fname.endswith(".csv"):
            continue
        table = fname[:-4]
        path = os.path.join(desc_dir, fname)
        # utf-8-sig strips a leading BOM (BIRD CSVs sometimes have one).
        rows = None
        for enc in ("utf-8-sig", "latin-1"):
            try:
                with open(path, encoding=enc) as f:
                    rows = list(csv.DictReader(f))
                break
            except UnicodeDecodeError:
                continue
        if not rows:
            continue
        for row in rows:
            col = (row.get("original_column_name") or "").strip()
            if not col:
                continue
            out[(table.lower(), col.lower())] = {
                "name": (row.get("column_name") or "").strip(),
                "desc": (row.get("column_description") or "").strip(),
                "val_desc": (row.get("value_description") or "").strip(),
            }
    return out


def get_schema_ddl(
    db_path: str,
    num_examples: int = 5,
    descriptions: dict | None = None,
    sample_rows: int = 3,
    include_fk_summary: bool = True,
) -> str:
    """Read schema DDL from a SQLite database, enriched with sample values.

    Augmentations applied (all reuse the single ``SELECT ... LIMIT`` per table —
    no extra DB scans):

    - Sample-rows block per table (markdown-style pipe table) using the same
      rows fetched for per-column examples.
    - Foreign-key summary line at the top of each table block.
    - Inline ``-- example: ... | name: ... | desc: ... | values: ...`` comment
      on each column line. ``name``, ``desc``, and ``values`` come from
      ``descriptions``; descriptions are NOT length-truncated (whitespace is
      collapsed so they stay one line).
    """
    descriptions = descriptions or {}
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.cursor()

    cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = cursor.fetchall()

    ddl_parts = []
    rows_to_fetch = max(num_examples, sample_rows, 1)

    for table_name, create_sql in tables:
        if not create_sql:
            continue

        col_examples: dict = {}
        col_names: list = []
        rows_for_table: list = []
        try:
            cursor.execute(f'SELECT * FROM "{table_name}" LIMIT {rows_to_fetch}')
            rows_for_table = cursor.fetchall()
            col_names = [desc[0] for desc in cursor.description]
            if num_examples > 0:
                for col_idx, col_name in enumerate(col_names):
                    vals = [row[col_idx] for row in rows_for_table[:num_examples] if row[col_idx] is not None]
                    if vals:
                        col_examples[col_name] = vals
        except sqlite3.Error:
            pass

        # Top-of-block header: FK summary + sample rows.
        header_lines: list = []
        if include_fk_summary:
            fks = _extract_fks(create_sql)
            if fks:
                header_lines.append(f"-- Foreign keys: {'; '.join(fks)}")
        if sample_rows > 0 and rows_for_table and col_names:
            header_lines.append(f'-- Sample rows from "{table_name}":')
            header_lines.append("-- | " + " | ".join(col_names) + " |")
            for row in rows_for_table[:sample_rows]:
                header_lines.append("-- | " + " | ".join(_short_value(v) for v in row) + " |")

        enriched_lines: list = list(header_lines)
        for line in create_sql.split("\n"):
            stripped = line.strip().rstrip(",")
            if stripped and not stripped.startswith("CREATE") and not stripped.startswith(")"):
                col_token = stripped.split()[0].strip('"').strip("'").strip("`")
                annots = []
                if col_token in col_examples:
                    annots.append(f"example: {col_examples[col_token]!r}")
                desc_key = (table_name.lower(), col_token.lower())
                if desc_key in descriptions:
                    d = descriptions[desc_key]
                    name = d.get("name", "")
                    if name and name.lower() != col_token.lower():
                        annots.append(f"name: {_flatten_inline(name)}")
                    if d["desc"]:
                        annots.append(f"desc: {_flatten_inline(d['desc'])}")
                    if d["val_desc"]:
                        annots.append(f"values: {_flatten_inline(d['val_desc'])}")
                if annots:
                    comment = " -- " + " | ".join(annots)
                    if line.rstrip().endswith(","):
                        line = line.rstrip()[:-1] + comment + ","
                    else:
                        line = line.rstrip() + comment
            enriched_lines.append(line)
        ddl_parts.append("\n".join(enriched_lines))

    conn.close()
    return "\n\n".join(ddl_parts)


# ---------------------------------------------------------------------------
# BIRD processing
# ---------------------------------------------------------------------------


def process_bird(
    bird_dir: str,
    split: str,
    num_examples: int = 5,
    use_descriptions: bool = True,
    sample_rows: int = 3,
    include_fk_summary: bool = True,
) -> list[dict]:
    """Process BIRD dataset from raw JSON + SQLite databases."""
    import json

    if split == "train":
        json_file = os.path.join(bird_dir, "train", "train.json")
        db_base = os.path.join(bird_dir, "train", "train_databases")
    else:
        json_file = os.path.join(bird_dir, "dev", "dev.json")
        db_base = os.path.join(bird_dir, "dev", "dev_databases")

    with open(json_file) as f:
        data = json.load(f)

    print(f"  BIRD {split}: {len(data)} raw samples from {json_file}")

    schema_cache = {}
    records = []

    for idx, item in enumerate(data):
        db_id = item["db_id"]
        question = item["question"]
        evidence = item.get("evidence", "")
        gold_sql = item["SQL"]

        db_path = os.path.join(db_base, db_id, f"{db_id}.sqlite")
        if not os.path.exists(db_path):
            continue

        if db_id not in schema_cache:
            try:
                db_dir = os.path.join(db_base, db_id)
                descs = load_db_descriptions(db_dir) if use_descriptions else {}
                schema_cache[db_id] = get_schema_ddl(
                    db_path,
                    num_examples=num_examples,
                    descriptions=descs,
                    sample_rows=sample_rows,
                    include_fk_summary=include_fk_summary,
                )
            except Exception as e:
                print(f"    Warning: failed to read schema for {db_id}: {e}")
                continue

        if evidence and evidence.strip():
            full_question = f"{question}\n\nEvidence:\n{evidence}"
        else:
            full_question = question

        records.append(
            {
                "data_source": "bird",
                "prompt": build_r1_messages(schema_cache[db_id], full_question),
                "ability": "sql",
                "reward_model": {"style": "rule", "ground_truth": gold_sql},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "db_id": db_id,
                    "db_path": db_path,
                    "question": question,
                },
            }
        )

    print(f"    -> {len(records)} samples")
    return records


# ---------------------------------------------------------------------------
# Spider processing
# ---------------------------------------------------------------------------


def process_spider(spider_dir: str) -> list[dict]:
    """Process Spider dataset from raw JSON + SQLite databases."""
    import json

    db_base = os.path.join(spider_dir, "database")
    records = []

    for json_name in ["train_spider.json", "train_others.json"]:
        json_file = os.path.join(spider_dir, json_name)
        if not os.path.exists(json_file):
            print(f"    Warning: {json_file} not found, skipping")
            continue

        with open(json_file) as f:
            data = json.load(f)
        print(f"  Spider {json_name}: {len(data)} raw samples")

        schema_cache = {}
        for idx, item in enumerate(data):
            db_id = item["db_id"]
            question = item["question"]
            gold_sql = item["query"]

            db_path = os.path.join(db_base, db_id, f"{db_id}.sqlite")
            if not os.path.exists(db_path):
                continue

            if db_id not in schema_cache:
                try:
                    schema_cache[db_id] = get_schema_ddl(db_path, num_examples=3)
                except Exception:
                    continue

            records.append(
                {
                    "data_source": "spider",
                    "prompt": build_r1_messages(schema_cache[db_id], question),
                    "ability": "sql",
                    "reward_model": {"style": "rule", "ground_truth": gold_sql},
                    "extra_info": {
                        "split": "train",
                        "index": idx,
                        "db_id": db_id,
                        "db_path": db_path,
                        "question": question,
                    },
                }
            )

    print(f"    -> {len(records)} samples total")
    return records


# ---------------------------------------------------------------------------
# GretelAI processing
# ---------------------------------------------------------------------------


def _create_gretelai_db(sql_context: str, db_path: str) -> bool:
    """Create a SQLite database from GretelAI's sql_context (CREATE+INSERT statements)."""
    try:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        conn.executescript(sql_context)
        conn.close()
        return True
    except Exception:
        return False


def process_gretelai(
    gretelai_parquet: str,
    r1_parquet: str,
    gretelai_db_dir: str,
) -> list[dict]:
    """Process GretelAI using R1 parquet for cleaned gold SQLs + original parquet for sql_context.

    Creates per-sample SQLite databases from sql_context for reward execution.
    """
    r1_df = pd.read_parquet(r1_parquet)
    r1_gretel = r1_df[r1_df["data_source"] == "gretelai"].copy()
    r1_gretel["db_id_int"] = r1_gretel["db_id"].astype(int)

    orig_df = pd.read_parquet(gretelai_parquet)
    orig_lookup = orig_df.set_index("id")

    print(f"  GretelAI: {len(r1_gretel)} samples from R1 parquet")

    os.makedirs(gretelai_db_dir, exist_ok=True)
    records = []
    skipped = 0

    for _, row in r1_gretel.iterrows():
        sample_id = row["db_id_int"]

        if sample_id not in orig_lookup.index:
            skipped += 1
            continue

        orig_row = orig_lookup.loc[sample_id]
        sql_context = orig_row["sql_context"]
        domain = orig_row["domain"]

        db_path = os.path.join(gretelai_db_dir, f"{sample_id}.sqlite")
        if not os.path.exists(db_path):
            if not _create_gretelai_db(sql_context, db_path):
                skipped += 1
                continue

        schema_ddl = row["omnisql_schema"]
        question = row["question"]
        gold_sql = row["ground_truth"]

        records.append(
            {
                "data_source": "gretelai",
                "prompt": build_r1_messages(schema_ddl, question),
                "ability": "sql",
                "reward_model": {"style": "rule", "ground_truth": gold_sql},
                "extra_info": {
                    "split": "train",
                    "index": int(row["extra_info"].get("index", 0)),
                    "db_id": str(sample_id),
                    "db_path": db_path,
                    "question": question,
                    "domain": domain,
                },
            }
        )

    print(f"    -> {len(records)} samples, {skipped} skipped")
    return records


# ---------------------------------------------------------------------------
# Token-length filtering
# ---------------------------------------------------------------------------


def filter_by_token_length(
    records: list[dict],
    tokenizer_name: str,
    max_tokens: int,
    batch_size: int = 64,
) -> list[dict]:
    """Drop records whose tokenized prompt exceeds ``max_tokens``."""
    if max_tokens is None or max_tokens <= 0 or not records:
        return records

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    texts = ["".join(m.get("content", "") for m in r["prompt"]) for r in records]

    kept, dropped_per_source = [], {}
    lens = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i : i + batch_size], add_special_tokens=False)
        for j, ids in enumerate(enc["input_ids"]):
            n = len(ids)
            lens.append(n)
            rec = records[i + j]
            if n <= max_tokens:
                kept.append(rec)
            else:
                dropped_per_source[rec["data_source"]] = dropped_per_source.get(rec["data_source"], 0) + 1

    n_total = len(records)
    n_kept = len(kept)
    print(
        f"  Token filter (<= {max_tokens} tokens, {tokenizer_name}): "
        f"{n_kept}/{n_total} kept ({n_total - n_kept} dropped)"
    )
    if dropped_per_source:
        for src, cnt in sorted(dropped_per_source.items()):
            print(f"    dropped {cnt} from {src}")
    if lens:
        import numpy as np

        arr = np.array(lens)
        print(
            f"  Pre-filter token stats: median={int(np.median(arr))} "
            f"p90={int(np.percentile(arr, 90))} "
            f"p99={int(np.percentile(arr, 99))} max={int(arr.max())}"
        )
    return kept


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_PATHS = {
    "bird_dir": "/data/ruofan/bird_wiki_original",
    "spider_dir": "/data/ruofan/spider_data",
    "gretelai_parquet": "/data/ruofan/gretelai/synthetic_text_to_sql_train.snappy.parquet",
    "r1_parquet": (
        "/code/users/lukasz/process/snowflake_v1/merged_train_model_type-qwen_coder-sft_maxtoken-8192_maxtime-10_len-16459.parquet"
    ),
}


def main():
    parser = argparse.ArgumentParser(description="Preprocess SQL data for verl RL training")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["bird"],
        choices=["bird", "spider", "gretelai"],
        help="Data sources to include (default: bird only)",
    )
    parser.add_argument("--bird_dir", type=str, default=DEFAULT_PATHS["bird_dir"])
    parser.add_argument("--spider_dir", type=str, default=DEFAULT_PATHS["spider_dir"])
    parser.add_argument("--gretelai_parquet", type=str, default=DEFAULT_PATHS["gretelai_parquet"])
    parser.add_argument("--r1_parquet", type=str, default=DEFAULT_PATHS["r1_parquet"])
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=32768,
        help=(
            "Drop training samples whose prompt exceeds this many tokens. "
            "Set to 0 to disable filtering. Default: 32768. "
            "(BIRD's outlier DBs `works_cycles` and `movie_3` sit at >80K "
            "tokens with full augmentation, so 32K is the natural break.)"
        ),
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="HF tokenizer used for the token-length filter. Default: Qwen/Qwen3-1.7B.",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=10,
        help=(
            "Sample rows per column to include inline as `-- example: [...]`. "
            "Higher values produce longer prompts. Default: 10."
        ),
    )
    parser.add_argument(
        "--sample_rows",
        type=int,
        default=10,
        help=(
            "Full sample rows shown as a markdown table at the top of each "
            "CREATE TABLE block. Reuses the rows fetched for `--num_examples` "
            "(no extra DB queries). Set to 0 to disable. Default: 10."
        ),
    )
    parser.add_argument(
        "--no_descriptions",
        action="store_true",
        help="Disable BIRD `database_description/*.csv` schema enrichment (column descriptions and value semantics).",
    )
    parser.add_argument(
        "--no_fk_summary",
        action="store_true",
        help="Disable the `-- Foreign keys: ...` summary line at the top of each CREATE TABLE block.",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    print(f"Sources: {args.sources}")
    print(f"Output:  {args.output_dir}")
    print()

    # --- Training data ---
    train_records = []

    if "bird" in args.sources:
        print("Processing BIRD (train)...")
        train_records.extend(
            process_bird(
                args.bird_dir,
                "train",
                num_examples=args.num_examples,
                use_descriptions=not args.no_descriptions,
                sample_rows=args.sample_rows,
                include_fk_summary=not args.no_fk_summary,
            )
        )

    if "spider" in args.sources:
        print("Processing Spider...")
        train_records.extend(process_spider(args.spider_dir))

    if "gretelai" in args.sources:
        print("Processing GretelAI...")
        gretelai_db_dir = os.path.join(args.output_dir, "gretelai_dbs")
        train_records.extend(
            process_gretelai(
                args.gretelai_parquet,
                args.r1_parquet,
                gretelai_db_dir,
            )
        )

    if train_records and args.max_tokens and args.max_tokens > 0:
        print("\nFiltering train by token length...")
        train_records = filter_by_token_length(
            train_records,
            args.tokenizer,
            args.max_tokens,
        )

    if train_records:
        train_path = os.path.join(args.output_dir, "train.parquet")
        os.makedirs(args.output_dir, exist_ok=True)
        import datasets as hf_datasets

        hf_datasets.Dataset.from_list(train_records).to_parquet(train_path)
        print(f"\nTrain: {len(train_records)} samples -> {train_path}")

        from collections import Counter

        source_counts = Counter(r["data_source"] for r in train_records)
        for src, cnt in source_counts.most_common():
            print(f"  {src}: {cnt}")

    # --- Validation data (BIRD dev only) ---
    # Dev is intentionally kept *clean* — minimal schema augmentation, matching
    # how the model would be evaluated on raw BIRD. No `database_description`
    # CSV enrichment, no per-table sample-rows block, no FK-summary line, just
    # 3 inline example values per column.  This avoids leaking auxiliary
    # context into the validation prompts that might inflate eval signal.
    if "bird" in args.sources:
        print("\nProcessing BIRD (dev) for validation (clean / no augmentation)...")
        val_records = process_bird(
            args.bird_dir,
            "dev",
            num_examples=3,
            use_descriptions=False,
            sample_rows=0,
            include_fk_summary=False,
        )
        if val_records:
            val_path = os.path.join(args.output_dir, "val.parquet")
            import datasets as hf_datasets

            hf_datasets.Dataset.from_list(val_records).to_parquet(val_path)
            print(f"Val: {len(val_records)} samples -> {val_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()

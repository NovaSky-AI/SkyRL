import pandas as pd
import numpy as np
import json
import argparse


def convert_browsecomp_plus_to_skyagent(jsonl_file_path, output_path):
    """Convert BrowseComp Plus JSONL dataset to SkyRL-Agent format

    Note: We only keep minimal information (question and answer),
    dropping gold_docs, negative_docs, and evidence_docs as redundant.
    """
    print("Loading BrowseComp Plus dataset...")

    # Load JSONL file line by line
    records = []
    with open(jsonl_file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                records.append(data)
                if line_num % 100 == 0:
                    print(f"Loaded {line_num} records...")
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed line {line_num}: {e}")
                continue

    print(f"Loaded {len(records)} records")

    # Convert to SkyRL-Agent format
    skyagent_records = []

    for idx, record in enumerate(records):
        # Extract minimal fields only - question and answer
        query_id = record.get("query_id", str(idx))
        question = record.get("query", "")
        answer = record.get("answer", "")

        # Create SkyRL-Agent record with minimal information
        skyagent_record = {
            "id": f"browsecomp_plus_{query_id}",
            "question": question,  # Copied but ignored by SkyRL-Agent
            "golden_answers": (
                [answer] if isinstance(answer, str) else list(answer)
            ),  # Copied but ignored by SkyRL-Agent
            "data_source": "browsecomp_plus",
            "prompt": [{"content": f"Answer the given question: {question}", "role": "user"}],
            "ability": "fact-reasoning",
            "reward_model": {
                "ground_truth": {"target": np.array([answer]) if isinstance(answer, str) else np.array(answer)},
                "style": "rule",
            },
            "extra_info": {
                "original_query_id": query_id,
                # Dropping: gold_docs, negative_docs, evidence_docs (redundant)
            },
            "metadata": None,
        }

        skyagent_records.append(skyagent_record)

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} records...")

    # Convert to DataFrame
    skyagent_df = pd.DataFrame(skyagent_records)

    print(f"Conversion complete! Shape: {skyagent_df.shape}")
    print(f"Columns: {list(skyagent_df.columns)}")

    # Save to parquet
    skyagent_df.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")

    # Show sample
    print("\n" + "=" * 80)
    print("Sample Converted Record (first record):")
    print("=" * 80)
    sample = skyagent_df.iloc[0]
    for col in skyagent_df.columns:
        print(f"\n{col}:")
        if col in ["prompt", "reward_model", "extra_info"]:
            print(f"  {sample[col]}")
        else:
            val = str(sample[col])
            if len(val) > 200:
                print(f"  {val[:200]}...")
            else:
                print(f"  {val}")

    # Show statistics
    print("\n" + "=" * 80)
    print("Dataset Statistics:")
    print("=" * 80)
    print(f"Total records: {len(skyagent_df)}")
    print("Data source: browsecomp_plus")
    print(f"Average question length: {skyagent_df['question'].str.len().mean():.1f} characters")
    print(
        f"Average answer length: {skyagent_df['golden_answers'].apply(lambda x: len(str(x[0])) if x else 0).mean():.1f} characters"
    )

    return skyagent_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert BrowseComp Plus JSONL dataset to SkyRL-Agent parquet format")
    parser.add_argument("-i", "--input", required=True, help="Path to the BrowseComp Plus JSONL file")
    parser.add_argument("-o", "--output", required=True, help="Path to the output parquet file")
    args = parser.parse_args()
    jsonl_file_path = args.input
    output_path = args.output

    print("BrowseComp Plus to SkyRL-Agent Converter")
    print("=" * 80)
    print("Note: Only keeping minimal information (question + answer)")
    print("Dropping: gold_docs, negative_docs, evidence_docs (redundant)")
    print("=" * 80)
    print()

    convert_browsecomp_plus_to_skyagent(jsonl_file_path, output_path)

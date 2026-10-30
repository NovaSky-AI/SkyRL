"""Convert the pinned SDPO Section 3 splits to SkyRL JSONL without resplitting."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

UPSTREAM_COMMIT = "7c457fc1b1f636ae794eb0362ba37d4743b06fbc"
DATASETS = {
    "biology": "datasets/sciknoweval/biology",
    "chemistry": "datasets/sciknoweval/chemistry",
    "material": "datasets/sciknoweval/material",
    "physics": "datasets/sciknoweval/physics",
    "tooluse": "datasets/tooluse",
}


def convert_row(row, split, task):
    messages = []
    if row.get("system") is not None:
        messages.append({"role": "system", "content": row["system"]})
    messages.append({"role": "user", "content": row["prompt"]})
    return {
        "prompt": messages,
        "env_class": "sdpo_tooluse" if task == "tooluse" else "sdpo_mcq",
        "data_source": row["dataset"],
        "ability": row["kind"],
        "reward_model": {"style": row["kind"], "ground_truth": row["answer"]},
        "extra_info": {
            "split": split,
            "index": str(row["idx"]),
            "description": row["description"],
            "problem": row["prompt"],
            "elo": row["elo"],
            "achievement_prior": row.get("achievement_prior", 0) if split == "train" else 0,
        },
    }


def prepare(repo, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "upstream_repository": "https://github.com/lasgroup/SDPO",
        "commit": UPSTREAM_COMMIT,
        "splits": {},
        "format": "SkyRL JSONL; original row order, prompts, answers, split membership",
    }
    for task, directory in DATASETS.items():
        ids = {}
        for split in ("train", "test"):
            relative = f"{directory}/{split}.json"
            raw = subprocess.check_output(["git", "-C", str(repo), "show", f"{UPSTREAM_COMMIT}:{relative}"])
            rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
            ids[split] = {str(row["idx"]) for row in rows}
            if len(ids[split]) != len(rows):
                raise ValueError(f"Duplicate IDs in {task}/{split}")
            converted = [convert_row(row, split, task) for row in rows]
            payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in converted).encode()
            target = output / task / f"{split}.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            manifest["splits"][f"{task}/{split}"] = {
                "rows": len(rows),
                "source_path": relative,
                "source_sha256": hashlib.sha256(raw).hexdigest(),
                "converted_sha256": hashlib.sha256(payload).hexdigest(),
            }
        if ids["train"] & ids["test"]:
            raise ValueError(f"Train/test ID overlap for {task}")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdpo-repo", type=Path, required=True, help="Local clone of lasgroup/SDPO")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.sdpo_repo, args.output_dir), indent=2))


if __name__ == "__main__":
    main()

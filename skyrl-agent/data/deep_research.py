import argparse
import os
import shutil
from pathlib import Path
from typing import List, Tuple

from huggingface_hub import hf_hub_download


# (remote path in repo, desired local filename)
FILES: List[Tuple[str, str]] = [
    ("train/train.parquet", "textbook_stem_5k_with_instance.parquet"),
    ("validation/validation.parquet", "hle_webthinker_converted.parquet"),
]


def download_files(repo: str, output_dir: str, files: List[Tuple[str, str]]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for remote_path, local_name in files:
        local_tmp = hf_hub_download(
            repo_id=repo,
            filename=remote_path,
            repo_type="dataset",
            local_dir=None,
        )
        target = os.path.join(output_dir, local_name)
        shutil.copyfile(local_tmp, target)
        print(f"Saved {remote_path} -> {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Deep Research parquet files.")
    parser.add_argument(
        "--repo",
        default="NovaSky-AI/Deep-Research",
        help="Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "DeepResearch-Data"),
        help="Destination directory for parquet files (defaults next to this script).",
    )
    args = parser.parse_args()

    download_files(args.repo, args.output_dir, FILES)


if __name__ == "__main__":
    main()

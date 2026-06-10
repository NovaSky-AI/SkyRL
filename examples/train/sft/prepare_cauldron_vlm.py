"""Convert HuggingFaceM4/the_cauldron to OpenAI messages format for VLM SFT.

the_cauldron ships each row as ``images`` (a list of PIL images) plus
``texts`` (a list of ``{"user", "assistant", "source"}`` turns). ``SFTTrainer``
expects an OpenAI-style ``messages`` column where image content is encoded as
``{"type": "image", "image": <data-uri>}``. This script does that conversion
and writes the result as Parquet so ``load_dataset(<dir>)`` picks it up.

All images for a row are attached to the first user turn (the standard layout
for chat-template VLM tokenization); remaining turns are text-only. The VLM SFT
path trains on the last assistant message only, so multi-turn rows still
contribute their final response as the supervised target.

Usage::

    uv run examples/train/sft/prepare_cauldron_vlm.py \\
        --output-dir ~/data/cauldron-vlm --config ai2d --num-rows 512
"""

import argparse
import base64
import io
import os

from datasets import load_dataset
from PIL import Image


def _pil_to_data_uri(img: Image.Image) -> str:
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def convert_example(example: dict) -> dict:
    image_content = [{"type": "image", "image": _pil_to_data_uri(img)} for img in example["images"]]

    messages = []
    for i, turn in enumerate(example["texts"]):
        user_content = list(image_content) if i == 0 else []
        user_content.append({"type": "text", "text": turn["user"]})
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": turn["assistant"]})

    return {"messages": messages}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=os.path.expanduser("~/data/cauldron-vlm"),
        help="Directory to write the converted parquet shard into.",
    )
    parser.add_argument(
        "--config",
        default="ai2d",
        help="the_cauldron subset/config name (e.g. ai2d, chart2text, vqav2).",
    )
    parser.add_argument(
        "--num-rows",
        type=int,
        default=None,
        help="Cap on number of rows. Use a small number for smoke tests; None for all.",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    parquet_path = os.path.join(output_dir, "train.parquet")

    split = "train" if args.num_rows is None else f"train[:{args.num_rows}]"
    print(f"[prepare_cauldron_vlm] Loading HuggingFaceM4/the_cauldron config={args.config} split={split} ...")
    ds = load_dataset("HuggingFaceM4/the_cauldron", args.config, split=split)

    print(f"[prepare_cauldron_vlm] Converting {len(ds)} rows -> OpenAI messages ...")
    ds = ds.map(convert_example, remove_columns=ds.column_names, desc="convert")

    os.makedirs(output_dir, exist_ok=True)
    print(f"[prepare_cauldron_vlm] Writing {parquet_path} ...")
    ds.to_parquet(parquet_path)
    print(f"[prepare_cauldron_vlm] Done. {len(ds)} rows -> {parquet_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build SkyRL-format parquet for CholecT50 surgical triplet recognition (RLVR).

Input: verl-format parquet (train/val), as produced by the reference
``verl/scripts/cholec_data.py`` in ft-rl-platform-benchmark:

    data_source   : "cholec_t50"
    prompt        : [{"role": "user", "content": "<image>\\n" + PROMPT}]
    images        : [frame]          (HF Image feature; one frame per row)
    ability       : "recognition"
    reward_model  : {"style": "rule", "ground_truth": <json str>}
    extra_info    : {"video": ..., "frame": ...}

Output: SkyRL-format parquet (one row per frame) that ``PromptDataset`` and
``CholecEnv`` consume directly:

    prompt      : [{"role": "user", "content": [
                       {"type": "image_url", "image_url": {"url": <data uri>}},
                       {"type": "text", "text": PROMPT}]}]
    env_class   : "cholec"
    reward_spec : {"method": "rule", "ground_truth": <json str>}
    extra_info  : {"data_source": ..., "index": ...}

The image is embedded in the prompt as a data URI because SkyRL's neuron renderer
(``NeuronInferenceEngine._extract_images``) reads images from the prompt's
``image_url`` content parts, not a separate ``images`` column. Every frame is
resized to a constant 854x480 grid (matching the TRL trainium2 leg) so the vision
patch count is fixed, then losslessly re-encoded as PNG.

The reward contract matches ``CholecEnv.reward_single``:
    reward = 0.4 * triplet_F1 + 0.6 * mean(instrument_F1, verb_F1, target_F1)
and ``ground_truth`` is the same JSON object
``{"triplets": [...], "instruments": [...], "verbs": [...], "targets": [...]}``
that ``cholec_reward.py`` / ``reward.py`` in ft-rl-platform-benchmark use.

Usage:
    python cholec_data.py --train ~/cholec/parquet/train.parquet \
        --val ~/cholec/parquet/val.parquet --out-dir ~/cholec/skyrl_parquet
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

PROMPT = (
    "You are a surgical activity recognition model. List every surgical action "
    "triplet (instrument-verb-target) visible in the endoscopic frame. Reply only "
    "as comma-separated triplets, each formatted instrument-verb-target. If no "
    "action is visible, reply 'none'."
)

# The CholecT50 frames arrive mixed-size (854x480 and 1920x1080). The TRL
# trainium2 leg resizes every frame to a constant 854x480 grid so the vision
# patch count is fixed and the scoring-forward shape stays stable. Mirror that.
IMAGE_SIZE = (854, 480)


def _image_data_uri(img) -> str:
    """Encode one ``images[0]`` element (a ``{"bytes": ...}`` struct from pyarrow,
    a PIL image, or raw bytes) as a PNG data URI, resized to ``IMAGE_SIZE``."""
    if isinstance(img, dict):
        raw = img.get("bytes")
        if raw is None:
            raise ValueError(f"images[0] struct has no 'bytes': {list(img.keys())}")
        pil = Image.open(io.BytesIO(bytes(raw))).convert("RGB")
    elif isinstance(img, (bytes, bytearray)):
        pil = Image.open(io.BytesIO(bytes(img))).convert("RGB")
    else:
        pil = img.convert("RGB")

    if pil.size != IMAGE_SIZE:
        pil = pil.resize(IMAGE_SIZE)

    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _row_to_skyrl(row, idx: int) -> dict:
    """Convert one verl-format row to a SkyRL-format record."""
    gt = row["reward_model"]["ground_truth"]
    if not isinstance(gt, str):
        gt = json.dumps(gt)
    content = [
        {"type": "image_url", "image_url": {"url": _image_data_uri(row["images"][0])}},
        {"type": "text", "text": PROMPT},
    ]
    return {
        "prompt": [{"role": "user", "content": content}],
        "env_class": "cholec",
        "reward_spec": {"method": "rule", "ground_truth": gt},
        "extra_info": {
            "data_source": row.get("data_source", "cholec_t50"),
            "ability": row.get("ability", "recognition"),
            "index": idx,
        },
    }


def _convert(src: str, dst: str) -> int:
    df = pq.read_table(src).to_pandas()
    records = [_row_to_skyrl(row, i) for i, (_, row) in enumerate(df.iterrows())]
    out = pa.Table.from_pylist(records)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    pq.write_table(out, dst)
    return len(records)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="verl-format train.parquet")
    ap.add_argument("--val", required=True, help="verl-format val.parquet")
    ap.add_argument("--out-dir", default=os.path.expanduser("~/cholec/skyrl_parquet"))
    args = ap.parse_args()

    for split, src in (("train", args.train), ("val", args.val)):
        dst = os.path.join(args.out_dir, f"{split}.parquet")
        n = _convert(src, dst)
        print(f"wrote {n} {split} rows -> {dst}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

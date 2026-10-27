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
``image_url`` content parts, not a separate ``images`` column. The raw encoded
frame bytes are preserved byte-for-byte (PNG/JPEG sniffed by magic bytes) so the
frame is passed to the processor exactly as verl did; no lossy re-encode.

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
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq

PROMPT = (
    "You are a surgical activity recognition model. List every surgical action "
    "triplet (instrument-verb-target) visible in the endoscopic frame. Reply only "
    "as comma-separated triplets, each formatted instrument-verb-target. If no "
    "action is visible, reply 'none'."
)


def _sniff_mime(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _image_data_uri(img) -> str:
    """Encode one ``images[0]`` element (a ``{"bytes": ...}`` struct from pyarrow,
    a PIL image, or raw bytes) as a data URI, preserving the original bytes."""
    if isinstance(img, dict):
        raw = img.get("bytes")
        if raw is None:
            raise ValueError(f"images[0] struct has no 'bytes': {list(img.keys())}")
        raw = bytes(raw)
    elif isinstance(img, (bytes, bytearray)):
        raw = bytes(img)
    else:
        # A PIL image (datasets read path) -- no original bytes, so encode losslessly.
        import io

        from PIL import Image

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw = buf.getvalue()
    mime = _sniff_mime(raw)
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


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

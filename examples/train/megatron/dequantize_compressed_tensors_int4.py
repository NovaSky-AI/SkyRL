"""Dequantize a compressed-tensors pack-quantized INT4 checkpoint to BF16.

Produces the BF16 "master" checkpoint that the Megatron trainer loads via
``trainer.policy.model.fake_int4_qat.bf16_base_path`` when ``model.path`` is an
INT4 release served by the inference engine (Megatron-Bridge cannot load
compressed-tensors checkpoints). See
``skyrl/backends/skyrl_train/workers/megatron/fake_int4_qat.py``.

For QAT-produced INT4 checkpoints (Kimi K2-Thinking / K2.6 / K2.7:
``scale_divisor=7.0, q_min=-7``) the dequantized weights are a fixed point of
the fake-quant STE: re-quantizing them reproduces the stored ``weight_scale``
and codes bit-for-bit, so the trainer's fake-quantized experts equal the grid
the inference engine serves. This property is checked for every quantized
tensor during conversion (disable with ``--no-verify``; RTN checkpoints such as
llm-compressor ``scale_divisor=7.5`` releases will report violations because a
dequantized RTN grid is NOT its own quantization fixed point -- for those, use
the original BF16 release as masters instead of this script).

Unpacking follows compressed-tensors ``unpack_from_int32`` semantics (nibble =
q + 8, little-endian within each int32) and dequantization is the bf16
``q * weight_scale`` per group of ``group_size`` along the input dim -- the
exact arithmetic pinned by ``tests/backends/skyrl_train/test_fake_int4_qat.py``.

Example (Kimi-K2.7-Code, ~595 GB INT4 -> ~2.1 TB BF16):

    uv run --isolated examples/train/megatron/dequantize_compressed_tensors_int4.py \
        --input /path/to/Kimi-K2.7-Code \
        --output /data/skyrl/models/Kimi-K2.7-Code-BF16 \
        --workers 12
"""

import argparse
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

_PACKED_SUFFIX = ".weight_packed"
_SCALE_SUFFIX = ".weight_scale"
_SHAPE_SUFFIX = ".weight_shape"

# Files copied verbatim from the INT4 release (weights + index are regenerated,
# large asset dirs like docs/ and figures/ are skipped).
_SKIP_COPY_DIRS = {"docs", "figures"}


def unpack_int4(packed: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """compressed-tensors ``unpack_from_int32`` (offset encoding: nibble = q + 8)."""
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=packed.device)
    # (rows, cols//8, 8) -> (rows, cols_padded); nibble j of word m is element m*8+j.
    nibbles = (packed.unsqueeze(-1) >> shifts) & 0xF
    return (nibbles.reshape(rows, -1) - 8).to(torch.int8)[:, :cols]


def dequantize(q: torch.Tensor, scale: torch.Tensor, group_size: int) -> torch.Tensor:
    """BF16 ``q * scale`` per group, matching compressed-tensors ``dequantize()``."""
    rows, cols = q.shape
    grouped = q.reshape(rows, cols // group_size, group_size).to(torch.bfloat16)
    return (grouped * scale.unsqueeze(-1)).reshape(rows, cols)


def verify_fixed_point(
    deq: torch.Tensor,
    q: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    scale_divisor: float,
    q_min: float,
) -> tuple[int, int]:
    """Check the dequantized weight is a fixed point of the fake-quant STE.

    Recomputes the INT4 grid from ``deq`` with the same arithmetic as
    ``fake_int4_quantize_ste`` and compares against the stored artifact.
    Returns (bad_scale_groups, bad_code_elements).
    """
    rows, cols = deq.shape
    grouped = deq.reshape(rows, cols // group_size, group_size)
    amax = grouped.abs().amax(dim=-1, keepdim=True).to(torch.float32)
    re_scale = (amax / scale_divisor).to(deq.dtype)
    bad_scale = int((re_scale.squeeze(-1) != scale).sum())
    safe_scale = torch.where(re_scale == 0, torch.ones_like(re_scale), re_scale)
    re_q = torch.clamp(torch.round(grouped / safe_scale), q_min, 7.0)
    bad_codes = int((re_q.to(torch.int8).reshape(rows, cols) != q).sum())
    return bad_scale, bad_codes


def convert_shard(
    input_dir: str,
    output_dir: str,
    shard_name: str,
    group_size: int,
    scale_divisor: float,
    q_min: float,
    verify: bool,
) -> dict:
    """Convert one safetensors shard; returns per-shard verification stats."""
    torch.set_num_threads(max(1, (os.cpu_count() or 8) // 8))
    in_path = Path(input_dir) / shard_name
    out_path = Path(output_dir) / shard_name
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    stats = {"shard": shard_name, "quantized": 0, "copied": 0, "bad_scale": 0, "bad_codes": 0}
    if out_path.exists():
        stats["skipped"] = True
        return stats

    out_tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(in_path), framework="pt") as sf:
        keys = set(sf.keys())
        for key in sorted(keys):
            if key.endswith(_SCALE_SUFFIX) or key.endswith(_SHAPE_SUFFIX):
                continue  # consumed alongside their .weight_packed
            if key.endswith(_PACKED_SUFFIX):
                base = key[: -len(_PACKED_SUFFIX)]
                scale_key, shape_key = base + _SCALE_SUFFIX, base + _SHAPE_SUFFIX
                if scale_key not in keys or shape_key not in keys:
                    raise RuntimeError(f"{shard_name}: {key} missing {scale_key} or {shape_key} in same shard")
                packed = sf.get_tensor(key)
                scale = sf.get_tensor(scale_key)
                rows, cols = sf.get_tensor(shape_key).tolist()
                if scale.dtype != torch.bfloat16:
                    raise RuntimeError(f"{shard_name}: {scale_key} is {scale.dtype}, expected bf16")
                q = unpack_int4(packed, rows, cols)
                deq = dequantize(q, scale, group_size)
                if verify:
                    bad_scale, bad_codes = verify_fixed_point(deq, q, scale, group_size, scale_divisor, q_min)
                    stats["bad_scale"] += bad_scale
                    stats["bad_codes"] += bad_codes
                out_tensors[base + ".weight"] = deq
                stats["quantized"] += 1
            else:
                out_tensors[key] = sf.get_tensor(key)
                stats["copied"] += 1

    save_file(out_tensors, str(tmp_path), metadata={"format": "pt"})
    os.replace(tmp_path, out_path)
    return stats


def strip_quantization_config(config: dict) -> bool:
    """Remove quantization_config (top-level or nested in text_config). Returns True if removed."""
    removed = config.pop("quantization_config", None) is not None
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        removed |= text_config.pop("quantization_config", None) is not None
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="INT4 checkpoint dir (a HF snapshot dir or local path)")
    parser.add_argument("--output", required=True, help="Output dir for the BF16 master checkpoint")
    parser.add_argument("--workers", type=int, default=12, help="Parallel shard converters")
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument(
        "--scale-divisor", type=float, default=7.0, help="Convention to verify against (7.0 Kimi QAT, 7.5 RTN)"
    )
    parser.add_argument("--q-min", type=float, default=-7.0, help="-7 for Kimi QAT, -8 for RTN")
    parser.add_argument("--no-verify", action="store_true", help="Skip the fake-quant fixed-point verification")
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = input_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    shards = sorted(set(index["weight_map"].values()))
    print(f"{len(shards)} shards, {len(index['weight_map'])} tensors, input={input_dir}")

    start = time.time()
    totals = {"quantized": 0, "copied": 0, "bad_scale": 0, "bad_codes": 0}
    skipped_shards = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                convert_shard,
                str(input_dir),
                str(output_dir),
                shard,
                args.group_size,
                args.scale_divisor,
                args.q_min,
                not args.no_verify,
            )
            for shard in shards
        ]
        for i, fut in enumerate(futures):
            stats = fut.result()
            for k in totals:
                totals[k] += stats.get(k, 0)
            skipped_shards += bool(stats.get("skipped"))
            flag = " (already done, skipped)" if stats.get("skipped") else ""
            print(
                f"[{i + 1}/{len(shards)}] {stats['shard']}: {stats['quantized']} dequantized, "
                f"{stats['copied']} copied{flag} ({time.time() - start:.0f}s elapsed)",
                flush=True,
            )

    # Rebuild the index from the converted shards.
    weight_map: dict[str, str] = {}
    total_size = 0
    for shard in shards:
        with safe_open(str(output_dir / shard), framework="pt") as sf:
            for key in sf.keys():
                weight_map[key] = shard
                sl = sf.get_slice(key)
                numel = 1
                for dim in sl.get_shape():
                    numel *= dim
                dtype = str(sl.get_dtype()).lower()
                bytes_per = 2 if ("16" in dtype) else 4 if ("32" in dtype or dtype == "f32") else 1
                total_size += numel * bytes_per
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f, indent=2, sort_keys=True)

    # Copy aux files (tokenizer, modeling code, templates, ...) and patch config.json.
    for entry in sorted(input_dir.iterdir()):
        # Never copy weight files: shards are regenerated by convert_shard, and any
        # safetensors outside the index (consolidated single-file exports, ...) would
        # be stale INT4 data silently overwriting or shadowing the BF16 masters.
        if entry.name.endswith(".safetensors"):
            continue
        if entry.name == "model.safetensors.index.json":
            continue
        if entry.is_dir():
            if entry.name not in _SKIP_COPY_DIRS:
                shutil.copytree(entry, output_dir / entry.name, dirs_exist_ok=True)
            continue
        shutil.copy2(entry, output_dir / entry.name)

    with open(input_dir / "config.json") as f:
        config = json.load(f)
    if strip_quantization_config(config):
        print("stripped quantization_config from config.json")
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, sort_keys=False)

    print(
        f"done in {time.time() - start:.0f}s: {totals['quantized']} tensors dequantized, "
        f"{totals['copied']} copied, total_size={total_size / 1e12:.3f} TB"
    )
    if not args.no_verify:
        if totals["bad_scale"] or totals["bad_codes"]:
            print(
                f"WARNING: fixed-point verification FAILED for scale_divisor={args.scale_divisor}, "
                f"q_min={args.q_min}: {totals['bad_scale']} scale groups, {totals['bad_codes']} codes differ. "
                "The trainer's fake-quantized experts will NOT be bit-exact with the served INT4 grid "
                "(is this an RTN checkpoint? RTN masters must come from the original BF16 release)."
            )
            raise SystemExit(2)
        if skipped_shards:
            # Pre-existing output shards were resumed without re-checking, so a full
            # bit-for-bit claim would be false (e.g. they could come from an earlier
            # --no-verify or differently-parameterized run).
            print(
                f"fixed-point verification PASSED for the {len(shards) - skipped_shards} newly converted "
                f"shard(s); {skipped_shards} pre-existing shard(s) were NOT re-verified -- delete them "
                "(or the output dir) and re-run for a full check"
            )
        else:
            print(
                f"fixed-point verification PASSED: re-quantizing the BF16 masters reproduces the stored "
                f"INT4 grid bit-for-bit (scale_divisor={args.scale_divisor}, q_min={args.q_min})"
            )


if __name__ == "__main__":
    main()

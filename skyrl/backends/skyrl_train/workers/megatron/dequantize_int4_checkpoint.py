"""Dequantize a compressed-tensors pack-quantized INT4 checkpoint into BF16 masters.

Megatron-Bridge cannot load compressed-tensors checkpoints, so fake-INT4 QAT
(see ``fake_int4_qat``) trains from BF16 master weights pointed to by
``trainer.policy.model.fake_int4_qat.bf16_base_path``. For models whose original
BF16 weights were never released (Kimi K2-Thinking / K2.6 / Miles), the only
valid masters are the dequantized INT4 weights themselves: under the Kimi QAT
convention ``(scale_divisor=7.0, q_min=-7)``, fake-quantize of a dequant is the
identity bit-for-bit, so training from the dequant reproduces the served grid
exactly.

This tool streams a pack-quantized safetensors checkpoint shard-by-shard and
writes a plain BF16 HF checkpoint. Each quantized module

    <module>.weight_packed  int32 ``[out, in/8]`` (8 offset-encoded nibbles each)
    <module>.weight_scale   bf16  ``[out, in/group_size]``
    <module>.weight_shape   int64 ``[2]`` (optional)

becomes ``<module>.weight = unpack(weight_packed) * weight_scale`` in the scale
dtype, matching compressed-tensors ``dequantize()`` bit-for-bit (nibble ``j`` of
each int32 holds code ``j``; code = nibble - 8). Every other tensor (attention,
shared experts, dense MLPs, vision tower, lm_head, ...) is copied through
unchanged, and the output ``config.json`` drops the quantization config so the
result loads as a regular BF16 checkpoint.

``--verify`` re-derives each sampled module's grid from the dequantized weights
(recomputed ``weight_scale`` must equal the stored one; fake-quantize must be
the identity). This pins the *convention* claim -- unpack bit-exactness itself
is pinned against real checkpoint slices by
``tests/backends/skyrl_train/test_dequantize_int4_checkpoint.py``.

Usage::

    uv run --isolated -m skyrl.backends.skyrl_train.workers.megatron.dequantize_int4_checkpoint \\
        /path/to/Kimi-K2.6-INT4 /path/to/Kimi-K2.6-BF16-masters --verify sample

Only ``(7.0, -7)``-convention checkpoints yield valid QAT masters through
dequantization. llm-compressor RTN releases (``scale_divisor=7.5``, e.g.
``Qwen3.6-35B-A3B-INT4-RTN``) require the original BF16 weights instead; their
dequant does not reproduce the ``/7.5`` grid, and ``--verify`` fails on them by
design.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from typing import Dict, List

import torch
from loguru import logger
from safetensors import safe_open
from safetensors.torch import save_file

from skyrl.backends.skyrl_train.workers.megatron.fake_int4_qat import (
    fake_int4_quantize_ste,
)

_PACKED_SUFFIX = ".weight_packed"
_SCALE_SUFFIX = ".weight_scale"
_SHAPE_SUFFIX = ".weight_shape"
# Companions of a packed weight that are consumed by dequantization rather than
# copied through. zero_point / g_idx never appear in the symmetric group scheme
# this tool supports; their presence is rejected in _dequantize_module.
_CONSUMED_SUFFIXES = (_SCALE_SUFFIX, _SHAPE_SUFFIX, ".weight_zero_point", ".weight_g_idx")

_PACK_FACTOR = 8  # int4 codes per int32
_CODE_OFFSET = 8  # stored nibble = code + 8

_INDEX_FILE = "model.safetensors.index.json"
_SINGLE_FILE = "model.safetensors"


def unpack_int4_codes(packed: torch.Tensor, out_features: int, in_features: int) -> torch.Tensor:
    """Unpack compressed-tensors int32-packed INT4 codes to int8 ``[out, in]``.

    Nibble ``j`` (bits ``4j..4j+3``) of each int32 holds code ``j`` of its group
    of 8, offset-encoded as ``code + 8``. Columns past ``in_features`` are
    packing padding and are sliced off.
    """
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=packed.device)
    nibbles = (packed.unsqueeze(-1) >> shifts) & 0xF
    codes = (nibbles - _CODE_OFFSET).to(torch.int8).reshape(out_features, -1)
    return codes[:, :in_features]


def dequantize_pack_quantized(
    packed: torch.Tensor,
    scale: torch.Tensor,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """Dequantize one pack-quantized weight to the scale dtype (normally bf16).

    The group multiply happens in the scale dtype, matching compressed-tensors
    ``dequantize()`` (and the grid the inference engine serves) bit-for-bit.
    """
    num_groups = scale.shape[1]
    if in_features % num_groups != 0:
        raise ValueError(f"in_features={in_features} not divisible by weight_scale groups={num_groups}")
    group_size = in_features // num_groups
    q = unpack_int4_codes(packed, out_features, in_features)
    g = q.view(out_features, num_groups, group_size).to(scale.dtype)
    return (g * scale.unsqueeze(-1)).reshape(out_features, in_features)


def _verify_convention(
    name: str, weight: torch.Tensor, scale: torch.Tensor, scale_divisor: float, q_min: float
) -> None:
    """Check that the dequantized weight is a fixed point of the QAT grid.

    Re-derives ``weight_scale`` from the dequant (must equal the stored scale)
    and fake-quantizes the dequant (must be the identity). Holds for the
    ``(7.0, -7)`` Kimi/Miles convention, where every group's max-|w| element
    codes to +-7; fails for llm-compressor RTN (``/7.5``) checkpoints, whose
    dequant is not valid as QAT masters.
    """
    out_features, in_features = weight.shape
    num_groups = scale.shape[1]
    group_size = in_features // num_groups
    g = weight.view(out_features, num_groups, group_size)
    amax = g.abs().amax(dim=-1).to(torch.float32)
    rescale = (amax / scale_divisor).to(weight.dtype)
    scale_ok = torch.equal(rescale, scale)
    requant = fake_int4_quantize_ste(weight, group_size, scale_divisor, q_min)
    identity_ok = torch.equal(requant, weight)
    if scale_ok and identity_ok:
        return
    mismatched = int((requant != weight).sum())
    raise RuntimeError(
        f"{name}: dequantized weights are not a fixed point of the "
        f"(scale_divisor={scale_divisor}, q_min={q_min}) grid "
        f"(scale match: {scale_ok}, {mismatched}/{weight.numel()} elements requantize differently). "
        f"This checkpoint was not produced with that convention -- for llm-compressor RTN "
        f"(scale_divisor=7.5) models, use the original BF16 release as QAT masters instead "
        f"of a dequant."
    )


class _CheckpointReader:
    """Lazy tensor lookup over a sharded (or single-file) safetensors checkpoint."""

    def __init__(self, input_dir: str):
        self.input_dir = input_dir
        index_path = os.path.join(input_dir, _INDEX_FILE)
        single_path = os.path.join(input_dir, _SINGLE_FILE)
        if os.path.exists(index_path):
            with open(index_path, encoding="utf-8") as f:
                self.weight_map: Dict[str, str] = json.load(f)["weight_map"]
        elif os.path.exists(single_path):
            with safe_open(single_path, framework="pt") as f:
                self.weight_map = {name: _SINGLE_FILE for name in f.keys()}
        else:
            raise FileNotFoundError(f"no {_INDEX_FILE} or {_SINGLE_FILE} in {input_dir}")
        self._handles: Dict[str, object] = {}

    def shard_files(self) -> List[str]:
        return sorted(set(self.weight_map.values()))

    def names_in(self, shard: str) -> List[str]:
        return sorted(name for name, f in self.weight_map.items() if f == shard)

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def get(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        if shard not in self._handles:
            self._handles[shard] = safe_open(os.path.join(self.input_dir, shard), framework="pt")
        return self._handles[shard].get_tensor(name)


class _ShardWriter:
    """Buffers output tensors and flushes ~max_shard_bytes safetensors shards.

    Shards are written under temporary names and renamed on finalize, once the
    total count is known: HF-style ``model-XXXXX-of-YYYYY.safetensors`` plus an
    index when there is more than one shard, plain ``model.safetensors``
    otherwise.
    """

    def __init__(self, output_dir: str, max_shard_bytes: int):
        self.output_dir = output_dir
        self.max_shard_bytes = max_shard_bytes
        self._buffer: Dict[str, torch.Tensor] = {}
        self._buffered_bytes = 0
        self._shard_names: List[List[str]] = []
        self.total_bytes = 0

    def add(self, name: str, tensor: torch.Tensor) -> None:
        tensor = tensor.contiguous()
        self._buffer[name] = tensor
        nbytes = tensor.numel() * tensor.element_size()
        self._buffered_bytes += nbytes
        self.total_bytes += nbytes
        if self._buffered_bytes >= self.max_shard_bytes:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        tmp_name = f"tmp-shard-{len(self._shard_names):05d}.safetensors"
        save_file(self._buffer, os.path.join(self.output_dir, tmp_name))
        self._shard_names.append(list(self._buffer.keys()))
        self._buffer = {}
        self._buffered_bytes = 0

    def finalize(self) -> int:
        self._flush()
        num_shards = len(self._shard_names)
        if num_shards == 1:
            os.rename(
                os.path.join(self.output_dir, "tmp-shard-00000.safetensors"),
                os.path.join(self.output_dir, _SINGLE_FILE),
            )
            return num_shards
        weight_map = {}
        for i, names in enumerate(self._shard_names):
            final = f"model-{i + 1:05d}-of-{num_shards:05d}.safetensors"
            os.rename(
                os.path.join(self.output_dir, f"tmp-shard-{i:05d}.safetensors"),
                os.path.join(self.output_dir, final),
            )
            for name in names:
                weight_map[name] = final
        index = {"metadata": {"total_size": self.total_bytes}, "weight_map": weight_map}
        with open(os.path.join(self.output_dir, _INDEX_FILE), "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, sort_keys=True)
        return num_shards


@dataclass
class ConversionStats:
    modules_dequantized: int = 0
    tensors_copied: int = 0
    modules_verified: int = 0
    output_shards: int = 0
    output_bytes: int = 0
    sidecar_files: List[str] = field(default_factory=list)


def _dequantize_module(reader: _CheckpointReader, module: str) -> tuple[torch.Tensor, torch.Tensor]:
    for suffix in (".weight_zero_point", ".weight_g_idx"):
        if reader.has(module + suffix):
            raise NotImplementedError(
                f"{module}{suffix} present: only symmetric group quantization without "
                f"activation reordering is supported"
            )
    packed = reader.get(module + _PACKED_SUFFIX)
    scale = reader.get(module + _SCALE_SUFFIX)
    if reader.has(module + _SHAPE_SUFFIX):
        shape = reader.get(module + _SHAPE_SUFFIX)
        out_features, in_features = int(shape[0]), int(shape[1])
    else:
        # Grouped INT4 checkpoints have in_features % group_size == 0, which is a
        # multiple of the pack factor, so the packed dim carries no padding.
        out_features, in_features = packed.shape[0], packed.shape[1] * _PACK_FACTOR
    weight = dequantize_pack_quantized(packed, scale, out_features, in_features)
    return weight, scale


def _write_config(input_dir: str, output_dir: str) -> bool:
    config_path = os.path.join(input_dir, "config.json")
    if not os.path.exists(config_path):
        return False
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    removed = False
    sections = [config] + [v for v in config.values() if isinstance(v, dict)]
    for section in sections:
        for key in ("quantization_config", "compression_config"):
            removed |= section.pop(key, None) is not None
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    return removed


def _copy_sidecar_files(input_dir: str, output_dir: str) -> List[str]:
    copied = []
    for entry in sorted(os.listdir(input_dir)):
        src = os.path.join(input_dir, entry)
        if not os.path.isfile(src):
            continue
        if entry.endswith(".safetensors") or entry in (_INDEX_FILE, "config.json"):
            continue
        shutil.copy2(src, os.path.join(output_dir, entry))
        copied.append(entry)
    return copied


def convert_checkpoint(
    input_dir: str,
    output_dir: str,
    max_shard_bytes: int = 5 * 1024**3,
    verify: str = "sample",
    scale_divisor: float = 7.0,
    q_min: float = -7.0,
    verify_sample_every: int = 25,
) -> ConversionStats:
    """Convert a pack-quantized INT4 checkpoint directory to BF16 masters.

    ``verify`` is one of ``"off"``, ``"sample"`` (first quantized module and
    every ``verify_sample_every``-th after), or ``"full"``; verification raises
    on the first module whose dequant is not a fixed point of the
    ``(scale_divisor, q_min)`` grid.
    """
    if verify not in ("off", "sample", "full"):
        raise ValueError(f"verify must be off|sample|full, got {verify!r}")
    os.makedirs(output_dir, exist_ok=True)
    reader = _CheckpointReader(input_dir)
    writer = _ShardWriter(output_dir, max_shard_bytes)
    stats = ConversionStats()

    for shard in reader.shard_files():
        logger.info(f"processing {shard}")
        for name in reader.names_in(shard):
            if name.endswith(_PACKED_SUFFIX):
                module = name[: -len(_PACKED_SUFFIX)]
                weight, scale = _dequantize_module(reader, module)
                if verify == "full" or (verify == "sample" and stats.modules_dequantized % verify_sample_every == 0):
                    _verify_convention(module, weight, scale, scale_divisor, q_min)
                    stats.modules_verified += 1
                writer.add(module + ".weight", weight)
                stats.modules_dequantized += 1
            elif any(name.endswith(s) for s in _CONSUMED_SUFFIXES) and reader.has(
                re.sub(r"\.weight_[a-z_]+$", _PACKED_SUFFIX, name)
            ):
                continue
            else:
                writer.add(name, reader.get(name))
                stats.tensors_copied += 1

    stats.output_shards = writer.finalize()
    stats.output_bytes = writer.total_bytes
    if not _write_config(input_dir, output_dir):
        logger.warning("input config.json had no quantization_config to strip (or was missing)")
    stats.sidecar_files = _copy_sidecar_files(input_dir, output_dir)
    logger.info(
        f"dequantized {stats.modules_dequantized} modules "
        f"(verified {stats.modules_verified}), copied {stats.tensors_copied} tensors "
        f"and {len(stats.sidecar_files)} sidecar files -> {stats.output_shards} shard(s), "
        f"{stats.output_bytes / 1024**3:.1f} GiB in {output_dir}"
    )
    return stats


def _parse_size(size: str) -> int:
    match = re.fullmatch(r"(\d+)\s*(GB|MB)", size.strip(), flags=re.IGNORECASE)
    if not match:
        raise argparse.ArgumentTypeError(f"expected e.g. '5GB' or '500MB', got {size!r}")
    return int(match.group(1)) * (1024**3 if match.group(2).upper() == "GB" else 1024**2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input_dir", help="compressed-tensors INT4 checkpoint directory")
    parser.add_argument("output_dir", help="destination for the BF16 masters checkpoint")
    parser.add_argument("--max-shard-size", type=_parse_size, default="5GB", help="output shard size, e.g. 5GB")
    parser.add_argument(
        "--verify",
        choices=("off", "sample", "full"),
        default="sample",
        help="fixed-point verification of the dequant against the QAT grid",
    )
    parser.add_argument(
        "--scale-divisor",
        type=float,
        default=7.0,
        help="QAT grid convention to verify against (7.0 = Kimi/Miles)",
    )
    parser.add_argument("--q-min", type=float, default=-7.0, help="lower INT4 clamp of the verified convention")
    args = parser.parse_args()
    convert_checkpoint(
        args.input_dir,
        args.output_dir,
        max_shard_bytes=args.max_shard_size,
        verify=args.verify,
        scale_divisor=args.scale_divisor,
        q_min=args.q_min,
    )


if __name__ == "__main__":
    main()

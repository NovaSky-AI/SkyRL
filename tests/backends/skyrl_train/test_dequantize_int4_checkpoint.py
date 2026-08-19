"""CPU unit tests for the INT4 -> BF16 checkpoint dequantization tool.

Run with:
uv run --isolated --extra skyrl-train --extra dev pytest -s tests/backends/skyrl_train/test_dequantize_int4_checkpoint.py

The golden tests pin unpack/dequant semantics to slices of REAL quantization
artifacts (see ``_fake_int4_qat_golden.py``): any drift in nibble order, offset
encoding, or the group-multiply dtype diverges from what the inference engine
dequantizes and fails bit-exactly, without needing compressed-tensors or a GPU.
"""

import json
import os

import pytest
import torch
from safetensors.torch import save_file

from skyrl.backends.skyrl_train.workers.megatron.dequantize_int4_checkpoint import (
    convert_checkpoint,
    dequantize_pack_quantized,
    unpack_int4_codes,
)
from tests.backends.skyrl_train._fake_int4_qat_golden import KIMI_QAT, QWEN_RTN

GS = 32


def _load_golden(case: dict):
    rows, cols = case["rows"], case["cols"]
    packed = torch.tensor(case["packed_i32"], dtype=torch.int32)
    scale = torch.tensor(case["scale_u16"], dtype=torch.uint16).view(torch.bfloat16).reshape(rows, cols // GS)
    q_ref = torch.zeros(rows, packed.shape[1] * 8, dtype=torch.int32)
    for j in range(8):
        q_ref[:, j::8] = (packed >> (4 * j)) & 0xF
    q_ref = (q_ref - 8).to(torch.int8)[:, :cols]
    served = (q_ref.view(rows, cols // GS, GS).to(torch.bfloat16) * scale.unsqueeze(-1)).reshape(rows, cols)
    return packed, scale, q_ref, served


def _pack_codes(q: torch.Tensor) -> torch.Tensor:
    """Reference compressed-tensors packer: nibble j of each int32 = code j + 8."""
    rows, cols = q.shape
    pad = (-cols) % 8
    if pad:
        q = torch.cat([q, torch.zeros(rows, pad, dtype=q.dtype)], dim=1)
    nibbles = q.to(torch.int64) + 8
    packed = torch.zeros(rows, q.shape[1] // 8, dtype=torch.int64)
    for j in range(8):
        packed |= nibbles[:, j::8] << (4 * j)
    return torch.where(packed >= 2**31, packed - 2**32, packed).to(torch.int32)


def _make_kimi_module(rows: int, cols: int, seed: int):
    """Random weights quantized with the (7.0, -7) convention; identity-stable."""
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(rows, cols, generator=gen).to(torch.bfloat16)
    g = w.view(rows, cols // GS, GS)
    amax = g.abs().amax(dim=-1, keepdim=True).to(torch.float32)
    scale = (amax / 7.0).to(torch.bfloat16)
    safe = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.clamp(torch.round(g / safe), -7.0, 7.0).to(torch.int8).reshape(rows, cols)
    dequant = (q.view(rows, cols // GS, GS).to(torch.bfloat16) * scale).reshape(rows, cols)
    return _pack_codes(q), scale.squeeze(-1), dequant


@pytest.mark.parametrize("case", [QWEN_RTN, KIMI_QAT], ids=["qwen_rtn_7p5", "kimi_qat_7p0"])
def test_unpack_matches_golden_codes(case):
    packed, _, q_ref, _ = _load_golden(case)
    q = unpack_int4_codes(packed, case["rows"], case["cols"])
    assert q.dtype == torch.int8
    assert torch.equal(q, q_ref)


@pytest.mark.parametrize("case", [QWEN_RTN, KIMI_QAT], ids=["qwen_rtn_7p5", "kimi_qat_7p0"])
def test_dequantize_matches_served_grid(case):
    """Dequant must reproduce the served compressed-tensors grid bit-for-bit."""
    packed, scale, _, served = _load_golden(case)
    weight = dequantize_pack_quantized(packed, scale, case["rows"], case["cols"])
    assert weight.dtype == torch.bfloat16
    assert torch.equal(weight, served)


def test_pack_unpack_roundtrip_with_padding():
    gen = torch.Generator().manual_seed(0)
    q = torch.randint(-8, 8, (16, 40), dtype=torch.int8, generator=gen)
    packed = _pack_codes(q)
    assert packed.shape == (16, 5)
    assert torch.equal(unpack_int4_codes(packed, 16, 40), q)


def _write_checkpoint(path, shards: dict, config: dict | None = None):
    """shards: {shard_filename: {tensor_name: tensor}}. Single-file checkpoints
    use the filename ``model.safetensors`` and get no index."""
    os.makedirs(path, exist_ok=True)
    weight_map = {}
    for shard_name, tensors in shards.items():
        save_file(tensors, os.path.join(path, shard_name))
        for name in tensors:
            weight_map[name] = shard_name
    if list(shards) != ["model.safetensors"]:
        index = {"metadata": {"total_size": 0}, "weight_map": weight_map}
        with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f)
    if config is not None:
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f)


def _load_output(path) -> dict:
    from safetensors import safe_open

    index_path = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            files = sorted(set(json.load(f)["weight_map"].values()))
    else:
        files = ["model.safetensors"]
    out = {}
    for fname in files:
        with safe_open(os.path.join(path, fname), framework="pt") as f:
            for name in f.keys():
                out[name] = f.get_tensor(name)
    return out


_QUANT_CONFIG = {
    "model_type": "test",
    "quantization_config": {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "config_groups": {
            "group_0": {"weights": {"num_bits": 4, "group_size": 32, "symmetric": True, "strategy": "group"}}
        },
        "ignore": ["lm_head"],
    },
}


def test_convert_checkpoint_end_to_end(tmp_path):
    """Sharded input with quantized + passthrough tensors and a cross-shard
    scale; verify=full passes on Kimi-convention weights; config is stripped;
    sidecar files are copied."""
    packed0, scale0, dequant0 = _make_kimi_module(8, 64, seed=1)
    packed1, scale1, dequant1 = _make_kimi_module(4, 128, seed=2)
    bias = torch.randn(8, dtype=torch.bfloat16)
    head = torch.randn(4, 16, dtype=torch.float32)

    in_dir, out_dir = str(tmp_path / "int4"), str(tmp_path / "bf16")
    _write_checkpoint(
        in_dir,
        {
            "model-00001-of-00002.safetensors": {
                "model.layers.0.mlp.experts.0.gate_proj.weight_packed": packed0,
                "model.layers.0.mlp.experts.0.gate_proj.weight_shape": torch.tensor([8, 64]),
                "model.layers.0.mlp.experts.0.gate_proj.bias": bias,
            },
            "model-00002-of-00002.safetensors": {
                # scale of module 0 lives in the other shard: exercises lazy lookup
                "model.layers.0.mlp.experts.0.gate_proj.weight_scale": scale0,
                "model.layers.1.mlp.experts.0.up_proj.weight_packed": packed1,
                "model.layers.1.mlp.experts.0.up_proj.weight_scale": scale1,
                "lm_head.weight": head,
            },
        },
        config=_QUANT_CONFIG,
    )
    with open(os.path.join(in_dir, "tokenizer.json"), "w") as f:
        f.write("{}")

    stats = convert_checkpoint(in_dir, out_dir, verify="full")
    assert stats.modules_dequantized == 2
    assert stats.modules_verified == 2
    assert stats.tensors_copied == 2

    out = _load_output(out_dir)
    assert set(out) == {
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.gate_proj.bias",
        "model.layers.1.mlp.experts.0.up_proj.weight",
        "lm_head.weight",
    }
    assert torch.equal(out["model.layers.0.mlp.experts.0.gate_proj.weight"], dequant0)
    assert torch.equal(out["model.layers.1.mlp.experts.0.up_proj.weight"], dequant1)
    assert torch.equal(out["model.layers.0.mlp.experts.0.gate_proj.bias"], bias)
    assert torch.equal(out["lm_head.weight"], head)

    with open(os.path.join(out_dir, "config.json")) as f:
        out_config = json.load(f)
    assert "quantization_config" not in out_config
    assert out_config["model_type"] == "test"
    assert os.path.exists(os.path.join(out_dir, "tokenizer.json"))


def test_convert_single_file_and_output_sharding(tmp_path):
    """Single-file input; a tiny shard cap forces a sharded, indexed output."""
    packed, scale, dequant = _make_kimi_module(8, 64, seed=3)
    extra = torch.randn(64, 64, dtype=torch.bfloat16)
    in_dir, out_dir = str(tmp_path / "int4"), str(tmp_path / "bf16")
    _write_checkpoint(
        in_dir,
        {
            "model.safetensors": {
                "m.gate_proj.weight_packed": packed,
                "m.gate_proj.weight_scale": scale,
                "m.norm.weight": extra,
            }
        },
        config=_QUANT_CONFIG,
    )
    stats = convert_checkpoint(in_dir, out_dir, max_shard_bytes=1024)
    assert stats.output_shards > 1
    assert os.path.exists(os.path.join(out_dir, "model.safetensors.index.json"))
    out = _load_output(out_dir)
    assert torch.equal(out["m.gate_proj.weight"], dequant)
    assert torch.equal(out["m.norm.weight"], extra)


def test_verify_rejects_rtn_convention(tmp_path):
    """An RTN (/7.5) artifact's dequant is not a fixed point of the (7.0, -7)
    grid, so verification must fail: RTN models need original BF16 masters."""
    packed, scale, _, _ = _load_golden(QWEN_RTN)
    in_dir, out_dir = str(tmp_path / "int4"), str(tmp_path / "bf16")
    _write_checkpoint(
        in_dir,
        {"model.safetensors": {"m.down_proj.weight_packed": packed, "m.down_proj.weight_scale": scale}},
        config=_QUANT_CONFIG,
    )
    with pytest.raises(RuntimeError, match="not a fixed point"):
        convert_checkpoint(in_dir, out_dir, verify="full")
    # the same checkpoint converts fine with verification off
    stats = convert_checkpoint(in_dir, str(tmp_path / "bf16_noverify"), verify="off")
    assert stats.modules_dequantized == 1 and stats.modules_verified == 0


def test_zero_point_rejected(tmp_path):
    packed, scale, _ = _make_kimi_module(8, 64, seed=4)
    in_dir = str(tmp_path / "int4")
    _write_checkpoint(
        in_dir,
        {
            "model.safetensors": {
                "m.gate_proj.weight_packed": packed,
                "m.gate_proj.weight_scale": scale,
                "m.gate_proj.weight_zero_point": torch.zeros(8, 2, dtype=torch.int8),
            }
        },
    )
    with pytest.raises(NotImplementedError, match="weight_zero_point"):
        convert_checkpoint(in_dir, str(tmp_path / "bf16"))


def test_kimi_golden_slice_end_to_end(tmp_path):
    """The real Kimi-K2.6 slice must convert with verification on and reproduce
    the served grid (== its masters, which only exist as the dequant)."""
    packed, scale, _, served = _load_golden(KIMI_QAT)
    in_dir, out_dir = str(tmp_path / "int4"), str(tmp_path / "bf16")
    _write_checkpoint(
        in_dir,
        {"model.safetensors": {"m.gate_proj.weight_packed": packed, "m.gate_proj.weight_scale": scale}},
        config=_QUANT_CONFIG,
    )
    stats = convert_checkpoint(in_dir, out_dir, verify="full")
    assert stats.modules_verified == 1
    out = _load_output(out_dir)
    assert torch.equal(out["m.gate_proj.weight"], served)

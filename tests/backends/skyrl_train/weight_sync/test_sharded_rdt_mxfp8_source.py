"""The MXFP8 wire on sharded_rdt: ``SerializedMxfp8GroupedSource``, CPU-only.

The wrapper must emit exactly what the NCCL MXFP8 wire emits (``iter_serialized_fp8_tensors``)
while keeping the grouped-source contract: whole-model metadata, per-rank ownership, groups in
metadata order, and ``None`` for names this rank does not hold.
"""

from functools import cached_property
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm", reason="sharded_rdt_base imports vllm's weight-transfer ABCs")

pytestmark = pytest.mark.vllm

from skyrl.backends.skyrl_train.weight_sync.fp8 import (  # noqa: E402
    SerializedFp8Config,
    iter_serialized_fp8_tensors,
    resolve_fp8_spec,
)
from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.rdt_fp8 import (  # noqa: E402
    SerializedMxfp8GroupedSource,
)
from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.sharded_rdt_base import (  # noqa: E402
    GroupedWeightSource,
    ParamMeta,
    layerwise_groups,
)

H, F, KV = 64, 32, 96


def _glm5_config(wire="mxfp8") -> SerializedFp8Config:
    spec = resolve_fp8_spec(SimpleNamespace(model_type="glm_moe_dsa"))
    assert spec is not None and spec.name == "glm5"
    return SerializedFp8Config(spec=spec, wire_format=wire)


def _tensors() -> dict:
    g = torch.Generator().manual_seed(0)

    def r(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=g).to(dtype)

    t = {"model.embed_tokens.weight": r(128, H)}
    for layer in range(2):
        p = f"model.layers.{layer}"
        t[f"{p}.input_layernorm.weight"] = r(H)
        t[f"{p}.self_attn.q_a_proj.weight"] = r(H, H)
        t[f"{p}.self_attn.kv_b_proj.weight"] = r(KV, H)  # stays BF16 on the MXFP8 wire
        t[f"{p}.self_attn.indexer.wk.weight"] = r(H, H)  # not a quantized suffix
        t[f"{p}.mlp.gate.weight"] = r(4, H)  # router stays BF16
        t[f"{p}.mlp.gate.e_score_correction_bias"] = r(4, dtype=torch.float32)
        for e in range(4):
            t[f"{p}.mlp.experts.{e}.gate_proj.weight"] = r(F, H)
            t[f"{p}.mlp.experts.{e}.up_proj.weight"] = r(F, H)
            t[f"{p}.mlp.experts.{e}.down_proj.weight"] = r(H, F)
    t["model.norm.weight"] = r(H)
    t["lm_head.weight"] = r(128, H)
    return t


class _DictSource(GroupedWeightSource):
    """Inner source: holds ``held`` (None = everything) and yields None for the rest."""

    def __init__(self, tensors: dict, held=None):
        self._t = tensors
        self._names = [n for g in layerwise_groups(list(tensors)) for n in g]
        self._held = held

    @cached_property
    def metadata(self):
        return [ParamMeta(n, self._t[n].dtype, tuple(self._t[n].shape)) for n in self._names]

    def held_names(self):
        return None if self._held is None else list(self._held)

    def iter_full(self):
        for n in self._names:
            yield n, (self._t[n] if self._held is None or n in self._held else None)


def _nccl_wire(tensors: dict) -> dict:
    """What the NCCL MXFP8 wire sends for the same weights."""
    cfg = _glm5_config()
    out = {}
    for name, tensor in tensors.items():
        for n, t in iter_serialized_fp8_tensors(name, tensor, tensor.dtype, cfg):
            out[n] = t
    return out


def test_stream_is_the_nccl_mxfp8_wire_bit_for_bit():
    t = _tensors()
    src = SerializedMxfp8GroupedSource(_DictSource(t), _glm5_config())
    got = dict(src.iter_full())
    want = _nccl_wire(t)
    assert list(got) == [m.name for m in src.metadata]
    assert set(got) == set(want)
    for name, ref in want.items():
        assert got[name].dtype == ref.dtype, name
        assert got[name].shape == ref.shape, name
        assert torch.equal(got[name].view(torch.uint8), ref.view(torch.uint8)), name


def test_metadata_declares_codes_scales_and_passthrough_dtypes():
    t = _tensors()
    meta = {m.name: m for m in SerializedMxfp8GroupedSource(_DictSource(t), _glm5_config()).metadata}
    q = "model.layers.0.mlp.experts.1.down_proj.weight"
    assert meta[q].dtype == torch.float8_e4m3fn and tuple(meta[q].shape) == (H, F)
    s = "model.layers.0.mlp.experts.1.down_proj.weight_scale"
    assert meta[s].dtype == torch.uint8 and tuple(meta[s].shape) == (H, F // 32)
    for passthrough in (
        "model.layers.0.self_attn.kv_b_proj.weight",
        "model.layers.0.mlp.gate.weight",
        "model.layers.0.input_layernorm.weight",
        "lm_head.weight",
    ):
        assert meta[passthrough].dtype == torch.bfloat16, passthrough
        assert passthrough.replace(".weight", ".weight_scale") not in meta
    assert meta["model.layers.0.mlp.gate.e_score_correction_bias"].dtype == torch.float32


def test_metadata_matches_the_yielded_tensors():
    t = _tensors()
    src = SerializedMxfp8GroupedSource(_DictSource(t), _glm5_config())
    yielded = dict(src.iter_full())
    for m in src.metadata:
        assert yielded[m.name].dtype == m.dtype, m.name
        assert tuple(yielded[m.name].shape) == tuple(m.shape), m.name


def test_groups_are_the_inner_groups_with_scales_added():
    t = _tensors()
    inner = _DictSource(t)
    src = SerializedMxfp8GroupedSource(inner, _glm5_config())
    groups = src.groups()
    assert len(groups) == len(inner.groups())
    for outer, base in zip(groups, inner.groups()):
        assert [n for n in outer if not n.endswith(".weight_scale")] == base
    # iter_groups yields exactly groups(), in order.
    assert [names for names, _ in src.iter_groups()] == groups


def test_ownership_carries_each_scale_with_its_weight_and_foreign_names_stay_none():
    t = _tensors()
    held = [n for n in t if ".experts.3." not in n]  # another EP rank owns expert 3
    src = SerializedMxfp8GroupedSource(_DictSource(t, held=held), _glm5_config())
    held_out = set(src.held_names())
    w = "model.layers.1.mlp.experts.0.up_proj.weight"
    assert w in held_out and w + "_scale" in held_out
    foreign = "model.layers.1.mlp.experts.3.up_proj.weight"
    assert foreign not in held_out and foreign + "_scale" not in held_out
    for names, tensors in src.iter_groups():
        for name, tensor in zip(names, tensors):
            assert (tensor is not None) == (name in held_out), name
    # Whole-model metadata on every rank: foreign weights and scales are still described.
    assert {foreign, foreign + "_scale"} <= {m.name for m in src.metadata}


def test_holding_everything_stays_none():
    src = SerializedMxfp8GroupedSource(_DictSource(_tensors()), _glm5_config())
    assert src.held_names() is None


def test_only_the_mxfp8_wire_is_accepted():
    with pytest.raises(ValueError, match="only the MXFP8"):
        SerializedMxfp8GroupedSource(_DictSource(_tensors()), _glm5_config("blockwise"))

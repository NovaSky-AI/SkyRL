"""``expert_nvfp4`` over sharded_rdt, CPU-only: ``SerializedNvfp4GroupedSource`` emits the push serializer's
ModelOpt NVFP4 tensors and keeps the grouped-source contract; the bake records vLLM's per-expert scalar loads."""

from functools import cached_property

import pytest
import torch

pytest.importorskip("vllm", reason="sharded_rdt_base imports vllm's weight-transfer ABCs")

pytestmark = pytest.mark.vllm

from skyrl.backends.skyrl_train.weight_sync import nvfp4  # noqa: E402
from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.rdt_nvfp4 import (  # noqa: E402
    SerializedNvfp4GroupedSource,
)
from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.sharded_rdt_base import (  # noqa: E402
    GroupedWeightSource,
    ParamMeta,
    layerwise_groups,
)
from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.sharded_rdt_fake import (  # noqa: E402
    BakeSink,
    FakeRDTTensor,
    _UnsupportedFakeOp,
)

H, F = 64, 32
META = torch.device("meta")
_SETTINGS = nvfp4.Nvfp4WeightQuantSettings("four_over_six")


@pytest.fixture(autouse=True)
def fake_quantizer(monkeypatch):
    """The FlashInfer kernel is GPU-only; a content-derived stub keeps the bytes comparable."""

    def quantize(weight, settings):
        rows, cols = weight.shape
        packed = (weight[:, ::2].abs() * 16).to(torch.uint8)
        block_scale = weight.reshape(rows, cols // 16, 16).abs().amax(-1).to(torch.float8_e4m3fn)
        return packed, block_scale, weight.abs().amax().to(torch.float32)

    monkeypatch.setattr(nvfp4, "quantize_nvfp4_matrix", quantize)


def _tensors() -> dict:
    g = torch.Generator().manual_seed(0)

    def r(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=g).to(dtype)

    t = {"model.embed_tokens.weight": r(128, H)}
    for layer in range(3):
        p = f"model.layers.{layer}"
        t[f"{p}.input_layernorm.weight"] = r(H)
        t[f"{p}.self_attn.q_proj.weight"] = r(H, H)
        t[f"{p}.mlp.gate.weight"] = r(4, H)
        t[f"{p}.mlp.gate.e_score_correction_bias"] = r(4, dtype=torch.float32)
        for e in range(4):
            t[f"{p}.mlp.experts.{e}.gate_proj.weight"] = r(F, H)
            t[f"{p}.mlp.experts.{e}.up_proj.weight"] = r(F, H)
            t[f"{p}.mlp.experts.{e}.down_proj.weight"] = r(H, F)
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


def _push_stream(source: _DictSource, bf16_layers=()) -> dict:
    """What the push backends' ``Nvfp4ExpertSerializer`` sends for the same stream."""
    serializer = nvfp4.Nvfp4ExpertSerializer(_SETTINGS, bf16_layers)
    out = dict(item for name, tensor in source.iter_full() for item in serializer(name, tensor))
    serializer.finish()
    return out


def _raw(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(-1).view(torch.uint8)


@pytest.mark.parametrize("bf16_layers", [(), (2,)])
def test_stream_is_the_push_serializers_bytes(bf16_layers):
    inner = _DictSource(_tensors())
    src = SerializedNvfp4GroupedSource(inner, _SETTINGS, bf16_layers)
    got = dict(src.iter_full())
    want = _push_stream(inner, bf16_layers)
    assert list(got) == [m.name for m in src.metadata]
    assert set(got) == set(want)
    for name, ref in want.items():
        assert got[name].dtype == ref.dtype and got[name].shape == ref.shape, name
        assert torch.equal(_raw(got[name]), _raw(ref)), name


def test_metadata_matches_the_stream_and_only_routed_experts_quantize():
    src = SerializedNvfp4GroupedSource(_DictSource(_tensors()), _SETTINGS, bf16_layers=[2])
    yielded = dict(src.iter_full())
    for m in src.metadata:
        assert yielded[m.name].dtype == m.dtype and tuple(yielded[m.name].shape) == tuple(m.shape), m.name
    meta = {m.name: m for m in src.metadata}
    e = "model.layers.0.mlp.experts.1"
    assert meta[f"{e}.gate_proj.weight"].dtype == torch.uint8
    assert tuple(meta[f"{e}.down_proj.weight_scale"].shape) == (H, F // 16)
    assert meta[f"{e}.up_proj.weight_scale_2"].dtype == torch.float32
    assert torch.equal(yielded[f"{e}.gate_proj.weight_scale_2"], yielded[f"{e}.up_proj.weight_scale_2"])
    assert meta["model.layers.2.mlp.experts.1.gate_proj.weight"].dtype == torch.bfloat16  # BF16 layer
    assert meta["model.layers.0.mlp.gate.weight"].dtype == torch.bfloat16
    assert meta["model.layers.0.mlp.gate.e_score_correction_bias"].dtype == torch.float32


def test_ownership_carries_all_four_tensors_and_foreign_experts_stay_none():
    t = _tensors()
    held = [n for n in t if ".experts.3." not in n]  # another EP rank owns expert 3
    src = SerializedNvfp4GroupedSource(_DictSource(t, held=held), _SETTINGS)
    held_out = set(src.held_names())
    base = "model.layers.1.mlp.experts.0.up_proj"
    assert {f"{base}.{s}" for s in ("weight", "weight_scale", "weight_scale_2", "input_scale")} <= held_out
    assert not any(".experts.3." in n for n in held_out)
    for names, tensors in src.iter_groups():
        for name, tensor in zip(names, tensors):
            assert (tensor is not None) == (name in held_out), name
    assert "model.layers.1.mlp.experts.3.gate_proj.weight_scale_2" in {m.name for m in src.metadata}
    groups = src.groups()
    assert [names for names, _ in src.iter_groups()] == groups


def test_gate_and_up_must_be_held_together():
    t = _tensors()
    held = [n for n in t if n != "model.layers.0.mlp.experts.0.up_proj.weight"]
    src = SerializedNvfp4GroupedSource(_DictSource(t, held=held), _SETTINGS)
    with pytest.raises(RuntimeError, match="unpaired gate/up"):
        list(src.iter_groups())


def test_a_source_without_per_expert_weights_is_rejected():
    t = {k: v for k, v in _tensors().items() if ".experts." not in k}
    with pytest.raises(ValueError, match="no per-expert routed-expert weights"):
        SerializedNvfp4GroupedSource(_DictSource(t), _SETTINGS).metadata


class TestBakeRecordsVllmScalarLoads:
    """vLLM's fused-MoE loader writes ModelOpt's per-expert scalars as ``param[e][i] = w.reshape(())``
    (``_load_per_tensor_weight_scale``) and moves input scales with ``w.to(param.device)``."""

    def _fake(self, rec, shape=()):
        return FakeRDTTensor(name="s", shape=torch.Size(shape), dtype=torch.float32, device=META, sink=rec)

    def test_indexed_assignment_records_the_slot(self):
        rec = BakeSink()
        layer = torch.nn.Module()
        param = torch.empty((8, 2), dtype=torch.float32, device=META)
        rec.current = (layer, "w13_weight_scale_2")
        param[5][1] = self._fake(rec).reshape(())
        (sc,) = rec.copies_by_layer[layer]
        assert sc.src == ("s", (("reshape", ((),), ()),))
        assert sc.offset == 11 and sc.shape == () and sc.nbytes == 4
        assert "s" in rec.copied_names

    def test_device_only_to_is_identity_and_dtype_change_still_rejected(self):
        rec = BakeSink()
        fake = self._fake(rec)
        assert fake.to(META) is fake
        assert fake.to(device=META, non_blocking=True) is fake
        assert fake.to(torch.float32) is fake
        with pytest.raises(_UnsupportedFakeOp):
            fake.to(torch.bfloat16)

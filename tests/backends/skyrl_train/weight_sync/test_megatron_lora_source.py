"""MegatronLoraAdapterSource on a shared-outer expert adapter with per-expert HF keys, and the sender
manifest of the in-memory LoRA sync."""

import json
import os
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron.bridge")
pytest.importorskip("vllm")

from skyrl.backends.skyrl_train.weight_sync.sources import (  # noqa: E402
    LoraAdapterWeightSource,
    MegatronLoraAdapterSource,
)

P = "base_model.model.language_model.model.layers.1.mlp."
E, R, H, F = 3, 4, 8, 6


def _export():
    """What Megatron-Bridge exports for one shared-outer MoE layer of a per-expert-key model (ML4)."""
    g = torch.Generator().manual_seed(0)
    out = {
        P + "experts.gate_proj.lora_A.weight": torch.randn(1, R, H, generator=g),
        P + "experts.up_proj.lora_A.weight": torch.randn(1, R, H, generator=g),
        P + "experts.down_proj.lora_B.weight": torch.randn(1, H, R, generator=g),
    }
    for e in range(E):
        out[P + f"experts.{e}.gate_proj.lora_B.weight"] = torch.randn(F, R, generator=g)
        out[P + f"experts.{e}.up_proj.lora_B.weight"] = torch.randn(F, R, generator=g)
        out[P + f"experts.{e}.down_proj.lora_A.weight"] = torch.randn(R, F, generator=g)
    return out


def _source(num_moe_experts):
    return MegatronLoraAdapterSource(
        dtype=torch.float32,
        experts_per_shared_adapter=1,
        bridge=None,
        actor_module=None,
        lora_cls=SimpleNamespace(dim=R, alpha=2 * R, dropout=0.0),
        base_model_name_or_path="m",
        num_moe_experts=num_moe_experts,
    )


class TestSharedOuterExpansion:
    def test_shared_side_expanded_to_every_expert(self):
        exported = _export()
        state, config = _source(E).finalize_adapter(dict(exported))
        assert config["r"] == config["lora_alpha"] == R
        for e in range(E):
            for proj, side in (("gate_proj", "A"), ("up_proj", "A"), ("down_proj", "B")):
                key = P + f"experts.{e}.{proj}.lora_{side}.weight"
                shared = exported[P + f"experts.{proj}.lora_{side}.weight"][0]
                assert torch.equal(state[key], shared * 2 if side == "B" else shared), key
            assert torch.equal(
                state[P + f"experts.{e}.gate_proj.lora_B.weight"],
                exported[P + f"experts.{e}.gate_proj.lora_B.weight"] * 2,
            )
        assert not any(".experts.gate_proj." in k or ".experts.down_proj." in k for k in state)
        assert len(state) == 6 * E

    def test_missing_expert_count_is_an_error(self):
        with pytest.raises(ValueError, match="num_moe_experts"):
            _source(None).finalize_adapter(_export())


class _Stub(LoraAdapterWeightSource):
    def __init__(self, state):
        super().__init__(dtype=torch.bfloat16, experts_per_shared_adapter=2)
        self._state = state

    def export_adapter_stream(self):
        yield from self._state.items()


class TestSenderManifest:
    def test_manifest_is_the_wire(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SKYRL_LORA_MANIFEST_DIR", str(tmp_path))
        a = torch.randn(R, H)
        state = {
            P + "experts.0.down_proj.lora_A.weight": a,
            P + "experts.1.down_proj.lora_A.weight": a.clone(),
            P + "experts.2.down_proj.lora_A.weight": torch.randn(R, H),
            P + "shared_experts.down_proj.lora_A.weight": torch.randn(R, H),
        }
        src = _Stub(state)
        src.set_lora_name("x")
        src.prepare()
        wire = dict(iter(src))
        files = os.listdir(tmp_path)
        assert len(files) == 1 and files[0].startswith("sender-x-")
        m = json.load(open(tmp_path / files[0]))
        from skyrl.backends.skyrl_train.weight_sync.lora_manifest import tensor_entry

        assert set(m["tensors"]) == set(state)
        for k, t in wire.items():
            assert m["tensors"][k] == tensor_entry(t)
        assert (
            m["tensors"][P + "experts.1.down_proj.lora_A.weight"]
            == m["tensors"][P + "experts.0.down_proj.lora_A.weight"]
        )
        assert m["tensors"][P + "experts.0.down_proj.lora_A.weight"]["dtype"] == "bfloat16"

    def test_no_manifest_when_disabled(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SKYRL_LORA_MANIFEST_DIR", raising=False)
        src = _Stub({P + "shared_experts.down_proj.lora_A.weight": torch.randn(R, H)})
        src.set_lora_name("x")
        src.prepare()
        list(iter(src))
        assert os.listdir(tmp_path) == []


class TestHostStagedExport:
    def test_export_is_staged_on_the_host(self):
        seen = {}

        class Bridge:
            def export_adapter_weights(self, module, **kw):
                seen.update(kw)
                yield "language_model.model.layers.0.self_attn.q_proj.lora_A.weight", torch.ones(2, 2)

        src = _source(E)
        src._bridge = Bridge()
        names = [n for n, _ in src.export_adapter_stream()]
        assert names == ["base_model.model.language_model.model.layers.0.self_attn.q_proj.lora_A.weight"]
        assert seen["cpu"] is True

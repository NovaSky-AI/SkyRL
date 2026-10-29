"""Shared-outer expert LoRA on ML4 (``lora.experts_shared_outer_loras``).

Layout (Megatron-Bridge ``SharedOuterGroupedExpertAdapter``, baseline hparams 1.1): per MoE layer,
gate / up lora_A and down lora_B are one tensor shared by all routed experts; gate / up lora_B and
down lora_A stay per expert. The PEFT state dict, the in-memory sync stream and the PEFT export are
the expanded per-expert adapter in vLLM's default format, with the outer factors bitwise identical
across the experts of a layer (baseline hparams 4).
"""

import re
from types import SimpleNamespace

import pytest
import torch

OUTER = (("gate_proj", "lora_A"), ("up_proj", "lora_A"), ("down_proj", "lora_B"))
INNER = (("gate_proj", "lora_B"), ("up_proj", "lora_B"), ("down_proj", "lora_A"))
EXPERT_KEY = re.compile(
    r"(?P<layer>.*\.layers\.\d+)\.mlp\.experts\.(?P<e>\d+)\.(?P<proj>\w+)\.(?P<side>lora_[AB])\.weight"
)


def _moe_layers(peft_model):
    return [m for m in peft_model.base_model.model.language_model.model.layers if hasattr(m.mlp, "experts")]


def _factor(expert, proj, side):
    return getattr(getattr(expert, proj), side)["default"].weight


def _randomize(peft_model, seed=1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in peft_model.named_parameters():
            if "lora_" in n:
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)


def _per_expert_copy(shared_model, build):
    """An untied model whose adapter is the expansion of ``shared_model``'s (its state dict)."""
    per = build()
    per.load_state_dict(shared_model.state_dict(), strict=True)
    assert not any(
        _factor(layer.mlp.experts[0], proj, side) is _factor(layer.mlp.experts[1], proj, side)
        for layer in _moe_layers(per)
        for proj, side in OUTER
    )
    return per


def _assert_expanded(adapter: dict, n_experts: int):
    """Baseline hparams 4: every routed expert present; outer factors bitwise equal within a layer."""
    groups = {}
    for k, t in adapter.items():
        m = EXPERT_KEY.fullmatch(k)
        if m:
            groups.setdefault((m["layer"], m["proj"], m["side"]), {})[int(m["e"])] = t
    assert groups
    for (layer, proj, side), per_e in groups.items():
        assert sorted(per_e) == list(range(n_experts)), (layer, proj, side)
        same = all(torch.equal(per_e[0], t) for t in per_e.values())
        if (proj, side) in OUTER:
            assert same, (layer, proj, side)
        else:
            assert not same, (layer, proj, side)  # randomized inner factors stay per expert
    return groups


class TestLayout:
    def test_outer_tied_inner_per_expert(self, ml4_model_dir):
        from conftest import load_ml4_wrapper

        per, _ = load_ml4_wrapper(ml4_model_dir)
        shared, _ = load_ml4_wrapper(ml4_model_dir, experts_shared_outer_loras=True)
        layers = _moe_layers(shared.model)
        n_exp = len(layers[0].mlp.experts)
        assert layers and n_exp > 1
        for layer in layers:
            e0 = layer.mlp.experts[0]
            for e in layer.mlp.experts[1:]:
                assert all(_factor(e, p, s) is _factor(e0, p, s) for p, s in OUTER)
                assert all(_factor(e, p, s) is not _factor(e0, p, s) for p, s in INNER)
            # Init as Megatron-Bridge's shared side: lora_A from the A init, lora_B zero.
            assert all(_factor(e0, p, "lora_A").abs().sum() > 0 for p in ("gate_proj", "up_proj"))
            assert not _factor(e0, "down_proj", "lora_B").any()
            # Shared experts and the gate/up pair are not tied.
            se = layer.mlp.shared_experts
            assert _factor(se, "gate_proj", "lora_A") is not _factor(e0, "gate_proj", "lora_A")
            assert _factor(e0, "gate_proj", "lora_A") is not _factor(e0, "up_proj", "lora_A")

        # One trainable tensor per shared factor: 3 * (E - 1) fewer per MoE layer.
        n_per = sum(p.requires_grad for p in per.model.parameters())
        n_shared = sum(p.requires_grad for p in shared.model.parameters())
        assert n_per - n_shared == 3 * (n_exp - 1) * len(layers)
        # The state dict keeps every per-expert key (expanded), as the untied model has.
        assert set(shared.model.state_dict()) == set(per.model.state_dict())

    def test_no_expert_linears_raises(self):
        from peft import LoraConfig, get_peft_model
        from torch import nn

        from skyrl.backends.skyrl_train.workers.model_wrapper import (
            tie_shared_outer_expert_loras,
        )

        model = get_peft_model(nn.Sequential(nn.Linear(4, 4)), LoraConfig(r=2, target_modules=["0"]))
        with pytest.raises(ValueError, match="no per-expert LoRA linears"):
            tie_shared_outer_expert_loras(model)


def _nvfp4_lora_model(ckpt, dtype, shared, **over):
    """ML4 on the fake NVFP4 checkpoint (activation QAT on, rollout TP 2) with PEFT LoRA r=4."""
    from fake_nvfp4 import nvfp4_params
    from ml4_train.configuration_ml4 import config_from_params
    from ml4_train.loading import load_sharded
    from ml4_train.lora import LORA_TARGET_REGEX
    from ml4_train.modeling_ml4 import ML4ForCausalLM
    from peft import LoraConfig, get_peft_model

    from skyrl.backends.skyrl_train.workers.model_wrapper import (
        tie_shared_outer_expert_loras,
    )

    cfg = config_from_params(
        nvfp4_params(), rollout_tp=2, rollout_moe_tp=2, rollout_moe_activation="nvfp4_per_token", **over
    )
    with torch.device("meta"):
        m = ML4ForCausalLM(cfg)
    m.to(dtype)
    load_sharded(m, ckpt)
    m = get_peft_model(m, LoraConfig(r=4, lora_alpha=8, target_modules=LORA_TARGET_REGEX))
    if shared:
        tie_shared_outer_expert_loras(m)
    _randomize(m)
    return m


@pytest.fixture(scope="module")
def nvfp4_ckpt(tmp_path_factory):
    from conftest import ML4_TRAIN_PATH

    if not ML4_TRAIN_PATH:
        pytest.skip("ML4_TRAIN_PATH not set")
    from fake_nvfp4 import write_fake_nvfp4_ckpt

    d = str(tmp_path_factory.mktemp("ml4_nvfp4"))
    write_fake_nvfp4_ckpt(d)
    return d


def _packed():
    """Two packed sequences (20 + 13 tokens)."""
    from fake_ckpt import TINY_PARAMS

    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, TINY_PARAMS["vocab_size"], (1, 33), generator=g)
    return ids, torch.cat([torch.arange(20), torch.arange(13)])[None]


def _fwd_bwd(model):
    ids, pos = _packed()
    logits = model(input_ids=ids, position_ids=pos).logits
    (logits.float() * torch.linspace(-1, 1, logits.numel()).view_as(logits)).sum().backward()
    return logits.detach()


class TestNumerics:
    @pytest.mark.parametrize(
        "dtype,grouped,rtol",
        [(torch.float32, True, 1e-5), (torch.float32, False, 1e-5), (torch.bfloat16, True, 2e-2)],
        ids=["fp32-grouped", "fp32-loop", "bf16-grouped"],
    )
    def test_forward_equal_and_shared_grad_is_sum(self, nvfp4_ckpt, dtype, grouped, rtol):
        """Shared-outer vs the per-expert model holding the expanded factors, NVFP4 experts with QAT.

        Forward and every non-shared gradient are bitwise equal (same values, same ops); the shared
        factor's gradient is the sum of the expanded copies' gradients (up to summation order).
        """
        torch.manual_seed(0)
        shared = _nvfp4_lora_model(nvfp4_ckpt, dtype, True, moe_grouped_mm=grouped)
        per = _per_expert_copy(shared, lambda: _nvfp4_lora_model(nvfp4_ckpt, dtype, False, moe_grouped_mm=grouped))
        assert all(m.mlp.nvfp4_fc2_shards == 2 and m.mlp.grouped_mm == grouped for m in _moe_layers(shared))
        assert torch.equal(_fwd_bwd(shared), _fwd_bwd(per))

        shared_ids = {id(_factor(layer.mlp.experts[0], p, s)) for layer in _moe_layers(shared) for p, s in OUTER}
        g_per = {n: p.grad for n, p in per.named_parameters() if p.requires_grad}
        n_checked = 0
        for n, p in shared.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) not in shared_ids:
                assert torch.equal(p.grad, g_per[n]), n
                continue
            m = EXPERT_KEY.fullmatch(n.replace(".default.weight", ".weight"))
            expanded = [
                k
                for k in g_per
                if k.startswith(f"{m['layer']}.mlp.experts.") and k.endswith(f".{m['proj']}.{m['side']}.default.weight")
            ]
            want = torch.stack([g_per[k].float() for k in expanded]).sum(0)
            assert len(expanded) == len(_moe_layers(shared)[0].mlp.experts)
            assert want.abs().sum() > 0, n
            torch.testing.assert_close(p.grad.float(), want, rtol=rtol, atol=rtol * want.abs().max().item(), msg=n)
            n_checked += 1
        assert n_checked == len(shared_ids)


class TestExport:
    def test_sync_stream_is_the_expanded_adapter(self, ml4_model_dir, vllm_weight_transfer):
        """In-memory sync of a shared-outer adapter = the stream of the untied model holding its expansion."""
        from conftest import load_ml4_wrapper

        from skyrl.backends.skyrl_train.distributed.fsdp_utils import (
            lora_adapter_config,
        )
        from skyrl.backends.skyrl_train.weight_sync.sources import FsdpLoraAdapterSource
        from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import (
            FSDPPolicyWorkerBase,
        )

        def stream(wrapper):
            worker = SimpleNamespace(model=wrapper, _is_multimodal_lm_only=False)
            source = FsdpLoraAdapterSource(
                dtype=torch.bfloat16,
                adapter_state=lambda: FSDPPolicyWorkerBase._lora_adapter_state(worker),
                adapter_config=lora_adapter_config(wrapper.model),
            )
            source.set_lora_name("ml4")
            source.prepare()
            meta = [(m.name, m.shape) for m in source.metadata()]
            sent = list(iter(source))
            assert meta == [(k, tuple(t.shape)) for k, t in sent]
            return sent

        shared, _ = load_ml4_wrapper(ml4_model_dir, experts_shared_outer_loras=True)
        _randomize(shared.model)
        per, _ = load_ml4_wrapper(ml4_model_dir)
        per.model.load_state_dict(shared.model.state_dict(), strict=True)

        sent_shared, sent_per = stream(shared), stream(per)
        assert [k for k, _ in sent_shared] == [k for k, _ in sent_per]
        for (k, a), (_, b) in zip(sent_shared, sent_per):
            assert a.is_contiguous() and torch.equal(a, b), k
        _assert_expanded(dict(sent_shared), len(_moe_layers(shared.model)[0].mlp.experts))

    def test_peft_export_round_trip(self, ml4_model_dir):
        """The PEFT export (disk sync and checkpoint ``lora_adapter``) loaded into an untied per-expert
        model, as vLLM loads it, reproduces the shared-outer model's logits bitwise."""
        from conftest import load_ml4_wrapper
        from peft import set_peft_model_state_dict

        from skyrl.backends.skyrl_train.distributed.fsdp_utils import (
            collect_lora_params,
            layered_summon_lora_params,
        )

        shared, _ = load_ml4_wrapper(ml4_model_dir, experts_shared_outer_loras=True)
        _randomize(shared.model)
        exported = collect_lora_params(shared.model)
        summoned = layered_summon_lora_params(shared.model)
        assert set(summoned) == set(exported) and all(torch.equal(summoned[k], exported[k]) for k in exported)
        _assert_expanded(exported, len(_moe_layers(shared.model)[0].mlp.experts))

        per, _ = load_ml4_wrapper(ml4_model_dir)
        res = set_peft_model_state_dict(per.model, exported)
        assert not [k for k in res.missing_keys if "lora_" in k] and not res.unexpected_keys

        ids, pos = _packed()
        with torch.no_grad():
            want = shared.model(input_ids=ids, position_ids=pos).logits
            got = per.model(input_ids=ids, position_ids=pos).logits
        assert torch.equal(got, want)

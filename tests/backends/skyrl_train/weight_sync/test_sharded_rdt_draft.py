"""CPU tests for syncing a spec-decode drafter over ``sharded_rdt``.

The consumer engine bakes the drafter's own modules in the target's sessions: the target's
``load_weights`` skips the MTP layer's names, and the drafter's picks them out of the rest, as both do
from a full checkpoint at engine load. Runs vLLM's real layerwise reload over a toy target and a toy
drafter that shares the target's embedding and LM head, as vLLM's MTP drafters do.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm")

from vllm.model_executor.model_loader import reload  # noqa: E402
from vllm.model_executor.model_loader.reload import (  # noqa: E402
    record_metadata_for_reloading,
)
from vllm.model_executor.model_loader.weight_utils import (  # noqa: E402
    default_weight_loader,
)

from skyrl.backends.skyrl_train.inference_servers.draft_model import (  # noqa: E402
    attach_draft_model,
)
from skyrl.backends.skyrl_train.patches.vllm import (  # noqa: E402
    patch_moe_kernel_constants,
)
from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.sharded_rdt_engine import (  # noqa: E402
    SkyRLShardedRDTWeightTransferEngine,
    _unshared_submodules,
)

pytestmark = pytest.mark.vllm


def _load(module, weights, own):
    """Load the names ``own`` accepts and skip the rest, as vLLM's models do with a full checkpoint."""
    params = dict(module.named_parameters())
    loaded = set()
    for name, tensor in weights:
        if not own(name):
            continue
        param = params[name]
        getattr(param, "weight_loader", default_weight_loader)(param, tensor)
        loaded.add(name)
    return loaded


class _Target(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(8, 4)
        self.lm_head = torch.nn.Linear(4, 8, bias=False)

    def load_weights(self, weights):
        return _load(self, weights, lambda name: not name.startswith("layers.2."))


class _SharedHead(torch.nn.Module):
    def __init__(self, head):
        super().__init__()
        self.norm = torch.nn.LayerNorm(4)
        self.head = head


class _MtpLayer(torch.nn.Module):
    def __init__(self, head):
        super().__init__()
        self.eh_proj = torch.nn.Linear(8, 4, bias=False)
        self.shared_head = _SharedHead(head)


class _Draft(torch.nn.Module):
    """vLLM's MTP drafter, reduced: its embedding and output head are the target's modules."""

    def __init__(self, target):
        super().__init__()
        self.embed_tokens = target.embed_tokens
        self.layers = torch.nn.ModuleDict({"2": _MtpLayer(target.lm_head)})

    def load_weights(self, weights):
        return _load(self, weights, lambda name: name.startswith("layers.2."))


class _DraftWithOwnHead(_Draft):
    """A drafter whose LM head is its own, not the target's: the target's load_weights takes its name."""

    def __init__(self, target):
        super().__init__(target)
        self.lm_head = torch.nn.Linear(4, 8, bias=False)

    def load_weights(self, weights):
        return _load(self, weights, lambda name: name.startswith("layers.2.") or name == "lm_head.weight")


class _DraftWithKvScales(_Draft):
    """A drafter holding vLLM's attention layer under a quantized KV cache: scale params no checkpoint needs."""

    def __init__(self, target):
        super().__init__(target)
        self.attn = torch.nn.Module()
        self.attn.k_scale = torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False)


class _DraftSharingNothing(_Draft):
    """A drafter with its own embedding and LM head: it shares no module with the target, so it is one unshared unit."""

    def __init__(self, target):
        super().__init__(target)
        self.embed_tokens = torch.nn.Embedding(8, 4)
        self.layers["2"].shared_head.head = torch.nn.Linear(4, 8, bias=False)


TARGET_NAMES = {"embed_tokens.weight": [8, 4], "lm_head.weight": [8, 4]}
DRAFT_NAMES = {
    "layers.2.eh_proj.weight": [4, 8],
    "layers.2.shared_head.norm.weight": [4],
    "layers.2.shared_head.norm.bias": [4],
}


def _engine(names, draft_cls=_Draft):
    target = _Target()
    draft = draft_cls(target)
    record_metadata_for_reloading(target)
    record_metadata_for_reloading(draft)
    engine = SkyRLShardedRDTWeightTransferEngine.__new__(SkyRLShardedRDTWeightTransferEngine)
    engine.model, engine.device = target, torch.device("cpu")
    engine._name_to_plan, engine._post_load_fns, engine._draft = {}, {}, None
    engine.skyrl_set_draft_model(draft, SimpleNamespace())
    init_info = SimpleNamespace(names=list(names), dtype_names=["float32"] * len(names), shapes=list(names.values()))
    return engine, target, draft, init_info


def test_unshared_submodules_keeps_the_target_modules_out():
    target = _Target()
    draft = _Draft(target)
    layer = draft.layers["2"]
    assert list(_unshared_submodules(draft, target)) == [layer.eh_proj, layer.shared_head.norm]


def test_bake_plans_the_drafter_layers_on_the_drafter_and_restores_both_models():
    engine, target, draft, init_info = _engine({**TARGET_NAMES, **DRAFT_NAMES})

    engine._bake(init_info)

    layer = draft.layers["2"]
    owner = {name: {sc.layer for sc in plan} for name, plan in engine._name_to_plan.items()}
    assert owner == {
        "embed_tokens.weight": {target.embed_tokens},
        "lm_head.weight": {target.lm_head},
        "layers.2.eh_proj.weight": {layer.eh_proj},
        "layers.2.shared_head.norm.weight": {layer.shared_head.norm},
        "layers.2.shared_head.norm.bias": {layer.shared_head.norm},
    }
    assert engine._live_names == set(TARGET_NAMES) | set(DRAFT_NAMES)
    # The dry run leaves every param real, the shared ones included.
    assert not any(p.is_meta for p in [*target.parameters(), *draft.parameters()])


def test_start_weight_update_reloads_the_drafter_but_not_the_shared_modules_twice():
    engine, target, draft, _ = _engine(TARGET_NAMES)
    engine.start_weight_update()
    layer = draft.layers["2"]
    assert layer.eh_proj.weight.is_meta and layer.shared_head.norm.weight.is_meta
    # The target's own reload owns the shared modules.
    assert target.embed_tokens.weight.is_meta and target.lm_head.weight.is_meta


def test_finish_weight_update_restores_the_drafters_moe_kernel_constants_too(monkeypatch):
    """A sleep for the sync zeroes the MoE kernel constants of the drafter's experts as well as the target's."""
    restored = []
    monkeypatch.setattr(reload, "finalize_layerwise_reload", lambda model, model_config: None)
    monkeypatch.setattr(patch_moe_kernel_constants, "restore_moe_kernel_constants", restored.append)
    engine, target, draft, _ = _engine(TARGET_NAMES)
    engine.model_config = None
    engine.drain_pending = lambda: None

    engine.finish_weight_update()

    assert restored == [target, draft]


def test_bake_refuses_a_plan_without_drafter_layers():
    engine, _, _, init_info = _engine(TARGET_NAMES)
    with pytest.raises(RuntimeError, match="loads nothing into the spec-decode drafter"):
        engine._bake(init_info)


def test_bake_refuses_a_drafter_module_the_plan_leaves_unloaded():
    engine, _, _, init_info = _engine({**TARGET_NAMES, **DRAFT_NAMES}, draft_cls=_DraftWithOwnHead)
    with pytest.raises(RuntimeError, match=r"drafter's \['lm_head'\]"):
        engine._bake(init_info)


def test_bake_refuses_any_unloaded_module_of_a_drafter_that_shares_nothing():
    names = {**TARGET_NAMES, **DRAFT_NAMES, "layers.2.shared_head.head.weight": [8, 4]}
    engine, _, draft, init_info = _engine(names, draft_cls=_DraftSharingNothing)
    assert list(engine._draft.unshared) == [draft]
    # Its MTP layer loads; its embedding, whose name the target consumed, does not.
    with pytest.raises(RuntimeError, match=r"drafter's \['embed_tokens'\]"):
        engine._bake(init_info)


def test_bake_accepts_a_drafter_module_with_only_kv_cache_scales():
    engine, _, _, init_info = _engine({**TARGET_NAMES, **DRAFT_NAMES}, draft_cls=_DraftWithKvScales)
    engine._bake(init_info)


def _worker(engine, draft):
    draft_cfg = SimpleNamespace()
    return SimpleNamespace(
        weight_transfer_engine=engine,
        model_runner=SimpleNamespace(get_draft_model=lambda: draft),
        vllm_config=SimpleNamespace(speculative_config=SimpleNamespace(draft_model_config=draft_cfg)),
    )


def test_attach_draft_model_hands_the_drafter_to_the_rdt_engine():
    engine = SkyRLShardedRDTWeightTransferEngine.__new__(SkyRLShardedRDTWeightTransferEngine)
    engine.model, engine._draft = _Target(), None
    draft = _Draft(engine.model)
    worker = _worker(engine, draft)
    assert attach_draft_model(worker) is True
    assert engine._draft.model is draft
    assert engine._draft.config is worker.vllm_config.speculative_config.draft_model_config


@pytest.mark.parametrize("engine", [SimpleNamespace(), None])  # no skyrl_set_draft_model, or no engine at all
def test_attach_draft_model_fails_on_an_engine_that_cannot_take_the_drafter(engine):
    with pytest.raises(AttributeError):
        attach_draft_model(_worker(engine, torch.nn.Linear(1, 1)))


def test_attach_draft_model_reports_a_worker_without_a_drafter():
    engine = SkyRLShardedRDTWeightTransferEngine.__new__(SkyRLShardedRDTWeightTransferEngine)
    engine.model, engine._draft = _Target(), None
    assert attach_draft_model(_worker(engine, None)) is False
    assert engine._draft is None

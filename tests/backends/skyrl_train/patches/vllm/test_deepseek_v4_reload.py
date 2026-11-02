"""CPU tests for ``patch_deepseek_v4_reload`` (backports of vllm-project/vllm#57797 and #64357).

Run with:
uv run --isolated --extra fsdp --extra dev pytest -m vllm tests/backends/skyrl_train/patches/vllm/test_deepseek_v4_reload.py
"""

import types

import pytest
import torch

pytestmark = pytest.mark.vllm


@pytest.fixture
def reload_patch():
    pytest.importorskip("vllm")
    from skyrl.backends.skyrl_train.patches.vllm import patch_deepseek_v4_reload

    return patch_deepseek_v4_reload


def _tp(monkeypatch, rank, size):
    import vllm.distributed as distributed

    monkeypatch.setattr(distributed, "get_tensor_model_parallel_rank", lambda: rank)
    monkeypatch.setattr(distributed, "get_tensor_model_parallel_world_size", lambda: size)


@pytest.mark.parametrize(
    "checkpoint_heads,runtime_heads,tp_size",
    [(8, 4, 2), (8, 8, 1), (8, 6, 2), (16, 12, 4)],
)
def test_attention_sink_loader_shapes(monkeypatch, reload_patch, checkpoint_heads, runtime_heads, tp_size):
    # Same cases as upstream's test_attention_sink_loader_shapes: each rank loads its heads, -inf on padding.
    checkpoint = torch.arange(checkpoint_heads, dtype=torch.float32)
    for tp_rank in range(tp_size):
        _tp(monkeypatch, tp_rank, tp_size)
        param = torch.nn.Parameter(torch.zeros(runtime_heads), requires_grad=False)
        reload_patch.load_attention_sink(param, checkpoint)
        local_heads = checkpoint.chunk(tp_size)[tp_rank]
        expected = torch.cat((local_heads, torch.full((runtime_heads - local_heads.numel(),), -torch.inf)))
        assert torch.equal(param, expected)


def test_attention_sink_loader_rejects_indivisible_heads(monkeypatch, reload_patch):
    _tp(monkeypatch, 0, 2)
    param = torch.nn.Parameter(torch.ones(4), requires_grad=False)
    with pytest.raises(ValueError):
        reload_patch.load_attention_sink(param, torch.zeros(7))
    assert torch.equal(param, torch.ones(4))


def test_sinks_load_through_their_weight_loader(reload_patch):
    calls = []

    class FakeModel:
        def __init__(self):
            sink = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
            sink.weight_loader = lambda param, weight: calls.append(("sink", weight.tolist()))
            self._params = {"layers.0.attn.attn_sink": sink, "layers.0.attn.wq.weight": torch.zeros(1)}

        def named_parameters(self):
            return self._params.items()

        def load_weights(self, weights):
            names = set()
            for name, _ in weights:
                assert "attn_sink" not in name  # the direct-copy branch never sees a sink
                names.add(name)
            return names

    reload_patch._patch_sink_loading(FakeModel)
    loaded = FakeModel().load_weights(
        [
            ("layers.0.attn.attn_sink", torch.tensor([1.0, 2.0])),
            ("layers.0.attn.wq.weight", torch.zeros(1)),
            ("layers.9.attn.attn_sink", torch.zeros(2)),  # another pipeline stage's layer
        ]
    )
    assert calls == [("sink", [1.0, 2.0])]
    assert loaded == {"layers.0.attn.attn_sink", "layers.0.attn.wq.weight"}


def test_model_hook_runs_once_after_reload_not_per_chunk(monkeypatch, reload_patch):
    finalized = []

    class FakeCausalLM:
        hf_to_vllm_mapper = None

        def process_weights_after_loading(self):
            finalized.append(True)

    class FakeLoader:
        def __init__(self, model):
            pass

        def load_weights(self, weights, mapper=None):
            return {name for name, _ in weights}

    reload_patch._patch_load_weights(FakeCausalLM, FakeLoader)
    monkeypatch.setattr(reload_patch, "_FINALIZE_AFTER_RELOAD", (FakeCausalLM,))
    model = FakeCausalLM()
    for chunk in (["a"], ["b"]):
        model.load_weights([(name, None) for name in chunk])
    assert finalized == []  # load_weights no longer runs the hook

    reload_patch.finalize_model_weights(model)
    assert finalized == [True]
    reload_patch.finalize_model_weights(types.SimpleNamespace(process_weights_after_loading=lambda: 1 / 0))
    assert finalized == [True]  # other models' hooks are left alone


def test_extra_post_load_step_runs_after_the_hook(reload_patch):
    # V4.1: the Engram page-cache step moves from load_weights into the post-load hook, after the model's own.
    calls = []

    class FakeCausalLM:
        hf_to_vllm_mapper = None

        def process_weights_after_loading(self):
            calls.append("hook")

    class FakeLoader:
        def __init__(self, model):
            pass

        def load_weights(self, weights, mapper=None):
            return {name for name, _ in weights}

    reload_patch._patch_load_weights(FakeCausalLM, FakeLoader, lambda model: calls.append("extra"))
    model = FakeCausalLM()
    assert model.load_weights([("a", None)]) == {"a"}
    assert calls == []  # load_weights only loads
    model.process_weights_after_loading()
    assert calls == ["hook", "extra"]


def test_patched_reload_finalize_runs_the_hook_once(monkeypatch, reload_patch):
    # Through the patched entry point: chunks only load; finishing the reload runs the layerwise finalize, then the
    # model hook and V4.1's extra step once; other models only get the layerwise finalize.
    import vllm.model_executor.model_loader.reload as reload_pkg
    from vllm.model_executor.model_loader.reload import layerwise

    calls = []

    def original_finalize(model, model_config):
        calls.append("layers")

    monkeypatch.setattr(layerwise, "finalize_layerwise_reload", original_finalize)
    monkeypatch.setattr(reload_pkg, "finalize_layerwise_reload", original_finalize)

    class FakeCausalLM:
        hf_to_vllm_mapper = None

        def process_weights_after_loading(self):
            calls.append("hook")

    class FakeLoader:
        def __init__(self, model):
            pass

        def load_weights(self, weights, mapper=None):
            return {name for name, _ in weights}

    reload_patch._patch_load_weights(FakeCausalLM, FakeLoader, lambda model: calls.append("extra"))
    monkeypatch.setattr(reload_patch, "_FINALIZE_AFTER_RELOAD", (FakeCausalLM,))
    reload_patch._patch_finalize_layerwise_reload()

    model = FakeCausalLM()
    for chunk in (["a"], ["b"]):
        model.load_weights([(name, None) for name in chunk])
    assert calls == []
    reload_pkg.finalize_layerwise_reload(model, None)
    assert calls == ["layers", "hook", "extra"]
    reload_pkg.finalize_layerwise_reload(object(), None)
    assert calls == ["layers", "hook", "extra", "layers"]

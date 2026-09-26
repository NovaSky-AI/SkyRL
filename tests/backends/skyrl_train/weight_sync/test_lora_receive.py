"""Receive side of the LoRA weight-sync target.

Two things are load-bearing and both are asserted here against a stand-in for
vLLM's engine (the real ones need the wheel, and neither behaviour depends on
it):

* the **branch**. An armed round must not run the base model's layerwise reload
  and must not load anything into the model; an unarmed round must behave
  exactly as before this feature existed.
* the **copy**. Chunks arrive as views into a transport buffer the sender reuses
  immediately, so staging has to clone.
"""

from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.skyrl_train.patches.vllm import patch_lora_in_memory as patch
from skyrl.backends.skyrl_train.weight_sync.lora_target import build_lora_receive_target
from skyrl.backends.skyrl_train.weight_sync.weight_receivers import (
    SkyrlReceiveLifecycleMixin,
)


class _FakeVllmEngine:
    """Stands in for vLLM's NCCL/IPC receive engine.

    Records the lifecycle calls SkyRL's mixin delegates, and loads through
    ``self.model.load_weights`` exactly as the real engines do -- which is the
    handle the staging proxy replaces.
    """

    def __init__(self, weights):
        self._weights = weights
        self.calls = []

    def start_weight_update(self):
        self.calls.append("start")

    def receive_weights(self, update_info):
        self.calls.append("receive")
        self.model.load_weights(self._weights)

    def finish_weight_update(self):
        self.calls.append("finish")


class _FakeModel:
    """The base model. Records what reached it, with the signature the unarmed
    path uses: ``_load_checkpoint_weights`` calls ``load_weights(weights=...)``
    after splitting off any compact batched-MoE FP8 tensors."""

    def __init__(self, loaded: list) -> None:
        self._loaded = loaded

    def load_weights(self, weights=None, **kwargs):
        pairs = list(weights)
        self._loaded.append(pairs)
        return {name for name, _ in pairs}

    def named_parameters(self):
        return iter(())


class _Engine(SkyrlReceiveLifecycleMixin, _FakeVllmEngine):
    def __init__(self, weights, lora_enabled=True):
        super().__init__(weights)
        self.device = "cpu"
        self.loaded = []
        self.model = _FakeModel(self.loaded)
        self.vllm_config = SimpleNamespace(lora_config=object() if lora_enabled else None)


@pytest.fixture(autouse=True)
def _clear_staged():
    for name in patch.staged_adapter_names():
        patch.discard_in_memory_adapter(name)
    yield
    for name in patch.staged_adapter_names():
        patch.discard_in_memory_adapter(name)


def _run_round(engine, update_info=None):
    engine.start_weight_update()
    engine.receive_weights(update_info)
    engine.finish_weight_update()


class TestArmedRound:
    def test_stages_the_adapter_and_never_touches_the_model(self):
        buffer = torch.arange(6, dtype=torch.float32)
        engine = _Engine([("experts.0.lora_A.weight", buffer[:3]), ("dense", buffer[3:])])
        engine.skyrl_set_lora_receive_target(
            build_lora_receive_target("tenant", {"r": 4}, {"experts.1.lora_A.weight": "experts.0.lora_A.weight"})
        )

        engine.start_weight_update()
        # No layerwise reload: the base model is untouched by an adapter update.
        assert engine.calls == []
        engine.receive_weights(None)
        assert engine.loaded == []  # staged, not loaded
        buffer.fill_(-1.0)  # the transport buffer is reused right after the chunk
        engine.finish_weight_update()
        assert engine.calls == ["receive"]

        staged = patch._STAGED["tenant"]
        assert staged.peft_config == {"r": 4}
        assert set(staged.tensors) == {"experts.0.lora_A.weight", "experts.1.lora_A.weight", "dense"}
        assert torch.equal(staged.tensors["dense"], torch.tensor([3.0, 4.0, 5.0]))
        # The alias shares one storage rather than a second copy.
        assert (
            staged.tensors["experts.1.lora_A.weight"].data_ptr() == staged.tensors["experts.0.lora_A.weight"].data_ptr()
        )

    def test_arming_lasts_exactly_one_round(self):
        engine = _Engine([("a", torch.ones(1))])
        engine.skyrl_set_lora_receive_target(build_lora_receive_target("t", {}, {}))
        assert engine.skyrl_lora_armed()
        _run_round(engine)
        assert not engine.skyrl_lora_armed()

        # The next round, unarmed, is an ordinary base-model update again.
        _run_round(engine)
        assert engine.calls == ["receive", "start", "receive", "finish"]
        ((name, tensor),) = engine.loaded[0]
        assert name == "a" and torch.equal(tensor, torch.ones(1))

    def test_double_arm_is_rejected(self):
        engine = _Engine([])
        engine.skyrl_set_lora_receive_target(build_lora_receive_target("t", {}, {}))
        with pytest.raises(RuntimeError, match="already armed"):
            engine.skyrl_set_lora_receive_target(build_lora_receive_target("t2", {}, {}))

    def test_a_non_lora_target_is_rejected(self):
        engine = _Engine([])
        with pytest.raises(ValueError, match="Not a LoRA receive target"):
            engine.skyrl_set_lora_receive_target({"kind": "model"})

    def test_requires_a_lora_enabled_engine(self):
        engine = _Engine([], lora_enabled=False)
        with pytest.raises(RuntimeError, match="enable-lora"):
            engine.skyrl_set_lora_receive_target(build_lora_receive_target("t", {}, {}))

    def test_duplicate_name_in_one_round_is_rejected(self):
        engine = _Engine([("a", torch.ones(1)), ("a", torch.ones(1))])
        engine.skyrl_set_lora_receive_target(build_lora_receive_target("t", {}, {}))
        engine.start_weight_update()
        with pytest.raises(ValueError, match="received twice"):
            engine.receive_weights(None)

    def test_round_without_tensors_fails_and_disarms(self):
        engine = _Engine([])
        engine.skyrl_set_lora_receive_target(build_lora_receive_target("t", {}, {}))
        engine.start_weight_update()
        engine.receive_weights(None)
        with pytest.raises(RuntimeError, match="without receiving any tensors"):
            engine.finish_weight_update()
        # Disarmed even on failure, so the next round cannot land on a stale target.
        assert not engine.skyrl_lora_armed()
        assert patch.staged_adapter_names() == []

    def test_staging_restores_the_model_handle(self):
        """The proxy must compose with set_weight_update_target / the drafter swap."""
        engine = _Engine([("a", torch.ones(1))])
        model = engine.model
        engine.skyrl_set_lora_receive_target(build_lora_receive_target("t", {}, {}))
        _run_round(engine)
        assert engine.model is model


class TestUnarmedRound:
    def test_is_an_ordinary_base_model_update(self):
        engine = _Engine([("w", torch.ones(2))])
        _run_round(engine)
        assert engine.calls == ["start", "receive", "finish"]
        ((name, tensor),) = engine.loaded[0]
        assert name == "w" and torch.equal(tensor, torch.ones(2))
        assert patch.staged_adapter_names() == []


class TestStagingRegistry:
    def test_stage_and_discard(self):
        patch.stage_in_memory_adapter("a", {"k": torch.ones(1)}, {"r": 1})
        assert patch.staged_adapter_names() == ["a"]
        assert patch.discard_in_memory_adapter("a")
        assert not patch.discard_in_memory_adapter("a")

    def test_empty_stage_rejected(self):
        with pytest.raises(ValueError):
            patch.stage_in_memory_adapter("a", {}, {})
        with pytest.raises(ValueError):
            patch.stage_in_memory_adapter("", {"k": torch.ones(1)}, {})

    def test_restaging_replaces_the_previous_generation(self):
        patch.stage_in_memory_adapter("a", {"k": torch.ones(1)}, {"r": 1})
        patch.stage_in_memory_adapter("a", {"k": torch.zeros(1)}, {"r": 2})
        assert patch._STAGED["a"].peft_config == {"r": 2}
        assert torch.equal(patch._STAGED["a"].tensors["k"], torch.zeros(1))

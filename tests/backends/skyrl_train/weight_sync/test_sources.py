"""Tests for the shared ``WeightSource`` implementations.

Two properties matter:

* **channel agreement.** ``metadata()`` declares what iteration will yield, and
  the trainer engine sizes the worker's receive buffers -- and in packed mode
  cuts its chunk boundaries -- from it. A source that disagrees splits the stream
  differently on each side, which hangs in NCCL rather than erroring.
* **laziness.** The packed producers bound their memory only because they consume
  the source lazily; a source that materialized eagerly would silently hold the
  whole model.
"""

import pytest
import torch

pytest.importorskip("vllm", reason="sources.py implements vLLM's WeightSource contract")

pytestmark = pytest.mark.vllm

from skyrl.backends.skyrl_train.weight_sync import sources as sources_mod  # noqa: E402
from skyrl.backends.skyrl_train.weight_sync.sources import (  # noqa: E402
    FsdpWeightSource,
    LoraAdapterWeightSource,
    MegatronWeightSource,
)


def _model():
    model = torch.nn.Module()
    model.register_parameter("w", torch.nn.Parameter(torch.ones(2, 3, dtype=torch.float32)))
    model.register_parameter("b", torch.nn.Parameter(torch.zeros(3, dtype=torch.float32)))
    return model


def _assert_channels_agree(source):
    """The contract vLLM validates at runtime in ``_checked_iter``."""
    meta = source.metadata()
    pairs = list(source)
    assert len(meta) == len(pairs)
    for m, (name, tensor) in zip(meta, pairs):
        assert m.name == name
        assert m.dtype == tensor.dtype
        assert m.shape == tuple(tensor.shape)
    return meta, pairs


class TestFsdpWeightSource:
    def test_channels_agree(self):
        meta, pairs = _assert_channels_agree(FsdpWeightSource(_model(), torch.bfloat16))
        assert [m.name for m in meta] == ["w", "b"]
        assert [m.shape for m in meta] == [(2, 3), (3,)]

    def test_casts_to_the_inference_dtype(self):
        source = FsdpWeightSource(_model(), torch.bfloat16)
        # Declared as well as yielded: the worker allocates from metadata().
        assert {m.dtype for m in source.metadata()} == {torch.bfloat16}
        assert {t.dtype for _, t in source} == {torch.bfloat16}

    def test_yields_contiguous_tensors(self):
        # NCCL sends `numel` elements straight from data_ptr(), so a
        # non-contiguous view would ship whatever follows its base pointer.
        assert all(t.is_contiguous() for _, t in FsdpWeightSource(_model(), torch.bfloat16))

    def test_weight_prefix_applies_to_both_channels(self):
        source = FsdpWeightSource(_model(), torch.bfloat16, weight_prefix="language_model.")
        meta, pairs = _assert_channels_agree(source)
        assert [m.name for m in meta] == ["language_model.w", "language_model.b"]
        # The prefix is a *wire* name; the state_dict is still keyed without it.
        assert set(source.model.state_dict()) == {"w", "b"}

    def test_is_re_iterable(self):
        source = FsdpWeightSource(_model(), torch.bfloat16)
        assert [n for n, _ in source] == [n for n, _ in source]

    def test_metadata_does_not_gather(self, monkeypatch):
        """FSDP2 ``DTensor.shape`` is already the global shape, so declaring the
        stream must not run the gather collective.

        Asserted at the seam rather than with a fake DTensor: ``state_dict()``
        returns detached tensors, so a hand-attached ``full_tensor`` would not
        survive it."""
        monkeypatch.setattr(
            sources_mod,
            "materialize_full_tensor",
            lambda t: pytest.fail("metadata() must not materialize"),
        )
        meta = FsdpWeightSource(_model(), torch.bfloat16).metadata()
        assert [m.shape for m in meta] == [(2, 3), (3,)]

    def test_iteration_gathers_every_parameter(self, monkeypatch):
        """The collective must run for every parameter on every rank: under
        pipeline parallelism a rank may not own one, but iterating still drives
        the collective its peers are waiting in."""
        gathered = []

        def _spy(tensor):
            gathered.append(tuple(tensor.shape))
            return tensor

        monkeypatch.setattr(sources_mod, "materialize_full_tensor", _spy)
        list(FsdpWeightSource(_model(), torch.bfloat16))
        assert gathered == [(2, 3), (3,)]


class _FakeBridge:
    """Stands in for Megatron-Bridge. Records calls so laziness is observable."""

    def __init__(self, tensors, *, expect_module=None):
        self._tensors = tensors
        self.export_calls = 0
        self.yielded = 0
        self.tasks_args = []
        self._expect_module = expect_module

    def export_hf_weights(self, module, show_progress=False, conversion_tasks=None):
        assert self._expect_module is None or module is self._expect_module
        self.export_calls += 1
        self.tasks_args.append(conversion_tasks)

        def gen():
            for name, tensor in self._tensors:
                self.yielded += 1
                yield name, tensor

        return gen()


class TestMegatronWeightSource:
    def _tensors(self):
        return [
            ("model.embed_tokens.weight", torch.ones(4, 2, dtype=torch.float32)),
            ("model.layers.0.self_attn.q_proj.weight", torch.ones(2, 2, dtype=torch.float32)),
        ]

    def test_channels_agree(self):
        bridge = _FakeBridge(self._tensors())
        meta, pairs = _assert_channels_agree(MegatronWeightSource(bridge, object(), torch.bfloat16))
        assert [m.name for m in meta] == [n for n, _ in self._tensors()]

    def test_exports_the_whole_model_in_one_call(self):
        """`_accumulate_grouped_export` needs every task of a `group_key` in one
        call, or expert weights are silently never yielded."""
        bridge = _FakeBridge(self._tensors())
        list(MegatronWeightSource(bridge, object(), torch.bfloat16))
        assert bridge.tasks_args == [None]

    def test_metadata_caches_its_dry_export(self):
        bridge = _FakeBridge(self._tensors())
        source = MegatronWeightSource(bridge, object(), torch.bfloat16)
        first = source.metadata()
        assert bridge.export_calls == 1
        # Engines call metadata() every round; it must not re-run the export.
        assert source.metadata() is first
        assert bridge.export_calls == 1

    def test_iteration_is_lazy(self):
        """Nothing may be pulled from the bridge until the consumer asks -- the
        packed producers consume lazily into a fixed buffer, so streaming alone
        bounds peak memory."""
        bridge = _FakeBridge(self._tensors())
        source = MegatronWeightSource(bridge, object(), torch.bfloat16)
        it = iter(source)
        assert bridge.yielded == 0
        next(it)
        assert bridge.yielded == 1
        next(it)
        assert bridge.yielded == 2

    def test_is_re_iterable_and_re_exports(self):
        bridge = _FakeBridge(self._tensors())
        source = MegatronWeightSource(bridge, object(), torch.bfloat16)
        assert [n for n, _ in source] == [n for n, _ in source]
        assert bridge.export_calls == 2

    def test_reads_the_module_it_was_given(self):
        module = object()
        bridge = _FakeBridge(self._tensors(), expect_module=module)
        list(MegatronWeightSource(bridge, module, torch.bfloat16))
        assert bridge.export_calls == 1


def _expert_key(expert: int, which: str) -> str:
    return f"base_model.model.model.layers.0.mlp.experts.{expert}.down_proj.lora_{which}.weight"


class _FakeLoraSource(LoraAdapterWeightSource):
    """A LoRA source whose export and finalize are scripted."""

    def __init__(self, state, adapter_config=None, **kwargs):
        super().__init__(**kwargs)
        self._state = state
        self._config = adapter_config or {}
        self.exports = 0
        self.finalized_keys = None

    def export_adapter_stream(self):
        self.exports += 1
        yield from self._state.items()

    def finalize_adapter(self, adapter_state):
        self.finalized_keys = list(adapter_state)
        return adapter_state, dict(self._config)


def _shared_expert_state(num_experts=4, experts_per_shared_adapter=2):
    """Per-expert keys where every expert in a group carries the same values."""
    state = {}
    for group in range(num_experts // experts_per_shared_adapter):
        a = torch.randn(2, 3, dtype=torch.float32)
        for e in range(group * experts_per_shared_adapter, (group + 1) * experts_per_shared_adapter):
            state[_expert_key(e, "A")] = a.clone()
    state["base_model.model.model.layers.0.self_attn.o_proj.lora_A.weight"] = torch.randn(2, 3, dtype=torch.float32)
    return state


class TestLoraAdapterWeightSource:
    """The adapter source. Same two properties as the model sources, plus the
    ``receive_target``, which the model sources have no equivalent of."""

    def _source(self, **kwargs):
        state = kwargs.pop("state", None) or _shared_expert_state()
        source = _FakeLoraSource(state, dtype=torch.bfloat16, experts_per_shared_adapter=2, **kwargs)
        source.set_lora_name("tenant")
        return source

    def test_channels_agree(self):
        source = self._source()
        source.prepare()
        _assert_channels_agree(source)

    def test_one_export_serves_both_channels_and_the_target(self):
        source = self._source()
        source.prepare()
        assert source.receive_target["lora_name"] == "tenant"
        names = [m.name for m in source.metadata()]
        sent = [n for n, _ in source]
        assert source.exports == 1
        assert names == sent
        # finalize only ever sees the canonical keys.
        assert source.finalized_keys == sent

    def test_only_unique_tensors_are_sent_and_the_rest_are_aliased(self):
        state = _shared_expert_state(num_experts=4, experts_per_shared_adapter=2)
        source = self._source(state=state)
        source.prepare()
        target = source.receive_target
        sent = {m.name for m in source.metadata()}
        # 2 groups of 2 experts -> 2 expert tensors sent, 2 aliased, plus 1 dense.
        assert len(sent) == 3
        assert set(target["aliases"]) | sent == set(state)
        assert all(v in sent for v in target["aliases"].values())

    def test_casts_to_the_wire_dtype_on_both_channels(self):
        source = self._source()
        source.prepare()
        assert {m.dtype for m in source.metadata()} == {torch.bfloat16}
        assert {t.dtype for _, t in source} == {torch.bfloat16}

    def test_yields_contiguous_tensors(self):
        source = self._source()
        source.prepare()
        assert all(t.is_contiguous() for _, t in source)

    def test_the_target_carries_the_name_and_adapter_config(self):
        source = self._source(adapter_config={"r": 4, "lora_alpha": 8})
        source.prepare()
        target = source.receive_target
        assert target["kind"] == "lora"
        assert target["lora_name"] == "tenant"
        assert target["adapter_config"] == {"r": 4, "lora_alpha": 8}

    def test_consuming_the_stream_re_exports_next_round(self):
        source = self._source()
        source.prepare()
        list(source)
        assert source.exports == 1
        source.set_lora_name("next")
        source.prepare()
        assert source.exports == 2
        assert source.receive_target["lora_name"] == "next"

    def test_prepare_is_idempotent_within_a_round(self):
        source = self._source()
        source.prepare()
        source.prepare()
        assert source.exports == 1

    def test_name_required_before_publishing(self):
        source = _FakeLoraSource({"k": torch.ones(2)}, dtype=torch.bfloat16, experts_per_shared_adapter=1)
        with pytest.raises(RuntimeError, match="set_lora_name"):
            _ = source.receive_target

    def test_finalize_must_keep_aliased_keys(self):
        class _Dropper(_FakeLoraSource):
            def finalize_adapter(self, adapter_state):
                return {k: v for k, v in adapter_state.items() if ".experts." not in k}, {}

        source = _Dropper(_shared_expert_state(), dtype=torch.bfloat16, experts_per_shared_adapter=2)
        source.set_lora_name("tenant")
        with pytest.raises(RuntimeError, match="dropped aliased keys"):
            source.prepare()

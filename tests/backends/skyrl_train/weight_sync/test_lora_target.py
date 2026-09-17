"""The LoRA adapter as a weight-sync target: dedupe, chunking, receive_target."""

import pytest
import torch

from skyrl.backends.skyrl_train.weight_sync.lora_target import (
    IN_MEMORY_LORA_PATH_PREFIX,
    LoraAdapterExtractor,
    build_lora_receive_target,
    chunk_adapter_state,
    dedupe_shared_expert_adapters,
    expand_lora_aliases,
    in_memory_lora_path,
    is_lora_receive_target,
    lora_name_from_in_memory_path,
)


def _expert_key(layer: int, expert: int, proj: str, which: str) -> str:
    return f"base_model.model.model.layers.{layer}.mlp.experts.{expert}.{proj}.lora_{which}.weight"


def _shared_expert_adapter(num_experts: int, experts_per_shared_adapter: int, rank: int = 4, dim: int = 8):
    """Per-expert keys where every expert in a group carries the same tensor values."""
    state = {}
    groups = num_experts // experts_per_shared_adapter
    for layer in range(2):
        for g in range(groups):
            a = torch.randn(rank, dim)
            b = torch.randn(dim, rank)
            for e in range(g * experts_per_shared_adapter, (g + 1) * experts_per_shared_adapter):
                state[_expert_key(layer, e, "down_proj", "A")] = a.clone()
                state[_expert_key(layer, e, "down_proj", "B")] = b.clone()
        state[f"base_model.model.model.layers.{layer}.self_attn.o_proj.lora_A.weight"] = torch.randn(rank, dim)
        state[f"base_model.model.model.layers.{layer}.self_attn.o_proj.lora_B.weight"] = torch.randn(dim, rank)
    return state


class TestDedupeSharedExpertAdapters:
    def test_one_tensor_per_group_and_aliases_for_the_rest(self):
        state = _shared_expert_adapter(num_experts=8, experts_per_shared_adapter=4)
        to_send, aliases = dedupe_shared_expert_adapters(state, experts_per_shared_adapter=4)

        # 2 layers x 2 groups x (A, B) expert tensors + 2 layers x (A, B) dense.
        assert len(to_send) == 2 * 2 * 2 + 2 * 2
        assert len(aliases) == len(state) - len(to_send)
        for public_key, sent_key in aliases.items():
            assert sent_key in to_send
            assert torch.equal(state[public_key], to_send[sent_key])
        # Order of sent keys follows the input.
        assert list(to_send) == [k for k in state if k in to_send]

    def test_expansion_restores_every_public_key_sharing_storage(self):
        state = _shared_expert_adapter(num_experts=8, experts_per_shared_adapter=4)
        to_send, aliases = dedupe_shared_expert_adapters(state, experts_per_shared_adapter=4)
        expanded = expand_lora_aliases(to_send, aliases)
        assert set(expanded) == set(state)
        for public_key, sent_key in aliases.items():
            assert expanded[public_key].data_ptr() == expanded[sent_key].data_ptr()
            assert torch.equal(expanded[public_key], state[public_key])

    def test_differing_member_is_sent_not_aliased(self):
        """The grouping is a hint; values are verified, so a mismatch costs bandwidth only."""
        state = _shared_expert_adapter(num_experts=4, experts_per_shared_adapter=4)
        odd = _expert_key(0, 2, "down_proj", "B")
        state[odd] = state[odd] + 1.0
        to_send, aliases = dedupe_shared_expert_adapters(state, experts_per_shared_adapter=4)
        assert odd in to_send
        assert odd not in aliases
        assert torch.equal(expand_lora_aliases(to_send, aliases)[odd], state[odd])

    def test_group_of_one_sends_everything(self):
        state = _shared_expert_adapter(num_experts=4, experts_per_shared_adapter=1)
        to_send, aliases = dedupe_shared_expert_adapters(state, experts_per_shared_adapter=1)
        assert aliases == {}
        assert list(to_send) == list(state)

    def test_dense_keys_never_alias(self):
        state = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(2, 2),
            "base_model.model.model.layers.1.self_attn.q_proj.lora_A.weight": torch.ones(2, 2),
        }
        to_send, aliases = dedupe_shared_expert_adapters(state, experts_per_shared_adapter=64)
        assert aliases == {} and len(to_send) == 2

    def test_rejects_non_positive_group(self):
        with pytest.raises(ValueError, match="experts_per_shared_adapter"):
            dedupe_shared_expert_adapters({}, experts_per_shared_adapter=0)

    def test_expand_unknown_alias_target_raises(self):
        with pytest.raises(KeyError, match="was not received"):
            expand_lora_aliases({"a": torch.ones(1)}, {"b": "missing"})


class TestChunkAdapterState:
    def test_chunks_respect_byte_budget_and_cast_to_wire_dtype(self):
        state = {f"k{i}": torch.randn(64, 64, dtype=torch.float32) for i in range(6)}  # 16 KiB each
        chunks = list(chunk_adapter_state(state, torch.bfloat16, bucket_size_bytes=3 * 64 * 64 * 2))
        assert [len(c) for c in chunks] == [3, 3]
        assert [n for c in chunks for n in c.names] == list(state)
        for chunk in chunks:
            assert all(t.dtype == torch.bfloat16 for t in chunk.tensors)
            assert chunk.dtypes == ["torch.bfloat16"] * len(chunk)
            assert chunk.shapes == [[64, 64]] * len(chunk)

    def test_oversized_tensor_gets_its_own_chunk(self):
        state = {"small": torch.zeros(4), "big": torch.zeros(1024), "small2": torch.zeros(4)}
        chunks = list(chunk_adapter_state(state, torch.float32, bucket_size_bytes=64))
        assert [c.names for c in chunks] == [["small"], ["big"], ["small2"]]

    def test_empty_state_yields_nothing(self):
        assert list(chunk_adapter_state({}, torch.float32, bucket_size_bytes=1)) == []


class TestReceiveTarget:
    def test_marker_path_round_trips(self):
        path = in_memory_lora_path("tenant-a")
        assert path == f"{IN_MEMORY_LORA_PATH_PREFIX}tenant-a"
        assert lora_name_from_in_memory_path(path) == "tenant-a"
        assert lora_name_from_in_memory_path("/tmp/skyrl_lora_sync") is None

    def test_build_and_detect(self):
        target = build_lora_receive_target("skyrl_lora", {"r": 32, "lora_alpha": 64}, {"b": "a"})
        assert target == {
            "kind": "lora",
            "lora_name": "skyrl_lora",
            "adapter_config": {"r": 32, "lora_alpha": 64},
            "aliases": {"b": "a"},
        }
        assert is_lora_receive_target(target)
        assert not is_lora_receive_target(None)
        assert not is_lora_receive_target({"kind": "model"})

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError):
            build_lora_receive_target("", {}, {})


class _FakeExtractor(LoraAdapterExtractor):
    def __init__(self, state, config, **kwargs):
        super().__init__(**kwargs)
        self._state = state
        self._config = config
        self.exports = 0
        self.finalized_keys = None

    def export_adapter_stream(self):
        self.exports += 1
        yield from self._state.items()

    def finalize_adapter(self, adapter_state):
        self.finalized_keys = list(adapter_state)
        return adapter_state, dict(self._config)


class TestLoraAdapterExtractor:
    def test_single_export_serves_target_metadata_and_chunks(self):
        state = _shared_expert_adapter(num_experts=8, experts_per_shared_adapter=4)
        extractor = _FakeExtractor(state, {"r": 4}, experts_per_shared_adapter=4, bucket_size_threshold_GB=1.0)
        extractor.set_lora_name("skyrl_lora")

        target = extractor.receive_target  # read first, as WeightTransferSender.send does
        metadata = extractor.get_weight_metadata(torch.bfloat16)
        chunks = list(extractor.extract_weights(torch.bfloat16))

        assert extractor.exports == 1
        assert extractor.derives_metadata_from_chunks
        sent_names = [n for c in chunks for n in c.names]
        assert extractor.finalized_keys == sent_names  # finalize sees only unique keys
        assert sent_names == metadata["names"]
        assert set(sent_names) | set(target["aliases"]) == set(state)
        assert target["lora_name"] == "skyrl_lora" and target["adapter_config"] == {"r": 4}

        # Consuming the stream drops the cache; the next sync exports again.
        extractor.set_lora_name("other")
        assert extractor.receive_target["lora_name"] == "other"
        assert extractor.exports == 2

    def test_finalize_must_keep_aliased_keys(self):
        class _Dropper(_FakeExtractor):
            def finalize_adapter(self, adapter_state):
                return {k: v for k, v in adapter_state.items() if ".experts." not in k}, {}

        state = _shared_expert_adapter(num_experts=4, experts_per_shared_adapter=4)
        extractor = _Dropper(state, {}, experts_per_shared_adapter=4)
        extractor.set_lora_name("x")
        with pytest.raises(RuntimeError, match="dropped aliased keys"):
            _ = extractor.receive_target

    def test_dedupe_accepts_a_lazy_stream(self):
        state = _shared_expert_adapter(num_experts=4, experts_per_shared_adapter=2)
        to_send, aliases = dedupe_shared_expert_adapters(iter(state.items()), experts_per_shared_adapter=2)
        assert set(to_send) | set(aliases) == set(state)

    def test_name_required_before_publishing(self):
        extractor = _FakeExtractor({"k": torch.ones(2)}, {}, experts_per_shared_adapter=1)
        with pytest.raises(RuntimeError, match="set_lora_name"):
            _ = extractor.receive_target

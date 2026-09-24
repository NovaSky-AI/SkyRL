"""CPU tests for the SkyRL-Train backend side of Rollout Routing Replay (R3).

Routing is stashed on the vLLM servers at sample time and forward_backward
pulls it back by digest (see routed_experts_stash.py); the backend holds no
routing state and nothing is exposed through the client-facing Tinker types.
Covers the digest fetch (dedupe, dtype narrowing, shape validation,
live-router fallback), stash-key model-name resolution, sample-time gating,
the lifecycle fan-outs, and the replay input packing. No GPU needed. Run:
  uv run --extra dev --extra fsdp pytest tests/tinker/skyrl_train/test_router_replay_backend.py
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

# Skip cleanly if the SkyRL-Train backend (ray/vllm) can't be imported.
skyrl_train_backend = pytest.importorskip("skyrl.backends.skyrl_train_backend")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from skyrl.backends.skyrl_train.inference_servers.routed_experts_stash import (  # noqa: E402
    sequence_digest,
)
from skyrl.tinker import types  # noqa: E402
from skyrl.tinker.engine import prepare_sample_batch  # noqa: E402
from skyrl.train.config import SkyRLTrainConfig  # noqa: E402
from skyrl.train.dataset.preprocess import (  # noqa: E402
    convert_prompts_responses_to_batch_tensors,
    make_router_padding_mask,
)

_build_rollout_expert_indices = skyrl_train_backend._build_rollout_expert_indices
SkyRLTrainBackend = skyrl_train_backend.SkyRLTrainBackend

BASE_MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"


def _routing(seq_len: int, num_layers: int, topk: int) -> np.ndarray:
    """Routing array whose entry (t, layer, k) encodes t*1000 + layer*10 + k."""
    return np.array(
        [[[t * 1000 + layer * 10 + k for k in range(topk)] for layer in range(num_layers)] for t in range(seq_len)],
        dtype=np.int32,
    )


class _SpyStashClient:
    """Fake RemoteInferenceClient exposing only the R3 stash methods."""

    def __init__(self, stashed: dict[str, dict[str, np.ndarray]] | None = None):
        # {model_name: {digest_hex: routing}}
        self.stashed = stashed or {}
        self.fetch_calls: list[tuple[str, list[str]]] = []
        self.weight_sync_calls: list[tuple[str, int]] = []
        self.clear_calls: list[str] = []

    async def fetch_routed_experts(self, model, digest_hexes):
        self.fetch_calls.append((model, list(digest_hexes)))
        hits = self.stashed.get(model, {})
        return {h: hits[h] for h in digest_hexes if h in hits}

    async def routed_experts_weight_sync(self, model, max_staleness=1):
        self.weight_sync_calls.append((model, max_staleness))
        return {}

    async def clear_routed_experts(self, model):
        self.clear_calls.append(model)
        return {}

    async def aclose(self):
        pass


def _fetch_backend(stashed=None, *, lora=False, model_ids_to_role=None):
    """Stand-in with the attributes the R3 fetch/lifecycle helpers touch."""
    return SimpleNamespace(
        _inference_engine_client=_SpyStashClient(stashed),
        _base_lora_signature=(8, 16) if lora else None,
        _model_ids_to_role=model_ids_to_role or {},
        _cfg=None,
        _routed_experts_missing_warned=False,
        config=SimpleNamespace(routed_experts_stash_max_staleness=1),
    )


def _bind(fake):
    """Bind the real backend methods onto the stand-in."""
    for name in (
        "_resolve_inference_model_name",
        "_run_client_call",
        "_fetch_rollout_routing",
        "_notify_routed_experts_weight_sync",
        "_clear_routed_experts_for_model",
        "_warn_on_missing_routing",
    ):
        setattr(fake, name, getattr(SkyRLTrainBackend, name).__get__(fake))
    return fake


def test_resolve_inference_model_name(monkeypatch):
    monkeypatch.setattr(skyrl_train_backend, "resolve_policy_model_name", lambda cfg: "policy-alias")

    # Multi-LoRA: the adapter registered on vLLM IS the Tinker model_id.
    fake = _bind(_fetch_backend(lora=True, model_ids_to_role={"model_a": "policy"}))
    fake._cfg = SkyRLTrainConfig()
    fake._cfg.trainer.policy.model.path = BASE_MODEL
    assert fake._resolve_inference_model_name("model_a") == "model_a"
    # Unknown ids and non-LoRA setups fall back to the policy name.
    assert fake._resolve_inference_model_name("unknown") == "policy-alias"
    assert _bind(_fetch_backend(lora=False))._resolve_inference_model_name("model_a") == "policy-alias"
    # An empty id is base-model sampling: the served base model, never the adapter alias.
    assert fake._resolve_inference_model_name("") == BASE_MODEL
    fake._cfg.generator.inference_engine.served_model_name = "served-base"
    assert fake._resolve_inference_model_name("") == "served-base"


def test_fetch_maps_digests_back_to_samples(monkeypatch):
    """Each training sequence gets the routing stashed under its own digest,
    with one fan-out per (single-model) batch and duplicates deduplicated."""
    monkeypatch.setattr(skyrl_train_backend, "resolve_policy_model_name", lambda cfg: BASE_MODEL)

    seq_a, seq_b = [10, 11, 12, 20, 21], [10, 11, 12, 30, 31, 32]
    routing_a, routing_b = _routing(4, 2, 3), _routing(5, 2, 3) + 1
    stashed = {
        BASE_MODEL: {
            sequence_digest(seq_a).hex(): routing_a,
            sequence_digest(seq_b).hex(): routing_b,
        }
    }
    fake = _bind(_fetch_backend(stashed))

    per_sample = fake._fetch_rollout_routing(["m", "m", "m"], [seq_a, seq_b, seq_a])
    np.testing.assert_array_equal(np.asarray(per_sample[0], dtype=np.int64), routing_a.astype(np.int64))
    np.testing.assert_array_equal(np.asarray(per_sample[1], dtype=np.int64), routing_b.astype(np.int64))
    np.testing.assert_array_equal(np.asarray(per_sample[2], dtype=np.int64), routing_a.astype(np.int64))

    # One fan-out under the resolved model name, digests deduplicated.
    assert fake._inference_engine_client.fetch_calls == [
        (BASE_MODEL, sorted({sequence_digest(seq_a).hex(), sequence_digest(seq_b).hex()}))
    ]


def test_fetch_missing_digests_fall_back_to_none_with_warning(monkeypatch):
    monkeypatch.setattr(skyrl_train_backend, "resolve_policy_model_name", lambda cfg: BASE_MODEL)
    seq_hit, seq_miss = [1, 2, 3], [4, 5, 6]
    stashed = {BASE_MODEL: {sequence_digest(seq_hit).hex(): _routing(2, 1, 2)}}
    fake = _bind(_fetch_backend(stashed))

    per_sample = fake._fetch_rollout_routing(["m", "m"], [seq_hit, seq_miss])
    assert per_sample[0] is not None and per_sample[1] is None

    fake._warn_on_missing_routing(per_sample)
    assert fake._routed_experts_missing_warned is True


def test_fetch_narrows_dtype_and_rejects_non_3d(monkeypatch):
    monkeypatch.setattr(skyrl_train_backend, "resolve_policy_model_name", lambda cfg: BASE_MODEL)
    seq_small, seq_big, seq_bad = [1], [2], [3]
    stashed = {
        BASE_MODEL: {
            sequence_digest(seq_small).hex(): np.ones((2, 1, 2), dtype=np.uint16),  # < 256 -> uint8
            sequence_digest(seq_big).hex(): np.full((2, 1, 2), 300, dtype=np.uint16),  # torch-unsafe -> int16
            sequence_digest(seq_bad).hex(): np.zeros((3, 4), dtype=np.int32),  # non-3D -> dropped
        }
    }
    fake = _bind(_fetch_backend(stashed))

    per_sample = fake._fetch_rollout_routing(["m", "m", "m"], [seq_small, seq_big, seq_bad])
    assert per_sample[0].dtype == np.uint8
    assert per_sample[1].dtype == np.int16
    assert per_sample[2] is None


def test_lifecycle_fanouts_carry_args_and_are_best_effort(monkeypatch):
    """Weight-sync and clear fan-outs pass the resolved name/staleness, and a
    failing fan-out must not raise out of the training operation."""
    monkeypatch.setattr(skyrl_train_backend, "resolve_policy_model_name", lambda cfg: BASE_MODEL)
    fake = _bind(_fetch_backend(lora=True, model_ids_to_role={"model_a": "policy"}))
    fake.config = SimpleNamespace(routed_experts_stash_max_staleness=3)

    fake._notify_routed_experts_weight_sync("model_a")
    fake._clear_routed_experts_for_model("model_a")
    assert fake._inference_engine_client.weight_sync_calls == [("model_a", 3)]
    assert fake._inference_engine_client.clear_calls == ["model_a"]

    async def _boom(*args, **kwargs):
        raise RuntimeError("server down")

    fake._inference_engine_client.routed_experts_weight_sync = _boom
    fake._inference_engine_client.clear_routed_experts = _boom
    fake._notify_routed_experts_weight_sync("model_a")  # no exception
    fake._clear_routed_experts_for_model("model_a")  # no exception


def _left_padded_attention_mask(full_sequences: list[list[int]]) -> torch.Tensor:
    max_seq_len = max(len(seq) for seq in full_sequences)
    return torch.tensor([[0] * (max_seq_len - len(seq)) + [1] * len(seq) for seq in full_sequences])


def test_build_rollout_expert_indices_matches_native_path():
    """Routing packs exactly as the native RL path packs it: one segment per
    sample, captured rows first, dummy routes for the uncaptured last token."""
    # Two samples of differing length; routing has one fewer row than the full
    # sequence (the last token has no routing), matching the inference engine.
    full_sequences = [[10, 11, 12, 13], [20, 21]]
    routing_a = (np.arange(3 * 2 * 3).reshape(3, 2, 3) % 7).astype(np.uint8)
    routing_b = np.full((1, 2, 3), 5, dtype=np.uint8)
    attention_mask = _left_padded_attention_mask(full_sequences)

    packed, router_padding_mask = _build_rollout_expert_indices(full_sequences, [routing_a, routing_b], attention_mask)

    # Same inputs through the native trainer's collation + mask.
    native = convert_prompts_responses_to_batch_tensors(
        0,
        [seq[:1] for seq in full_sequences],
        [seq[1:] for seq in full_sequences],
        [[0.0] * (len(seq) - 1) for seq in full_sequences],
        [[1] * (len(seq) - 1) for seq in full_sequences],
        rollout_expert_indices=[routing_a, routing_b],
    )
    native_packed, native_attention_mask = native[6], native[1]
    assert torch.equal(attention_mask, native_attention_mask)
    assert torch.equal(packed.values, native_packed.values)
    assert torch.equal(packed.cu_seqlens, native_packed.cu_seqlens)
    assert torch.equal(router_padding_mask, make_router_padding_mask(native_attention_mask, [3, 1]))

    assert packed.values.dtype == torch.uint8
    assert packed.cu_seqlens.tolist() == [0, 4, 6]
    np.testing.assert_array_equal(packed.segment(0)[:3].numpy(), routing_a)
    # Only real tokens that have a captured route are replayed.
    assert router_padding_mask.tolist() == [[False, False, False, True], [True, True, False, True]]


def test_build_rollout_expert_indices_requires_routing_for_every_sample():
    """Megatron replays a whole batch, so one sample without routing means no replay."""
    full_sequences = [[1, 2, 3], [4, 5, 6]]
    attention_mask = _left_padded_attention_mask(full_sequences)
    routing = np.zeros((2, 1, 2), dtype=np.uint8)
    assert _build_rollout_expert_indices(full_sequences, None, attention_mask) is None
    assert _build_rollout_expert_indices(full_sequences, [routing, None], attention_mask) is None
    assert _build_rollout_expert_indices(full_sequences, [routing, routing], attention_mask) is not None


def _sample_input(**kwargs) -> types.SampleInput:
    return types.SampleInput(
        base_model=BASE_MODEL,
        prompt=types.ModelInput(chunks=[types.EncodedTextChunk(tokens=[1, 2, 3])]),
        sampling_params=types.SamplingParams(temperature=1.0, max_tokens=4, seed=0),
        num_samples=1,
        checkpoint_id="",
        prompt_logprobs=False,
        **kwargs,
    )


class _SpyClient:
    def __init__(self):
        self.payloads = []

    async def sample(self, request_payload):
        self.payloads.append(request_payload)
        return {}

    async def aclose(self):
        pass


def test_stash_staleness_config_is_declared_not_forwarded():
    """The staleness knob is a declared backend-config field, so it must not leak into model_extra.

    ``_build_skyrl_train_config`` forwards ``model_extra`` as SkyRL-Train config
    overrides; a leaked key would be applied to SkyRLTrainConfig and error out.
    """
    overrides = skyrl_train_backend.MegatronBackendOverrides(
        routed_experts_stash_max_staleness=3,
        **{"trainer.micro_train_batch_size_per_gpu": 2},
    )
    assert overrides.routed_experts_stash_max_staleness == 3
    assert "routed_experts_stash_max_staleness" not in overrides.model_extra
    assert overrides.model_extra.get("trainer.micro_train_batch_size_per_gpu") == 2
    # Default preserved when unset; staleness must be non-negative.
    assert skyrl_train_backend.MegatronBackendOverrides().routed_experts_stash_max_staleness == 1
    with pytest.raises(Exception):
        skyrl_train_backend.MegatronBackendOverrides(routed_experts_stash_max_staleness=-1)


@pytest.mark.parametrize("replay_enabled", [True, False])
def test_sample_requests_stash_gated_on_replay(monkeypatch, replay_enabled):
    """The sample body sets stash_routed_experts iff R3 is enabled on the backend."""
    monkeypatch.setattr(skyrl_train_backend, "resolve_policy_model_name", lambda cfg: BASE_MODEL)

    spy = _SpyClient()
    fake_self = SimpleNamespace(
        _cfg=SkyRLTrainConfig(),
        _base_lora_signature=None,
        _model_ids_to_role={},
        _inference_engine_client=spy,
        _router_replay_enabled=lambda: replay_enabled,
        _aggregate_sample_results=lambda prepared_batch, outputs: {},
    )
    fake_self._resolve_inference_model_name = SkyRLTrainBackend._resolve_inference_model_name.__get__(fake_self)
    sample_async = SkyRLTrainBackend._sample_with_remote_client_async

    batch = prepare_sample_batch({"req": ("", _sample_input())})
    asyncio.run(sample_async(fake_self, batch, close_client=True))

    assert len(spy.payloads) == 1
    assert spy.payloads[0]["json"]["stash_routed_experts"] is replay_enabled

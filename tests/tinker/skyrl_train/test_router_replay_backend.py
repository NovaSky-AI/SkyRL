"""CPU tests for the SkyRL-Train backend side of Rollout Routing Replay (R3).

Routing is stashed on the vLLM servers at sample time and forward_backward
pulls it back by digest (see routed_experts_stash.py); the backend holds no
routing state and nothing is exposed through the client-facing Tinker types.
Covers the digest fetch (dedupe, dtype narrowing, shape validation,
live-router fallback), stash-key model-name resolution, sample-time gating,
the lifecycle fan-outs, and the replay tensor assembly. No GPU needed. Run:
  uv run --extra dev --extra fsdp pytest tests/tinker/skyrl_train/test_router_replay_backend.py
"""

from __future__ import annotations

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

_build_rollout_expert_indices = skyrl_train_backend._build_rollout_expert_indices
_narrow_routing_dtype = skyrl_train_backend._narrow_routing_dtype
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
    monkeypatch.setattr(skyrl_train_backend, "resolve_policy_model_name", lambda cfg: BASE_MODEL)

    # Multi-LoRA: the adapter registered on vLLM IS the Tinker model_id.
    fake = _bind(_fetch_backend(lora=True, model_ids_to_role={"model_a": "policy"}))
    assert fake._resolve_inference_model_name("model_a") == "model_a"
    # Unknown / empty ids and non-LoRA setups fall back to the policy name.
    assert fake._resolve_inference_model_name("unknown") == BASE_MODEL
    assert fake._resolve_inference_model_name("") == BASE_MODEL
    assert _bind(_fetch_backend(lora=False))._resolve_inference_model_name("model_a") == BASE_MODEL


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


def test_build_rollout_expert_indices_left_pads_and_aligns():
    # Two samples of differing length; routing has one fewer row than the full
    # sequence (the last token has no routing), matching the inference engine.
    full_sequences = [[10, 11, 12, 13], [20, 21]]  # lens 4 and 2
    num_layers, topk = 2, 3
    routing_a = np.array([[[t] * topk for _ in range(num_layers)] for t in range(3)], dtype=np.int32)
    routing_b = np.array([[[100 + t] * topk for _ in range(num_layers)] for t in range(1)], dtype=np.int32)
    max_seq_len = 4

    tensor = _build_rollout_expert_indices(full_sequences, [routing_a, routing_b], max_seq_len)
    assert tensor is not None
    assert tuple(tensor.shape) == (2, max_seq_len, num_layers, topk)

    # Sample A: no left pad; routing at positions [0,1,2]; last position (3) zero.
    assert tensor[0, 2, 1, 2].item() == 2
    assert tensor[0, 3].sum().item() == 0
    # Sample B: left pad of 2; routing lands at position 2; last position (3) zero.
    assert tensor[1, 0].sum().item() == 0
    assert tensor[1, 1].sum().item() == 0
    assert tensor[1, 2, 0, 0].item() == 100
    assert tensor[1, 3].sum().item() == 0


def test_build_rollout_expert_indices_none_when_absent():
    assert _build_rollout_expert_indices([[1, 2]], None, 2) is None
    assert _build_rollout_expert_indices([[1, 2]], [None], 2) is None


def test_build_rollout_expert_indices_downcasts_dtype():
    full_sequences = [[1, 2, 3]]
    small = np.ones((2, 1, 2), dtype=np.int32)  # expert ids < 256 -> uint8
    assert _build_rollout_expert_indices(full_sequences, [small], 3).dtype == torch.uint8
    big = np.full((2, 1, 2), 300, dtype=np.int32)  # 256..2**15 -> int16
    assert _build_rollout_expert_indices(full_sequences, [big], 3).dtype == torch.int16


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
        _cfg=None,
        _base_lora_signature=None,
        _model_ids_to_role={},
        _inference_engine_client=spy,
        _router_replay_enabled=lambda: replay_enabled,
        _aggregate_sample_results=lambda prepared_batch, outputs: {},
    )
    fake_self._resolve_inference_model_name = SkyRLTrainBackend._resolve_inference_model_name.__get__(fake_self)
    sample = SkyRLTrainBackend._sample_with_remote_client

    batch = prepare_sample_batch({"req": ("", _sample_input())})
    sample(fake_self, batch)

    assert len(spy.payloads) == 1
    assert spy.payloads[0]["json"]["stash_routed_experts"] is replay_enabled

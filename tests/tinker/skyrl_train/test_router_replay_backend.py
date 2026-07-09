"""CPU tests for the SkyRL-Train backend side of Rollout Routing Replay (R3).

R3 stays a server-side launch flag: routing captured at sample time is cached on
the backend keyed by the full sampled token sequence, and forward_backward looks
it up by the training sequence. Nothing is exposed through the client-facing
Tinker types. These tests cover that cache roundtrip, the left-padded tensor
assembly, and the sample-time gating — no GPU or inference engine needed. Run:
  uv run --extra dev --extra fsdp pytest tests/tinker/skyrl_train/test_router_replay_backend.py
"""

from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

import pytest

# Skip cleanly if the SkyRL-Train backend (ray/vllm) can't be imported.
skyrl_train_backend = pytest.importorskip("skyrl.backends.skyrl_train_backend")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from skyrl.tinker import types  # noqa: E402
from skyrl.tinker.engine import prepare_sample_batch  # noqa: E402

_build_rollout_expert_indices = skyrl_train_backend._build_rollout_expert_indices
SkyRLTrainBackend = skyrl_train_backend.SkyRLTrainBackend

BASE_MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"


def _routing(seq_len: int, num_layers: int, topk: int) -> np.ndarray:
    """Routing array whose entry (t, layer, k) encodes t*1000 + layer*10 + k."""
    return np.array(
        [[[t * 1000 + layer * 10 + k for k in range(topk)] for layer in range(num_layers)] for t in range(seq_len)],
        dtype=np.int32,
    )


def _cache_backend():
    """A stand-in exposing just the cache attributes the R3 helpers touch."""
    return SimpleNamespace(_routed_experts_cache=OrderedDict(), _routed_experts_cache_cap=4)


def test_store_and_lookup_roundtrip_by_full_sequence():
    fake = _cache_backend()
    prompt, response = [10, 11, 12], [20, 21]
    # Routing covers every forwarded token (prompt + response minus last): 4 rows.
    routing = _routing(seq_len=4, num_layers=2, topk=3)
    SkyRLTrainBackend._store_routed_experts(fake, prompt, response, routing)

    # forward_backward reconstructs the key as prompt + response.
    key = tuple(prompt + response)
    assert key in fake._routed_experts_cache
    got = fake._routed_experts_cache[key]
    assert got.shape == (4, 2, 3)
    assert int(got[3, 1, 2]) == 3 * 1000 + 1 * 10 + 2


def test_store_downcasts_dtype_and_ignores_non_3d():
    fake = _cache_backend()
    SkyRLTrainBackend._store_routed_experts(fake, [1], [2], _routing(1, 1, 2))  # small ids -> uint8
    assert next(iter(fake._routed_experts_cache.values())).dtype == np.uint8

    fake2 = _cache_backend()
    big = _routing(1, 1, 2) + 300  # > 255 -> int16
    SkyRLTrainBackend._store_routed_experts(fake2, [1], [2], big)
    assert next(iter(fake2._routed_experts_cache.values())).dtype == np.int16

    fake3 = _cache_backend()
    SkyRLTrainBackend._store_routed_experts(fake3, [1], [2], np.zeros((3, 4)))  # 2-D -> skipped
    assert len(fake3._routed_experts_cache) == 0


def test_cache_is_bounded_fifo():
    fake = _cache_backend()  # cap = 4
    for i in range(6):
        SkyRLTrainBackend._store_routed_experts(fake, [i], [i + 100], _routing(1, 1, 2))
    assert len(fake._routed_experts_cache) == 4
    # Oldest two keys evicted.
    assert (0, 100) not in fake._routed_experts_cache
    assert (5, 105) in fake._routed_experts_cache


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


def test_end_to_end_cache_to_tensor_alignment():
    """Store at sample time, then rebuild the training tensor via a cache lookup."""
    fake = _cache_backend()
    prompt, response = [10, 11, 12], [20, 21]  # full sampled sequence len 5
    routing = _routing(seq_len=4, num_layers=2, topk=3)  # forwarded tokens = 4
    SkyRLTrainBackend._store_routed_experts(fake, prompt, response, routing)

    # forward_backward: full_sequences[i] == prompt + response.
    full_sequences = [prompt + response]
    per_sample = [fake._routed_experts_cache.get(tuple(fs)) for fs in full_sequences]
    tensor = _build_rollout_expert_indices(full_sequences, per_sample, max_seq_len=5)
    assert tuple(tensor.shape) == (1, 5, 2, 3)
    # Routing occupies positions [0,4); the final token has none.
    assert tensor[0, 3, 1, 2].item() == 3 * 1000 + 1 * 10 + 2
    assert tensor[0, 4].sum().item() == 0


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


def test_routed_experts_cache_cap_is_declared_not_forwarded():
    """The cap is a declared backend-config field, so it must not leak into model_extra.

    ``_build_skyrl_train_config`` forwards ``model_extra`` as SkyRL-Train config
    overrides; a leaked key would be applied to SkyRLTrainConfig and error out.
    """
    overrides = skyrl_train_backend.MegatronBackendOverrides(
        routed_experts_cache_cap=123, **{"trainer.micro_train_batch_size_per_gpu": 2}
    )
    assert overrides.routed_experts_cache_cap == 123
    assert "routed_experts_cache_cap" not in overrides.model_extra
    assert overrides.model_extra.get("trainer.micro_train_batch_size_per_gpu") == 2
    # Default preserved when unset; must be positive.
    assert skyrl_train_backend.MegatronBackendOverrides().routed_experts_cache_cap == 8192
    with pytest.raises(Exception):
        skyrl_train_backend.MegatronBackendOverrides(routed_experts_cache_cap=0)


def test_backend_honors_configured_cache_cap():
    """The configured cap reaches the backend's live cache bound (no ray init in __init__)."""
    backend = SkyRLTrainBackend(BASE_MODEL, skyrl_train_backend.MegatronBackendOverrides(routed_experts_cache_cap=321))
    assert backend._routed_experts_cache_cap == 321
    assert (
        SkyRLTrainBackend(BASE_MODEL, skyrl_train_backend.MegatronBackendOverrides())._routed_experts_cache_cap == 8192
    )


@pytest.mark.parametrize("replay_enabled", [True, False])
def test_sample_requests_routed_experts_gated_on_replay(monkeypatch, replay_enabled):
    """The sample body sets return_routed_experts iff R3 is enabled on the backend."""
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
    sample = SkyRLTrainBackend._sample_with_remote_client

    batch = prepare_sample_batch({"req": ("", _sample_input())})
    sample(fake_self, batch)

    assert len(spy.payloads) == 1
    assert spy.payloads[0]["json"]["return_routed_experts"] is replay_enabled

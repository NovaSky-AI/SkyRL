"""CPU tests for the SkyRL-Train backend side of Rollout Routing Replay (R3).

R3 stays a server-side launch flag: routing captured at sample time is cached on
the backend keyed by the full sampled token sequence, and forward_backward looks
it up by the training sequence. Nothing is exposed through the client-facing
Tinker types. These tests cover that cache roundtrip, the lifecycle eviction
(consumed entries at weight sync, never-trained entries after the staleness
window), the left-padded tensor assembly, the sample-time gating, and the
on-disk spool fallback used when sampling is forwarded by the API process
(non-colocated mode) — no GPU or inference engine needed. Run:
  uv run --extra dev --extra fsdp pytest tests/tinker/skyrl_train/test_router_replay_backend.py
"""

from __future__ import annotations

import io
from types import SimpleNamespace

import pytest

# Skip cleanly if the SkyRL-Train backend (ray/vllm) can't be imported.
skyrl_train_backend = pytest.importorskip("skyrl.backends.skyrl_train_backend")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from skyrl.tinker import types  # noqa: E402
from skyrl.tinker.engine import prepare_sample_batch  # noqa: E402
from skyrl.tinker.routed_experts_spool import RoutedExpertsSpool  # noqa: E402

_build_rollout_expert_indices = skyrl_train_backend._build_rollout_expert_indices
_routing_cache_key = skyrl_train_backend._routing_cache_key
RoutedExpertsCache = skyrl_train_backend.RoutedExpertsCache
SkyRLTrainBackend = skyrl_train_backend.SkyRLTrainBackend

BASE_MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"


def _routing(seq_len: int, num_layers: int, topk: int) -> np.ndarray:
    """Routing array whose entry (t, layer, k) encodes t*1000 + layer*10 + k."""
    return np.array(
        [[[t * 1000 + layer * 10 + k for k in range(topk)] for layer in range(num_layers)] for t in range(seq_len)],
        dtype=np.int32,
    )


def _cache_backend():
    """A stand-in exposing just the cache attribute the R3 helpers touch."""
    return SimpleNamespace(_routed_experts_cache=RoutedExpertsCache(cap=4), _routed_experts_spool=None)


def _spool_backend(tmp_path, cap: int = 64):
    """A stand-in with both the in-memory cache and an on-disk spool wired."""
    return SimpleNamespace(
        _routed_experts_cache=RoutedExpertsCache(cap=cap),
        _routed_experts_spool=RoutedExpertsSpool(str(tmp_path / "spool")),
    )


def _npy_bytes(arr: np.ndarray) -> bytes:
    """Serialize an array the way the API-side spool writer receives it (.npy)."""
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def _peek(cache: "RoutedExpertsCache", model_id: str, token_ids: list[int]) -> np.ndarray | None:
    """Presence check that does NOT mark the entry consumed (unlike get)."""
    entry = cache._entries.get(_routing_cache_key(model_id, token_ids))
    return None if entry is None else entry.routing


def test_store_and_lookup_roundtrip_by_model_and_sequence():
    fake = _cache_backend()
    model_id, prompt, response = "model_a", [10, 11, 12], [20, 21]
    # Routing covers every forwarded token (prompt + response minus last): 4 rows.
    routing = _routing(seq_len=4, num_layers=2, topk=3)
    SkyRLTrainBackend._store_routed_experts(fake, model_id, prompt, response, routing)

    # forward_backward reconstructs the key from (model_id, prompt + response).
    got = fake._routed_experts_cache.get(model_id, prompt + response)
    assert got is not None
    assert got.shape == (4, 2, 3)
    assert int(got[3, 1, 2]) == 3 * 1000 + 1 * 10 + 2


def test_key_is_compact_and_hashed():
    """The key is (model_id, 16-byte digest), not a per-token tuple."""
    key = _routing_cache_key("model_a", list(range(4096)))
    assert isinstance(key, tuple) and len(key) == 2
    assert key[0] == "model_a"
    assert isinstance(key[1], bytes) and len(key[1]) == 16
    # Digest is exact over order + values.
    assert key[1] == _routing_cache_key("model_a", list(range(4096)))[1]
    assert key[1] != _routing_cache_key("model_a", list(range(4096))[::-1])[1]


def test_multi_tenant_same_tokens_do_not_collide():
    """Two adapters that sample identical tokens must not share a cache entry."""
    fake = _cache_backend()
    prompt, response = [10, 11, 12], [20, 21]
    routing_a = _routing(seq_len=4, num_layers=2, topk=3)
    routing_b = routing_a + 1
    SkyRLTrainBackend._store_routed_experts(fake, "model_a", prompt, response, routing_a)
    SkyRLTrainBackend._store_routed_experts(fake, "model_b", prompt, response, routing_b)

    assert len(fake._routed_experts_cache) == 2
    got_a = fake._routed_experts_cache.get("model_a", prompt + response)
    got_b = fake._routed_experts_cache.get("model_b", prompt + response)
    assert int(got_a[0, 0, 0]) + 1 == int(got_b[0, 0, 0])


def test_store_downcasts_dtype_and_ignores_non_3d():
    fake = _cache_backend()
    SkyRLTrainBackend._store_routed_experts(fake, "m", [1], [2], _routing(1, 1, 2))  # small ids -> uint8
    assert fake._routed_experts_cache.get("m", [1, 2]).dtype == np.uint8

    fake2 = _cache_backend()
    big = _routing(1, 1, 2) + 300  # > 255 -> int16
    SkyRLTrainBackend._store_routed_experts(fake2, "m", [1], [2], big)
    assert fake2._routed_experts_cache.get("m", [1, 2]).dtype == np.int16

    fake3 = _cache_backend()
    SkyRLTrainBackend._store_routed_experts(fake3, "m", [1], [2], np.zeros((3, 4)))  # 2-D -> skipped
    assert len(fake3._routed_experts_cache) == 0


def test_cache_is_bounded_fifo():
    fake = _cache_backend()  # cap = 4
    for i in range(6):
        SkyRLTrainBackend._store_routed_experts(fake, "m", [i], [i + 100], _routing(1, 1, 2))
    assert len(fake._routed_experts_cache) == 4
    # Oldest two keys evicted.
    assert fake._routed_experts_cache.get("m", [0, 100]) is None
    assert fake._routed_experts_cache.get("m", [5, 105]) is not None


def test_consumed_entries_evicted_at_next_weight_sync():
    """Routing that made it through the trainer is dropped once the model syncs weights."""
    cache = RoutedExpertsCache(cap=64)
    cache.put("m", [1, 2], _routing(1, 1, 2))
    cache.put("m", [3, 4], _routing(1, 1, 2))

    # forward_backward replays [1, 2]; [3, 4] was filtered out client-side.
    assert cache.get("m", [1, 2]) is not None
    # Consumption alone must not evict: minibatches/epochs re-read the entry.
    assert cache.get("m", [1, 2]) is not None

    cache.on_weight_sync("m")
    assert _peek(cache, "m", [1, 2]) is None  # trained -> evicted
    assert _peek(cache, "m", [3, 4]) is not None  # not yet stale -> kept


def test_never_trained_entries_expire_after_staleness_window():
    """Entries the trainer never read (client-side filtered) expire after max_staleness + 1 syncs."""
    cache = RoutedExpertsCache(cap=64, max_staleness=1)
    cache.put("m", [1, 2], _routing(1, 1, 2))

    cache.on_weight_sync("m")  # first sync after storage: within staleness window
    assert _peek(cache, "m", [1, 2]) is not None
    cache.on_weight_sync("m")  # second sync: abandoned -> evicted
    assert _peek(cache, "m", [1, 2]) is None


def test_zero_staleness_expires_unconsumed_entries_at_first_sync():
    cache = RoutedExpertsCache(cap=64, max_staleness=0)
    cache.put("m", [1, 2], _routing(1, 1, 2))
    cache.on_weight_sync("m")
    assert len(cache) == 0


def test_entries_stored_after_a_sync_age_from_that_sync():
    """The staleness clock starts at the entry's storage generation, not at zero."""
    cache = RoutedExpertsCache(cap=64, max_staleness=1)
    cache.on_weight_sync("m")  # gen 1
    cache.put("m", [1, 2], _routing(1, 1, 2))  # stored at gen 1

    cache.on_weight_sync("m")  # gen 2: entry one sync old -> kept
    assert _peek(cache, "m", [1, 2]) is not None
    cache.on_weight_sync("m")  # gen 3: entry two syncs old -> evicted
    assert _peek(cache, "m", [1, 2]) is None


def test_weight_sync_is_scoped_per_model():
    """Syncs for one adapter must not consume or age another adapter's entries."""
    cache = RoutedExpertsCache(cap=64, max_staleness=1)
    cache.put("model_a", [1, 2], _routing(1, 1, 2))
    cache.put("model_b", [1, 2], _routing(1, 1, 2))
    assert cache.get("model_a", [1, 2]) is not None  # trained on model_a only

    for _ in range(3):
        cache.on_weight_sync("model_a")

    assert _peek(cache, "model_a", [1, 2]) is None
    assert _peek(cache, "model_b", [1, 2]) is not None


def test_restore_resets_consumption_and_age():
    """Re-sampling the same sequence refreshes the entry for the new rollout round."""
    cache = RoutedExpertsCache(cap=64, max_staleness=1)
    cache.put("m", [1, 2], _routing(1, 1, 2))
    assert cache.get("m", [1, 2]) is not None  # consumed in round 1
    cache.put("m", [1, 2], _routing(1, 1, 2))  # sampled again before the sync

    cache.on_weight_sync("m")
    assert _peek(cache, "m", [1, 2]) is not None  # fresh entry survives


def test_evict_model_and_clear():
    cache = RoutedExpertsCache(cap=64)
    cache.put("model_a", [1, 2], _routing(1, 1, 2))
    cache.put("model_b", [1, 2], _routing(1, 1, 2))

    assert cache.evict_model("model_a") == 1
    assert _peek(cache, "model_a", [1, 2]) is None
    assert _peek(cache, "model_b", [1, 2]) is not None

    cache.clear()
    assert len(cache) == 0
    assert cache.nbytes == 0


def test_nbytes_tracks_stored_routing():
    cache = RoutedExpertsCache(cap=64)
    routing = _routing(4, 2, 3)
    cache.put("m", [1, 2], routing)
    assert cache.nbytes == routing.nbytes
    cache.get("m", [1, 2])
    cache.on_weight_sync("m")
    assert cache.nbytes == 0


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
    model_id, prompt, response = "model_a", [10, 11, 12], [20, 21]  # full sampled sequence len 5
    routing = _routing(seq_len=4, num_layers=2, topk=3)  # forwarded tokens = 4
    SkyRLTrainBackend._store_routed_experts(fake, model_id, prompt, response, routing)

    # forward_backward: full_sequences[i] == prompt + response, keyed by model_id.
    full_sequences = [prompt + response]
    per_sample = [fake._routed_experts_cache.get(model_id, fs) for fs in full_sequences]
    tensor = _build_rollout_expert_indices(full_sequences, per_sample, max_seq_len=5)
    assert tuple(tensor.shape) == (1, 5, 2, 3)
    # Routing occupies positions [0,4); the final token has none.
    assert tensor[0, 3, 1, 2].item() == 3 * 1000 + 1 * 10 + 2
    assert tensor[0, 4].sum().item() == 0


def test_lookup_falls_back_to_spool_and_promotes_to_cache(tmp_path):
    """Routing spooled by the API process (non-colocated) is found by the
    training-sequence lookup, promoted into the in-memory cache for later
    epochs, and evicted at the model's next weight sync like any consumed entry."""
    fake = _spool_backend(tmp_path)
    model_id, seq = "model_a", [10, 11, 12, 20, 21]
    routing = _routing(seq_len=4, num_layers=2, topk=3)
    # API side: raw .npy bytes keyed by the full sampled sequence.
    fake._routed_experts_spool.write(model_id, seq, _npy_bytes(routing))

    lookup = SkyRLTrainBackend._lookup_rollout_routing
    got = lookup(fake, model_id, seq)
    assert got is not None and got.shape == (4, 2, 3)
    np.testing.assert_array_equal(np.asarray(got, dtype=np.int64), routing.astype(np.int64))

    # The spool file was consumed, but later epochs replay from the cache.
    assert fake._routed_experts_spool.consume(model_id, seq) is None
    assert lookup(fake, model_id, seq) is not None

    # Promoted entries follow the cache's consumed-at-sync eviction.
    fake._routed_experts_cache.on_weight_sync(model_id)
    assert _peek(fake._routed_experts_cache, model_id, seq) is None
    assert lookup(fake, model_id, seq) is None


def test_lookup_without_spool_is_cache_only():
    """With no spool wired (colocated mode, unit tests) the lookup is exactly the cache."""
    fake = _cache_backend()
    model_id, prompt, response = "model_a", [10, 11, 12], [20, 21]
    SkyRLTrainBackend._store_routed_experts(fake, model_id, prompt, response, _routing(4, 2, 3))

    lookup = SkyRLTrainBackend._lookup_rollout_routing
    assert lookup(fake, model_id, prompt + response) is not None
    assert lookup(fake, model_id, [99]) is None


def test_spooled_routing_is_narrowed_like_local_capture(tmp_path):
    """Spooled int32 routing is downcast on consume exactly like locally-captured routing."""
    fake = _spool_backend(tmp_path)
    seq = [1, 2, 3]
    fake._routed_experts_spool.write("m", seq, _npy_bytes(np.ones((2, 1, 2), dtype=np.int32)))
    assert SkyRLTrainBackend._lookup_rollout_routing(fake, "m", seq).dtype == np.uint8

    big_seq = [4, 5, 6]
    fake._routed_experts_spool.write("m", big_seq, _npy_bytes(np.full((2, 1, 2), 300, dtype=np.int32)))
    assert SkyRLTrainBackend._lookup_rollout_routing(fake, "m", big_seq).dtype == np.int16


def test_spooled_routing_rejects_non_3d(tmp_path):
    fake = _spool_backend(tmp_path)
    seq = [1, 2, 3]
    fake._routed_experts_spool.write("m", seq, _npy_bytes(np.zeros((3, 4), dtype=np.int32)))
    assert SkyRLTrainBackend._lookup_rollout_routing(fake, "m", seq) is None
    assert len(fake._routed_experts_cache) == 0


def test_set_spool_dir_wipes_leftovers_and_none_disables(tmp_path):
    """Backend startup wipes files left by a previous server run; None turns the fallback off."""
    spool_dir = tmp_path / "spool"
    leftover = RoutedExpertsSpool(str(spool_dir))
    leftover.write("dead_model", [1, 2], _npy_bytes(_routing(1, 1, 2)))

    fake = SimpleNamespace(_routed_experts_spool=None, _routed_experts_cache=RoutedExpertsCache(cap=4))
    SkyRLTrainBackend.set_routed_experts_spool_dir(fake, str(spool_dir))
    assert fake._routed_experts_spool is not None
    assert not spool_dir.exists()

    SkyRLTrainBackend.set_routed_experts_spool_dir(fake, None)
    assert fake._routed_experts_spool is None


def test_end_to_end_spool_to_tensor_alignment(tmp_path):
    """Non-colocated flow: routing spooled by the API process is looked up by the
    training sequence and assembled into the same left-padded replay tensor."""
    fake = _spool_backend(tmp_path)
    model_id, prompt, response = "model_a", [10, 11, 12], [20, 21]
    routing = _routing(seq_len=4, num_layers=2, topk=3)  # forwarded tokens = 4
    fake._routed_experts_spool.write(model_id, prompt + response, _npy_bytes(routing))

    full_sequences = [prompt + response]
    per_sample = [SkyRLTrainBackend._lookup_rollout_routing(fake, model_id, fs) for fs in full_sequences]
    tensor = _build_rollout_expert_indices(full_sequences, per_sample, max_seq_len=5)
    assert tuple(tensor.shape) == (1, 5, 2, 3)
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


def test_routed_experts_cache_config_is_declared_not_forwarded():
    """Cache knobs are declared backend-config fields, so they must not leak into model_extra.

    ``_build_skyrl_train_config`` forwards ``model_extra`` as SkyRL-Train config
    overrides; a leaked key would be applied to SkyRLTrainConfig and error out.
    """
    overrides = skyrl_train_backend.MegatronBackendOverrides(
        routed_experts_cache_cap=123,
        routed_experts_cache_max_staleness=3,
        **{"trainer.micro_train_batch_size_per_gpu": 2},
    )
    assert overrides.routed_experts_cache_cap == 123
    assert overrides.routed_experts_cache_max_staleness == 3
    assert "routed_experts_cache_cap" not in overrides.model_extra
    assert "routed_experts_cache_max_staleness" not in overrides.model_extra
    assert overrides.model_extra.get("trainer.micro_train_batch_size_per_gpu") == 2
    # Defaults preserved when unset; cap must be positive, staleness non-negative.
    defaults = skyrl_train_backend.MegatronBackendOverrides()
    assert defaults.routed_experts_cache_cap == 8192
    assert defaults.routed_experts_cache_max_staleness == 1
    with pytest.raises(Exception):
        skyrl_train_backend.MegatronBackendOverrides(routed_experts_cache_cap=0)
    with pytest.raises(Exception):
        skyrl_train_backend.MegatronBackendOverrides(routed_experts_cache_max_staleness=-1)


def test_backend_honors_configured_cache_settings():
    """The configured knobs reach the backend's live cache (no ray init in __init__)."""
    backend = SkyRLTrainBackend(
        BASE_MODEL,
        skyrl_train_backend.MegatronBackendOverrides(
            routed_experts_cache_cap=321, routed_experts_cache_max_staleness=2
        ),
    )
    assert backend._routed_experts_cache.cap == 321
    assert backend._routed_experts_cache.max_staleness == 2
    default_cache = SkyRLTrainBackend(BASE_MODEL, skyrl_train_backend.MegatronBackendOverrides())._routed_experts_cache
    assert default_cache.cap == 8192
    assert default_cache.max_staleness == 1


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

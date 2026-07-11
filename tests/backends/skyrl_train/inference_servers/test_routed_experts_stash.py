"""CPU tests for the server-side R3 routing stash (`routed_experts_stash.py`):
key/digest, bulk lookup with consumption, weight-sync eviction, model
eviction, the FIFO entry cap, and the npz/base64 transport helpers. No vLLM
or GPU needed. Run:
  uv run --extra dev --extra fsdp pytest tests/backends/skyrl_train/inference_servers/test_routed_experts_stash.py
"""

from __future__ import annotations

import base64
import io

import numpy as np

from skyrl.backends.skyrl_train.inference_servers.routed_experts_stash import (
    RoutedExpertsStash,
    decode_routed_experts_b64,
    dump_arrays_npz,
    load_arrays_npz,
    sequence_digest,
)


def _routing(seq_len: int = 4, num_layers: int = 2, topk: int = 3) -> np.ndarray:
    return np.arange(seq_len * num_layers * topk, dtype=np.uint8).reshape(seq_len, num_layers, topk)


def _peek(stash: RoutedExpertsStash, model: str, token_ids: list[int]) -> bool:
    """Presence check that does NOT mark the entry consumed (unlike get_many)."""
    return (model, sequence_digest(token_ids)) in stash._entries


def test_put_get_many_roundtrip_never_deletes():
    stash = RoutedExpertsStash()
    routing = _routing()
    tokens = [10, 11, 12, 20, 21]
    stash.put("model_a", tokens, routing)

    # The digest key is order-exact: a permuted sequence must miss.
    digest_hex = sequence_digest(tokens).hex()
    hits = stash.get_many("model_a", [digest_hex, sequence_digest(tokens[::-1]).hex()])
    assert set(hits) == {digest_hex}
    np.testing.assert_array_equal(hits[digest_hex], routing)

    # Fetches never delete: multi-epoch training fetches the same digests again.
    assert stash.get_many("model_a", [digest_hex])
    assert len(stash) == 1


def test_consumed_entries_deleted_at_next_weight_sync():
    """Routing the trainer fetched is deleted at the model's very next weight
    sync — not after the staleness window — even when fetched repeatedly
    (multi-epoch) beforehand."""
    stash = RoutedExpertsStash()
    stash.put("m", [1, 2], _routing())
    stash.put("m", [3, 4], _routing())
    digest_trained = sequence_digest([1, 2]).hex()

    # forward_backward fetches [1, 2] across two epochs; [3, 4] was filtered
    # out client-side and never fetched.
    assert stash.get_many("m", [digest_trained])
    assert stash.get_many("m", [digest_trained])
    assert len(stash) == 2  # consumption alone must not delete (epochs re-fetch)

    assert stash.on_weight_sync("m", max_staleness=1) == 1
    assert not _peek(stash, "m", [1, 2])  # trained -> deleted at the next sync
    assert _peek(stash, "m", [3, 4])  # never trained, not yet stale -> kept


def test_put_same_key_refreshes_entry_nbytes_and_consumption():
    """Re-stashing the same sequence replaces the entry (with correct byte
    accounting) and starts a fresh, unconsumed one for the new round."""
    stash = RoutedExpertsStash()
    stash.put("m", [1, 2], _routing())
    assert stash.get_many("m", [sequence_digest([1, 2]).hex()])  # consumed in round 1

    newer = _routing(seq_len=8)
    stash.put("m", [1, 2], newer)  # sampled again before the sync
    assert len(stash) == 1
    assert stash.nbytes == newer.nbytes

    assert stash.on_weight_sync("m", max_staleness=1) == 0
    assert _peek(stash, "m", [1, 2])  # fresh entry survives the sync


def test_keys_are_scoped_per_model():
    """Two adapters that sample identical tokens must not share an entry."""
    stash = RoutedExpertsStash()
    routing_a = _routing()
    routing_b = routing_a + 1
    stash.put("model_a", [1, 2], routing_a)
    stash.put("model_b", [1, 2], routing_b)

    digest_hex = sequence_digest([1, 2]).hex()
    assert int(stash.get_many("model_a", [digest_hex])[digest_hex][0, 0, 0]) + 1 == int(
        stash.get_many("model_b", [digest_hex])[digest_hex][0, 0, 0]
    )


def test_fifo_cap_bounds_entries():
    stash = RoutedExpertsStash(max_entries=4)
    for i in range(6):
        stash.put("m", [i], _routing())
    assert len(stash) == 4
    assert not stash.get_many("m", [sequence_digest([0]).hex()])  # oldest evicted
    assert stash.get_many("m", [sequence_digest([5]).hex()])


def test_weight_sync_drops_never_consumed_entries_past_staleness_window():
    """A never-fetched entry stashed at generation g dies at the sync producing
    generation g + max_staleness + 1 — its rollout round was abandoned."""
    stash = RoutedExpertsStash()
    stash.put("m", [1, 2], _routing())  # stashed at gen 0, never fetched

    assert stash.on_weight_sync("m", max_staleness=1) == 0  # gen 1: within window
    assert _peek(stash, "m", [1, 2])
    assert stash.on_weight_sync("m", max_staleness=1) == 1  # gen 2: 0 <= 2 - 1 - 1
    assert not _peek(stash, "m", [1, 2])
    assert stash.nbytes == 0


def test_entries_stashed_after_a_sync_age_from_that_sync():
    stash = RoutedExpertsStash()
    stash.on_weight_sync("m", max_staleness=1)  # gen 1
    stash.put("m", [1, 2], _routing())  # stashed at gen 1, never fetched

    assert stash.on_weight_sync("m", max_staleness=1) == 0  # gen 2: one sync old
    assert _peek(stash, "m", [1, 2])
    assert stash.on_weight_sync("m", max_staleness=1) == 1  # gen 3: two syncs old
    assert not _peek(stash, "m", [1, 2])


def test_weight_sync_is_scoped_per_model():
    stash = RoutedExpertsStash()
    stash.put("model_a", [1, 2], _routing())
    stash.put("model_b", [1, 2], _routing())

    for _ in range(3):
        stash.on_weight_sync("model_a", max_staleness=1)

    digest_hex = sequence_digest([1, 2]).hex()
    assert not stash.get_many("model_a", [digest_hex])
    assert stash.get_many("model_b", [digest_hex])


def test_evict_model():
    stash = RoutedExpertsStash()
    stash.put("model_a", [1, 2], _routing())
    stash.put("model_a", [3, 4], _routing())
    stash.put("model_b", [1, 2], _routing())

    assert stash.evict_model("model_a") == 2
    assert len(stash) == 1
    assert stash.nbytes == _routing().nbytes
    # The sync generation resets with the model.
    stash.put("model_a", [5, 6], _routing())
    assert stash.on_weight_sync("model_a", max_staleness=0) == 1


def test_npz_transport_roundtrip():
    arrays = {
        sequence_digest([1, 2]).hex(): _routing(),
        sequence_digest([3, 4]).hex(): _routing(seq_len=7).astype(np.uint16),
    }
    payload = dump_arrays_npz(arrays)
    loaded = load_arrays_npz(payload)
    assert set(loaded) == set(arrays)
    for name, arr in arrays.items():
        np.testing.assert_array_equal(loaded[name], arr)
        assert loaded[name].dtype == arr.dtype
    assert load_arrays_npz(dump_arrays_npz({})) == {}


def test_decode_routed_experts_b64_matches_vllm_encoding():
    """Decode the exact transport vLLM's completions endpoint uses (base64 .npy)."""
    arr = _routing()
    buf = io.BytesIO()
    np.save(buf, arr)
    payload = base64.b64encode(buf.getvalue()).decode("ascii")
    np.testing.assert_array_equal(decode_routed_experts_b64(payload), arr)

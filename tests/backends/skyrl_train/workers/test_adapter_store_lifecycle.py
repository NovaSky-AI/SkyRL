"""CPU tests for AdapterStore's register/delete state machine.

Pins the stale-live invariant: after the *current* adapter is deleted, the
live GPU state still mirrors the deleted tenant, so a subsequently created
adapter must be seeded from the pristine slot (and only become live via
swap_to) instead of silently inheriting the deleted tenant's weights.

Tensor plumbing (`_allocate_empty_slot`, `_copy_slot`) is stubbed out — these
tests exercise only the slot bookkeeping, which is what the multi-tenant
warm-runtime lifecycle (delete last tenant, create a new one) relies on.
"""

from __future__ import annotations

import pytest

try:
    from skyrl.backends.skyrl_train.workers.megatron.adapter_store import AdapterStore
except Exception as e:  # noqa: BLE001 — megatron/TE also raise RuntimeError without CUDA libs
    pytest.skip(f"megatron adapter store unavailable: {e}", allow_module_level=True)

SIGNATURE = object()


def _store() -> tuple[AdapterStore, list[tuple[object, object]]]:
    """AdapterStore with tensor plumbing stubbed; returns (store, copy_log)."""
    store = AdapterStore()
    store._signature = SIGNATURE
    store._pristine = object()
    copies: list[tuple[object, object]] = []
    store._allocate_empty_slot = lambda model_chunks, optimizer: object()
    store._copy_slot = lambda src, dst: copies.append((src, dst))
    return store, copies


def test_first_create_treats_live_as_authoritative():
    store, copies = _store()

    store.create("model-a", model_chunks=[], optimizer=None, signature=SIGNATURE)

    assert store.current_id == "model-a"
    assert copies == []


def test_create_after_deleting_current_seeds_from_pristine():
    store, copies = _store()
    store.create("model-a", model_chunks=[], optimizer=None, signature=SIGNATURE)
    store.delete("model-a")

    store.create("model-b", model_chunks=[], optimizer=None, signature=SIGNATURE)

    # Live state still mirrors deleted model-a: model-b's slot must be a
    # pristine copy, and it must not be considered live until swap_to runs.
    assert copies == [(store._pristine, store._slots["model-b"])]
    assert store.current_id is None


def test_delete_of_non_current_adapter_keeps_live_authoritative():
    store, copies = _store()
    store.create("model-a", model_chunks=[], optimizer=None, signature=SIGNATURE)
    store.create("model-b", model_chunks=[], optimizer=None, signature=SIGNATURE)
    copies.clear()

    store.delete("model-b")
    store.create("model-c", model_chunks=[], optimizer=None, signature=SIGNATURE)

    # model-a is still current and live; model-c seeds from pristine as any
    # subsequent registration does.
    assert store.current_id == "model-a"
    assert copies == [(store._pristine, store._slots["model-c"])]

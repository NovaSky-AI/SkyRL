"""CPU tests for handling stub sub-optimizers in Megatron's ChainedOptimizer.

With LoRA targeting only expert linears there are no trainable dense params, so the dense
DistributedOptimizer is a stub (``is_stub_optimizer=True``, ``optimizer=None``). Optimizer
offload/reload must skip it, and dist-checkpoint save/load must leave it out of the chain,
since megatron-core cannot build a stub's ``sharded_state_dict``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.megatron

try:
    from skyrl.backends.skyrl_train.distributed.megatron.megatron_strategy import (
        _without_stub_optimizers,
    )
    from skyrl.backends.skyrl_train.distributed.megatron.megatron_utils import (
        load_megatron_optimizer,
        offload_megatron_optimizer,
    )
except ModuleNotFoundError as e:
    if not e.name or (e.name != "megatron" and not e.name.startswith("megatron.")):
        raise
    pytest.skip(f"megatron unavailable: {e}", allow_module_level=True)


def _stub():
    return SimpleNamespace(is_stub_optimizer=True, optimizer=None)


def _real():
    return SimpleNamespace(is_stub_optimizer=False, optimizer=object())


def test_without_stub_optimizers_drops_stubs_and_restores():
    stub, real = _stub(), _real()
    chained = SimpleNamespace(chained_optimizers=[stub, real])

    with _without_stub_optimizers(chained):
        assert chained.chained_optimizers == [real]
    assert chained.chained_optimizers == [stub, real]

    with pytest.raises(RuntimeError):
        with _without_stub_optimizers(chained):
            raise RuntimeError("save failed")
    assert chained.chained_optimizers == [stub, real]


def test_without_stub_optimizers_is_noop_without_stubs():
    chain = [_real(), _real()]
    chained = SimpleNamespace(chained_optimizers=chain)
    with _without_stub_optimizers(chained):
        assert chained.chained_optimizers is chain

    plain = _real()  # not a ChainedOptimizer
    with _without_stub_optimizers(plain):
        pass
    assert not hasattr(plain, "chained_optimizers")


def test_offload_and_load_skip_stub_optimizer():
    # A stub has no torch optimizer or param shards; touching either would raise.
    offload_megatron_optimizer(_stub())
    load_megatron_optimizer(_stub())
